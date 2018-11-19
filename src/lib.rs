#![feature(asm)]

/// SoftMMU implementation for vectorized JITs
///
/// Design:
///
/// In this design there are effectively 2 parts.
/// A page table which is walked to find the backing memory which contains
/// both the memory and permissions for memory.
///
/// And the backing permissions and memory themselves.
///
/// The page tables actually have no permissions. If a corresponding entry
/// is NULL then that page is not mapped and the page walk is terminated. If
/// it is valid the whole page walk is completed and permissions are checked
/// once at the end in the contents of the final page.
///
/// No skip levels or large pages are supported in any way currently.
///
/// This codebase is designed for 64-bit lanes, 512-bit vectors, and 64-bit
/// address spaces and will not work in cases other than this.
///
/// This design decision was made to reduce templating and casting and to
/// keep the code simple. This is designed for one thing, no reason to template
/// everything to theoretically handle something it's not designed for.

extern crate vectorized;
extern crate safecast;
extern crate falkasm;
extern crate conststore;

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashSet;
use safecast::SafeCast;
use vectorized::{Lane, Vector, Mask, VECTOR_WIDTH};

pub mod avx512_routines;
pub mod avx512_jit;

/// Maximum number of dirty entries in the fast dirty list
const MAX_DIRTY: usize = 8192;

/// 64-bit is the only support. This allows us to use `usize` and decreases
/// casting usage. This library is pointless on 32-bits anyways as any
/// realistic target will use more than 4 GiB of RAM.
///
/// Further we assume the vector lanes are also 64-bits
#[cfg(not(target_pointer_width = "64"))]
compile_error!("Only designed for 64-bit systems");

/// Allocation base
const DEFAULT_ALLOC_BASE: VirtAddr = VirtAddr(0x1bb700000000);

// Do not change these permissions without changing the JIT!!!!
pub const PERM_READ:  u8 = 1;
pub const PERM_WRITE: u8 = 2;
pub const PERM_EXEC:  u8 = 4;
pub const PERM_RAW:   u8 = 8;

/// The page referenced by this table is aliased. This page should not be
/// freed during a drop as it's lifetime is managed elsewhere. Further this
/// page can never have its contents modified, regardless of the permissions
/// on the page. Aliased memory can never have SPECIAL_BIT_DIRTY set
pub const SPECIAL_BIT_ALIASED: usize = 1;

/// The page referenced by this table can only be written to if the backing page
/// is copied
pub const SPECIAL_BIT_COW:     usize = 2;

/// Means that the page is dirty
/// Dirty pages can never have SPECIAL_BIT_ALIASED set
pub const SPECIAL_BIT_DIRTY:   usize = 4;

/// Mask to extract special bits from a page table entry
/// Special bits are _ONLY_ valid on the final table and thus does not need
/// to be masked off until translating the page itself
pub const SPECIAL_BIT_MASK: usize = 7;

/// Virtual address type which allows for strong typing on parameters and
/// fields that expect a virtual address
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct VirtAddr(pub usize);

/// Number of levels in the page table
const PAGE_TABLE_DEPTH: usize = 5;

/// Number of bits used from the address from their respective levels of
/// the page table. This list must sum up to 64.
///
/// This must always at least have 2 elements. This is because there must
/// be at least one level of page table translation and a definition of the
/// page size
const PAGE_TABLE_LAYOUT: [u32; PAGE_TABLE_DEPTH] = [16, 16, 16, 11, 5];

/// Size of a page
pub const PAGE_SIZE: usize = 1 << PAGE_TABLE_LAYOUT[PAGE_TABLE_DEPTH - 1];

/// Mask to use on the bottom of an address to find the bits which index the
/// page itself.
pub const PAGE_INDEX_MASK: usize = PAGE_SIZE - 1;

/// Actual backing size in bytes of a page
const PAGE_SIZE_BACKING: usize = PAGE_SIZE * VECTOR_WIDTH * 2;

/// Number of vectors per page
const VECTORS_PER_PAGE: usize = PAGE_SIZE_BACKING / 64;

#[derive(Default, Debug)]
pub struct SoftMMUStats {
    /// Number of bytes allocated for page tables
    page_table_utilization: usize,

    /// Number of bytes allocated for pages that are owned by this table,
    /// thus this excludes aliased memory
    page_utilization: usize,

    /// Number of pages which aliased
    aliased_pages: usize,

    /// Number of pages owned by this MMU (not aliased)
    owned_pages: usize,
}

pub trait ExceptionHandler: Send + Sync {
    /// Exception handler when a virtual address is not present in the MMU
    /// Return `true` if you handle the exception, if not `false`
    /// 
    /// This is unsafe as the MMU is not truly mutable, it is possible there
    /// are readers. For this exception handler to be safe it can only
    /// atomically add pages, it can never remove or change pages as they
    /// may already be referenced by others. This exception handler is called
    /// during a mutex that ensures only one writer, but there could be readers
    /// while writing is happening
    unsafe fn exception(&mut self, mmu: &mut SoftMMU, vaddr: VirtAddr) -> bool;
}

/// Software MMU implementation with easily modified layout. Designed for
/// emulators and is designed to be interfaced with JITs.
pub struct SoftMMU {
    /// Root level of the page table
    page_table: Vec<usize>,

    /// Dirty list, contains a list of dirty guest virtual addresses and
    /// a pointer to the page table entry which describes them such that
    /// we can quickly clear the dirty bits without page table walks
    dirty: Vec<(VirtAddr, usize)>,

    /// Number of dirty page slots remaining
    dirty_remain: usize,

    /// Determines if AVX-512 is supported and thus we can use the accelerated
    /// routines for memory operations
    avx512_supported: bool,

    /// Statistics about the page table
    stats: SoftMMUStats,

    /// Deduplicated pages
    unique_pages: Mutex<HashSet<Vec<Vector>>>,
    
    /// Handler to get invoked on exceptions for page level exceptions
    exception_handler: Option<Box<ExceptionHandler>>,

    /// Virtual address to use for the next allocation
    alloc_base: VirtAddr,

    /// Optional master which this VM is based from
    master: Option<Arc<SoftMMU>>,

    /// Lock used for master VMs to allow for mutable updates even though
    /// the `master` is shared immutably via an `Arc`.
    ///
    /// We need this because we may cause the master to be updated when
    /// lazily paging in memory. Since the master copy might also lazily page
    /// in memory from an exception handler we need to allow updating of the
    /// master VM. This allows us this ability.
    ///
    /// Neither a `Mutex` nor `RwLock` are appropriate in this case as we have
    /// active JITs using the master. `RwLock`s are also way too slow. Thus
    /// this lock is used when accessing the masters memory. Since exception
    /// handlers are `unsafe`, it is up to the author of the exception handler
    /// to make sure they only ever insert to the masters virtual address
    /// space. Removing or modifying entries is not safe.
    ///
    /// TL;DR: Prevents multiple forked MMUs from stomping over eachother when
    ///        inserting into the master MMUs address space
    lock: AtomicUsize,
}

/// Used as a helper in the `Drop` implementation for `SoftMMU`
///
/// The `address` is a pointer to either a table or page. The `idx` is the
/// index into the PAGE_TABLE_LAYOUT for which this `address` refers to.
/// For example, an `idx` of PAGE_TABLE_LAYOUT.len()-1 indicates the `address`
/// points to a last level entry in the table, thus it's actually a page. This
/// index is used to determine the size of the entry, as well as whether it
/// points to a table or a page.
unsafe fn recursive_drop(st: &mut SoftMMUStats, address: usize, idx: usize) {
    // Assert the index is well formed. Zero should never be passed as the
    // level 1 table is statically inlined with the SoftMMU and the index
    // should always be below the number of entries in the page table
    // layout.
    assert!(idx > 0 && idx < PAGE_TABLE_LAYOUT.len(),
        "Invalid usage of recursive_drop()");

    // Don't free null addresses. This allows us to not have to check for
    // null before calling this function
    if address == 0 { return; }

    if (idx + 1) == PAGE_TABLE_LAYOUT.len() {
        // Skip freeing the entry if it's aliased
        if (address & SPECIAL_BIT_ALIASED) != 0 {
            st.aliased_pages = st.aliased_pages
                .checked_sub(1).expect("Integer underflow on aliased pages");
            return;
        }
    }

    // Mask off special bits
    let address = address & !SPECIAL_BIT_MASK;

    if (idx + 1) == PAGE_TABLE_LAYOUT.len() {
        // Entry is a page as it's the last index
        
        // Re-create the original box used to create the page
        let page = Vec::from_raw_parts(address as *mut Vector,
            VECTORS_PER_PAGE, VECTORS_PER_PAGE);
        
        st.page_utilization = st.page_utilization
            .checked_sub(std::mem::size_of_val(page.as_slice()))
            .expect("Page utilization integer underflow");
        st.owned_pages = st.owned_pages
            .checked_sub(1).expect("Integer underflow on owned pages");
    } else {
        // This entry is a pointer to a table

        // Re-create the originally allocated table which will be dropped
        // at the end of this scope
        let table = Vec::from_raw_parts(address as *mut usize,
            1 << PAGE_TABLE_LAYOUT[idx], 1 << PAGE_TABLE_LAYOUT[idx]);

        // For every entry in this table recurse into them
        for &ent in &table {
            recursive_drop(st, ent, idx + 1);
        }

        st.page_table_utilization = st.page_table_utilization
            .checked_sub(std::mem::size_of_val(table.as_slice()))
            .expect("Page table utilization integer underflow");
    }
}

impl Drop for SoftMMU {
    fn drop(&mut self) {
        // For each entry in the page table recursively drop the entries
        for &ent in self.page_table.iter() {
            unsafe { recursive_drop(&mut self.stats, ent, 1); }
        }
        
        self.stats.page_table_utilization = self.stats.page_table_utilization
            .checked_sub(std::mem::size_of_val(self.page_table.as_slice()))
            .expect("Page table utilization integer underflow");

        assert!(self.stats.page_table_utilization == 0 &&
                self.stats.page_utilization == 0 &&
                self.stats.owned_pages == 0 &&
                self.stats.aliased_pages == 0,
                "Oh no! We somehow did not free everything we used");
    }
}

impl SoftMMU {
    /// Creates a new, empty, soft MMU
    pub fn new() -> SoftMMU {
        // Make sure there are at least 2 entries in the page table layout
        // format
        assert!(PAGE_TABLE_LAYOUT.len() >= 2, "Invalid page table format");

        // Compute the number of accounted for bits in the page table shape
        let cbits = PAGE_TABLE_LAYOUT.iter().fold(0u32, |acc, &x| {
            acc.checked_add(x).expect("Integer overflow on PAGE_TABLE_LAYOUT")
        });

        // Check that the number of accounted bits in the page table shape
        // matches up with the virtual address type size
        assert!(cbits as usize == std::mem::size_of::<VirtAddr>() * 8,
            "Invalid page table shape, does not match VirtAddr");

        // Validate the vector width is 512-bits
        assert!(std::mem::size_of::<Vector>() == 64,
            "Vector does not add up to 512-bits");
        
        // Validate the lane size is 64-bits
        assert!(std::mem::size_of::<Lane>() == 8, "Lane is not 64-bits");
        
        // Make sure lanes don't split between pages
        assert!(PAGE_SIZE >= std::mem::size_of::<Lane>(),
            "Page size must be large enough to hold a lane");

        // Safely compute the root page table size
        let rpts = 1usize.checked_shl(PAGE_TABLE_LAYOUT[0])
            .expect("Invalid root page table size");

        let mut ret = SoftMMU {
            page_table:        vec![0usize; rpts],
            dirty:             vec![(VirtAddr(0), 0); MAX_DIRTY],
            dirty_remain:      MAX_DIRTY,
            avx512_supported:  is_x86_feature_detected!("avx512f"),
            alloc_base:        DEFAULT_ALLOC_BASE,
            stats:             Default::default(),
            unique_pages:      Default::default(),
            exception_handler: None,
            master:            None,
            lock:              AtomicUsize::new(0),
        };

        // Record usage stats for the root page table which is always
        // allocated above
        ret.stats.page_table_utilization =
            std::mem::size_of_val(ret.page_table.as_slice());

        ret
    }

    /// Get a pointer to the root page table
    pub fn backing(&mut self) -> *mut usize {
        self.page_table.as_mut_ptr()
    }

    /// Creates a new VM which is forked from an existing one
    pub fn fork_from(master: Arc<SoftMMU>) -> SoftMMU {
        // We don't right now support nested masters
        assert!(master.master.is_none(),
            "Nested masters not currently supported");

        let mut mmu = SoftMMU::new();
        mmu.master = Some(master);
        mmu
    }

    /// Restore VM state to the master VM state
    pub fn reset(&mut self) {
        let master: &SoftMMU = self.master.as_ref()
            .expect("Attempted to reset MMU without master");

        for &(vaddr, page_entry) in &self.dirty[self.dirty_remain..] {
            //print!("{:x} {:x}\n", vaddr.0, page_entry);

            unsafe {
                let master: &mut SoftMMU =
                    &mut *(master as *const SoftMMU as *mut SoftMMU);
                let page_entry = page_entry as *mut usize;
                
                // Clear the dirty bit
                *page_entry &= !SPECIAL_BIT_DIRTY;

                // Get a mutable pointer to the memory we must restore
                let memory = std::slice::from_raw_parts_mut(
                    (*page_entry & !SPECIAL_BIT_MASK) as *mut u8,
                    PAGE_SIZE_BACKING);

                // Translate the master MMU's memory
                let master_mem =
                    master.virt_to_phys_int(vaddr, false, false, false, None);

                if let Some(master_mem) = master_mem {
                    memory.copy_from_slice(master_mem);
                } else {
                    panic!("Memory not present in master");
                }
            }
        }

        // Reset dirty page list
        self.dirty_remain = MAX_DIRTY;

        // Reset the allocation base
        self.alloc_base = master.alloc_base;
    }
    
    /// Sets up an exception handler
    pub fn set_exception_handler(&mut self, handler: Box<ExceptionHandler>) {
        self.exception_handler = Some(handler);
    }

    /// Adds to the page table structure such that the `vaddr` is valid for
    /// `size` bytes.
    ///
    /// Panics if the memory already exists
    pub fn add_memory(&mut self, vaddr: VirtAddr, size: usize) {
        // Do nothing if 0 was requested for the size
        if size == 0 { return; }

        // Make sure size does not overflow the virtual address
        let vend = vaddr.0.checked_add(size - 1)
            .expect("Integer overflow on `add_memory` bounds");

        // Get the backing address from the vaddr and round it down to the
        // nearest page
        let vaddr_align = vaddr.0 & !PAGE_INDEX_MASK;

        // Allocate all pages in this range
        for vaddr in (vaddr_align..=vend).step_by(PAGE_SIZE) {
            unsafe {
                assert!(self.virt_to_phys_int(
                        VirtAddr(vaddr), false, false, false, None).is_none(),
                        "Attempted to add memory that already exists");
                
                assert!(self.virt_to_phys_int(
                        VirtAddr(vaddr), true, false, true, None).is_some(),
                        "Failed to add memory");
            }
        }
    }
    
    /// Get a reference to the page containing `vaddr`. Note this returns
    /// a slice to the entire page rather than the page starting at `vaddr`
    /// thus the user must be aware of the offset into the page.
    ///
    /// Returns `None` if the page is not mapped
    pub fn virt_to_phys(&mut self, vaddr: VirtAddr) -> Option<&[u8]> {
        unsafe {
            self.virt_to_phys_int(vaddr, false, false, true, None).map(|x| &*x)
        }
    }
    
    /// Get a mutable reference to the page containing `vaddr`. Note this
    /// returns a slice to the entire page rather than the page starting at
    /// `vaddr` thus the user must be aware of the offset into the page.
    ///
    /// Note that use of this function implies intent to modify the memory and
    /// thus it will trigger CoW. This will also cause the page to be marked
    /// as dirty.
    ///
    /// Returns `None` if the page is not mapped. Panics if mutable access
    /// is attempted on an aliased page which is not marked as CoW.
    pub fn virt_to_phys_mut(&mut self, vaddr: VirtAddr) -> Option<&mut [u8]> {
        unsafe {
            self.virt_to_phys_int(vaddr, false, true, true, None)
        }
    }

    /// Translate the page containing `vaddr` and optionally create page
    /// tables and pages if they do not already exist if `create_tables`
    /// is true.
    ///
    /// Returns `None` if the page is not mapped and `create_tables` is
    /// `false`.
    ///
    /// Otherwise returns a slice to the _beginning_ of the page which
    /// contains `vaddr`
    ///
    /// If `page` is not `None` and `create_tables` is `true`, the `page` will
    /// be used to creating the backing page rather than a newly allocated page
    /// This is used for aliasing memory and also makes this function unsafe.
    /// This `page` value is always ored with `SPECIAL_BIT_ALIASED` before
    /// being placed into the page table. It's up to the caller to set
    /// `SPECIAL_BIT_COW` if desired.
    ///
    /// `mutate` specifies whether or not the request is to get mutable
    /// buffer. This determines whether or not CoW occurs. It's up to the
    /// caller of this unsafe function to downcast the returned mutable
    /// reference to an immutable reference if this is set to `false`.
    ///
    /// This pointing to the start of the page is a bit weird, but it is
    /// required to keep the layout of an individual page
    /// (eg. vectorized layout) opaque to the core page table routines
    ///
    /// Note this does no permission checks. This only checking this does
    /// is for non-present pages, CoW handling, and aliased memory handling.
    /// It is up to you to build permissions checking abstractions on top
    /// of this.
    ///
    /// `can_mutate` is used to block all mutable changes. If set to `false`
    /// then this function will never create new page tables or new page table
    /// entries regardless of any of the settings. It will only be used to
    /// walk the page tables to get a mapping. Since even a non-mutable page
    /// walk can mutate page tables (via exception handlers faulting in memory,
    /// or pulling from master), this is needed to block these behaviors when
    /// needed.
    unsafe fn virt_to_phys_int(&mut self, orig_vaddr: VirtAddr,
                               create_tables: bool,
                               mutate: bool,
                               can_mutate: bool,
                               page: Option<usize>)
            -> Option<&mut [u8]> {
        // Reference the raw page table. This variable gets updated
        // as we traverse the table
        let mut cur_table: &mut [usize] = &mut self.page_table;

        // Enforce mutate == false and create_tables == false if
        // can_mutate == false
        if !can_mutate {
            assert!(!mutate && !create_tables, "Invalid mutation options");
        }

        // Rip out the inner value of the virtual address
        let mut vaddr = orig_vaddr.0;

        for ii in 0..PAGE_TABLE_LAYOUT.len() {
            // Get the current number of bits for this table
            let shift = PAGE_TABLE_LAYOUT[ii];

            // Rotate the virtual address by shift, causing the bits we are
            // going to use as the index to be in the LSB
            vaddr = vaddr.rotate_left(shift);

            // Create an index for the current table
            let idx = vaddr & ((1 << shift) - 1);

            // If the current entry is zero and we were invoked to create
            // pages and tables if they do not exist, create the correct table
            // or entry.
            if create_tables && cur_table[idx] == 0 {
                if (ii + 2) == PAGE_TABLE_LAYOUT.len() {
                    // Entry needs to be a page

                    if let Some(page) = page {
                        // Use the specified page
                        self.stats.aliased_pages += 1;
                        assert!((page & SPECIAL_BIT_DIRTY) == 0,
                            "Dirty and aliased not allowed");
                        cur_table[idx] = page | SPECIAL_BIT_ALIASED;
                    } else {
                        // Create a zeroed out page
                        let mut buf = vec![Vector::splat(0);
                            VECTORS_PER_PAGE];
                    
                        assert!(buf.len() == buf.capacity(),
                            "Rust over-allocated a page, need to rethink");
                    
                        // Update stats
                        self.stats.page_utilization +=
                            std::mem::size_of_val(buf.as_slice());
                        self.stats.owned_pages += 1;

                        // Insert the page into the table
                        cur_table[idx] = buf.as_mut_ptr() as usize;
                        std::mem::forget(buf);
                    }
                } else {
                    // Allocate a new table capable of holding enough entries
                    // for the next level of indexing.
                    let mut table =
                        vec![0usize; 1 << PAGE_TABLE_LAYOUT[ii + 1]];

                    // Validate Rust didn't over-allocate. This is an
                    // assumption we make during our `Drop` handler.
                    // If this is an issue we'll have to update our `Drop`
                    // handler to be smarter
                    assert!(table.len() == table.capacity(),
                        "Rust over-allocated a page table, need to rethink");
                    
                    // Update stats
                    self.stats.page_table_utilization +=
                        std::mem::size_of_val(table.as_slice());

                    // Insert a pointer to this table into the current table
                    cur_table[idx] = table.as_mut_ptr() as usize;

                    // Forget the allocation so we don't free it as now we must
                    // manage the lifetimes
                    std::mem::forget(table);
                }
            }

            // We failed to translate the address due to a table missing and
            // we were not configured to create tables if they do not exist.
            if cur_table[idx] == 0 {
                // If we cannot mutate we just return that there is no mapping
                if !can_mutate { return None; }

                // Memory does not exist in this VM, but it's possible the
                // master has it and we should lazily fill it in now
                if let Some(ref master) = self.master {
                    // Global master lock, ugly
                    while master.lock.compare_and_swap(0, 1,
                        Ordering::SeqCst) != 0 {}

                    let bb: &mut SoftMMU = {
                        &mut *(&**master as *const SoftMMU as *mut _)
                    };

                    // Attempt to fault in the masters memory via their
                    // exception handler. This can add entries to the
                    // master and thus must be done under the master lock.
                    bb.virt_to_phys_int(orig_vaddr, false, false, true, None);

                    // Release the lock
                    master.lock.store(0, Ordering::SeqCst);

                    // If the virtual address is mapped in the master
                    // then we will make a local aliased and CoW mapping
                    // to that same page
                    if let Some(trans) = bb.virt_to_phys(orig_vaddr) {
                        self.virt_to_phys_int(orig_vaddr, true, false, true,
                            Some(trans.as_ptr() as usize |
                                 SPECIAL_BIT_ALIASED |
                                 SPECIAL_BIT_COW));

                        // Try handling again
                        return self.virt_to_phys_int(
                            orig_vaddr, create_tables, mutate, true, page);
                    }
                }
                
                if self.exception_handler.is_some() {
                    // Take ownership of exception handler so we don't have
                    // lifetime issues
                    let mut eh = self.exception_handler.take().unwrap();

                    // Invoke exception handler
                    if eh.exception(self, orig_vaddr) {
                        // Retry translation if we handled exception
                        self.exception_handler = Some(eh);

                        // Try handling again
                        return self.virt_to_phys_int(
                            orig_vaddr, create_tables, mutate, true, page);
                    } else {
                        // Exception handler did not handle the exception
                        self.exception_handler = Some(eh);
                        return None;
                    }
                } else {
                    // Page doesn't exist and there's no exception handler
                    return None;
                }
            }

            // Mask off the special bits
            let mut table_entry = cur_table[idx] & !SPECIAL_BIT_MASK;
            
            // If the current table entry points to a page and not a table then
            // we terminate the loop
            if ii == PAGE_TABLE_LAYOUT.len() - 2 {
                if mutate {
                    // Get a reference to the current data
                    let old_data = std::slice::from_raw_parts(
                        table_entry as *const Vector, VECTORS_PER_PAGE);

                    // Check if memory is aliased
                    if (cur_table[idx] & SPECIAL_BIT_ALIASED) != 0 {
                        // Check if memory is CoW
                        if (cur_table[idx] & SPECIAL_BIT_COW) != 0 {
                            // Memory is CoW, copy it to an owned page
                            let mut buf = vec![Vector::splat(0);
                                VECTORS_PER_PAGE];

                            // Copy old contents to the new page
                            buf.copy_from_slice(old_data);
                            
                            // Update stats
                            self.stats.page_utilization +=
                                std::mem::size_of_val(buf.as_slice());
                            self.stats.owned_pages   += 1;
                            self.stats.aliased_pages -= 1;

                            // Insert the new page into the table
                            let pageptr = buf.as_mut_ptr() as usize;
                            std::mem::forget(buf);

                            cur_table[idx] = pageptr;

                            // Update table entry as we've mapped in a new page
                            table_entry = pageptr;
                        } else {
                            // Mutable access to aliased but not-CoW memory is
                            // not allowed
                            panic!("Attempted to mutably reference aliased \
                                   memory");
                        }
                    }
                    
                    // Update dirty bit and add it to the dirty list if it's
                    // not already dirty
                    if (cur_table[idx] & SPECIAL_BIT_DIRTY) == 0 {
                        assert!(self.dirty_remain > 0, "Out of dirty entries");
                        self.dirty_remain -= 1;
                        let di = self.dirty_remain;
                        self.dirty[di] = (orig_vaddr,
                            &mut cur_table[idx] as *mut usize as usize);
                        cur_table[idx] |= SPECIAL_BIT_DIRTY;
                    }

                    // Return the slice
                    return Some(std::slice::from_raw_parts_mut(
                        table_entry as *mut u8, PAGE_SIZE_BACKING)
                    );
                } else {
                    // Return the slice unconditionally as we don't plan to
                    // mutate it
                    return Some(std::slice::from_raw_parts_mut(
                        table_entry as *mut u8, PAGE_SIZE_BACKING)
                    );
                }
            }
            
            // Go into the next table
            cur_table = std::slice::from_raw_parts_mut(
                table_entry as *mut usize, 1 << PAGE_TABLE_LAYOUT[ii + 1]);
        }

        panic!("Cannot hit this");
    }

    /// Add de-duplicated memory to the page tables starting at `vaddr` and
    /// containing `data`. All bytes will be mapped with the same `permissions`
    /// and the memory will be read-only unless `is_cow` is set in which case
    /// a new copy of the data will be created upon modification.
    ///
    /// The `vaddr` must be page aligned, and the `data` must be evenly
    /// divisible by PAGE_SIZE. This is because we do not support partial page
    /// de-duplication
    pub fn add_dedup(&mut self, vaddr: VirtAddr, data: &[u8], permissions: u8,
                     is_cow: bool)
    {
        // Nothing to do
        if data.len() <= 0 { return; }

        // Convert CoW boolean to special bit
        let is_cow = if is_cow { SPECIAL_BIT_COW } else { 0 };

        // Broadcast out the permissions
        let permissions = Vector::splat(read_usize(&[permissions; 8])); 

        // Go through each page size chunk
        for (page_id, chunk) in data.chunks(PAGE_SIZE).enumerate() {
            // Compute corresponding virtual address
            let vaddr = VirtAddr(vaddr.0 + (page_id * PAGE_SIZE));

            // Make this chunk uses a full page and doesn't straddle a boundary
            assert!((vaddr.0 & PAGE_INDEX_MASK) == 0 &&
                    chunk.len() == PAGE_SIZE,
                    "Partial page deduping not allowed");

            // Make sure the page is not already mapped
            assert!(self.virt_to_phys(vaddr) == None,
                "Attempted to add dedup memory to already mapped memory");

            let mut target_contents: Vec<Vector> = Vec::new();

            // Interleave VM contents at qword level
            for qword in chunk.chunks(8) {
                // Store the permissions
                target_contents.push(permissions);

                // Store the memory contents
                target_contents.push(Vector::splat(read_usize(qword)));
            }

            // Make sure we created a complete page
            assert!(std::mem::size_of_val(target_contents.as_slice()) ==
                    PAGE_SIZE_BACKING);

            let entry = {
                // Get a lock on the unique pages database.
                // Use the master's database if we have a master
                let mut up = if let Some(ref master) = self.master {
                    master.unique_pages.lock().unwrap()
                } else {
                    self.unique_pages.lock()
                        .expect("Failed to get unique_pages lock")
                };

                if let Some(existing_page) = up.get(&target_contents) {
                    // Page already exists, alias to this one
                    existing_page.as_ptr() as usize
                } else {
                    // Page does not exist in the database, add it

                    // Get the pointer to the newly created page
                    let tcptr = target_contents.as_ptr() as usize;

                    // Update the database
                    up.insert(target_contents);

                    tcptr
                }
            };

            // Map to the memory in the dedup set
            unsafe {
                self.virt_to_phys_int(vaddr, true, false, true,
                    Some(entry | SPECIAL_BIT_ALIASED | is_cow));
            }
        }
    }

    /// Compute the page table index for a specific byte for a specific mask
    /// when accessing permissions
    /// This handles the skipping/striding pattern with the interleaved
    /// permissions and qwords
    #[inline]
    fn permission_index(mask_idx: usize, vaddr: VirtAddr) -> usize {
        // Look up the index into the page for this virtual address
        let page_index = (vaddr.0 as usize) & PAGE_INDEX_MASK;

        let qword_index = (page_index / 8) * 128;
        let qword_mod   = page_index % 8;

        qword_index + mask_idx * 8 + qword_mod
    }
    
    /// Compute the page table index for a specific byte for a specific mask
    /// when accessing contents
    /// This handles the skipping/striding pattern with the interleaved
    /// permissions and qwords
    #[inline]
    fn content_index(mask_idx: usize, vaddr: VirtAddr) -> usize {
        // Contents always follow permissions by 64 bytes
        SoftMMU::permission_index(mask_idx, vaddr) + 64
    }

    /// Invoke a closure for every byte for every VM in [vaddr, vaddr+size)
    ///
    /// This function is unsafe for the same reasons as `virt_to_phys_int`
    /// being unsafe. It always returns mutable references to bytes even if
    /// you specify `mutate` as false. For this to be safe these references
    /// must be downcast for immutable calls.
    ///
    /// The `perm_func` closure is called first for every VM with a copy of
    /// the permission byte for the VM with the supplied index. If this
    /// closure returns `false` this function exits early and reports the
    /// index it was attempting to process which returned failure.
    ///
    /// The `cont_func` is then conditionally called for every byte in every
    /// VM again, allowing for modification of the contents or the permission
    /// byte. The syntax for this closure are (&mut perm, &mut content, index).
    /// This closure has no ability to terminate processing and must not fail.
    ///
    /// The permission closure and contents closure are separate as this allows
    /// us to validate permissions on all VMs for a given byte before actually
    /// accessing the memory. This allows us to maintain our API with returning
    /// a single size for read/written bytes even though we're operating with
    /// multiple VMs with potentially differing permissions.
    ///
    /// On error returns the index of the byte it was attempting to process
    /// when `func` returned false or translation failed. This is the length
    /// that has been successfully processed up until failure
    unsafe fn for_each_byte_int<P, C>(&mut self, mask: Mask, vaddr: VirtAddr,
                                      size: usize, mutate: bool,
                                      perm_func: P,
                                      mut cont_func: C) -> Result<(), usize>
        where P: Fn(u8, usize) -> bool,
              C: FnMut(&mut u8, &mut u8, usize)
    {
        if size == 0 { return Ok(()); }

        let vaddr = vaddr.0;

        // Compute the ending address
        let vend = vaddr.checked_add(size - 1)
            .expect("Integer overflow on for_each_byte size");

        // State keeping track of the current page translation
        let mut translated = None;

        for (ii, vaddr) in (vaddr..=vend).enumerate() {
            // If we have not translated the vaddr yet, or we have rolled
            // over to a new page, then we must rewalk the tables to
            // translate the address.
            if translated.is_none() ||
                    (vaddr & PAGE_INDEX_MASK) == 0 {
                // Get the backing page for this virtual address
                translated = Some(
                    self.virt_to_phys_int(VirtAddr(vaddr), false,
                        mutate, true, None)
                );
            }

            // CHeck if we failed to translate
            if translated == Some(None) {
                return Err(ii);
            }

            // Call the permission function on all bytes first
            for mask in mask.iter() {
                let vaddr = VirtAddr(vaddr);
                let idx   = SoftMMU::permission_index(mask, vaddr);
                
                // This can never fail as this is always Some by now
                if let Some(Some(ref mut x)) = translated {
                    // Call the closure, and bail out if it returns false
                    if !perm_func(x[idx], ii) {
                        return Err(ii);
                    }
                } else { panic!("This can never happen"); }
            }

            // Call the content function on all bytes
            for mask in mask.iter() {
                let vaddr    = VirtAddr(vaddr);
                let perm_idx = SoftMMU::permission_index(mask, vaddr);
                let cont_idx = SoftMMU::content_index(mask, vaddr);
                
                // This should never happen, but makes the following unsafe
                // code safe in all conditions
                assert!(perm_idx != cont_idx &&
                    perm_idx < PAGE_SIZE_BACKING &&
                    cont_idx < PAGE_SIZE_BACKING,
                    "Invalid permission/content index or translation");

                // This can never fail as this is always Some by now
                if let Some(Some(ref mut x)) = translated {
                    // Get mutable reference to the payload
                    let (x, y) = (
                        &mut *x.as_mut_ptr().offset(perm_idx as isize),
                        &mut *x.as_mut_ptr().offset(cont_idx as isize),
                    );

                    // Call the content closure
                    cont_func(x, y, ii);
                } else { panic!("This can never happen"); }
            }
        }

        Ok(())
    }

    /// Mutably access every byte in every VM specified by `mask` in the
    /// range [vaddr..vaddr+size)
    ///
    /// Read documentation for `for_each_byte_int` for more info
    fn for_each_byte_mut<P, C>(&mut self, mask: Mask,
                               vaddr: VirtAddr, size: usize,
                               perm_func: P,
                               mut cont_func: C) -> Result<(), usize>
        where P: Fn(u8, usize) -> bool,
              C: FnMut(&mut u8, &mut u8, usize)
    {
        unsafe {
            self.for_each_byte_int(mask, vaddr, size, true,
                |x, y|    { perm_func(x, y)    },
                |x, y, z| { cont_func(x, y, z) })
        }
    }

    /// Immutably access every byte in every VM specified by `mask` in the
    /// range [vaddr..vaddr+size)
    ///
    /// Read documentation for `for_each_byte_int` for more info
    fn for_each_byte<P, C>(&mut self, mask: Mask, vaddr: VirtAddr, size: usize,
                           perm_func: P,
                           mut cont_func: C) -> Result<(), usize>
        where P: Fn(u8, usize) -> bool,
              C: FnMut(&u8, &u8, usize)
    {
        unsafe {
            self.for_each_byte_int(mask, vaddr, size, false,
                |x, y|    { perm_func(x, y)    },
                |x, y, z| { cont_func(x, y, z) })
        }
    }
    
    /// Set a contiguous region of `size` bytes starting at `vaddr` to
    /// `permissions` using `mask` mask
    pub fn set_permissions_naive(&mut self, mask: Mask, vaddr: VirtAddr,
                                 size: usize,
                                 permissions: u8) -> usize {
        self.for_each_byte_mut(mask, vaddr, size, |_, _| { true },
        |perm, _, _| { *perm = permissions; }).err().unwrap_or(size)
    }

    /// Set a contiguous region of `size` bytes starting at `vaddr` to
    /// `permissions` using `mask` mask
    pub fn set_permissions(&mut self, mask: Mask, vaddr: VirtAddr,
                           size: usize, permissions: u8) -> usize {
        if self.avx512_supported {
            unsafe {
                self.avx512_memset_int(mask, vaddr, permissions, size, 0)
            }
        } else {
            self.set_permissions_naive(mask, vaddr, size, permissions)
        }
    }

    pub fn write_multiple_force_naive(&mut self, mask: Mask, ovaddr: VirtAddr,
                                      mems: &[Vec<u8>]) -> usize {
        let mut bwritten = 0usize;

        assert!(mems.len() == 8,
            "Invalid number of buffers for write_multiple");

        let size = mems[0].len();

        assert!((ovaddr.0 & 7) == 0, "Invalid alignment for write_multiple");
        assert!((size % 8) == 0, "Invalid size align for write_multiple");

        for ii in 0..8 {
            // Size must match original size
            assert!(mems[ii].len() == size);

            // Skip disabled VMs
            if mask.disabled(ii) { continue; }
            bwritten = self.write_force(Mask::single(ii), ovaddr,
                mems[ii].as_slice());
        }

        bwritten
    }

    pub fn write_multiple_force(&mut self, mask: Mask, ovaddr: VirtAddr,
                                mems: &[Vec<u8>]) -> usize {
        if self.avx512_supported {
            unsafe { self.avx512_write_multiple_force(mask, ovaddr, mems) }
        } else {
            self.write_multiple_force_naive(mask, ovaddr, mems)
        }
    }

    /// Naive memset
    pub fn memset_naive(&mut self, mask: Mask, vaddr: VirtAddr, byte: u8,
                        size: usize) -> usize {
        // Super slow but stable
        let bytes = vec![byte; size];
        self.write_force(mask, vaddr, &bytes[..])
    }

    /// Forcably write `byte` for `size` to `vaddr` in VMs enabled by `mask`
    pub fn memset(&mut self, mask: Mask, vaddr: VirtAddr, byte: u8,
                  size: usize) -> usize {
        if self.avx512_supported {
            unsafe { self.avx512_memset_int(mask, vaddr, byte, size, 64) }
        } else {
            self.memset_naive(mask, vaddr, byte, size)
        }
    }

    /// Validate the memory permissions are set on a given region
    pub fn check_permissions<F>(&mut self, mask: Mask, vaddr: VirtAddr,
                                size: usize, filter: F) -> usize
            where F: Fn(u8) -> bool
    {
        self.for_each_byte_mut(mask, vaddr, size,
            |perm, _| { filter(perm) },
            |_, _, _| {}
            ).err().unwrap_or(size)
    }

    /// Write to memory using a closure to filter the access
    ///
    /// Returns number of bytes written. It's possible that if there is a
    /// differeing permission on one of the VMs, that one VM might have written
    /// one more byte than this.
    ///
    /// The `filter` provided returns `true` if the access may proceed,
    /// otherwise this function panics. The filter is passed one argument which
    /// is the permission byte corresponding to the byte that is about to be
    /// written to
    pub fn write_int<F, T>(&mut self, mask: Mask, vaddr: VirtAddr, bytes: &T,
                           filter: F) -> usize
            where F: Fn(u8) -> bool,
                  T: SafeCast + ?Sized,
    {
        let bytes: &[u8] = bytes.cast();

        self.for_each_byte_mut(mask, vaddr, bytes.len(),
            |perm, _| { filter(perm) },
            |perm, contents, ii| {
                // Do RAW memory
                if (*perm & PERM_RAW) != 0 {
                    *perm |= PERM_READ;
                }

                // Write in the byte
                *contents = bytes[ii];
            }).err().unwrap_or(bytes.len())
    }
    /// Write to the memory regardless of permissions
    fn write_force_naive<T: SafeCast + ?Sized>(&mut self, mask: Mask,
                                               vaddr: VirtAddr,
                                               bytes: &T) -> usize {
        self.write_int(mask, vaddr, bytes, |_| true)
    }

    /// Write to the memory regardless of permissions
    pub fn write_force<T: SafeCast + ?Sized>(&mut self, mask: Mask,
                                             vaddr: VirtAddr,
                                             bytes: &T) -> usize {
        if self.avx512_supported {
            unsafe {
                self.avx512_write_force(mask, vaddr, bytes)
            }
        } else {
            self.write_force_naive(mask, vaddr, bytes)
        }
    }

    /// Write to memory only if it is writable
    pub fn write<T: SafeCast + ?Sized>(&mut self, mask: Mask, vaddr: VirtAddr,
                                       bytes: &T) -> usize {
        self.write_int(mask, vaddr, bytes, |perm| (perm & PERM_WRITE) != 0)
    }
    
    /// Read from memory using a closure to filter the access
    ///
    /// The `filter` provided returns `true` if the access may proceed,
    /// otherwise this function panics. The filter is passed one argument which
    /// is the permission byte corresponding to the byte that is about to be
    /// read from
    pub fn read_int<F, T>(&mut self, mask: Mask, vaddr: VirtAddr, bytes: &mut T,
                          filter: F) -> usize
        where F: Fn(u8) -> bool,
              T: SafeCast + ?Sized
    {
        let bytes: &mut [u8] = bytes.cast_mut();
        let mut last_byte = None;

        self.for_each_byte(mask, vaddr, bytes.len(),
            |perm, _| { filter(perm) },
            |_, contents, ii| {
                if Some(ii) == last_byte {
                    // Make sure all VMs have the same bytes
                    assert!(bytes[ii] == *contents,
                            "Content mismatch on mask");
                } else {
                    // Update byte tracking state machine
                    last_byte = Some(ii);

                    // Read the byte
                    bytes[ii] = *contents;
                }
            }).err().unwrap_or(bytes.len())
    }
    
    /// Read from the memory regardless of permissions
    pub fn read_force<T: SafeCast + ?Sized>(&mut self, mask: Mask,
                                            vaddr: VirtAddr,
                                            bytes: &mut T) -> usize {
        self.read_int(mask, vaddr, bytes, |_| true)
    }
    
    /// Read from memory only if it is readable
    pub fn read<T: SafeCast + ?Sized>(&mut self, mask: Mask, vaddr: VirtAddr,
                                      bytes: &mut T) -> usize {
        self.read_int(mask, vaddr, bytes, |perm| (perm & PERM_READ) != 0)
    }

    /// Pretty print the memory of all VMs and for all bytes contained in
    /// the page which contains `vaddr`
    pub fn pretty_print_page(&mut self, vaddr: VirtAddr) {
        let page = self.virt_to_phys(vaddr)
            .expect("Page not mapped for pretty printing");

        // Print a mask header
        print!("mask                   ");
        for mask in 0..8 { print!("{:16} ", mask); }
        print!("\n");

        for offset in (0..PAGE_SIZE_BACKING).step_by(8 * 8 * 2) {
            print!("Permissions {:08x} : ", vaddr.0 + offset / (8 * 2));
            for mask in 0..8 {
                let perms = read_usize(
                    &page[offset+mask*8..offset+(mask+1)*8]);
                print!("{:016x} ", perms);
            }
            print!("\n");

            print!("Contents    {:08x} : ", vaddr.0 + offset / (8 * 2));

            // Skip over permissions
            let offset = offset + 64;

            for mask in 0..8 {
                let contents = read_usize(
                    &page[offset+mask*8..offset+(mask+1)*8]);
                print!("{:016x} ", contents);
            }
            print!("\n");
        }
    }

    /// Create an allocation of RaW memory for `size` bytes
    pub fn alloc(&mut self, mask: Mask, size: usize) -> VirtAddr {
        assert!(size > 0, "Zero sized allocations not allowed");

        let alloc_base = self.alloc_base;

        // Add the size to the base and keep it 64-byte aligned
        self.alloc_base.0 += (size + 63) & !63;

        // Add some padding pages after the allocation
        self.alloc_base.0 += 1024 * 1024;

        // Add memory to the page tables
        self.add_memory(alloc_base, size);
        self.set_permissions(mask, alloc_base, size, PERM_RAW | PERM_WRITE);

        alloc_base
    }
}

/// Helper function to convert bytes to a usize
fn read_usize(bytes: &[u8]) -> usize {
    ((bytes[0] as usize) <<  0) |
    ((bytes[1] as usize) <<  8) |
    ((bytes[2] as usize) << 16) |
    ((bytes[3] as usize) << 24) |
    ((bytes[4] as usize) << 32) |
    ((bytes[5] as usize) << 40) |
    ((bytes[6] as usize) << 48) |
    ((bytes[7] as usize) << 56)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Instant;
    use std::cell::Cell;
    use super::{PAGE_SIZE, PAGE_SIZE_BACKING};
    use super::VirtAddr;
    use super::SoftMMU;
    use vectorized::Mask;
    use super::{PERM_RAW, PERM_READ, PERM_WRITE};
    use super::ExceptionHandler;
    use crate::MAX_DIRTY;

    pub const TEST_VADDR: usize = 0x1000_0000_0000;

    thread_local! {
        static SEED: Cell<usize> = Cell::new(0x36a62277216e7f1a);
    }

    pub fn xorshift() -> usize {
        SEED.with(|x| {
            let mut seed = x.get();
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 43;
            x.set(seed);
            seed
        })
    }
    
    /// Get elapsed time in seconds
    fn elapsed_from(start: &Instant) -> f64 {
        let dur = start.elapsed();
        dur.as_secs() as f64 + dur.subsec_nanos() as f64 / 1_000_000_000.0
    }

    #[test]
    #[ignore]
    fn benchmark() {
        const NUM_TESTS: usize = 10000;

        let vaddr = VirtAddr(TEST_VADDR);

        let it = Instant::now();
        let mut mmu = SoftMMU::new();
        let mmu_create_time = elapsed_from(&it);

        assert!(PAGE_SIZE <= 1 * 1024 * 1024,
                "Unsupported page size for benchmark");

        let mut meg = vec![0u8; 1 * 1024 * 1024];

        let it = Instant::now();
        mmu.add_dedup(vaddr, &meg, PERM_READ | PERM_WRITE | PERM_RAW, true);
        let deduped_meg = elapsed_from(&it);

        // Simulate reading a 1024 byte packet over and over
        let it = Instant::now();
        for _ in 0..NUM_TESTS {
            assert!(mmu.read(Mask::all(), vaddr, &mut meg[..1024]) == 1024);
        }
        let read_packet = NUM_TESTS as f64 / elapsed_from(&it);
        
        // Simulate writing a 1024 byte packet over and over
        let it = Instant::now();
        for _ in 0..NUM_TESTS {
            assert!(mmu.write_force(Mask::all(), vaddr, &meg[..1024]) == 1024);
        }
        let write_packet = NUM_TESTS as f64 / elapsed_from(&it);
        
        // Simulate memsetting a 1024 byte packet over and over
        let it = Instant::now();
        for _ in 0..NUM_TESTS {
            assert!(mmu.memset(Mask::all(), vaddr, 0x41, 1024) == 1024);
        }
        let memset_packet = NUM_TESTS as f64 / elapsed_from(&it);
        
        // Simulate setting permissions for a 1024 byte packet over and over
        let it = Instant::now();
        for _ in 0..NUM_TESTS {
            assert!(mmu.set_permissions(Mask::all(), vaddr,
                1024, PERM_READ) == 1024);
        }
        let set_permissions_packet = NUM_TESTS as f64 / elapsed_from(&it);

        let buffers = vec![vec![0x41u8; 1024]; 8];
        let it = Instant::now();
        for _ in 0..NUM_TESTS {
            assert!(mmu.write_multiple_force(Mask::all(), vaddr,
                buffers.as_slice()) == 1024);
        }
        let write_multiple_packet = NUM_TESTS as f64 / elapsed_from(&it);

        print!(
            "Empty SoftMMU created in                   {:15.4} seconds\n\
             1 MiB of deduped memory added in           {:15.4} seconds\n\
             1024 byte chunks read per second           {:15.4}\n\
             1024 byte chunks written per second        {:15.4}\n\
             1024 byte chunks memset per second         {:15.4}\n\
             1024 byte chunks permed per second         {:15.4}\n\
             1024 byte chunks write multiple per second {:15.4}\n\
             ",
            mmu_create_time,
            deduped_meg,
            read_packet * 8.,
            write_packet * 8.,
            memset_packet * 8.,
            set_permissions_packet * 8.,
            write_multiple_packet * 8.);

        panic!("Woo");
    }

    #[test]
    #[ignore]
    fn human_verification() {
        // Print the layout of a page table to the screen for a human to
        // look at and validate. Test not run by default as it requires
        // manual verification
        let mut mmu = SoftMMU::new();
        let vaddr = VirtAddr(TEST_VADDR);
        mmu.add_dedup(vaddr, &vec![0x41u8; PAGE_SIZE * 4],
                      PERM_READ | PERM_WRITE, true);

        for mask in 0..8 {
            let mut bytes =
                vec![0x12u8, 0x34, 0x56, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc];
            bytes[0] = mask + 0xc0;
            //while bytes.len() < PAGE_SIZE { bytes.push(0xcc); }
            assert!(
                mmu.write_force(Mask::from_raw(1 << mask),
                    VirtAddr(vaddr.0 + 6), bytes.as_slice()) == bytes.len());
        }

        mmu.pretty_print_page(VirtAddr(vaddr.0 + PAGE_SIZE*0));
        mmu.pretty_print_page(VirtAddr(vaddr.0 + PAGE_SIZE*1));
        mmu.pretty_print_page(VirtAddr(vaddr.0 + PAGE_SIZE*2));
        mmu.pretty_print_page(VirtAddr(vaddr.0 + PAGE_SIZE*3));
        panic!("Woo");
    }

    #[test]
    fn basic_mapping_tests() {
        // Make sure memory returns not mapped until it is mapped, then it
        // should return mapped.
        let mut mmu = SoftMMU::new();
        let vaddr = VirtAddr(TEST_VADDR);
        assert!(mmu.virt_to_phys(vaddr).is_none());
        assert!(mmu.virt_to_phys_mut(vaddr).is_none());
        mmu.add_memory(vaddr, 1);
        assert!(mmu.virt_to_phys(vaddr).is_some());
        assert!(mmu.virt_to_phys_mut(vaddr).is_some());
    }

    #[test]
    fn dedup_working() {
        // Make sure deduping memory actually dedups it
        let mut mmu = SoftMMU::new();
        let vaddr = VirtAddr(TEST_VADDR);

        // Map 2 regions of memory with the same contents
        mmu.add_dedup(vaddr, &vec![0x41u8; PAGE_SIZE], 0xd0, true);
        mmu.add_dedup(VirtAddr(vaddr.0 + PAGE_SIZE),
            &vec![0x41u8; PAGE_SIZE], 0xd0, true);

        // Make sure the memory is mapped as aliased
        assert!(mmu.stats.aliased_pages == 2);
        assert!(mmu.stats.owned_pages   == 0);

        // Make sure there is only one entry in the unique pages database
        assert!(mmu.unique_pages.lock().unwrap().len() == 1);

        let page_0 = mmu.virt_to_phys(vaddr).unwrap().as_ptr();
        let page_1 =
            mmu.virt_to_phys(VirtAddr(vaddr.0 + PAGE_SIZE)).unwrap().as_ptr();

        // Make sure they're using the same backing for both pages
        assert!(page_0 == page_1);
        
        let page_0_cow = mmu.virt_to_phys_mut(vaddr).unwrap().as_ptr();
        let page_1_cow = mmu.virt_to_phys_mut(
            VirtAddr(vaddr.0 + PAGE_SIZE)).unwrap().as_ptr();
        
        // Make sure the memory got CoWed
        assert!(mmu.stats.aliased_pages == 0);
        assert!(mmu.stats.owned_pages   == 2);

        // Make sure both addresses got their own CoW pages and that they
        // did not somehow CoW to the original page backing
        assert!(page_0_cow != page_1_cow && page_0_cow != page_0 &&
                page_1_cow != page_0);
        
        // Make sure the one unique page entry still exists in the database
        assert!(mmu.unique_pages.lock().unwrap().len() == 1);
    }

    #[test]
    fn immutable_noncow_dedup_memory() {
        // Validate basic dedup operation
        
        // Create a new MMU
        let mut mmu = SoftMMU::new();

        let vaddr = VirtAddr(TEST_VADDR);
        
        // Create non-CoW aliased memory and attempt to get access to it
        mmu.add_dedup(vaddr, &vec![0u8; PAGE_SIZE], 0, false);
        assert!(mmu.virt_to_phys(vaddr).is_some());

        assert!(mmu.stats.aliased_pages == 1);
        assert!(mmu.stats.owned_pages   == 0);
    }

    #[test]
    #[should_panic(expected = "Attempted to mutably reference aliased memory")]
    fn mutable_noncow_dedup_memory() {
        // Validate mutable accessed to non-CoW but aliased memory panics

        // Create a new MMU
        let mut mmu = SoftMMU::new();

        let vaddr = VirtAddr(TEST_VADDR);
        
        // Create non-CoW aliased memory and attempt to mutably get access
        // to it
        mmu.add_dedup(vaddr, &vec![0u8; PAGE_SIZE], 0, false);
        assert!(mmu.virt_to_phys_mut(vaddr).is_some());
    }

    #[test]
    fn mutable_cow_dedup_memory() {
        // Validate memory that is marked as CoW actually gets CoWed on mutable
        // accesses only

        // Create a new MMU
        let mut mmu = SoftMMU::new();

        let vaddr = VirtAddr(TEST_VADDR);
        
        // Create CoW aliased memory and attempt to get access to it
        mmu.add_dedup(vaddr, &vec![0u8; PAGE_SIZE], 0, true);

        // Save off the pointer for the deduped memory
        let existing_page = mmu.virt_to_phys(vaddr).unwrap().as_ptr();

        assert!(mmu.stats.aliased_pages == 1);
        assert!(mmu.stats.owned_pages   == 0);
        
        // Mutably access page
        let new_page = mmu.virt_to_phys_mut(vaddr).unwrap().as_ptr();
        
        // Memory should be CoWed, thus should no longer be aliased and should
        // be owned by us
        assert!(mmu.stats.aliased_pages == 0);
        assert!(mmu.stats.owned_pages   == 1);

        // Make sure the backing changed but the contents did not
        assert!(existing_page != new_page);

        unsafe {
            assert!(
                std::slice::from_raw_parts(existing_page, PAGE_SIZE_BACKING) ==
                std::slice::from_raw_parts(new_page, PAGE_SIZE_BACKING));
        }
    }

    #[test]
    fn dirty_working() {
        let mut mmu = SoftMMU::new();
        let vaddr = VirtAddr(TEST_VADDR);
        mmu.add_memory(vaddr, 1);
        assert!(mmu.dirty_remain == MAX_DIRTY);
        assert!(mmu.virt_to_phys(vaddr).is_some());
        assert!(mmu.dirty_remain == MAX_DIRTY);
        assert!(mmu.virt_to_phys_mut(vaddr).is_some());
        assert!(mmu.dirty_remain == MAX_DIRTY-1);
        assert!(mmu.dirty[MAX_DIRTY-1].0 == vaddr);
    }

    #[test]
    fn validate_read_different_perms() {
        let mut mmu = SoftMMU::new();
        let vaddr = VirtAddr(TEST_VADDR);
        mmu.add_dedup(vaddr, &vec![0x41u8; PAGE_SIZE], PERM_READ, true);

        // Punch a one byte hole in one of the VMs
        mmu.set_permissions(Mask::from_raw(0x10), VirtAddr(vaddr.0 + 4), 1, 0);

        let mut buffer = vec![0xffu8; PAGE_SIZE];
        assert!(mmu.read(Mask::all(), vaddr, &mut buffer[..]) == 4);

        // Ensure the memory has been read exactly and no other bytes have
        // been clobbered in our buffer. This validates that permission checks
        // occur on all VMs first, before bytes are read into the buffer
        assert!(&buffer[..4] == &vec![0x41u8; 4][..]);
        assert!(&buffer[4..] == &vec![0xffu8; PAGE_SIZE - 4][..]);
    }
    
    #[test]
    fn validate_read() {
        // Make sure basic reads are working, and partial reads are returning
        // the correct lengths
        let mut mmu = SoftMMU::new();
        let vaddr = VirtAddr(TEST_VADDR);
        mmu.add_dedup(vaddr, &vec![0x41u8; PAGE_SIZE], PERM_READ, true);
        let mut buffer = vec![0u8; PAGE_SIZE + 1024];
        assert!(mmu.read(Mask::all(), vaddr,
            &mut buffer[..PAGE_SIZE]) == PAGE_SIZE);
        assert!(&buffer[..PAGE_SIZE] == &vec![0x41u8; PAGE_SIZE][..]);
        assert!(&buffer[PAGE_SIZE..] == &vec![0x00u8; 1024][..]);
        assert!(mmu.read(Mask::all(), vaddr, &mut buffer[..]) == PAGE_SIZE);
        assert!(mmu.read(Mask::all(), VirtAddr(0x100), &mut buffer[..]) == 0);
    }

    #[test]
    fn validate_write() {
        // Make sure basic writes are working, and partial writes are returning
        // the correct lengths
        let mut mmu = SoftMMU::new();
        let vaddr = VirtAddr(TEST_VADDR);
        mmu.add_dedup(vaddr, &vec![0x41u8; PAGE_SIZE],
                      PERM_WRITE | PERM_RAW, true);
        assert!(mmu.write(Mask::all(), vaddr,
            &vec![0x41u8; PAGE_SIZE][..]) == PAGE_SIZE);
        assert!(mmu.write(Mask::all(), vaddr,
            &vec![0x41u8; PAGE_SIZE+1024][..]) == PAGE_SIZE);
        assert!(mmu.write(Mask::all(), VirtAddr(0x100), &[0x41; 2048][..]) == 0);
    }

    #[test]
    fn validate_perms() {
        // Validate that nothing works if permissions are zero
        let mut mmu = SoftMMU::new();
        let vaddr = VirtAddr(TEST_VADDR);
        mmu.add_dedup(vaddr, &vec![0x41u8; PAGE_SIZE], 0, true);
        let mut buf = [0; 32];
        assert!(mmu.read(Mask::all(), vaddr, &mut buf[..]) == 0);
        assert!(mmu.write(Mask::all(), vaddr, &buf) == 0);
        assert!(mmu.read(Mask::all(), vaddr, &mut buf[..]) == 0);
    }

    #[test]
    fn validate_content_update() {
        // Validate that writes affect reads
        let mut mmu = SoftMMU::new();
        let vaddr = VirtAddr(TEST_VADDR);
        mmu.add_dedup(vaddr, &vec![0x41u8; PAGE_SIZE],
                      PERM_READ | PERM_WRITE, true);
        
        let mut buf = vec![0u8; PAGE_SIZE];
        assert!(mmu.read(Mask::all(), vaddr, &mut buf[..]) == PAGE_SIZE);
        assert!(&buf[..] == &vec![0x41u8; PAGE_SIZE][..]);
        
        // Write in a byte
        assert!(mmu.write(Mask::all(), VirtAddr(vaddr.0 + 1), &[0x77u8]) == 1);

        let mut buf = vec![0u8; PAGE_SIZE];
        assert!(mmu.read(Mask::all(), vaddr, &mut buf[..]) == PAGE_SIZE);
        assert!(&buf[..1] == &[0x41u8; 1][..]);
        assert!(buf[1] == 0x77);
        assert!(&buf[2..] == &[0x41u8; PAGE_SIZE - 2][..]);
    }

    #[test]
    #[should_panic(expected = "Content mismatch on mask")]
    fn validate_faulting_differing_reads() {
        // Validate that reading different bytes from different vmss causes
        // a panic
        let mut mmu = SoftMMU::new();
        let vaddr = VirtAddr(TEST_VADDR);
        mmu.add_dedup(vaddr, &vec![0x41u8; PAGE_SIZE],
                      PERM_READ | PERM_WRITE, true);

        let mut buf = vec![0u8; PAGE_SIZE];
        assert!(mmu.read(Mask::all(), vaddr, &mut buf[..]) == PAGE_SIZE);
        assert!(&buf[..] == &vec![0x41u8; PAGE_SIZE][..]);
        
        // Zap in a single byte into one VM that is the same
        assert!(mmu.write(Mask::from_raw(0x10), vaddr, &[0x41u8]) == 1);
        
        // Validate we're still fine
        assert!(mmu.read(Mask::all(), vaddr, &mut buf[..]) == PAGE_SIZE);
        assert!(&buf[..] == &vec![0x41u8; PAGE_SIZE][..]);
        
        // Zap in a single byte into one VM that differs
        assert!(mmu.write(Mask::from_raw(0x10), vaddr, &[0x77u8]) == 1);
        
        // Cause a panic as now we cannot read the same memory from all VMs
        assert!(mmu.read(Mask::all(), vaddr, &mut buf[..]) == PAGE_SIZE);
    }

    #[test]
    fn valid_reads_after_write() {
        // Validate RaW bit is being updated
        let mut mmu = SoftMMU::new();
        let vaddr = VirtAddr(TEST_VADDR);
        mmu.add_dedup(vaddr, &vec![0x41u8; PAGE_SIZE],
                      PERM_WRITE | PERM_RAW, true);
        let mut buf = vec![0u8; PAGE_SIZE];
        assert!(mmu.read(Mask::all(), vaddr, &mut buf[..]) == 0);
        assert!(mmu.write(Mask::all(), vaddr, &buf[..]) == PAGE_SIZE);
        assert!(mmu.read(Mask::all(), vaddr, &mut buf[..]) == PAGE_SIZE);
    }

    #[test]
    fn shared_dedup_masters() {
        // Validate that multiple MMUs forked from the same master still
        // continue to use the same dedup database and benefit from cross-MMU
        // deduplication
        let master = Arc::new(SoftMMU::new());
        let mut mmu_a = SoftMMU::fork_from(master.clone());
        let mut mmu_b = SoftMMU::fork_from(master.clone());

        let vaddr = VirtAddr(TEST_VADDR);

        // Create mappings of the same thing in both VMs
        mmu_a.add_dedup(vaddr, &vec![0x41u8; PAGE_SIZE][..], PERM_READ, false);
        mmu_b.add_dedup(vaddr, &vec![0x41u8; PAGE_SIZE][..], PERM_READ, false);

        // Get their backing pages
        let page_a = mmu_a.virt_to_phys(vaddr).unwrap().as_ptr();
        let page_b = mmu_b.virt_to_phys(vaddr).unwrap().as_ptr();

        // Make sure they use the same backing
        assert!(page_a == page_b);
    }

    #[test]
    #[should_panic(expected = "Got exception at vaddr 100000000000")]
    fn test_exception_handler() {
        struct Eh {}

        impl ExceptionHandler for Eh {
            unsafe fn exception(&mut self, _mmu: &mut SoftMMU,
                                vaddr: VirtAddr) -> bool {
                panic!("Got exception at vaddr {:x}", vaddr.0);
            }
        }
        
        let mut mmu = SoftMMU::new();
        mmu.set_exception_handler(Box::new(Eh {}));
        mmu.write(Mask::all(), VirtAddr(TEST_VADDR), b"APPLES");
    }

    #[test]
    fn test_master_lazy_eh_filling() {
        // Validate that a child can fill memory from a master which is lazily
        // backed by an exception handler

        struct Eh {}

        impl ExceptionHandler for Eh {
            unsafe fn exception(&mut self, mmu: &mut SoftMMU,
                                vaddr: VirtAddr) -> bool {
                assert!(vaddr == VirtAddr(TEST_VADDR));

                // Add dedup memory in
                mmu.add_dedup(vaddr, &vec![0x41u8; PAGE_SIZE], PERM_READ, false);

                // Handled exception
                true
            }
        }
        
        // Create master MMU with no memory at all, but does have an
        // exception handler to lazily fault in pages
        let mut master = SoftMMU::new();
        master.set_exception_handler(Box::new(Eh {}));

        // Wrap it up
        let master = Arc::new(master);

        // Fork from the master
        let mut child = SoftMMU::fork_from(master);

        // Validate we can read memory
        // This will not be present in the child, causing a fault
        // We will then go to the master to fill this memory, which will fault
        // The master fault handler will add memory at this location
        // We then will reattempt the reads and they are no longer faulting
        // so we return out with the right memory :)
        let mut buf = vec![0u8; PAGE_SIZE];
        assert!(child.read(Mask::all(),
            VirtAddr(TEST_VADDR), &mut buf[..]) == PAGE_SIZE);
        assert!(&buf[..] == &vec![0x41u8; PAGE_SIZE][..]);
    }

    #[test]
    fn validate_avx512_write_force() {
        let mut naive_mmu = SoftMMU::new();
        let mut fast_mmu  = SoftMMU::new();
        
        // This test is pointless without avx512
        if !naive_mmu.avx512_supported { return; }

        let vaddr = VirtAddr(TEST_VADDR);
        naive_mmu.add_dedup(vaddr, &vec![0x41u8; PAGE_SIZE * 4],
                            PERM_READ | PERM_WRITE, true);

        fast_mmu.add_dedup(vaddr, &vec![0x41u8; PAGE_SIZE * 4],
                           PERM_READ | PERM_WRITE, true);

        for _ in 0..100000 {
            let ii    = xorshift() % (PAGE_SIZE * 4);
            let vaddr = VirtAddr(vaddr.0 + ii);
            let size  = xorshift() % (PAGE_SIZE * 4 + 1);
            
            let mask = xorshift() as u8;
            if mask == 0 { continue; }

            let mut payload = Vec::new();
            for _ in 0..size {
                payload.push(xorshift() as u8);
            }

            let bread = naive_mmu.write_force_naive(Mask::from_raw(mask),
                vaddr, &payload[..]);

            assert!(fast_mmu.write_force(Mask::from_raw(mask),
                vaddr, &payload[..]) == bread);

            for page in 0..4 {
                let vaddr = VirtAddr(vaddr.0 + PAGE_SIZE * page);
                let b1 = naive_mmu.virt_to_phys(vaddr);
                let b2 = fast_mmu.virt_to_phys(vaddr);
                assert!(b1 == b2);
            }
        }
    }

    #[test]
    fn validate_avx512_memset() {
        let mut naive_mmu = SoftMMU::new();
        let mut fast_mmu  = SoftMMU::new();
        
        // This test is pointless without avx512
        if !naive_mmu.avx512_supported { return; }

        let vaddr = VirtAddr(TEST_VADDR);
        naive_mmu.add_dedup(vaddr, &vec![0x41u8; PAGE_SIZE * 4],
                            PERM_READ | PERM_WRITE, true);

        fast_mmu.add_dedup(vaddr, &vec![0x41u8; PAGE_SIZE * 4],
                           PERM_READ | PERM_WRITE, true);

        for _ in 0..100000 {
            let ii    = xorshift() % (PAGE_SIZE * 4);
            let vaddr = VirtAddr(vaddr.0 + ii);
            let size  = xorshift() % (PAGE_SIZE * 4 + 1);

            let byte = xorshift() as u8;
            let mask = xorshift() as u8;
            if mask == 0 { continue; }

            let bread = naive_mmu.memset_naive(Mask::from_raw(mask),
                vaddr, byte, size);

            assert!(fast_mmu.memset(Mask::from_raw(mask),
                vaddr, byte, size) == bread);

            for page in 0..4 {
                let vaddr = VirtAddr(vaddr.0 + PAGE_SIZE * page);
                let b1 = naive_mmu.virt_to_phys(vaddr);
                let b2 = fast_mmu.virt_to_phys(vaddr);
                assert!(b1 == b2);
            }
        }
    }

   #[test]
    fn validate_avx512_set_permissions() {
        let mut naive_mmu = SoftMMU::new();
        let mut fast_mmu  = SoftMMU::new();
        
        // This test is pointless without avx512
        if !naive_mmu.avx512_supported { return; }

        let vaddr = VirtAddr(TEST_VADDR);
        naive_mmu.add_dedup(vaddr, &vec![0x41u8; PAGE_SIZE * 4],
                            PERM_READ | PERM_WRITE, true);

        fast_mmu.add_dedup(vaddr, &vec![0x41u8; PAGE_SIZE * 4],
                           PERM_READ | PERM_WRITE, true);

        for _ in 0..100000 {
            let ii    = xorshift() % (PAGE_SIZE * 4);
            let vaddr = VirtAddr(vaddr.0 + ii);
            let size  = xorshift() % (PAGE_SIZE * 4 + 1);

            let byte = xorshift() as u8;
            let mask = xorshift() as u8;
            if mask == 0 { continue; }

            let bread = naive_mmu.set_permissions_naive(Mask::from_raw(mask),
                vaddr, size, byte);

            assert!(fast_mmu.set_permissions(Mask::from_raw(mask),
                vaddr, size, byte) == bread);

            for page in 0..4 {
                let vaddr = VirtAddr(vaddr.0 + PAGE_SIZE * page);
                let b1 = naive_mmu.virt_to_phys(vaddr);
                let b2 = fast_mmu.virt_to_phys(vaddr);
                assert!(b1 == b2);
            }
        }
    }

    #[test]
    fn validate_avx512_write_multiple() {
        let mut naive_mmu = SoftMMU::new();
        let mut fast_mmu  = SoftMMU::new();
        
        // This test is pointless without avx512
        if !naive_mmu.avx512_supported { return; }

        let vaddr = VirtAddr(TEST_VADDR);
        naive_mmu.add_dedup(vaddr, &vec![0x41u8; PAGE_SIZE * 4],
                            PERM_READ | PERM_WRITE, true);

        fast_mmu.add_dedup(vaddr, &vec![0x41u8; PAGE_SIZE * 4],
                           PERM_READ | PERM_WRITE, true);

        for _ in 0..100000 {
            let ii    = (xorshift() % (PAGE_SIZE * 4)) & !7;
            let vaddr = VirtAddr(vaddr.0 + ii);
            let size  = (xorshift() % (PAGE_SIZE * 4 + 1)) & !7;

            let mask = xorshift() as u8;
            if mask == 0 { continue; }

            let mut buffers = Vec::new();
            for _ in 0..8 {
                let mut buf = Vec::new();
                for _ in 0..size {
                    buf.push(xorshift() as u8);
                }
                buffers.push(buf);
            }

            let bread = naive_mmu.write_multiple_force_naive(Mask::from_raw(mask),
                vaddr, buffers.as_slice());

            assert!(fast_mmu.write_multiple_force(Mask::from_raw(mask),
                vaddr, buffers.as_slice()) == bread);

            for page in 0..4 {
                let vaddr = VirtAddr(vaddr.0 + PAGE_SIZE * page);
                let b1 = naive_mmu.virt_to_phys(vaddr);
                let b2 = fast_mmu.virt_to_phys(vaddr);
                assert!(b1 == b2);
            }
        }
    }
}

