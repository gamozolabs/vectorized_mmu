use crate::{PAGE_TABLE_LAYOUT, PAGE_INDEX_MASK};
use crate::{PERM_READ, PERM_WRITE, PERM_RAW};
use crate::{SPECIAL_BIT_MASK, SPECIAL_BIT_DIRTY, SPECIAL_BIT_ALIASED};
use conststore::ConstStore;
use falkasm::{AsmStream, KmaskType, KmaskReg, Operand};
use falkasm::Reg::*;
use falkasm::VecReg::*;
use falkasm::Operand::MemoryBcast    as Membc;
use falkasm::Operand::MemoryVidx     as Mvidx;
use falkasm::Operand::Immediate      as Imm;
use falkasm::Operand::Register       as Reg;
use falkasm::Operand::Memory         as Mem;
use falkasm::Operand::VecRegister    as Vreg;
use falkasm::Operand::BranchNear     as BranchNear;
use falkasm::Operand::BranchShort    as BranchShort;

#[cfg(target_os="windows")]
pub fn alloc_rwx(size: usize) -> &'static mut [u8] {
    extern {
        fn VirtualAlloc(lpAddress: *const u8, dwSize: usize,
                        flAllocationType: u32, flProtect: u32) -> *mut u8;
    }

    unsafe {
        const PAGE_EXECUTE_READWRITE: u32 = 0x40;

        const MEM_COMMIT:  u32 = 0x00001000;
        const MEM_RESERVE: u32 = 0x00002000;

        let ret = VirtualAlloc(0 as *const _, size, MEM_COMMIT | MEM_RESERVE,
                               PAGE_EXECUTE_READWRITE);
        assert!(!ret.is_null());

        std::slice::from_raw_parts_mut(ret, size)
    }
}

#[cfg(target_os="linux")]
pub fn alloc_rwx(size: usize) -> &'static mut [u8] {
    extern {
        fn mmap(addr: *mut u8, length: usize, prot: i32, flags: i32, fd: i32,
                offset: usize) -> *mut u8;
    }

    unsafe {
        // Alloc RWX and MAP_PRIVATE | MAP_ANON
        let ret = mmap(0 as *mut u8, size, 7, 34, -1, 0);
        assert!(!ret.is_null());
        
        std::slice::from_raw_parts_mut(ret, size)
    }
}

// Expects:
//
// rax - scratch
// rbx - scratch
// rcx - scratch
// rdx - scratch
// rsi - scratch
// rdi - scratch
// R8  - Points to 64-byte aligned scratch memory
// R9  - Points to root page table
// R11 - Points to conststore table
// R13 - Fast dirty list
// R14 - Fast dirty remaining free entries
//
// 68 cycles per read with no divergence
// 141 cycles per read with divergence
//
pub fn mmu_translate(asm: &mut AsmStream, conststore: &mut ConstStore,
        bytes: usize, addr: Operand, transaddr: Operand, is_write: bool) {
    const OFFSETS: [usize; 8] = [0x00, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38];

    assert!(bytes == 8 || bytes == 4 || bytes == 2 || bytes == 1,
        "Unsupported operation size for mmu_jit");

    let diverge         = format!("diverge{}", asm.len());
    let page_fault      = format!("page_fault{}", asm.len());
    let page_fault_all  = format!("page_fault_all{}", asm.len());
    let page_fault_vectorized  = format!("page_fault_vectorized{}", asm.len());
    let bad_alignment   = format!("bad_alignment{}", asm.len());
    let fast_dirty_full = format!("fast_dirty_full{}", asm.len());
    let already_dirty   = format!("already_dirty{}", asm.len());
    let already_dirty_vec = format!("already_dirty_vec{}", asm.len());
    let dirtyloop = format!("dirtyloop{}", asm.len());
    let nextiter = format!("nextiter{}", asm.len());

    // Permute register which contains all lanes set to the index of any
    // of the VMs that is actively running
    let aperm = Vreg(Zmm30);

    let nokmask   = Operand::KmaskRegister(KmaskType::Merge(KmaskReg::K0));
    let kmask     = Operand::KmaskRegister(KmaskType::Merge(KmaskReg::K1));
    let kmaskz    = Operand::KmaskRegister(KmaskType::Zero(KmaskReg::K1));
    let tmpkmask  = Operand::KmaskRegister(KmaskType::Merge(KmaskReg::K2));
    //let tmpkmaskz = Operand::KmaskRegister(KmaskType::Zero(KmaskReg::K2));

    // Allocate a new constant variable in the constant storage database
    // and construct a memory operand that refers to it
    /*macro_rules! cs {
        ($constant:expr) => ({
            let idx = conststore.add_const($constant) as i64;
            Mem(Some(R11), None, idx * 8)
        })
    }*/

    macro_rules! csbc {
        ($constant:expr) => ({
            let idx = conststore.add_const($constant) as i64;
            Membc(Some(R11), None, idx * 8)
        })
    }

    let is_read = !is_write;

    let mut perms  = if is_read { PERM_READ } else { PERM_WRITE } as u64;
    let mut rawbit = PERM_RAW as u64;
    for _ in 0..bytes - 1 {
        perms  = perms.rotate_left(8)  | perms;
        rawbit = rawbit.rotate_left(8) | rawbit;
    }

    // Broadcast one of the active VMs to all lanes
    // It's important we use nokmask here so we can extract the address easily
    // with a vmovq later
    asm.vpermq(&[Vreg(Zmm0), nokmask, aperm, addr]);

    // Compare all the addresses with this one active VM to look for census
    asm.vpcmpeqq(&[tmpkmask, kmask, Vreg(Zmm0), addr]);
    
    // kxorw k2, k2, k1
    asm.raw_bytes(b"\xc5\xec\x47\xd1");
    
    // kortestw k2, k2
    asm.raw_bytes(b"\xc5\xf8\x98\xd2");
    asm.jnz(&[BranchNear(&diverge)]);

    // At this point all enabled VMs are reading the exact same address
    
    // Get the address to read into rcx (since we permed the active VM to all
    // lanes we can safely get the one from the bottom lane)
    asm.set_vecwidth(falkasm::VecWidth::Width128);
    asm.vmovq(&[Reg(Rcx), Vreg(Zmm0)]);
    asm.set_vecwidth(falkasm::VecWidth::Width512);

    // Check that the alignment is good
    if bytes > 1 {
        match bytes {
            2 => asm.test(&[Reg(Rcx), Imm(1)]),
            4 => asm.test(&[Reg(Rcx), Imm(3)]),
            8 => asm.test(&[Reg(Rcx), Imm(7)]),
            _ => panic!("Invalid state"),
        }
        asm.jnz(&[BranchNear(&bad_alignment)]);
    }

    // Set a page table pointer value we can update as we traverse the table
    asm.mov(&[Reg(Rdx), Reg(R9)]);

    // Start a page walk
    for ii in 0..PAGE_TABLE_LAYOUT.len()-1 {
        let shift = PAGE_TABLE_LAYOUT[ii] as i64;
        asm.rol(&[Reg(Rcx), Imm(shift)]);

        // Mask off bits not used in this table
        asm.mov(&[Reg(Rax), Reg(Rcx)]);
        asm.and(&[Reg(Rax), Imm((1 << shift) - 1)]);
        
        // Look up the entry and check for if it is present
        asm.mov(&[Reg(Rsi), Mem(Some(Rdx), Some((Rax, 8)), 0)]);
        asm.test(&[Reg(Rsi), Reg(Rsi)]);
        asm.jz(&[BranchNear(&page_fault_all)]);

        if ii == PAGE_TABLE_LAYOUT.len()-2 {
            // If this is the last translation, fixup the address by shifting it
            // again
            let shift = PAGE_TABLE_LAYOUT[ii+1] as i64;
            asm.rol(&[Reg(Rcx), Imm(shift)]);

            if is_write {
                // Finally on the final translation and if it is a write we
                // need to make sure we're not writing to aliased memory and
                // update dirty bits and dirty status

                // Check dirty bit
                asm.test(&[Reg(Rsi), Imm(SPECIAL_BIT_DIRTY as i64)]);
                asm.jnz(&[BranchShort(&already_dirty)]);

                // Warning: we assume dirty memory is never aliased
                // This allows us to check the aliased bit check if memory was
                // already marked as dirty. Saves us a branch!
                
                // Check if the memory is aliased. We can never write to
                // aliased memory so fault out
                asm.test(&[Reg(Rsi), Imm(SPECIAL_BIT_ALIASED as i64)]);
                asm.jnz(&[BranchNear(&page_fault_all)]);

                // Update fast dirty list
                asm.test(&[Reg(R14), Reg(R14)]);
                asm.jz(&[BranchNear(&fast_dirty_full)]);

                // Get a new free index into the fast dirty list
                asm.sub(&[Reg(R14), Imm(1)]);

                // rdi = r14 * 16
                asm.mov(&[Reg(Rdi), Reg(R14)]);
                asm.shl(&[Reg(Rdi), Imm(4)]);

                // rbx = pointer to page table entry
                asm.lea(&[Reg(Rbx), Mem(Some(Rdx), Some((Rax, 8)), 0)]);

                // Insert into the fast dirty list
                asm.mov(&[Mem(Some(R13), Some((Rdi, 1)), 0), Reg(Rcx)]);
                asm.mov(&[Mem(Some(R13), Some((Rdi, 1)), 8), Reg(Rbx)]);

                // Set the dirty bit on the page
                asm.or(&[Mem(Some(Rdx), Some((Rax, 8)), 0),
                    Imm(SPECIAL_BIT_DIRTY as i64)]);

                asm.label(&already_dirty);     
            }
        }
        
        // Update page table lookup
        asm.mov(&[Reg(Rdx), Reg(Rsi)]);
       
        if ii == PAGE_TABLE_LAYOUT.len()-2 {
            // Mask off the metadata on the last level translation as there
            // are special bits stored in this part of the table
            asm.and(&[Reg(Rdx), Imm(!(SPECIAL_BIT_MASK as i64))]);
        }
    }

    // Okay, at this point rdx points to the page contents of the page that
    // contains the address that was just translated.

    // Compute the offset for the 8-byte qword in the page which contains the
    // memory in question
    asm.and(&[Reg(Rcx), Imm(PAGE_INDEX_MASK as i64 & !7)]);
    asm.shl(&[Reg(Rcx), Imm(4)]);

    // Now rdx+rcx points to the permissions for the qword which contains the
    // address requested

    // Load the permissions
    asm.vmovdqa64(&[Vreg(Zmm0), kmaskz, Mem(Some(Rdx), Some((Rcx, 1)), 0)]);

    match bytes {
        1 | 2 | 4 => {
            // Grab the alignment from the address
            if bytes == 1 {
                asm.vpandq(&[Vreg(Zmm1), kmaskz, addr, csbc!(7)]);
            } else if bytes == 2 {
                asm.vpandq(&[Vreg(Zmm1), kmaskz, addr, csbc!(6)]);
            } else if bytes == 4 {
                asm.vpandq(&[Vreg(Zmm1), kmaskz, addr, csbc!(4)]);
            } else {
                panic!("How'd you get here?");
            }

            // Multiply byte alignment by 8
            asm.vpsllvq(&[Vreg(Zmm1), kmaskz, Vreg(Zmm1), csbc!(3)]);

            // Shift the permission into position
            asm.vpsrlvq(&[Vreg(Zmm2), kmaskz, Vreg(Zmm0), Vreg(Zmm1)]);

            // DO NOT TOUCH ZMM1 WE USE IT LATER
            
            // Check if permissions match
            asm.vpandq(&[Vreg(Zmm2), kmaskz, Vreg(Zmm2), csbc!(perms)]);
            asm.vpcmpeqq(&[tmpkmask, kmask, Vreg(Zmm2), csbc!(perms)]);
        }
        8 => {
            // Check if permissions match
            asm.vpandq(&[Vreg(Zmm2), kmaskz, Vreg(Zmm0), csbc!(perms)]);
            asm.vpcmpeqq(&[tmpkmask, kmask, Vreg(Zmm2), csbc!(perms)]);
        }
        _ => panic!("Invalid state"),
    }

    // Validate the permissions match

    // kxorw k2, k2, k1
    asm.raw_bytes(b"\xc5\xec\x47\xd1");

    // kortestw k2, k2
    asm.raw_bytes(b"\xc5\xf8\x98\xd2");
    asm.jnz(&[BranchNear(&page_fault)]);

    // At this point permissions are checked and valid!

    if is_write {
        match bytes {
            1 | 2 | 4 => {
                {
                    // Raw update
                    // Rotate permissions into the LSB
                    asm.vprorvq(&[Vreg(Zmm0), kmask, Vreg(Zmm0), Vreg(Zmm1)]);

                    asm.vpandq(&[Vreg(Zmm2), kmask, Vreg(Zmm0), csbc!(rawbit)]);
                    asm.vpsrlvq(&[Vreg(Zmm2), kmask, Vreg(Zmm2), csbc!(3)]);
                    asm.vporq(&[Vreg(Zmm0), kmask, Vreg(Zmm0), Vreg(Zmm2)]);
                    
                    asm.vprolvq(&[Vreg(Zmm0), kmask, Vreg(Zmm0), Vreg(Zmm1)]);

                    // Update permissions
                    asm.vmovdqa64(&[Mem(Some(Rdx), Some((Rcx, 1)), 0), kmask, Vreg(Zmm0)]);
                }
            }
            8 => {
                {
                    // Raw update
                    asm.vpandq(&[Vreg(Zmm2), kmask, Vreg(Zmm0), csbc!(rawbit)]);
                    asm.vpsrlvq(&[Vreg(Zmm2), kmask, Vreg(Zmm2), csbc!(3)]);
                    asm.vporq(&[Vreg(Zmm0), kmask, Vreg(Zmm0), Vreg(Zmm2)]);

                    // Update permissions
                    asm.vmovdqa64(&[Mem(Some(Rdx), Some((Rcx, 1)), 0), kmask, Vreg(Zmm0)]);
                }
            }
            _ => panic!("Unhandled byte combination"),
        }
    }

    asm.lea(&[Reg(Rax), Mem(Some(Rdx), Some((Rcx, 1)), 0)]);
    asm.set_vecwidth(falkasm::VecWidth::Width128);
    asm.vmovq(&[transaddr, Reg(Rax)]);
    asm.set_vecwidth(falkasm::VecWidth::Width512);
    asm.vpbroadcastq(&[transaddr, kmask, transaddr]);

    // Access complete
    asm.ret(&[]);

    // At this point we're handling multiple addresses at once
    asm.label(&diverge);

    // Save off the initial online state
    // kmovw k3, k1
    asm.raw_bytes(b"\xc5\xf8\x90\xd9");

    // Check alignment
    if bytes > 1 {
        match bytes {
            2 => asm.vptestmq(&[tmpkmask, kmask, addr, csbc!(1)]),
            4 => asm.vptestmq(&[tmpkmask, kmask, addr, csbc!(3)]),
            8 => asm.vptestmq(&[tmpkmask, kmask, addr, csbc!(7)]),
            _ => panic!("Invalid state"),
        }

        // kortestw k2, k2
        asm.raw_bytes(b"\xc5\xf8\x98\xd2");
        asm.jnz(&[BranchNear(&bad_alignment)]);
    }

    // Zmm0 = broadcasted page table base
    asm.set_vecwidth(falkasm::VecWidth::Width128);
    asm.vmovq(&[Vreg(Zmm0), Reg(R9)]);
    asm.set_vecwidth(falkasm::VecWidth::Width512);
    asm.vpbroadcastq(&[Vreg(Zmm0), kmask, Vreg(Zmm0)]);

    // Start a page walk
    let mut shiftsum = 3;
    for ii in 0..PAGE_TABLE_LAYOUT.len()-1 {
        let shift = PAGE_TABLE_LAYOUT[ii] as u64;
        shiftsum += shift;

        asm.vprolvq(&[Vreg(Zmm1), kmask, addr, csbc!(shiftsum)]);
        asm.vpandq(&[Vreg(Zmm1), kmask, Vreg(Zmm1), csbc!(((1 << (shift + 3)) - 1) & !0x7)]);
        asm.vpaddq(&[Vreg(Zmm1), kmask, Vreg(Zmm0), Vreg(Zmm1)]);
    
        asm.raw_bytes(b"\xc5\xf8\x90\xd1"); // kmovw k2, k1
        asm.vpgatherqq(&[Vreg(Zmm0), tmpkmask, Mvidx(None, (Zmm1, 1), 0)]);
        asm.vpcmpeqq(&[tmpkmask, kmask, Vreg(Zmm0), csbc!(0)]);

        // kandnw k1, k2, k1
        // This means kmask = (kmask & ~tmpkmask). Effectively the VM's that
        // were going to fault (set to one in tmpkmask) will be turned off
        // allowing us to continue gathering exception information at deeper
        // stages for the VMs remaining on
        asm.raw_bytes(b"\xc5\xec\x42\xc9");
        asm.raw_bytes(b"\xc5\xf8\x98\xc9"); // kortestw k1, k1
        asm.jz(&[BranchNear(&page_fault_vectorized)]);
        
        if ii == PAGE_TABLE_LAYOUT.len()-2 {
            if is_write {
                // Make sure all the dirty bits are set if we want to skip
                // updating dirty information
                asm.vptestmq(&[tmpkmask, kmask, Vreg(Zmm0), csbc!(SPECIAL_BIT_DIRTY as u64)]);
                asm.raw_bytes(b"\xc5\xec\x47\xd1"); // kxorw k2, k2, k1
                asm.raw_bytes(b"\xc5\xf8\x98\xd2"); // kortestw k2, k2
                asm.jz(&[BranchNear(&already_dirty_vec)]);

                // Warning: we assume dirty memory is never aliased
                // This allows us to check the aliased bit check if memory was
                // already marked as dirty. Saves us a branch!
                
                // If any of the VMs have memory marked as aliased then we
                // page fault
                asm.vptestmq(&[tmpkmask, kmask, Vreg(Zmm0), csbc!(SPECIAL_BIT_ALIASED as u64)]);

                // kandnw k1, k2, k1
                // Same as above, disable faulting VMs
                asm.raw_bytes(b"\xc5\xec\x42\xc9");
                asm.raw_bytes(b"\xc5\xf8\x98\xc9"); // kortestw k1, k1
                asm.jz(&[BranchNear(&page_fault_vectorized)]);

                // We could maybe do some AVX-512 implementation of the dirty
                // updates via compression and interleaving of addresses
                // however it would be hard to prevent duplicates in the dirty
                // list thus we just simply unroll. Dirty updates should
                // hopefully never really matter as they are self-silencing

                asm.raw_bytes(b"\xc5\xf8\x93\xc1"); // kmovw eax, k1
                asm.vmovdqa64(&[Mem(Some(R8), None, 0x00), kmask, Vreg(Zmm1)]);
                asm.vmovdqa64(&[Mem(Some(R8), None, 0x40), kmask, addr]);

                asm.xor(&[Reg(Rcx), Reg(Rcx)]);

                asm.label(&dirtyloop);

                asm.shr(&[Reg(Rax), Imm(1)]);
                asm.jnc(&[BranchShort(&nextiter)]);

                asm.mov(&[Reg(Rbx), Mem(Some(R8), Some((Rcx, 1)), 0)]);
                asm.mov(&[Reg(Rdx), Mem(Some(R8), Some((Rcx, 1)), 0x40)]);
                
                asm.test(&[Mem(Some(Rbx), None, 0), Imm(SPECIAL_BIT_DIRTY as i64)]);
                asm.jnz(&[BranchShort(&nextiter)]);

                // Update fast dirty list
                asm.test(&[Reg(R14), Reg(R14)]);
                asm.jz(&[BranchNear(&fast_dirty_full)]);

                // Get a new free index into the fast dirty list
                asm.sub(&[Reg(R14), Imm(1)]);

                // rdi = r14 * 16
                asm.mov(&[Reg(Rdi), Reg(R14)]);
                asm.shl(&[Reg(Rdi), Imm(4)]);

                // Insert into the fast dirty list
                asm.mov(&[Mem(Some(R13), Some((Rdi, 1)), 0), Reg(Rdx)]);
                asm.mov(&[Mem(Some(R13), Some((Rdi, 1)), 8), Reg(Rbx)]);

                // Set the dirty bit on the page
                asm.or(&[Mem(Some(Rbx), None, 0), Imm(SPECIAL_BIT_DIRTY as i64)]);

                asm.label(&nextiter);
                asm.add(&[Reg(Rcx), Imm(8)]);
                asm.test(&[Reg(Rax), Reg(Rax)]);
                asm.jnz(&[BranchShort(&dirtyloop)]);

                asm.label(&already_dirty_vec);
            }

            // Mask off the metadata on the last level translation as there
            // are special bits stored in this part of the table
            asm.vpandq(&[Vreg(Zmm0), kmask, Vreg(Zmm0), csbc!(!(SPECIAL_BIT_MASK as u64))]);
        }
    }

    // Compute the offset for the 8-byte qword in the page which contains the
    // memory in question
    asm.vpandq(&[Vreg(Zmm1), kmask, addr, csbc!(PAGE_INDEX_MASK as u64 & !7)]);
    asm.vpsllvq(&[Vreg(Zmm1), kmask, Vreg(Zmm1), csbc!(4)]);
    asm.vpaddq(&[Vreg(Zmm3), kmask, Vreg(Zmm1), Vreg(Zmm0)]);

    asm.mov(&[Reg(Rax), Imm(OFFSETS.as_ptr() as i64)]);
    asm.vpaddq(&[Vreg(Zmm3), kmask, Vreg(Zmm3), Mem(Some(Rax), None, 0)]);
   
    // Load the permissions
    asm.raw_bytes(b"\xc5\xf8\x90\xd1"); // kmovw k2, k1
    asm.vpgatherqq(&[Vreg(Zmm0), tmpkmask, Mvidx(None, (Zmm3, 1), 0)]);

    match bytes {
        1 | 2 | 4 => {
            // Grab the alignment from the address
            if bytes == 1 {
                asm.vpandq(&[Vreg(Zmm1), kmaskz, addr, csbc!(7)]);
            } else if bytes == 2 {
                asm.vpandq(&[Vreg(Zmm1), kmaskz, addr, csbc!(6)]);
            } else if bytes == 4 {
                asm.vpandq(&[Vreg(Zmm1), kmaskz, addr, csbc!(4)]);
            } else {
                panic!("How'd you get here?");
            }

            // Multiply byte alignment by 8
            asm.vpsllvq(&[Vreg(Zmm1), kmaskz, Vreg(Zmm1), csbc!(3)]);

            // Shift the permission into position
            asm.vpsrlvq(&[Vreg(Zmm2), kmaskz, Vreg(Zmm0), Vreg(Zmm1)]);

            // DO NOT TOUCH ZMM1 WE USE IT LATER
            
            // Check if permissions match
            asm.vpandq(&[Vreg(Zmm2), kmaskz, Vreg(Zmm2), csbc!(perms)]);
            asm.vpcmpeqq(&[tmpkmask, kmask, Vreg(Zmm2), csbc!(perms)]);
        }
        8 => {
            // Check if permissions match
            asm.vpandq(&[Vreg(Zmm2), kmaskz, Vreg(Zmm0), csbc!(perms)]);
            asm.vpcmpeqq(&[tmpkmask, kmask, Vreg(Zmm2), csbc!(perms)]);
        }
        _ => panic!("Invalid state"),
    }

    // Validate the permissions match

    // kandw k1, k2, k1
    // Same as above, disable faulting VMs
    asm.raw_bytes(b"\xc5\xec\x41\xc9");

    // We now deliver all the possible page faults at the same time if any
    // VM has been masked off up to this point.
    asm.raw_bytes(b"\xc5\xf4\x47\xd3"); // kxorw k2, k1, k3
    asm.raw_bytes(b"\xc5\xf8\x98\xd2"); // kortestw k2, k2
    asm.jnz(&[BranchNear(&page_fault)]);

    // At this point permissions are checked and valid!

    if is_write {
        match bytes {
            1 | 2 | 4 => {
                // Raw update
                // Rotate permissions into the LSB
                asm.vprorvq(&[Vreg(Zmm0), kmask, Vreg(Zmm0), Vreg(Zmm1)]);

                asm.vpandq(&[Vreg(Zmm2), kmask, Vreg(Zmm0), csbc!(rawbit)]);
                asm.vpsrlvq(&[Vreg(Zmm2), kmask, Vreg(Zmm2), csbc!(3)]);
                asm.vporq(&[Vreg(Zmm0), kmask, Vreg(Zmm0), Vreg(Zmm2)]);
                
                asm.vprolvq(&[Vreg(Zmm0), kmask, Vreg(Zmm0), Vreg(Zmm1)]);

                // Update permissions
                asm.raw_bytes(b"\xc5\xf8\x90\xd1"); // kmovw k2, k1
                asm.vpscatterqq(&[Mvidx(None, (Zmm3, 1), 0), tmpkmask, Vreg(Zmm0)]);
            }
            8 => {
                // Raw update
                asm.vpandq(&[Vreg(Zmm2), kmask, Vreg(Zmm0), csbc!(rawbit)]);
                asm.vpsrlvq(&[Vreg(Zmm2), kmask, Vreg(Zmm2), csbc!(3)]);
                asm.vporq(&[Vreg(Zmm0), kmask, Vreg(Zmm0), Vreg(Zmm2)]);

                // Update permissions
                asm.raw_bytes(b"\xc5\xf8\x90\xd1"); // kmovw k2, k1
                asm.vpscatterqq(&[Mvidx(None, (Zmm3, 1), 0), tmpkmask, Vreg(Zmm0)]);
            }
            _ => panic!("Unhandled byte combination"),
        }
    }
    
    // Update the translated address
    asm.vmovdqa64(&[transaddr, kmask, Vreg(Zmm3)]);

    asm.ret(&[]);

    // =======================================================================
    
    asm.label(&bad_alignment);

    // Alignment was bad, return out of the emulator with the correct error
    // code and information

    // Save the fault code into Rax
    if is_write {
        asm.mov(&[Reg(Rax), Imm(0x1dead040 | bytes as i64)]);
    } else {
        asm.mov(&[Reg(Rax), Imm(0x1dead030 | bytes as i64)]);
    }
    
    // Compute the caused_vmexit mask
    match bytes {
        1 => asm.int3(&[]), // This should never happen
        2 => asm.vptestmq(&[tmpkmask, kmask, addr, csbc!(1)]),
        4 => asm.vptestmq(&[tmpkmask, kmask, addr, csbc!(3)]),
        8 => asm.vptestmq(&[tmpkmask, kmask, addr, csbc!(7)]),
        _ => panic!("Invalid state"),
    }

    // Save the faulting address into Zmm0
    asm.vmovdqa64(&[Vreg(Zmm0), kmaskz, addr]);
    asm.add(&[Reg(Rsp), Imm(8)]);
    asm.ret(&[]);
    
    // =======================================================================

    asm.label(&page_fault_vectorized);

    // Vectorized faults result in VMs being disabled, thus the difference
    // between the initial mask and the remaining mask
    
    asm.raw_bytes(b"\xc5\xf4\x47\xd3"); // kxorw k2, k1, k3
    asm.jmp(&[BranchNear(&page_fault)]);

    asm.label(&page_fault_all);
    
    // All lanes caused a fault
    asm.raw_bytes(b"\xc5\xf8\x90\xd1"); // kmovw k2, k1

    asm.label(&page_fault);

    // Page fault! Return out of the emulator with the correct error
    // code and information

    // Save the fault code into Rax
    if is_write {
        asm.mov(&[Reg(Rax), Imm(0x1dead020 | bytes as i64)]);
    } else {
        asm.mov(&[Reg(Rax), Imm(0x1dead010 | bytes as i64)]);
    }

    // Save the faulting address into Zmm0
    asm.vmovdqa64(&[Vreg(Zmm0), kmaskz, addr]);
    asm.add(&[Reg(Rsp), Imm(8)]);
    asm.ret(&[]);
                
    // =======================================================================
    
    // The fast dirty list was full, we have to fault out requesting a flush
    // of the list to somewhere else

    asm.label(&fast_dirty_full);

    // Generic error, not specific to a lane so just report all lanes as
    // caused_vmexit.
    asm.raw_bytes(b"\xc5\xf8\x90\xd1"); // kmovw k2, k1

    asm.mov(&[Reg(Rax), Imm(0x1dead055)]);
    asm.add(&[Reg(Rsp), Imm(8)]);
    asm.ret(&[]);
}

// Expects:
//
// rax - scratch
// rbx - scratch
// rcx - scratch
// rdx - scratch
// rsi - scratch
// rdi - scratch
// R8  - Points to 64-byte aligned scratch memory
// R9  - Points to root page table
// R11 - Points to conststore table
// R13 - Fast dirty list
// R14 - Fast dirty remaining free entries
//
pub fn mmu_access(asm: &mut AsmStream, conststore: &mut ConstStore,
        bytes: usize, addr: Operand, transaddr: Operand,
        output: Option<Operand>, input: Option<Operand>) {
    assert!(bytes == 8 || bytes == 4 || bytes == 2 || bytes == 1,
        "Unsupported operation size for mmu_jit");

    let diverge         = format!("diverge{}", asm.len());

    // Permute register which contains all lanes set to the index of any
    // of the VMs that is actively running
    let aperm = Vreg(Zmm30);

    let nokmask   = Operand::KmaskRegister(KmaskType::Merge(KmaskReg::K0));
    let kmask     = Operand::KmaskRegister(KmaskType::Merge(KmaskReg::K1));
    let kmaskz    = Operand::KmaskRegister(KmaskType::Zero(KmaskReg::K1));
    let tmpkmask  = Operand::KmaskRegister(KmaskType::Merge(KmaskReg::K2));
    //let tmpkmaskz = Operand::KmaskRegister(KmaskType::Zero(KmaskReg::K2));

    // Allocate a new constant variable in the constant storage database
    // and construct a memory operand that refers to it
    /*macro_rules! cs {
        ($constant:expr) => ({
            let idx = conststore.add_const($constant) as i64;
            Mem(Some(R11), None, idx * 8)
        })
    }*/

    macro_rules! csbc {
        ($constant:expr) => ({
            let idx = conststore.add_const($constant) as i64;
            Membc(Some(R11), None, idx * 8)
        })
    }

    let is_read  = output.is_some() && input.is_none();
    let is_write = input.is_some() && output.is_none();
    assert!((is_read || is_write) && !(is_read && is_write),
        "Invalid arguments to mmu_jit");

    let addrreg = if let Vreg(addr) = transaddr { addr } else { panic!("Whoops") };

    // Compute shift amounts based on the alignment of the address
    match bytes {
        1 | 2 | 4 => {
            // Grab the alignment from the address
            if bytes == 1 {
                asm.vpandq(&[Vreg(Zmm1), kmaskz, addr, csbc!(7)]);
            } else if bytes == 2 {
                asm.vpandq(&[Vreg(Zmm1), kmaskz, addr, csbc!(6)]);
            } else if bytes == 4 {
                asm.vpandq(&[Vreg(Zmm1), kmaskz, addr, csbc!(4)]);
            } else {
                panic!("How'd you get here?");
            }

            // Multiply byte alignment by 8
            asm.vpsllvq(&[Vreg(Zmm1), kmaskz, Vreg(Zmm1), csbc!(3)]);
        }
        8 => {
            // No shift amount used in this case
        }
        _ => panic!("Invalid state"),
    }

    // Broadcast one of the active VMs to all lanes
    // It's important we use nokmask here so we can extract the address easily
    // with a vmovq later
    asm.vpermq(&[Vreg(Zmm0), nokmask, aperm, addr]);

    // Compare all the addresses with this one active VM to look for census
    asm.vpcmpeqq(&[tmpkmask, kmask, Vreg(Zmm0), addr]);
    
    // kxorw k2, k2, k1
    asm.raw_bytes(b"\xc5\xec\x47\xd1");
    
    // kortestw k2, k2
    asm.raw_bytes(b"\xc5\xf8\x98\xd2");
    asm.jnz(&[BranchNear(&diverge)]);

    // Get the address we want to read from
    asm.vpermq(&[Vreg(Zmm0), nokmask, aperm, transaddr]);
    asm.set_vecwidth(falkasm::VecWidth::Width128);
    asm.vmovq(&[Reg(Rdx), Vreg(Zmm0)]);
    asm.set_vecwidth(falkasm::VecWidth::Width512);

    if is_write {
        match bytes {
            1 | 2 | 4 => {
                // Read the original memory
                asm.vmovdqa64(&[Vreg(Zmm0), kmask,
                                Mem(Some(Rdx), None, 64)]);

                // Rotate the word so the memory we are inserting is at
                // the LSB
                //
                // Zmm1 comes from above and it's the shift amount to get
                // the byte/word into the LSB
                asm.vprorvq(&[Vreg(Zmm0), kmaskz, Vreg(Zmm0), Vreg(Zmm1)]);

                // Zero out the bottom byte
                asm.vpandq(&[Vreg(Zmm0), kmaskz, Vreg(Zmm0), csbc!(!((1 << (bytes * 8))-1))]);

                // Mask off everything but the bottom byte for the value
                // we want to insert.
                asm.vpandq(&[Vreg(Zmm2), kmaskz, input.unwrap(), csbc!((1 << (bytes * 8))-1)]);

                // Merge old and new values
                asm.vporq(&[Vreg(Zmm0), kmaskz, Vreg(Zmm0), Vreg(Zmm2)]);

                // Rotate value back into original position
                asm.vprolvq(&[Vreg(Zmm0), kmaskz, Vreg(Zmm0), Vreg(Zmm1)]);

                // Write out actual value
                asm.vmovdqa64(&[Mem(Some(Rdx), None, 64), kmask, Vreg(Zmm0)]);
            }
            8 => {
                // Write the actual value out
                asm.vmovdqa64(&[Mem(Some(Rdx), None, 64), kmask, input.unwrap()]);
            }
            _ => panic!("Unhandled byte combination"),
        }
    } else {
        let out = output.unwrap();

        // Load the actual memory contents
        asm.vmovdqa64(&[out, kmask, Mem(Some(Rdx), None, 64)]);

        match bytes {
            1 | 2 | 4 => {
                // Shift the memory into position based on the shift amount
                // calculated during the permission checks
                asm.vpsrlvq(&[out, kmask, out, Vreg(Zmm1)]);

                // Sign extend
                asm.vpsllvq(&[out, kmask, out,
                            csbc!(64 - (bytes as u64 * 8))]);
                asm.vpsravq(&[out, kmask, out,
                            csbc!(64 - (bytes as u64 * 8))]);
            }

            // Nothing to do if it was a qword value
            8 => {}

            _ => panic!("Invalid state"),
        }
    }

    // Access complete
    asm.ret(&[]);
    
    // At this point we're handling multiple addresses at once
    asm.label(&diverge);

    if is_write {
        match bytes {
            1 | 2 | 4 => {
                // Read the original memory
                asm.raw_bytes(b"\xc5\xf8\x90\xd1"); // kmovw k2, k1
                asm.vpgatherqq(&[Vreg(Zmm0), tmpkmask,
                                Mvidx(None, (addrreg, 1), 64)]);

                // Rotate the word so the memory we are inserting is at
                // the LSB
                //
                // Zmm1 comes from above and it's the shift amount to get
                // the byte/word into the LSB
                asm.vprorvq(&[Vreg(Zmm0), kmaskz, Vreg(Zmm0), Vreg(Zmm1)]);

                // Zero out the bottom byte
                asm.vpandq(&[Vreg(Zmm0), kmaskz, Vreg(Zmm0), csbc!(!((1 << (bytes * 8))-1))]);

                // Mask off everything but the bottom byte for the value
                // we want to insert.
                asm.vpandq(&[Vreg(Zmm2), kmaskz, input.unwrap(), csbc!((1 << (bytes * 8))-1)]);

                // Merge old and new values
                asm.vporq(&[Vreg(Zmm0), kmaskz, Vreg(Zmm0), Vreg(Zmm2)]);

                // Rotate value back into original position
                asm.vprolvq(&[Vreg(Zmm0), kmaskz, Vreg(Zmm0), Vreg(Zmm1)]);

                // Write out actual value
                asm.raw_bytes(b"\xc5\xf8\x90\xd1"); // kmovw k2, k1
                asm.vpscatterqq(&[Mvidx(None, (addrreg, 1), 64), tmpkmask, Vreg(Zmm0)]);
            }
            8 => {
                // Write the actual value out
                asm.raw_bytes(b"\xc5\xf8\x90\xd1"); // kmovw k2, k1
                asm.vpscatterqq(&[Mvidx(None, (addrreg, 1), 64), tmpkmask, input.unwrap()]);
            }
            _ => panic!("Unhandled byte combination"),
        }
    } else {
        let out = output.unwrap();

        // Load the actual memory contents
        asm.raw_bytes(b"\xc5\xf8\x90\xd1"); // kmovw k2, k1
        asm.vpgatherqq(&[out, tmpkmask, Mvidx(None, (addrreg, 1), 64)]);

        match bytes {
            1 | 2 | 4 => {
                // Shift the memory into position based on the shift amount
                // calculated during the permission checks
                asm.vpsrlvq(&[out, kmask, out, Vreg(Zmm1)]);

                // Sign extend
                asm.vpsllvq(&[out, kmask, out,
                            csbc!(64 - (bytes as u64 * 8))]);
                asm.vpsravq(&[out, kmask, out,
                            csbc!(64 - (bytes as u64 * 8))]);
            }

            // Nothing to do if it was a qword value
            8 => {}

            _ => panic!("Invalid state"),
        }
    }

    asm.ret(&[]);
}

#[cfg(test)]
mod jit_tests {
    use std::sync::Arc;
    use crate::{SoftMMU, VirtAddr};
    use crate::{PERM_READ, PERM_WRITE, PERM_RAW, PAGE_SIZE};
    use crate::tests::xorshift;
    use crate::tests::TEST_VADDR;
    use crate::avx512_jit::{alloc_rwx, mmu_translate, mmu_access};
    use falkasm::{AsmStream, VecWidth, AsmMode};
    use conststore::ConstStore;
    use falkasm::Operand::VecRegister as Vreg;
    use falkasm::VecReg::*;
    use vectorized::{Vector, Mask};
    use safecast::SafeCast;

    const TEST_ARENA_SIZE: usize = 1024;

    #[derive(Default)]
    #[repr(C)]
    struct Context {
        // Normal GPRs
        rax: u64,
        rcx: u64,
        rdx: u64,
        rbx: u64,
        rsp: u64,
        rbp: u64,
        rsi: u64,
        rdi: u64,
        r8:  u64,
        r9:  u64,
        r10: u64,
        r11: u64,
        r12: u64,
        r13: u64,
        r14: u64,
        r15: u64,

        // Location to jump to
        rip: u64,

        // All kmask registers
        kmask: [u16; 8],

        // All ZMM registers
        zmm: [Vector; 32],
    }

    // Transition into a VM with `context`, updating context on return.
    // This preserves the previous context of the VM and restores it when the
    // VM exits
    //
    // It's up to the VM code to not clobber state it is not supposed to. We
    // will end up jumping to `context.rip` and we expect it to return out
    //
    // We should be using xsave/xrstor but we want to just play it safe here
    // and since this is used for testing we want direct access to all the
    // registers from Rust.
    //
    // This hotloops the call for R15 iterations
    unsafe fn enter_vm(context: &mut Context) {
        // Storage for pre-VM context
        let mut saved_context = Context::default();

        asm!(r#"
            // Save the context pointers to the stack
            push $1
            push rax

            // Save off context
            mov [$1 + 0x00], rax
            mov [$1 + 0x08], rcx
            mov [$1 + 0x10], rdx
            mov [$1 + 0x18], rbx
            mov [$1 + 0x20], rsp
            mov [$1 + 0x28], rbp
            mov [$1 + 0x30], rsi
            mov [$1 + 0x38], rdi
            mov [$1 + 0x40], r8
            mov [$1 + 0x48], r9
            mov [$1 + 0x50], r10
            mov [$1 + 0x58], r11
            mov [$1 + 0x60], r12
            mov [$1 + 0x68], r13
            mov [$1 + 0x70], r14
            mov [$1 + 0x78], r15

            // RIP doesn't get saved/restored
            mov qword ptr [$1 + 0x80], 0

            kmovw [$1 + 0x88], k0
            kmovw [$1 + 0x8a], k1
            kmovw [$1 + 0x8c], k2
            kmovw [$1 + 0x8e], k3
            kmovw [$1 + 0x90], k4
            kmovw [$1 + 0x92], k5
            kmovw [$1 + 0x94], k6
            kmovw [$1 + 0x96], k7

            vmovdqa64 [$1 + 0x0c0], zmm0
            vmovdqa64 [$1 + 0x100], zmm1
            vmovdqa64 [$1 + 0x140], zmm2
            vmovdqa64 [$1 + 0x180], zmm3
            vmovdqa64 [$1 + 0x1c0], zmm4
            vmovdqa64 [$1 + 0x200], zmm5
            vmovdqa64 [$1 + 0x240], zmm6
            vmovdqa64 [$1 + 0x280], zmm7
            vmovdqa64 [$1 + 0x2c0], zmm8
            vmovdqa64 [$1 + 0x300], zmm9
            vmovdqa64 [$1 + 0x340], zmm10
            vmovdqa64 [$1 + 0x380], zmm11
            vmovdqa64 [$1 + 0x3c0], zmm12
            vmovdqa64 [$1 + 0x400], zmm13
            vmovdqa64 [$1 + 0x440], zmm14
            vmovdqa64 [$1 + 0x480], zmm15
            vmovdqa64 [$1 + 0x4c0], zmm16
            vmovdqa64 [$1 + 0x500], zmm17
            vmovdqa64 [$1 + 0x540], zmm18
            vmovdqa64 [$1 + 0x580], zmm19
            vmovdqa64 [$1 + 0x5c0], zmm20
            vmovdqa64 [$1 + 0x600], zmm21
            vmovdqa64 [$1 + 0x640], zmm22
            vmovdqa64 [$1 + 0x680], zmm23
            vmovdqa64 [$1 + 0x6c0], zmm24
            vmovdqa64 [$1 + 0x700], zmm25
            vmovdqa64 [$1 + 0x740], zmm26
            vmovdqa64 [$1 + 0x780], zmm27
            vmovdqa64 [$1 + 0x7c0], zmm28
            vmovdqa64 [$1 + 0x800], zmm29
            vmovdqa64 [$1 + 0x840], zmm30
            vmovdqa64 [$1 + 0x880], zmm31

            // Load new context
            kmovw k0, [rax + 0x88]
            kmovw k1, [rax + 0x8a]
            kmovw k2, [rax + 0x8c]
            kmovw k3, [rax + 0x8e]
            kmovw k4, [rax + 0x90]
            kmovw k5, [rax + 0x92]
            kmovw k6, [rax + 0x94]
            kmovw k7, [rax + 0x96]

            vmovdqa64 zmm0, [rax + 0x0c0]
            vmovdqa64 zmm1, [rax + 0x100]
            vmovdqa64 zmm2, [rax + 0x140]
            vmovdqa64 zmm3, [rax + 0x180]
            vmovdqa64 zmm4, [rax + 0x1c0]
            vmovdqa64 zmm5, [rax + 0x200]
            vmovdqa64 zmm6, [rax + 0x240]
            vmovdqa64 zmm7, [rax + 0x280]
            vmovdqa64 zmm8, [rax + 0x2c0]
            vmovdqa64 zmm9, [rax + 0x300]
            vmovdqa64 zmm10, [rax + 0x340]
            vmovdqa64 zmm11, [rax + 0x380]
            vmovdqa64 zmm12, [rax + 0x3c0]
            vmovdqa64 zmm13, [rax + 0x400]
            vmovdqa64 zmm14, [rax + 0x440]
            vmovdqa64 zmm15, [rax + 0x480]
            vmovdqa64 zmm16, [rax + 0x4c0]
            vmovdqa64 zmm17, [rax + 0x500]
            vmovdqa64 zmm18, [rax + 0x540]
            vmovdqa64 zmm19, [rax + 0x580]
            vmovdqa64 zmm20, [rax + 0x5c0]
            vmovdqa64 zmm21, [rax + 0x600]
            vmovdqa64 zmm22, [rax + 0x640]
            vmovdqa64 zmm23, [rax + 0x680]
            vmovdqa64 zmm24, [rax + 0x6c0]
            vmovdqa64 zmm25, [rax + 0x700]
            vmovdqa64 zmm26, [rax + 0x740]
            vmovdqa64 zmm27, [rax + 0x780]
            vmovdqa64 zmm28, [rax + 0x7c0]
            vmovdqa64 zmm29, [rax + 0x800]
            vmovdqa64 zmm30, [rax + 0x840]
            vmovdqa64 zmm31, [rax + 0x880]

            mov rcx, [rax + 0x08]
            mov rdx, [rax + 0x10]
            mov rbx, [rax + 0x18]
            //mov rsp, [rax + 0x20]
            mov rbp, [rax + 0x28]
            mov rsi, [rax + 0x30]
            mov rdi, [rax + 0x38]
            mov r8,  [rax + 0x40]
            mov r9,  [rax + 0x48]
            mov r10, [rax + 0x50]
            mov r11, [rax + 0x58]
            mov r12, [rax + 0x60]
            mov r13, [rax + 0x68]
            mov r14, [rax + 0x70]
            mov r15, [rax + 0x78]
            push qword ptr [rax + 0x80] // Save jump target to stack
            mov rax, [rax + 0x00]       // Load rax last

            call 2f
            jmp  3f

        2:
            // Call into jump target
            call qword ptr [rsp + 8]
            dec r15
            jnz 2b

            xor rax, rax // If we got to here we succeeded
            ret

        3:
            // Pop off call target
            add rsp, 8

            // Save rax value
            push rax
            mov  rax, qword ptr [rsp + 8] // Load context pointer

            // Save off context
            mov [rax + 0x08], rcx
            mov [rax + 0x10], rdx
            mov [rax + 0x18], rbx
            mov [rax + 0x20], rsp
            mov [rax + 0x28], rbp
            mov [rax + 0x30], rsi
            mov [rax + 0x38], rdi
            mov [rax + 0x40], r8
            mov [rax + 0x48], r9
            mov [rax + 0x50], r10
            mov [rax + 0x58], r11
            mov [rax + 0x60], r12
            mov [rax + 0x68], r13
            mov [rax + 0x70], r14
            mov [rax + 0x78], r15

            // Save off original rax value
            pop qword ptr [rax + 0x00]

            // Restore context pointers
            pop rax
            pop $1

            // RIP doesn't get saved/restored
            mov qword ptr [rax + 0x80], 0

            kmovw [rax + 0x88], k0
            kmovw [rax + 0x8a], k1
            kmovw [rax + 0x8c], k2
            kmovw [rax + 0x8e], k3
            kmovw [rax + 0x90], k4
            kmovw [rax + 0x92], k5
            kmovw [rax + 0x94], k6
            kmovw [rax + 0x96], k7

            vmovdqa64 [rax + 0x0c0], zmm0
            vmovdqa64 [rax + 0x100], zmm1
            vmovdqa64 [rax + 0x140], zmm2
            vmovdqa64 [rax + 0x180], zmm3
            vmovdqa64 [rax + 0x1c0], zmm4
            vmovdqa64 [rax + 0x200], zmm5
            vmovdqa64 [rax + 0x240], zmm6
            vmovdqa64 [rax + 0x280], zmm7
            vmovdqa64 [rax + 0x2c0], zmm8
            vmovdqa64 [rax + 0x300], zmm9
            vmovdqa64 [rax + 0x340], zmm10
            vmovdqa64 [rax + 0x380], zmm11
            vmovdqa64 [rax + 0x3c0], zmm12
            vmovdqa64 [rax + 0x400], zmm13
            vmovdqa64 [rax + 0x440], zmm14
            vmovdqa64 [rax + 0x480], zmm15
            vmovdqa64 [rax + 0x4c0], zmm16
            vmovdqa64 [rax + 0x500], zmm17
            vmovdqa64 [rax + 0x540], zmm18
            vmovdqa64 [rax + 0x580], zmm19
            vmovdqa64 [rax + 0x5c0], zmm20
            vmovdqa64 [rax + 0x600], zmm21
            vmovdqa64 [rax + 0x640], zmm22
            vmovdqa64 [rax + 0x680], zmm23
            vmovdqa64 [rax + 0x6c0], zmm24
            vmovdqa64 [rax + 0x700], zmm25
            vmovdqa64 [rax + 0x740], zmm26
            vmovdqa64 [rax + 0x780], zmm27
            vmovdqa64 [rax + 0x7c0], zmm28
            vmovdqa64 [rax + 0x800], zmm29
            vmovdqa64 [rax + 0x840], zmm30
            vmovdqa64 [rax + 0x880], zmm31

            // Restore context
            kmovw k0, [$1 + 0x88]
            kmovw k1, [$1 + 0x8a]
            kmovw k2, [$1 + 0x8c]
            kmovw k3, [$1 + 0x8e]
            kmovw k4, [$1 + 0x90]
            kmovw k5, [$1 + 0x92]
            kmovw k6, [$1 + 0x94]
            kmovw k7, [$1 + 0x96]

            vmovdqa64 zmm0, [$1 + 0x0c0]
            vmovdqa64 zmm1, [$1 + 0x100]
            vmovdqa64 zmm2, [$1 + 0x140]
            vmovdqa64 zmm3, [$1 + 0x180]
            vmovdqa64 zmm4, [$1 + 0x1c0]
            vmovdqa64 zmm5, [$1 + 0x200]
            vmovdqa64 zmm6, [$1 + 0x240]
            vmovdqa64 zmm7, [$1 + 0x280]
            vmovdqa64 zmm8, [$1 + 0x2c0]
            vmovdqa64 zmm9, [$1 + 0x300]
            vmovdqa64 zmm10, [$1 + 0x340]
            vmovdqa64 zmm11, [$1 + 0x380]
            vmovdqa64 zmm12, [$1 + 0x3c0]
            vmovdqa64 zmm13, [$1 + 0x400]
            vmovdqa64 zmm14, [$1 + 0x440]
            vmovdqa64 zmm15, [$1 + 0x480]
            vmovdqa64 zmm16, [$1 + 0x4c0]
            vmovdqa64 zmm17, [$1 + 0x500]
            vmovdqa64 zmm18, [$1 + 0x540]
            vmovdqa64 zmm19, [$1 + 0x580]
            vmovdqa64 zmm20, [$1 + 0x5c0]
            vmovdqa64 zmm21, [$1 + 0x600]
            vmovdqa64 zmm22, [$1 + 0x640]
            vmovdqa64 zmm23, [$1 + 0x680]
            vmovdqa64 zmm24, [$1 + 0x6c0]
            vmovdqa64 zmm25, [$1 + 0x700]
            vmovdqa64 zmm26, [$1 + 0x740]
            vmovdqa64 zmm27, [$1 + 0x780]
            vmovdqa64 zmm28, [$1 + 0x7c0]
            vmovdqa64 zmm29, [$1 + 0x800]
            vmovdqa64 zmm30, [$1 + 0x840]
            vmovdqa64 zmm31, [$1 + 0x880]

            mov rax, [$1 + 0x00]
            mov rcx, [$1 + 0x08]
            mov rdx, [$1 + 0x10]
            mov rbx, [$1 + 0x18]
            //mov rsp, [$1 + 0x20]
            mov rbp, [$1 + 0x28]
            mov rsi, [$1 + 0x30]
            mov rdi, [$1 + 0x38]
            mov r8,  [$1 + 0x40]
            mov r9,  [$1 + 0x48]
            mov r10, [$1 + 0x50]
            mov r11, [$1 + 0x58]
            mov r12, [$1 + 0x60]
            mov r13, [$1 + 0x68]
            mov r14, [$1 + 0x70]
            mov r15, [$1 + 0x78]

        "# ::
        "{rax}"(context as *mut Context as usize),
        "r"(&mut saved_context as *mut Context as usize) :
        "cc", "memory" : "volatile", "intel");
    }

    /// Performs a rdtscp instruction, returns 64-bit TSC value
    fn rdtsc() -> u64 {
	let high: u32;
	let low:  u32;

	unsafe {
	    asm!("rdtscp" :
		 "={edx}"(high), "={eax}"(low) :: "rcx", "memory" :
		 "volatile", "intel");
	}

	((high as u64) << 32) | (low as u64)
    }

    #[test]
    fn bench_mmujit() {
        const TEST_ITERS: usize = 100000000;

        // Bail out if AVX512 isn't supported
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        // Create a new master
        let mut mmu = SoftMMU::new();
        let vaddr = VirtAddr(TEST_VADDR);

        // Create random bytes for the initial fill of the memory
        let mut meg = Vec::new();
        for _ in 0..TEST_ARENA_SIZE {
            meg.push(xorshift() as u8);
        }

        let mask = Mask::all();
       
        // Map in the memory as RW
        mmu.add_memory(vaddr, meg.len());
        assert!(mmu.set_permissions(Mask::all(), vaddr, meg.len(),
            PERM_READ | PERM_WRITE) == meg.len());
        assert!(mmu.write(Mask::all(), vaddr, &meg[..]) == meg.len());

        // Create a new ASM stream
        let mut asm = AsmStream::new(AsmMode::Bits64);
        asm.set_vecwidth(VecWidth::Width512);
        
        // Create a small constant store
        let mut conststore = ConstStore::new(1024);

        // Keep track of the locations for the different variants of the
        // JIT implementations (read, write, varying sizes, etc)
        let mut jit_locations = Vec::new();

        // JIT out all the code
        for &is_write in &[false, true] {
            for &opsize in &[1, 2, 4, 8] {
                jit_locations.push(asm.len());
                
                // Translate addresses into zmm7
                mmu_translate(&mut asm, &mut conststore, opsize,
                              Vreg(Zmm4), Vreg(Zmm7), is_write);

                jit_locations.push(asm.len());

                if is_write {
                    mmu_access(&mut asm, &mut conststore, opsize,
                               Vreg(Zmm4), Vreg(Zmm7), None, Some(Vreg(Zmm5)));
                } else {
                    mmu_access(&mut asm, &mut conststore, opsize,
                               Vreg(Zmm4), Vreg(Zmm7), Some(Vreg(Zmm5)), None);
                }
            }
        }

        // Create RWX memory and copy the code into it
        let jitbuf = alloc_rwx(1024 * 1024);
        let backing = asm.backing();
        jitbuf[..backing.len()].copy_from_slice(&backing);

        let retbuf = alloc_rwx(1);
        retbuf[0] = 0xc3;
            
        // Scratch memory used by the JIT
        let mut scratchmem = [vectorized::Vector::default(); 64];

        let mut context = Context::default();

        let it = rdtsc();
        for _ in 0..1 {
            context.rip = retbuf.as_ptr() as u64;
            context.r15 = TEST_ITERS as u64;
            unsafe { enter_vm(&mut context); }
        }
        let overhead = rdtsc() - it;

        print!("Overhead {}\n", overhead);

        let same_addr = Vector::splat(TEST_VADDR);
        let mut diff_addr = [0; 8];

        // Randomly generate addresses (have a chance of generating
        // addresses outside of the arena)
        for addr in diff_addr.iter_mut() {
            *addr = 
                (TEST_VADDR + (xorshift() % (TEST_ARENA_SIZE + 64))) &
                !(8 - 1);
        }
        let diff_addr = Vector::new(diff_addr);

        for &addrs in &[same_addr, diff_addr] {
        for is_write in &[false, true] {
            for opsize in &[1, 2, 4, 8] {
                let (translate, access) = match (is_write, opsize) {
                    (false, 1) => (jit_locations[0], jit_locations[1]),
                    (false, 2) => (jit_locations[2], jit_locations[3]),
                    (false, 4) => (jit_locations[4], jit_locations[5]),
                    (false, 8) => (jit_locations[6], jit_locations[7]),
                    (true,  1) => (jit_locations[8], jit_locations[9]),
                    (true,  2) => (jit_locations[10], jit_locations[11]),
                    (true,  4) => (jit_locations[12], jit_locations[13]),
                    (true,  8) => (jit_locations[14], jit_locations[15]),
                    _ => panic!("Invalid test state"),
                };

                // Get the VMID number for one of the enabled VMs
                let an_enabled_vm = mask.iter().next().unwrap();

                // Broadcast out tde enabled VM ID
                let sel: [usize; 8] = [an_enabled_vm; 8];

                let mut context = Context::default();
                context.r11 = conststore.table().as_ptr() as u64;
                context.r9  = mmu.backing() as u64;
                context.kmask[1] = mask.raw() as u16;
                context.r13 = mmu.dirty.as_mut_ptr() as u64;
                context.r14 = mmu.dirty_remain as u64;
                context.r8  = &mut scratchmem as *mut _ as u64;
                context.zmm[4] = addrs;
                context.zmm[30] = Vector::new(sel);

                let it = rdtsc();
                for _ in 0..1 {
                    context.rip = jitbuf.as_ptr() as u64 + translate as u64;
                    context.r15 = TEST_ITERS as u64;
                    unsafe { enter_vm(&mut context); }
                    assert!(context.rax == 0);
                }
                let translate_time = rdtsc() - it;

                let it = rdtsc();
                for _ in 0..1 {
                    context.rip = jitbuf.as_ptr() as u64 + access as u64;
                    context.r15 = TEST_ITERS as u64;
                    unsafe { enter_vm(&mut context); }
                    assert!(context.rax == 0);
                }
                let access_time = rdtsc() - it;

                let tt  = (translate_time - overhead) as f64 / TEST_ITERS as f64;
                let at  = (access_time - overhead) as f64 / TEST_ITERS as f64;
                print!("Write: {:5} | opsize: {} | Diverge: {:5} | \
                       Translate {:10.4} cycles | Access {:10.4} cycles\n",
                       is_write, opsize, addrs != same_addr, tt, at);
                //print!("Translate {:.4} | Access {:.4}\n", tta, ata);
            }
        }
        }

        panic!("See ya");
    }
   
    #[test]
    fn test_mmujit() {
        // Bail out if AVX512 isn't supported
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        // Create a new master
        let mut master = SoftMMU::new();
        let vaddr = VirtAddr(TEST_VADDR);

        // Create random bytes for the initial fill of the memory
        let mut meg = Vec::new();
        for _ in 0..TEST_ARENA_SIZE {
            meg.push(xorshift() as u8);
        }
       
        // Map in the memory as RW
        master.add_memory(vaddr, meg.len());
        assert!(master.set_permissions(Mask::all(), vaddr, meg.len(),
            PERM_READ | PERM_WRITE) == meg.len());
        assert!(master.write(Mask::all(), vaddr, &meg[..]) == meg.len());
        assert!(master.set_permissions(Mask::all(), vaddr, meg.len(),
            PERM_RAW | PERM_WRITE) == meg.len());

        // Wrap up the master
        let master = Arc::new(master);

        // Fork from the master
        let mut mmu       = SoftMMU::fork_from(master.clone());
        let mut mmu_nojit = SoftMMU::fork_from(master.clone());

        // Force the memory to be paged in by writing to it
        assert!(mmu.write(Mask::all(), vaddr, &meg[..]) == meg.len());
        assert!(mmu_nojit.write(Mask::all(), vaddr, &meg[..]) == meg.len());

        // Restore state, this leaves the memory as paged in since we wrote
        // to all of it but resets permissions to their original states
        mmu.reset();
        mmu_nojit.reset();

        // Create a new ASM stream
        let mut asm = AsmStream::new(AsmMode::Bits64);
        asm.set_vecwidth(VecWidth::Width512);
        
        // Create a small constant store
        let mut conststore = ConstStore::new(1024);

        // Keep track of the locations for the different variants of the
        // JIT implementations (read, write, varying sizes, etc)
        let mut jit_locations = Vec::new();

        // JIT out all the code
        for &is_write in &[false, true] {
            for &opsize in &[1, 2, 4, 8] {
                jit_locations.push(asm.len());
                
                // Translate addresses into zmm7
                mmu_translate(&mut asm, &mut conststore, opsize,
                              Vreg(Zmm4), Vreg(Zmm7), is_write);

                jit_locations.push(asm.len());

                if is_write {
                    mmu_access(&mut asm, &mut conststore, opsize,
                               Vreg(Zmm4), Vreg(Zmm7), None, Some(Vreg(Zmm5)));
                } else {
                    mmu_access(&mut asm, &mut conststore, opsize,
                               Vreg(Zmm4), Vreg(Zmm7), Some(Vreg(Zmm5)), None);
                }
            }
        }

        // Create RWX memory and copy the code into it
        let jitbuf = alloc_rwx(1024 * 1024);
        let backing = asm.backing();
        jitbuf[..backing.len()].copy_from_slice(&backing);

        for _ in 0..1000000 {
            // Generate a random mask with at least one lane enabled
            let mut mask = Mask::from_raw(xorshift() as u8);
            if mask.all_disabled() { continue; }

            let status: u64;

            let mut addrs: [usize; 8] = [0; 8];

            let is_write = [false, true][xorshift() % 2];
            let opsize   = [1, 2, 4, 8][xorshift() % 4];

            // Have a random chance of resetting the MMUs, this tests the
            // resets are working correctly
            if xorshift() % 8192 == 0 {
                mmu.reset();
                mmu_nojit.reset();
            }

            // Random chance of randomly changing permissions
            if xorshift() % 64 == 0 {
                let addroff = xorshift() % meg.len();
                let size    =
                    std::cmp::min(xorshift() % 128, meg.len() - addroff);
                let randperm = xorshift() as u8;
                let vaddr = VirtAddr(TEST_VADDR + addroff);

                let rmask = Mask::from_raw(xorshift() as u8);
                if !rmask.all_disabled() {
                    assert!(mmu.set_permissions(rmask, vaddr, size,
                        randperm) == size);
                    assert!(mmu_nojit.set_permissions(rmask, vaddr, size,
                        randperm) == size);
                }
            }

            // 50% chance of all VMs being enabled
            if xorshift() % 2 == 0 {
                mask = Mask::all();
            }

            // Randomly generate addresses (have a chance of generating
            // addresses outside of the arena)
            for addr in addrs.iter_mut() {
                *addr = 
                    (TEST_VADDR + (xorshift() % (TEST_ARENA_SIZE + 64))) &
                    !(opsize - 1);
            }

            // 50% chance of using the same address for all lanes
            if xorshift() % 2 == 0 {
                addrs = [addrs[0]; 8];
            }

            // Get the VMID number for one of the enabled VMs
            let an_enabled_vm = mask.iter().next().unwrap();

            // Broadcast out the enabled VM ID
            let sel: [usize; 8] = [an_enabled_vm; 8];

            // Broadcast out the payload to write to memory
            let mut payload: [usize; 8] = [0; 8];
            for pl in payload.iter_mut() { *pl = xorshift(); }
            
            // 50% chance of using the same value for all lanes
            if xorshift() % 2 == 0 {
                payload = [payload[0]; 8];
            }

            // Scratch memory used by the JIT
            let mut scratchmem = [vectorized::Vector::default(); 64];

            /*print!("write: {:?} opsize: {:?} mask: {:02x}\n{:016x?}\n{:016x?}\n\n",
                   is_write, opsize, mask.raw(), addrs, payload);*/

            let (translate, access) = match (is_write, opsize) {
                (false, 1) => (jit_locations[0], jit_locations[1]),
                (false, 2) => (jit_locations[2], jit_locations[3]),
                (false, 4) => (jit_locations[4], jit_locations[5]),
                (false, 8) => (jit_locations[6], jit_locations[7]),
                (true,  1) => (jit_locations[8], jit_locations[9]),
                (true,  2) => (jit_locations[10], jit_locations[11]),
                (true,  4) => (jit_locations[12], jit_locations[13]),
                (true,  8) => (jit_locations[14], jit_locations[15]),
                _ => panic!("Invalid test state"),
            };

            let caused_vmexit: u64;

            let mut context = Context::default();
            context.r11 = conststore.table().as_ptr() as u64;
            context.r9  = mmu.backing() as u64;
            context.kmask[1] = mask.raw() as u16;
            context.r13 = mmu.dirty.as_mut_ptr() as u64;
            context.r14 = mmu.dirty_remain as u64;
            context.r8  = &mut scratchmem as *mut _ as u64;
            context.zmm[4] = Vector::new(addrs);
            context.zmm[30] = Vector::new(sel);
            context.rip = jitbuf.as_ptr() as u64 + translate as u64;
            context.r15 = 1;
            unsafe { enter_vm(&mut context); }

            if context.rax == 0 {
                context.rip = jitbuf.as_ptr() as u64 + access as u64;
                context.zmm[4] = Vector::new(addrs);
                context.zmm[5] = Vector::new(payload);
                context.r15 = 1;
                unsafe { enter_vm(&mut context); }
                scratchmem[0] = context.zmm[5];
            }

            mmu.dirty_remain = context.r14 as usize;
            status = context.rax;
            caused_vmexit = context.kmask[2] as u64;

            let caused_vmexit = caused_vmexit as u8;

            if !is_write {
                let mut vals    = [0usize; 8];
                let mut faulted = 0u8;

                // Perform all the reads first
                for vmid in mask.iter() {
                    let addr = addrs[vmid];
                    let mask = Mask::single(vmid);
                    let bread;

                    let actual = match opsize {
                        1 => {
                            let mut actual = 0u8;
                            bread = mmu_nojit.read(mask, VirtAddr(addr), &mut actual);
                            actual as i8 as i64 as usize
                        }
                        2 => {
                            let mut actual = 0u16;
                            bread = mmu_nojit.read(mask, VirtAddr(addr), &mut actual);
                            actual as i16 as i64 as usize
                        }
                        4 => {
                            let mut actual = 0u32;
                            bread = mmu_nojit.read(mask, VirtAddr(addr), &mut actual);
                            actual as i32 as i64 as usize
                        }
                        8 => {
                            let mut actual = 0u64;
                            bread = mmu_nojit.read(mask, VirtAddr(addr), &mut actual);
                            actual as usize
                        }
                        _ => panic!("Bad opsize"),
                    };

                    if bread != opsize {
                        // Report it faulted
                        faulted |= 1 << vmid;
                    } else {
                        // Store the expected value
                        vals[vmid] = actual;
                    }
                }

                assert!(faulted == caused_vmexit);

                // Make sure JIT reported fault if there was a fault, otherwise
                // expect no fault
                if faulted > 0 {
                    assert!(status == 0x1dead010 + opsize as u64);
                } else {
                    assert!(status == 0);

                    // Make sure all the values match
                    for vmid in mask.iter() {
                        assert!(vals[vmid] == scratchmem[0].extract(vmid));
                    }
                }
            } else {
                let mut faulted = 0u8;

                // First check if we can perform the writes
                for vmid in mask.iter() {
                    if mmu_nojit.check_permissions(Mask::single(vmid),
                            VirtAddr(addrs[vmid]), opsize,
                            |x| { (x & PERM_WRITE) != 0 }) != opsize {
                        faulted |= 1 << vmid;
                    }
                }
                
                assert!(faulted == caused_vmexit);
       
                // Make sure JIT reports faults if they happen
                if faulted > 0 {
                    assert!(status == 0x1dead020 + opsize as u64);
                } else {
                    // Make sure no faults were reported by the JIT
                    assert!(status == 0);

                    // Perform the writes in the second MMU
                    for vmid in mask.iter() {
                        let payload_bytes: &[u8] = payload[vmid].cast();

                        let addr = addrs[vmid];
                        assert!(mmu_nojit.write(Mask::single(vmid),
                            VirtAddr(addr),
                            &payload_bytes[..opsize]) == opsize);
                    }
                }
            }

            // Validate JIT and rust versions of the MMU have the exact same
            // results
            for saddr in (TEST_VADDR..TEST_VADDR+TEST_ARENA_SIZE).step_by(PAGE_SIZE) {
                let v1 = mmu.virt_to_phys(VirtAddr(saddr)).unwrap();
                let v2 = mmu_nojit.virt_to_phys(VirtAddr(saddr)).unwrap();
                assert!(v1 == v2, "Mismatch between JIT and normal write");
            }

            // Validate dirty lists have the same addresses in them
            let js  = &mmu.dirty[mmu.dirty_remain..];
            let njs = &mmu_nojit.dirty[mmu_nojit.dirty_remain..];

            assert!(mmu_nojit.dirty_remain == mmu.dirty_remain);
            for ((v1, _), (v2, _)) in js.iter().zip(njs.iter()) {
                assert!(v1 == v2);
            }
        }
    }
}

