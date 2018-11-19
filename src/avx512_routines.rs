/// High performance AVX-512 implementations for common MMU routines

use vectorized::Mask;
use super::VirtAddr;
use super::SoftMMU;
use super::PAGE_SIZE;
use super::PAGE_INDEX_MASK;
use safecast::SafeCast;

impl SoftMMU {
    /// Handwritten AVX-512 fast write_force implementation
    ///
    /// Technically we could check permissions (and originally did), but we've
    /// dropped this to only support write force.
    ///
    /// Checking permissions makes this about 30-40% slower, and also is
    /// inconsistent with how we report lengths during partial writes.
    ///
    /// This routine is really only designed for injecting fuzz inputs as
    /// fast as possible so there's no reason to really have it check
    /// permissions
    pub unsafe fn avx512_write_force<T>(&mut self, mask: Mask, ovaddr: VirtAddr,
                                        bytes: &T) -> usize
        where T: SafeCast + ?Sized
    {
        let bytes: &[u8] = bytes.cast();

        // We could soft handle this, but you should _never_ do this so we
        // panic to let you know your code is probably not doing what it
        // intends.
        assert!(!mask.all_disabled(), "Attempted to write with Mask 0");

        if bytes.len() == 0 { return 0; }

        let mut vaddr = ovaddr.0;

        // Compute the ending address
        let vend = vaddr.checked_add(bytes.len() - 1)
            .expect("Integer overflow on avx512_write_force size");

        // State keeping track of the current page translation
        let mut translated = 0 as *mut u8;

        while vaddr <= vend {
            // Compute the distance into the buffer we're going to use
            let ii = vaddr - ovaddr.0;

            // If we have not translated the vaddr yet, or we have rolled
            // over to a new page, then we must rewalk the tables to
            // translate the address.
            if translated == 0 as *mut u8 ||
                    (vaddr & PAGE_INDEX_MASK) == 0 {
                // Get the backing page for this virtual address
                translated = if let Some(addr) = 
                        self.virt_to_phys_mut(VirtAddr(vaddr)) {
                    addr.as_mut_ptr()
                } else {
                    // Failed to translate, bail early
                    return ii;
                }
            }

            let _dummy: u64;

            let pageoff = vaddr & PAGE_INDEX_MASK;

            // Compute the number of bytes to write. This is either the
            // remainder of bytes on the page or the number of bytes left
            // in the buffer, whichever is fewer
            let to_write = std::cmp::min(bytes.len() - ii, PAGE_SIZE - pageoff);

            asm!(r#"

                sub rsp, 16

                // Establish kmask
                kmovw k1, $6

                // Check if we're starting unaligned. If we aren't we can
                // skip the unaligned first copy
                test $7, $7
                jz   6f

                // Initialize temp buffer and mask
                mov qword ptr [rsp + 0], 0
                mov qword ptr [rsp + 8], ~0

                // Generate the relevant mask
                // For example if we were writing 2 bytes at offset 5 this will
                // produce:
                // 0xff0000ff_fffffff
                //
                lea   rdi, [rsp + 8 + $7]
                mov   rcx, 8
                sub   rcx, $7  // rcx = number of remaining bytes in this qword
                cmp   $2, rcx
                cmovb rcx, $2  // rcx = min(remaining_qword, remaining_buffer)
                mov   rdx, rcx // save off this size for the next copy
                xor   eax, eax // zero out the byte we write
                rep   stosb

                // Copy the bytes to the qword buffer
                lea rdi, [rsp + 0 + $7]
                mov rcx, rdx
                mov rsi, $1
                rep movsb

                // Read the original memory and mask and combine it with
                // the new memory we want to write
                vmovdqa64 zmm0, [$0 + 64]
                vpandq    zmm0, zmm0, qword ptr [rsp + 8] {1to8}
                vporq     zmm0, zmm0, qword ptr [rsp + 0] {1to8}
                vmovdqa64 [$0 + 64] {k1}, zmm0

                // Update pointers and lengths
                add $0, 128
                add $1, rdx
                sub $2, rdx

                // End if we're out of remaining bytes
                jz 5f

            6:
                // Check if we have any 64-byte aligned chunks
                //
                // Performance increases due to this unrolling to justify it.
                // Performance numbers are number of 1 KiB writes per second
                //
                // With 8-byte pages    99k -> 98k (slight loss as it's unused)
                // With 64-byte pages  660k -> 760k
                // With 256-byte pages 1.3M -> 2.1M
                // With 4 KiB pages    2.2M -> 4.3M
                test $2, ~0x3f
                jz   2f

            3:
                // Copy 64-bytes at a time
                // I tried unrolling as 8 broadcasts followed by 8
                // stores and this wasn't really any faster than doing them
                // interleaved so we're doing this as it reduces register
                // pressure.
                vpbroadcastq zmm0, qword ptr [$1 + 8*0]
                vmovdqa64    [$0 + 64 + 128*0] {k1}, zmm0
                vpbroadcastq zmm0, qword ptr [$1 + 8*1]
                vmovdqa64    [$0 + 64 + 128*1] {k1}, zmm0
                vpbroadcastq zmm0, qword ptr [$1 + 8*2]
                vmovdqa64    [$0 + 64 + 128*2] {k1}, zmm0
                vpbroadcastq zmm0, qword ptr [$1 + 8*3]
                vmovdqa64    [$0 + 64 + 128*3] {k1}, zmm0
                vpbroadcastq zmm0, qword ptr [$1 + 8*4]
                vmovdqa64    [$0 + 64 + 128*4] {k1}, zmm0
                vpbroadcastq zmm0, qword ptr [$1 + 8*5]
                vmovdqa64    [$0 + 64 + 128*5] {k1}, zmm0
                vpbroadcastq zmm0, qword ptr [$1 + 8*6]
                vmovdqa64    [$0 + 64 + 128*6] {k1}, zmm0
                vpbroadcastq zmm0, qword ptr [$1 + 8*7]
                vmovdqa64    [$0 + 64 + 128*7] {k1}, zmm0

                add $0, 128*8
                add $1, 64
                sub $2, 64
                cmp $2, 64
                jae 3b

            2:
                // Check if we have any 8-byte aligned chunks
                test $2, ~0x7
                jz   4f

            7:
                // Broadcast out memory as contents
                vpbroadcastq zmm0, qword ptr [$1]
                vmovdqa64    [$0 + 64] {k1}, zmm0

                add $0, 128
                add $1, 8
                sub $2, 8
                cmp $2, 8
                jae 7b

            4:
                // Check if we have any remaining bytes
                test $2, $2
                jz   5f
                
                // Initialize temp buffer and mask
                mov qword ptr [rsp + 0], 0
                mov qword ptr [rsp + 8], ~0

                // Generate the mask
                lea rdi, [rsp + 8]
                xor eax, eax
                mov rcx, $2
                rep stosb

                // Read remaining bytes into temp padding buffer
                mov rdi, rsp
                mov rsi, $1
                mov rcx, $2
                rep movsb

                // Read the original memory and mask and combine it with
                // the new memory we want to write
                vmovdqa64 zmm0, [$0 + 64]
                vpandq    zmm0, zmm0, qword ptr [rsp + 8] {1to8}
                vporq     zmm0, zmm0, qword ptr [rsp + 0] {1to8}
                vmovdqa64 [$0 + 64] {k1}, zmm0

            5:

                add rsp, 16

            "# :
            "=r"(_dummy),
            "=r"(_dummy),
            "=r"(_dummy) :
            "0"(translated as usize + (pageoff / 8) * 128),
            "1"(bytes[ii..].as_ptr() as usize),
            "2"(to_write),
            "r"(mask.raw() as u32),
            "r"(pageoff & 7) :
            "memory", "cc", "rax", "rcx", "rdx", "rdi", "rsi",
            "zmm0", "k1" : "intel");

            vaddr = if let Some(val) = vaddr.checked_add(to_write) {
                val
            } else {
                break;
            }
        }

        bytes.len()
    }

    /// Handwritten AVX-512 fast memset implementation
    ///
    /// When `offset` is 0 this sets permissions
    /// When `offset` is 64 this sets contents
    pub unsafe fn avx512_memset_int(&mut self, mask: Mask, ovaddr: VirtAddr,
                                    byte: u8, size: usize,
                                    offset: usize) -> usize {
        // We could soft handle this, but you should _never_ do this so we
        // panic to let you know your code is probably not doing what it
        // intends.
        assert!(!mask.all_disabled(), "Attempted to write with Mask 0");

        // Validate the offset is correct
        assert!(offset == 0 || offset == 64, "Invalid offset");

        if size == 0 { return 0; }

        let mut vaddr = ovaddr.0;

        // Compute the ending address
        let vend = vaddr.checked_add(size - 1)
            .expect("Integer overflow on avx512_memset size");

        // State keeping track of the current page translation
        let mut translated = 0 as *mut u8;

        // Make a qword filled of `byte`
        let byte: u64 =
            ((byte as u64) <<  0) |
            ((byte as u64) <<  8) |
            ((byte as u64) << 16) |
            ((byte as u64) << 24) |
            ((byte as u64) << 32) |
            ((byte as u64) << 40) |
            ((byte as u64) << 48) |
            ((byte as u64) << 56);

        while vaddr <= vend {
            // Compute the distance into the buffer we're going to use
            let ii = vaddr - ovaddr.0;

            // If we have not translated the vaddr yet, or we have rolled
            // over to a new page, then we must rewalk the tables to
            // translate the address.
            if translated == 0 as *mut u8 ||
                    (vaddr & PAGE_INDEX_MASK) == 0 {
                // Get the backing page for this virtual address
                translated = if let Some(addr) = 
                        self.virt_to_phys_mut(VirtAddr(vaddr)) {
                    addr.as_mut_ptr()
                } else {
                    // Failed to translate, bail early
                    return ii;
                }
            }

            let _dummy: u64;

            let pageoff = vaddr & PAGE_INDEX_MASK;

            // Compute the number of bytes to write. This is either the
            // remainder of bytes on the page or the number of bytes left
            // in the buffer, whichever is fewer
            let to_write = std::cmp::min(size - ii, PAGE_SIZE - pageoff);

            asm!(r#"

                sub rsp, 8

                // Establish kmask
                kmovw k1, $6

                // Broadcast out the qword to memset
                vpbroadcastq zmm1, $1

                // Check if we're starting unaligned. If we aren't we can
                // skip the unaligned first copy
                test $7, $7
                jz   6f

                // Initialize temp buffer and mask
                mov qword ptr [rsp], ~0

                // Generate the relevant mask
                // For example if we were writing 2 bytes at offset 5 this will
                // produce:
                // 0xff0000ff_fffffff
                //
                lea   rdi, [rsp + $7]
                mov   rcx, 8
                sub   rcx, $7  // rcx = number of remaining bytes in this qword
                cmp   $2, rcx
                cmovb rcx, $2  // rcx = min(remaining_qword, remaining_buffer)
                mov   rdx, rcx // save off this size for the next copy
                xor   eax, eax // zero out the byte we write
                rep   stosb

                // Mask the bytes to write with the inverse mask
                vpbroadcastq zmm2, qword ptr [rsp]
                vpandnq zmm2, zmm2, zmm1

                // Read the original memory and mask and combine it with
                // the new memory we want to write
                vmovdqa64 zmm0, [$0 + $8]
                vpandq    zmm0, zmm0, qword ptr [rsp] {1to8}
                vporq     zmm0, zmm0, zmm2
                vmovdqa64 [$0 + $8] {k1}, zmm0

                // Update pointers and lengths
                add $0, 128
                sub $2, rdx

                // End if we're out of remaining bytes
                jz 5f

            6:
                // Check if we have any 64-byte aligned chunks
                test $2, ~0x3f
                jz   2f

            3:
                // Copy 64-bytes at a time
                // I tried unrolling as 8 broadcasts followed by 8
                // stores and this wasn't really any faster than doing them
                // interleaved so we're doing this as it reduces register
                // pressure.
                vmovdqa64 [$0 + $8 + 128*0] {k1}, zmm1
                vmovdqa64 [$0 + $8 + 128*1] {k1}, zmm1
                vmovdqa64 [$0 + $8 + 128*2] {k1}, zmm1
                vmovdqa64 [$0 + $8 + 128*3] {k1}, zmm1
                vmovdqa64 [$0 + $8 + 128*4] {k1}, zmm1
                vmovdqa64 [$0 + $8 + 128*5] {k1}, zmm1
                vmovdqa64 [$0 + $8 + 128*6] {k1}, zmm1
                vmovdqa64 [$0 + $8 + 128*7] {k1}, zmm1

                add $0, 128*8
                sub $2, 64
                cmp $2, 64
                jae 3b

            2:
                // Check if we have any 8-byte aligned chunks
                test $2, ~0x7
                jz   4f

            7:
                // Broadcast out memory as contents
                vmovdqa64 [$0 + $8] {k1}, zmm1

                add $0, 128
                sub $2, 8
                cmp $2, 8
                jae 7b

            4:
                // Check if we have any remaining bytes
                test $2, $2
                jz   5f
                
                // Initialize mask
                mov qword ptr [rsp], ~0

                // Generate the mask
                lea rdi, [rsp]
                xor eax, eax
                mov rcx, $2
                rep stosb

                // Mask the bytes to write with the inverse mask
                vpbroadcastq zmm2, qword ptr [rsp]
                vpandnq zmm2, zmm2, zmm1

                // Read the original memory and mask and combine it with
                // the new memory we want to write
                vmovdqa64 zmm0, [$0 + $8]
                vpandq    zmm0, zmm0, qword ptr [rsp] {1to8}
                vporq     zmm0, zmm0, zmm2
                vmovdqa64 [$0 + $8] {k1}, zmm0

            5:

                add rsp, 8

            "# :
            "=r"(_dummy),
            "=r"(_dummy),
            "=r"(_dummy) :
            "0"(translated as usize + (pageoff / 8) * 128),
            "1"(byte),
            "2"(to_write),
            "r"(mask.raw() as u32),
            "r"(pageoff & 7),
            "r"(offset) :
            "memory", "cc", "rax", "rcx", "rdx", "rdi", "rsi",
            "zmm0", "zmm1", "zmm2", "k1" : "intel");

            vaddr = if let Some(val) = vaddr.checked_add(to_write) {
                val
            } else {
                break;
            }
        }

        size
    }

    /// Handwritten AVX-512 fast write_multiple_force implementation
    pub unsafe fn avx512_write_multiple_force(&mut self,
                                              mask: Mask, ovaddr: VirtAddr,
                                              mems: &[Vec<u8>]) -> usize {
        // We could soft handle this, but you should _never_ do this so we
        // panic to let you know your code is probably not doing what it
        // intends.
        assert!(!mask.all_disabled(), "Attempted to write with Mask 0");

        // We always expect 8 inputs
        assert!(mems.len() == 8, "Invalid number of arrays");

        // We must be 8-byte aligned
        assert!((ovaddr.0 % 8) == 0, "Unaligned address for write_multiple");

        // All arrays write the same size
        let size = mems[0].len();
        
        // Sanity check the inputs and save them as pointers that we can use
        // with gather instructions
        let mut ptrs = [0usize; 8];
        for (ii, bytes) in mems.iter().enumerate() {
            assert!(bytes.len() == size, "Differing size of memory");
            ptrs[ii] = bytes.as_ptr() as usize;
        }

        // Size must be 8-byte aligned
        assert!((size % 8) == 0, "Bytes not 8-byte aligned");

        // If we're trying to write 0 bytes, just bail out
        if size == 0 { return 0; }

        let mut vaddr = ovaddr.0;

        // Compute the ending address
        let vend = vaddr.checked_add(size - 1)
            .expect("Integer overflow on avx512_write_multiple_force size");

        // State keeping track of the current page translation
        let mut translated = 0 as *mut u8;

        while vaddr <= vend {
            const EIGHT:     usize = 8;
            const SIXTYFOUR: usize = 64;

            // Compute the distance into the buffer we're going to use
            let ii = vaddr - ovaddr.0;

            // If we have not translated the vaddr yet, or we have rolled
            // over to a new page, then we must rewalk the tables to
            // translate the address.
            if translated == 0 as *mut u8 ||
                    (vaddr & PAGE_INDEX_MASK) == 0 {
                // Get the backing page for this virtual address
                translated = if let Some(addr) = 
                        self.virt_to_phys_mut(VirtAddr(vaddr)) {
                    addr.as_mut_ptr()
                } else {
                    // Failed to translate, bail early
                    return ii;
                }
            }

            let _dummy: u64;

            let pageoff = vaddr & PAGE_INDEX_MASK;

            // Compute the number of bytes to write. This is either the
            // remainder of bytes on the page or the number of bytes left
            // in the buffer, whichever is fewer
            let to_write = std::cmp::min(size - ii, PAGE_SIZE - pageoff);

            asm!(r#"

                sub rsp, 16

                // Establish kmask
                kmovw k1, $6
                kmovw k2, k1

                // Load the pointers
                vmovdqu64 zmm1, [$1]
                vpaddq    zmm1, zmm1, qword ptr [$10] {1to8}

                // Constant 8
                vpbroadcastq zmm3, qword ptr [$8]

                // Constant 64
                vpbroadcastq zmm4, qword ptr [$9]

            6:
                // Check if we have any 64-byte aligned chunks
                test $2, ~0x3f
                jz   2f

            3:
                // Copy 64-bytes at a time
                vpgatherqq zmm0 {k1}, [zmm1 + 8*0]
                kmovw k1, k2
                vmovdqa64 [$0 + 64 + 128*0] {k1}, zmm0
                kmovw k1, k2
                vpgatherqq zmm0 {k1}, [zmm1 + 8*1]
                kmovw k1, k2
                vmovdqa64 [$0 + 64 + 128*1] {k1}, zmm0
                vpgatherqq zmm0 {k1}, [zmm1 + 8*2]
                kmovw k1, k2
                vmovdqa64 [$0 + 64 + 128*2] {k1}, zmm0
                vpgatherqq zmm0 {k1}, [zmm1 + 8*3]
                kmovw k1, k2
                vmovdqa64 [$0 + 64 + 128*3] {k1}, zmm0
                vpgatherqq zmm0 {k1}, [zmm1 + 8*4]
                kmovw k1, k2
                vmovdqa64 [$0 + 64 + 128*4] {k1}, zmm0
                vpgatherqq zmm0 {k1}, [zmm1 + 8*5]
                kmovw k1, k2
                vmovdqa64 [$0 + 64 + 128*5] {k1}, zmm0
                vpgatherqq zmm0 {k1}, [zmm1 + 8*6]
                kmovw k1, k2
                vmovdqa64 [$0 + 64 + 128*6] {k1}, zmm0
                vpgatherqq zmm0 {k1}, [zmm1 + 8*7]
                kmovw k1, k2
                vmovdqa64 [$0 + 64 + 128*7] {k1}, zmm0

                add    $0, 128*8
                vpaddq zmm1, zmm1, zmm4 // add 64
                sub    $2, 64
                cmp    $2, 64
                jae    3b

            2:
                // Check if we have any 8-byte aligned chunks
                test $2, ~0x7
                jz   4f

            7:
                // Broadcast out memory as contents
                vpgatherqq zmm0 {k1}, [zmm1]
                kmovw k1, k2
                vmovdqa64 [$0 + 64] {k1}, zmm0

                add    $0, 128
                vpaddq zmm1, zmm1, zmm3 // add 8
                sub    $2, 8
                jnz    7b

            4:
                add rsp, 16

            "# :
            "=r"(_dummy),
            "=r"(_dummy),
            "=r"(_dummy) :
            "0"(translated as usize + (pageoff / 8) * 128),
            "1"(ptrs.as_ptr() as usize),
            "2"(to_write),
            "r"(mask.raw() as u32),
            "r"(pageoff & 7),
            "r"(&EIGHT as *const usize as usize),
            "r"(&SIXTYFOUR as *const usize as usize),
            "r"(&ii as *const usize as usize) :
            "memory", "cc",
            "zmm0", "zmm1", "zmm3", "zmm4", "k1", "k2" : "intel");

            vaddr = if let Some(val) = vaddr.checked_add(to_write) {
                val
            } else {
                break;
            }
        }

        size
    }
}

