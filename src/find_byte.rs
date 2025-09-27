#[inline]
fn find_index_linear(input: &[u8], byte: u8) -> Option<usize> {
    for (idx, value) in input.iter().enumerate() {
        if *value == byte {
            return Some(idx);
        }
    }

    None
}

const SWAR_MASK_L: u64 = 0x7f7f7f7f7f7f7f7f;
const SWAR_MASK_H: u64 = 0x8080808080808080;
const SWAR_ADD: u64 = 0x0101010101010101;

#[inline]
fn find_index_swar(input: &[u8], byte: u8) -> Option<usize> {
    let search = u64::from_ne_bytes([byte; 8]);
    let tail_len = input.len() & 7;
    let len = input.len() - tail_len;
    let addr: *const u64 = input.as_ptr().cast();
    let mut offset = 0;

    while offset < len {
        let block: u64 = unsafe { addr.byte_add(offset).read() };
        let eq = block ^ search;
        let cmp = (!eq & SWAR_MASK_L).wrapping_add(SWAR_ADD) & (!eq & SWAR_MASK_H);

        if cmp > 0 {
            return Some(offset + (cmp.trailing_zeros() / 8) as usize);
        }

        offset += 8;
    }

    if tail_len > 0 {
        let offset = input.len() - 8;
        let block: u64 = unsafe { addr.byte_add(offset).read() };
        let eq = block ^ search;
        let cmp = (!eq & SWAR_MASK_L).wrapping_add(SWAR_ADD) & (!eq & SWAR_MASK_H);

        if cmp > 0 {
            Some(offset + (cmp.trailing_zeros() / 8) as usize)
        } else {
            None
        }
    } else {
        None
    }
}

/// Returns first occurance of a byte.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
fn find_byte_avx2(input: &[u8], byte: u8) -> Option<usize> {
    match input.len() {
        0..=7 => find_index_linear(input, byte),
        8..=63 => find_index_swar(input, byte),
        _ => {
            let search: std::arch::x86_64::__m256i =
                unsafe { std::arch::x86_64::_mm256_set1_epi8(byte as i8) };
            let addr: *const u8 = input.as_ptr();
            let tail_len = input.len() & 63;
            let block_len = input.len() - tail_len;
            let mut offset: usize = 0;

            while offset < block_len {
                let search_map = unsafe {
                    let ptr0 = addr.byte_add(offset).cast();
                    let ptr1 = addr.byte_add(offset + 32).cast();
                    let block0 = std::arch::x86_64::_mm256_loadu_si256(ptr0);
                    let block1 = std::arch::x86_64::_mm256_loadu_si256(ptr1);
                    let cmp0 = std::arch::x86_64::_mm256_cmpeq_epi8(block0, search);
                    let cmp1 = std::arch::x86_64::_mm256_cmpeq_epi8(block1, search);
                    let pos_l = std::arch::x86_64::_mm256_movemask_epi8(cmp0) as u32;
                    let pos_h = std::arch::x86_64::_mm256_movemask_epi8(cmp1) as u32;
                    ((pos_h as u64) << 32) | pos_l as u64
                };

                if search_map > 0 {
                    let bit_pos = search_map.trailing_zeros() as usize;
                    return Some(offset + bit_pos);
                }

                offset += 64;
            }

            if tail_len > 0 {
                offset = input.len() - 64;

                let search_map = unsafe {
                    let ptr0 = addr.byte_add(offset).cast();
                    let ptr1 = addr.byte_add(offset + 32).cast();
                    let block0 = std::arch::x86_64::_mm256_loadu_si256(ptr0);
                    let block1 = std::arch::x86_64::_mm256_loadu_si256(ptr1);
                    let cmp0 = std::arch::x86_64::_mm256_cmpeq_epi8(block0, search);
                    let cmp1 = std::arch::x86_64::_mm256_cmpeq_epi8(block1, search);
                    let pos_l = std::arch::x86_64::_mm256_movemask_epi8(cmp0) as u32;
                    let pos_h = std::arch::x86_64::_mm256_movemask_epi8(cmp1) as u32;
                    ((pos_h as u64) << 32) | pos_l as u64
                };

                if search_map > 0 {
                    let bit_pos = search_map.trailing_zeros() as usize;
                    Some(offset + bit_pos)
                } else {
                    None
                }
            } else {
                None
            }
        }
    }
}

#[inline]
pub fn fast_find(input: &[u8], byte: u8) -> Option<usize> {
    match input.len() {
        0..=7 => find_index_linear(input, byte),

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        _ => find_index_swar(input, byte),

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        8..=63 => find_index_swar(input, byte),

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        _ => find_byte_avx2(input, byte),
    }
}
