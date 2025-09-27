/// **Internal use only**
/// Splits buffer with avx2, **but only supports buffers with greater than or equals to 64 bytes
/// length**.
pub(crate) struct Avx2SplitIterator<'a> {
  input: &'a [u8],
  split_byte: u8,
  bitmap: u64,
  block_head: usize,
  read_head: usize,
  current_head: usize,
}

impl<'a> Avx2SplitIterator<'a> {
  pub fn new(buffer: &'a [u8], split: u8) -> Self {
    Self {
      input: buffer,
      split_byte: split,
      bitmap: 0,
      current_head: 0,
      read_head: 0,
      block_head: 0,
    }
  }

  #[inline]
  fn search_blocks(&mut self) {
    let mask = unsafe { std::arch::x86_64::_mm256_set1_epi8(self.split_byte as i8) };
    let addr = self.input.as_ptr();
    let tail_len = self.input.len() & 63;
    let block_len = self.input.len() - tail_len;
    let mut offset = self.read_head;

    while offset < block_len {
      let bitmap = unsafe {
        let ptr0 = addr.byte_add(offset).cast();
        let ptr1 = addr.byte_add(offset + 32).cast();
        let block0 = std::arch::x86_64::_mm256_loadu_si256(ptr0);
        let block1 = std::arch::x86_64::_mm256_loadu_si256(ptr1);
        let cmp0 = std::arch::x86_64::_mm256_cmpeq_epi8(block0, mask);
        let cmp1 = std::arch::x86_64::_mm256_cmpeq_epi8(block1, mask);
        let pos_l = std::arch::x86_64::_mm256_movemask_epi8(cmp0) as u32;
        let pos_h = std::arch::x86_64::_mm256_movemask_epi8(cmp1) as u32;
        ((pos_h as u64) << 32) | pos_l as u64
      };

      self.block_head = offset;
      offset += 64;
      self.read_head = offset;

      if bitmap > 0 {
        self.bitmap = bitmap;
        return;
      }
    }

    if tail_len > 0 {
      offset = self.input.len() - 64;

      let bitmap = unsafe {
        let ptr0 = addr.byte_add(offset).cast();
        let ptr1 = addr.byte_add(offset + 32).cast();
        let block0 = std::arch::x86_64::_mm256_loadu_si256(ptr0);
        let block1 = std::arch::x86_64::_mm256_loadu_si256(ptr1);
        let cmp0 = std::arch::x86_64::_mm256_cmpeq_epi8(block0, mask);
        let cmp1 = std::arch::x86_64::_mm256_cmpeq_epi8(block1, mask);
        let pos_l = std::arch::x86_64::_mm256_movemask_epi8(cmp0) as u32;
        let pos_h = std::arch::x86_64::_mm256_movemask_epi8(cmp1) as u32;
        ((pos_h as u64) << 32) | pos_l as u64
      };

      self.block_head = block_len;
      self.read_head += tail_len;
      self.bitmap = bitmap.wrapping_shr((64 - tail_len) as u32);
    }
  }
}

impl<'a> Iterator for Avx2SplitIterator<'a> {
  type Item = &'a [u8];

  #[inline]
  fn next(&mut self) -> Option<Self::Item> {
    let len = self.input.len();

    if self.current_head >= len {
      return None;
    }

    if self.bitmap == 0 && self.read_head < len {
      self.search_blocks();
    }

    if self.bitmap > 0 {
      let split_start = self.current_head;
      let bit_pos = self.bitmap.trailing_zeros();
      let split_end: usize = self.block_head + (bit_pos as usize);

      self.block_head = split_end + 1;
      self.current_head = split_end + 1;
      self.bitmap = self.bitmap.wrapping_shr(bit_pos + 1);

      Some(&self.input[split_start..split_end])
    } else if self.current_head < self.input.len() && self.read_head >= self.input.len() {
      let split_start = self.current_head;
      self.current_head = self.input.len();

      Some(&self.input[split_start..])
    } else {
      None
    }
  }
}
