const SWAR_MASK_L: u64 = 0x7f7f7f7f7f7f7f7f;
const SWAR_MASK_H: u64 = 0x8080808080808080;
const SWAR_ADD: u64 = 0x0101010101010101;

macro_rules! read {
  ($ptr:expr, $offset:expr) => {
    unsafe { $ptr.byte_add($offset).read() }
  };

  ($ptr:expr) => {
    unsafe { $ptr.read() }
  };
}

pub(crate) struct SwarSplitIterator<'a> {
  input: &'a [u8],
  split_byte: u8,
  bitmap: u64,
  block_head: usize,
  read_head: usize,
  current_head: usize,
}

impl<'a> SwarSplitIterator<'a> {
  pub fn new(input: &'a [u8], split: u8) -> Self {
    Self {
      input,
      split_byte: split,
      bitmap: 0,
      current_head: 0,
      read_head: 0,
      block_head: 0,
    }
  }

  #[inline]
  fn search_blocks(&mut self) {
    let search = u64::from_ne_bytes([self.split_byte; 8]);
    let block_ptr: *const u64 = self.input.as_ptr().cast();
    let tail_len = self.input.len() & 7;
    let block_len = self.input.len() - tail_len;
    let mut offset = self.read_head;

    while offset < block_len {
      let block = read!(block_ptr, offset);
      let eq = block ^ search;
      let cmp_result = (!eq & SWAR_MASK_L).wrapping_add(SWAR_ADD) & (!eq & SWAR_MASK_H);

      self.block_head = offset;
      offset += 8;
      self.read_head = offset;

      if cmp_result > 0 {
        self.bitmap = cmp_result;
        return;
      }
    }

    if tail_len > 0 {
      let last_block = read!(block_ptr, self.input.len() - 8);
      let eq = last_block ^ search;
      let cmp_result = ((!eq & SWAR_MASK_L).wrapping_add(SWAR_ADD) & (!eq & SWAR_MASK_H))
        .wrapping_shr((64 - (tail_len * 8)) as u32);

      self.bitmap = cmp_result;
      self.block_head = self.input.len() - tail_len;
      self.read_head += self.input.len();
    }
  }
}

impl<'a> Iterator for SwarSplitIterator<'a> {
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
      let split_end: usize = self.block_head + (bit_pos / 8) as usize;

      self.block_head = split_end + 1;
      self.current_head = split_end + 1;
      self.bitmap = self.bitmap.wrapping_shr(bit_pos).wrapping_shr(1);

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
