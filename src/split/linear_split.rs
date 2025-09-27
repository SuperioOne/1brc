pub(crate) struct LinearSplitIterator<'a> {
  input: &'a [u8],
  split_byte: u8,
  current_head: usize,
}

impl<'a> LinearSplitIterator<'a> {
  pub(crate) fn new(input: &'a [u8], split_byte: u8) -> Self {
    Self {
      input,
      split_byte,
      current_head: 0,
    }
  }
}

impl<'a> Iterator for LinearSplitIterator<'a> {
  type Item = &'a [u8];

  #[inline]
  fn next(&mut self) -> Option<Self::Item> {
    if self.current_head >= self.input.len() {
      None
    } else {
      let mut read_head = self.current_head;

      for byte in &self.input[read_head..] {
        if *byte == self.split_byte {
          let end = read_head;
          let start = self.current_head;

          self.current_head = read_head + 1;

          return Some(&self.input[start..end]);
        } else {
          read_head += 1;
        }
      }

      if self.current_head < self.input.len() {
        let start = self.current_head;
        self.current_head = self.input.len();

        Some(&self.input[start..])
      } else {
        None
      }
    }
  }
}
