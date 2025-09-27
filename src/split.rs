#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use self::avx2_split::Avx2SplitIterator;
use self::linear_split::LinearSplitIterator;
use self::swar_split::SwarSplitIterator;

mod avx2_split;
mod linear_split;
mod swar_split;

pub enum IteratorTypes<'a> {
    SWAR(SwarSplitIterator<'a>),
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    Avx2(Avx2SplitIterator<'a>),
    Linear(LinearSplitIterator<'a>),
}

pub struct FastSplitIterator<'a> {
    inner_iterator: IteratorTypes<'a>,
}

impl<'a> FastSplitIterator<'a> {
    pub fn new(input: &'a [u8], split_byte: u8) -> Self {
        match input.len() {
            0..=7 => Self {
                inner_iterator: IteratorTypes::Linear(LinearSplitIterator::new(input, split_byte)),
            },
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            8..=63 => Self {
                inner_iterator: IteratorTypes::SWAR(SwarSplitIterator::new(input, split_byte)),
            },
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            _ => Self {
                inner_iterator: IteratorTypes::Avx2(Avx2SplitIterator::new(input, split_byte)),
            },
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            _ => Self {
                inner_iterator: IteratorTypes::SWAR(SwarSplitIterator::new(input, split_byte)),
            },
        }
    }
}

impl<'a> Iterator for FastSplitIterator<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner_iterator {
            IteratorTypes::SWAR(i) => i.next(),
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            IteratorTypes::Avx2(i) => i.next(),
            IteratorTypes::Linear(i) => i.next(),
        }
    }
}
