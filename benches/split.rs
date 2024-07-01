use criterion::{black_box, criterion_group, criterion_main, Criterion};

use std::arch::x86_64::{
    _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_set1_epi8,
};

#[derive(Debug)]
pub struct AvxLineIterator<'a> {
    inner: &'a str,
    line_map: u64,
    read: usize,
    head: usize,
    map_offset: usize,
}

impl<'a> AvxLineIterator<'a> {
    pub fn new(text: &'a str) -> Self {
        Self {
            line_map: 0,
            map_offset: 0,
            read: 0,
            head: 0,
            inner: text,
        }
    }

    #[inline]
    fn search_next(&mut self) {
        let mask = unsafe { _mm256_set1_epi8(b'\n' as i8) };
        let addr = self.inner.as_ptr();
        let tail_len = self.inner.len() & 63;
        let len = self.inner.len() - tail_len;
        let mut offset = self.read;

        while offset < len {
            let ptr0 = unsafe { addr.byte_add(offset).cast() };
            let ptr1 = unsafe { addr.byte_add(offset + 32).cast() };
            let block0 = unsafe { _mm256_loadu_si256(ptr0) };
            let block1 = unsafe { _mm256_loadu_si256(ptr1) };
            let cmp0 = unsafe { _mm256_cmpeq_epi8(block0, mask) };
            let cmp1 = unsafe { _mm256_cmpeq_epi8(block1, mask) };
            let pos_l = unsafe { _mm256_movemask_epi8(cmp0) } as u32;
            let pos_h = unsafe { _mm256_movemask_epi8(cmp1) } as u32;
            let line_map = ((pos_h as u64) << 32) | pos_l as u64;

            self.map_offset = offset;
            offset += 64;
            self.read = offset;

            if line_map > 0 {
                self.line_map = line_map;
                return;
            }
        }

        if offset >= len && tail_len > 0 {
            let mut line_map: u64 = 0;
            let tail = self.inner[offset..].as_bytes();

            for (idx, val) in tail.iter().enumerate() {
                if *val == b'\n' {
                    line_map |= 0x1 << idx;
                }
            }

            self.map_offset = offset;
            self.line_map = line_map;
            self.read += tail_len;
        }
    }
}

impl<'a> Iterator for AvxLineIterator<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.head == self.inner.len() {
            return None;
        }

        if self.line_map == 0 && self.read < self.inner.len() {
            self.search_next();
        }

        if self.line_map > 0 {
            let head = self.head;
            let bit_pos = self.line_map.trailing_zeros();
            let line_end: usize = self.map_offset + (bit_pos as usize);

            self.map_offset = line_end + 1;
            self.head = line_end + 1;
            self.line_map = self.line_map.wrapping_shr(bit_pos + 1);

            Some(&self.inner[head..line_end])
        } else if self.head < self.inner.len() && self.read == self.inner.len() {
            let head = self.head;
            self.head = self.inner.len();
            Some(&self.inner[head..])
        } else {
            None
        }
    }
}

#[no_mangle]
pub fn simd_test(buff: &str) -> usize {
    AvxLineIterator::new(buff).count()
}

#[no_mangle]
pub fn regular(buff: &str) -> usize {
    buff.lines().count()
}

const LINES : &'static str= "I’d just like to interject for a moment. What you’re referring to as REST, is in fact, JSON/RPC, or as I’ve recently taken to calling it, REST-less. JSON is not a hypermedia unto itself, but rather a plain data format made useful by out of band information as defined by swagger documentation or similar.

Many computer users work with a canonical version of REST every day, without realizing it. Through a peculiar turn of events, the version of REST which is widely used today is often called “The Web”, and many of its users are not aware that it is basically the REST-ful architecture, defined by Roy Fielding.

There really is a REST, and these people are using it, but it is just a part of The Web they use. REST is the network architecture: hypermedia encodes the state of resources for hypermedia clients. JSON is an essential part of Single Page Applications, but useless by itself; it can only function in the context of a complete API specification. JSON is normally used in combination with SPA libraries: the whole system is basically RPC with JSON added, or JSON/RPC. All these so-called “REST-ful” APIs are really JSON/RPC.";

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("simd split", |b| b.iter(|| simd_test(black_box(LINES))));
    c.bench_function("regular split", |b| b.iter(|| regular(black_box(LINES))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
