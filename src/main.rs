use algorithms::hash::cityhash::hash_fn::cityhash_64_with_seed;
use core::slice;
use libc::{c_void, mmap64, munmap, MAP_FAILED, MAP_PRIVATE, PROT_READ};
use std::{
    arch::x86_64::{
        __m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_set1_epi8,
    },
    collections::{BTreeMap, HashMap},
    env::args,
    fs::File,
    hash::{BuildHasher, Hasher},
    io::{BufWriter, Write},
    os::{fd::AsRawFd, unix::fs::MetadataExt},
    ptr::null_mut,
    str::from_utf8_unchecked,
    sync::Arc,
    thread::{available_parallelism, ScopedJoinHandle},
};

struct City64Hasher {
    inner: u64,
}
struct City64HasherBuilder;

impl Hasher for City64Hasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.inner
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let seed = self.inner;
        self.inner = cityhash_64_with_seed(bytes, 0x9ae16a3b2f90404, seed);
    }
}

impl Default for City64Hasher {
    fn default() -> Self {
        Self { inner: 0 }
    }
}

impl BuildHasher for City64HasherBuilder {
    type Hasher = City64Hasher;

    fn build_hasher(&self) -> Self::Hasher {
        City64Hasher::default()
    }
}

impl Default for City64HasherBuilder {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug)]
pub enum BrcErrors {
    InvalidPath,
    IOError(String),
    MmapFailed,
}

impl From<std::io::Error> for BrcErrors {
    fn from(value: std::io::Error) -> Self {
        BrcErrors::IOError(value.to_string())
    }
}

#[derive(Debug)]
struct StationData {
    name: Box<str>,
    min: f64,
    max: f64,
    count: f64,
    total: f64,
}

impl StationData {
    #[inline]
    pub fn merge(&mut self, other: &StationData) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.total += other.total;
        self.count += other.count;
    }
}

/// Good'ol mmap, see [man(mmap)](https://www.man7.org/linux/man-pages/man2/mmap.2.html)
#[derive(Debug, Clone)]
pub struct Mmap {
    inner: Arc<MmapInner>,
}

impl Mmap {
    pub fn new(fd: File) -> Result<Self, BrcErrors> {
        const FLAGS: i32 = MAP_PRIVATE;

        let len = fd.metadata()?.size() as usize;
        let addr: *const c_void = unsafe {
            mmap64(
                null_mut::<c_void>(),
                len,
                PROT_READ,
                FLAGS,
                fd.as_raw_fd(),
                0,
            )
        };

        if addr == MAP_FAILED {
            eprintln!("{:?}", addr);
            return Err(BrcErrors::MmapFailed);
        }

        Ok(Self {
            inner: Arc::from(MmapInner { addr, len }),
        })
    }

    #[inline]
    pub fn as_byte_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.inner.addr.cast::<u8>(), self.inner.len) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len
    }
}

#[derive(Debug)]
struct MmapInner {
    addr: *const c_void,
    len: usize,
}

unsafe impl Send for MmapInner {}
unsafe impl Sync for MmapInner {}

impl Drop for MmapInner {
    fn drop(&mut self) {
        let res = unsafe { munmap(self.addr.cast_mut(), self.len) };
        if res != 0 {
            eprintln!("Unable to drop mmapped region, code : {}", res);
        }
    }
}

/// Splits a slice into chunks with respecting LF characters. It does not gurantees each chunk size
/// is exactly same since it looks for nearest LF from chunk offset.
pub struct ChunkedLinesIter<'a> {
    bytes: &'a [u8],
    pos: usize,
    chunk_size: usize,
}

impl<'a> ChunkedLinesIter<'a> {
    pub fn chunk_by(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }
}

impl<'a> From<&'a [u8]> for ChunkedLinesIter<'a> {
    fn from(value: &'a [u8]) -> Self {
        Self {
            pos: 0,
            bytes: value,
            chunk_size: 65536, // Default is 64KiB
        }
    }
}

impl<'a> Iterator for ChunkedLinesIter<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let start = self.pos;
        let end = start + self.chunk_size - 1;

        if end >= self.bytes.len() {
            if self.pos < self.bytes.len() {
                self.pos = self.bytes.len();
                Some(&self.bytes[start..])
            } else {
                None
            }
        } else {
            // End position could be middle of the row. We simply look ahead to find nearest LF (0x0A) before slicing.
            let look_ahead_chunk = &self.bytes[end..];

            if let Some(lf_idx) = find_index(look_ahead_chunk, b'\n') {
                let chunk_end = end + lf_idx;

                self.pos = chunk_end + 1;
                Some(&self.bytes[start..=chunk_end])
            } else {
                self.pos = self.bytes.len();
                Some(&self.bytes[start..])
            }
        }
    }
}

/// Custom line iterator, augmented with AVX2 instructions. This iterator processes 64-byte in a single
/// iteration. It keeps an internal bitmap (8-byte) to track upcoming LF locations.
///
/// Another possible optimization is using AVX-512 but I don't have hardware with AVX-512 support to test it.
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
            // reads 64-byte (cacheline size for most cpus) data per iteration
            let ptr0 = unsafe { addr.byte_add(offset).cast() };
            let ptr1 = unsafe { addr.byte_add(offset + 32).cast() };
            let block0 = unsafe { _mm256_loadu_si256(ptr0) };
            let block1 = unsafe { _mm256_loadu_si256(ptr1) };
            let cmp0 = unsafe { _mm256_cmpeq_epi8(block0, mask) };
            let cmp1 = unsafe { _mm256_cmpeq_epi8(block1, mask) };
            let pos_l = unsafe { _mm256_movemask_epi8(cmp0) } as u32;
            let pos_h = unsafe { _mm256_movemask_epi8(cmp1) } as u32;
            let line_map = ((pos_h as u64) << 32) | pos_l as u64; // bitmap for LF positions.

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

        // bitmap is empty and read is not done.
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

#[inline]
fn find_index_0_to_7(input: &[u8], char: u8) -> Option<usize> {
    for (idx, value) in input.iter().enumerate() {
        if *value == char {
            return Some(idx);
        }
    }

    None
}

const SWAR_MASK_L: u64 = 0x7f7f7f7f7f7f7f7f;
const SWAR_MASK_H: u64 = 0x8080808080808080;
const SWAR_ADD: u64 = 0x0101010101010101;

#[inline]
fn find_index_8_to_63(input: &[u8], char: u8) -> Option<usize> {
    let search = u64::from_ne_bytes([char; 8]);
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
        let tail = &input[offset..];
        find_index_0_to_7(tail, char).map(|v| v + offset)
    } else {
        None
    }
}

// Returns first occurance of a byte.
//
// Algoritm is:
// 0-7 bytes len -> linear search
// 8-63 bytes len -> SWAR string match
// 64-.. bytes len -> AVX matching
#[inline]
pub fn find_index(input: &[u8], char: u8) -> Option<usize> {
    match input.len() {
        0..=7 => find_index_0_to_7(input, char),
        8..=63 => find_index_8_to_63(input, char),
        _ => {
            let mask: __m256i = unsafe { _mm256_set1_epi8(char as i8) };
            let addr: *const u8 = input.as_ptr();
            let tail_len = input.len() & 63;
            let len = input.len() - tail_len;
            let mut offset: usize = 0;

            while offset < len {
                let ptr0 = unsafe { addr.byte_add(offset).cast() };
                let ptr1 = unsafe { addr.byte_add(offset + 32).cast() };
                let block0 = unsafe { _mm256_loadu_si256(ptr0) };
                let block1 = unsafe { _mm256_loadu_si256(ptr1) };
                let cmp0 = unsafe { _mm256_cmpeq_epi8(block0, mask) };
                let cmp1 = unsafe { _mm256_cmpeq_epi8(block1, mask) };
                let pos_l = unsafe { _mm256_movemask_epi8(cmp0) } as u32;
                let pos_h = unsafe { _mm256_movemask_epi8(cmp1) } as u32;
                let search_map = ((pos_h as u64) << 32) | pos_l as u64;

                if search_map > 0 {
                    let bit_pos = search_map.trailing_zeros() as usize;
                    return Some(offset + bit_pos);
                }

                offset += 64;
            }

            if tail_len > 0 {
                let tail = &input[offset..];
                let char_pos = match tail_len {
                    0..=7 => find_index_0_to_7(tail, char),
                    8..=63 => find_index_8_to_63(tail, char),
                    _ => unreachable!(),
                };

                char_pos.map(|v| v + offset)
            } else {
                None
            }
        }
    }
}

type StationHashMap = HashMap<HashKey, StationData, City64HasherBuilder>;
type HashKey = u128;

/// Truncates station names into u128 (16-byte) value with Big-Endian format. This version assumes
/// first 16 characters of station names are always unique.
///
/// Why 16 characters and not 8?
/// - Because of the Australian cities! See: Melbourne (Florida) and Melbourne (Australia)
///
/// Why Big-Endian?
/// - On little-endian systems first char reads as least significant bits which makes ordering with
/// std::collection::BTree problematic. For example; when we read "Melbourne" from memory as u128,
/// we endup with "enruobleM" as value. Applying bswap instruction fixes the ordering in a very cheap way.
#[inline]
fn get_key(input: &[u8]) -> HashKey {
    let ptr: *const HashKey = input.as_ptr().cast();
    let len = input.len();

    match len {
        0 => 0,
        1..=15 => {
            let offset_bit = (16 - len) * 8;
            let a = unsafe { ptr.read() };
            (a.to_be() >> offset_bit) << offset_bit
        }
        _ => unsafe { ptr.read() }.to_be(),
    }
}

fn main() -> Result<(), BrcErrors> {
    let worker_count = available_parallelism()?;
    let fd = if let Some(file_path) = args().nth(1) {
        let fd = std::fs::File::open(file_path)?;
        Ok(fd)
    } else {
        Err(BrcErrors::InvalidPath)
    }?;

    let mmap = Mmap::new(fd)?;

    let task_results = std::thread::scope(|s| {
        let chunk_size: usize = mmap.len() / worker_count;
        let mmap_bytes = mmap.as_byte_slice();
        let mut handles: Vec<ScopedJoinHandle<StationHashMap>> =
            Vec::with_capacity(worker_count.into());

        for chunk in ChunkedLinesIter::from(mmap_bytes).chunk_by(chunk_size) {
            // In the name of Utf-8 encoded input file we trust.
            let chunk_str: &str = { unsafe { from_utf8_unchecked(chunk) } };
            let handle = s.spawn(|| {
                let mut store: StationHashMap =
                    StationHashMap::with_capacity_and_hasher(512, City64HasherBuilder::default());

                for line in AvxLineIterator::new(chunk_str) {
                    let mut split_pos: usize = 0;

                    for (idx, char) in line.as_bytes().iter().enumerate().rev() {
                        if *char == b';' {
                            split_pos = idx;
                            break;
                        }
                    }

                    if let Ok(s_val) = (&line[(split_pos + 1)..line.len()]).parse::<f64>() {
                        let s_name = &line[0..split_pos];
                        let key = get_key(s_name.as_bytes());

                        match store.get_mut(&key) {
                            Some(entry) => {
                                entry.min = entry.min.min(s_val);
                                entry.max = entry.max.max(s_val);
                                entry.total += s_val;
                                entry.count += 1.0;
                            }
                            None => {
                                _ = store.insert(
                                    key,
                                    StationData {
                                        name: Box::from(s_name),
                                        min: s_val,
                                        max: s_val,
                                        count: 1.0,
                                        total: s_val,
                                    },
                                );
                            }
                        }
                    }
                }

                store
            });

            handles.push(handle);
        }

        handles
            .into_iter()
            .flat_map(|h| h.join())
            .collect::<Vec<StationHashMap>>()
    });

    let mut merge_btree: BTreeMap<HashKey, StationData> = BTreeMap::new();

    for result_block in task_results {
        for (key, value) in result_block {
            match merge_btree.get_mut(&key) {
                Some(station) => {
                    station.merge(&value);
                }
                None => {
                    merge_btree.insert(key, value);
                }
            }
        }
    }

    let mut stdout_writer = BufWriter::new(std::io::stdout());
    let last_idx = merge_btree.len() - 1;

    _ = stdout_writer.write(b"{")?;

    for (idx, (_, value)) in merge_btree.into_iter().enumerate() {
        if idx == last_idx {
            stdout_writer.write_fmt(format_args!(
                "{}={:.1}/{:.1}/{:.1}}}",
                value.name,
                value.min,
                value.total / value.count,
                value.max,
            ))?;
        } else {
            stdout_writer.write_fmt(format_args!(
                "{}={:.1}/{:.1}/{:.1}, ",
                value.name,
                value.min,
                value.total / value.count,
                value.max,
            ))?;
        }
    }

    stdout_writer.flush()?;

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{find_index, AvxLineIterator, ChunkedLinesIter};

    const TEST_VALUES: &str = "sentence 1
sentence 2
sentence 3
short
shr
kind of long sentence
block2
sentence 2
sentence 3
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
1
2

3
4
5
hello world-";

    const EXAMPLE_INPUT: &str = "
Dallas;13.6
Sacramento;33.9
Austin;20.2
Anchorage;-1.0
Djibouti;24.5
Pointe-Noire;19.3
Gabès;5.0
Guadalajara;29.2
Kumasi;17.1
Pyongyang;24.5
Milwaukee;3.9
Kunming;-1.9
Zanzibar City;23.1
Sana'a;10.3
Belize City;31.3
Belize City;41.8
Chișinău;3.2
Petropavlovsk-Kamchatsky;11.1
Dhaka;32.5
Brisbane;23.7
Ho Chi Minh City;27.5
Kolkata;20.4
Jayapura;33.8
Bulawayo;18.6
Zanzibar City;27.7
Baghdad;6.9
Guangzhou;22.2
Harare;-8.2
Hamilton;24.8
Dublin;12.5
Dili;28.6
Winnipeg;13.9
Fukuoka;27.0
Brazzaville;20.6
Dakar;17.2
Skopje;2.5
Zagreb;18.6
Sofia;-2.5
Oslo;-7.8
Oklahoma City;-1.8
Cairns;12.6
Da Lat;13.8
Memphis;25.5
Omaha;23.3
Havana;16.0
Tripoli;11.2
Darwin;17.9
El Paso;13.1
Surabaya;15.2
Kingston;12.2
Tehran;10.4
Reggane;39.9
Reggane;45.0
Surabaya;44.8
Petropavlovsk-Kamchatsky;2.9
Accra;29.7
Muscat;23.6
Oranjestad;22.4
Mek'ele;23.5
Kano;40.2
Tamanrasset;15.5
Maun;44.8
Murmansk;3.2
Karachi;32.1
Ifrane;-2.6
Tallinn;-2.8
Aden;19.7
Berlin;1.6
Cabo San Lucas;18.7
Tampa;26.6
Kunming;9.0";

    #[test]
    fn chunk_test() {
        let input: &[u8] = &[
            5, 5, 5, 5, 5, b'\n', 1, b'\n', 2, 2, b'\n', 4, 4, 4, 4, b'\n', 1, b'\n', 1, b'\n', 3,
            3, 3, b'\n',
        ];

        let mut iterator = ChunkedLinesIter::from(input).chunk_by(5);

        let first = iterator.next().unwrap();
        assert_eq!(first, &[5, 5, 5, 5, 5, b'\n']);

        let second = iterator.next().unwrap();
        assert_eq!(second, &[1, b'\n', 2, 2, b'\n']);

        let third = iterator.next().unwrap();
        assert_eq!(third, &[4, 4, 4, 4, b'\n']);

        let forth = iterator.next().unwrap();
        assert_eq!(forth, &[1, b'\n', 1, b'\n', 3, 3, 3, b'\n']);
    }

    #[test]
    fn find_test() {
        assert_eq!(find_index("cfg=".as_bytes(), b'c'), Some(0));
        assert_eq!(find_index("cfg=".as_bytes(), b'='), Some(3));
        assert_eq!(find_index("cfg-example=mkf".as_bytes(), b'='), Some(11));
        assert_eq!(find_index("cfg-example=mkf".as_bytes(), b'p'), Some(8));
        assert_eq!(
            find_index("cfg-example-longer-than-expected$=mkf".as_bytes(), b'$'),
            Some(32)
        );
        assert_eq!(
            find_index(TEST_VALUES.as_bytes(), b'5'),
            TEST_VALUES.find('5')
        );
        assert_eq!(
            find_index(TEST_VALUES.as_bytes(), b's'),
            TEST_VALUES.find('s')
        );
        assert_eq!(find_index(TEST_VALUES.as_bytes(), b's'), Some(0));

        assert_eq!(
            find_index(TEST_VALUES.as_bytes(), b'\n'),
            TEST_VALUES.find('\n')
        );
        assert_eq!(
            find_index(TEST_VALUES.as_bytes(), b'k'),
            TEST_VALUES.find('k')
        );
        assert_eq!(
            find_index(TEST_VALUES.as_bytes(), b'f'),
            TEST_VALUES.find('f')
        );
        assert_eq!(
            find_index(TEST_VALUES.as_bytes(), b'-'),
            TEST_VALUES.find('-')
        );
    }

    #[test]
    fn avx_line_iterator() {
        for (l1, l2) in AvxLineIterator::new(TEST_VALUES).zip(TEST_VALUES.lines()) {
            assert_eq!(l1, l2);
        }
    }

    #[test]
    fn avx_line_iterator_example_input() {
        for (l1, l2) in AvxLineIterator::new(EXAMPLE_INPUT).zip(EXAMPLE_INPUT.lines()) {
            assert_eq!(l1, l2);
        }
    }
}
