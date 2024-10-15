use algorithms_buffer_utils::FastBufferUtils;
use algorithms_hash::cityhash::hash_fn::cityhash_64_with_seed;
use core::slice;
use libc::{c_void, mmap64, munmap, MAP_FAILED, MAP_PRIVATE, PROT_READ};
use std::{
    collections::{BTreeMap, HashMap},
    env::args,
    fs::File,
    hash::{BuildHasher, Hasher},
    io::{BufWriter, Write},
    os::{fd::AsRawFd, unix::fs::MetadataExt},
    ptr::null_mut,
    str::from_utf8_unchecked,
    sync::Arc,
    thread::{self, available_parallelism, JoinHandle},
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
    name: Box<[u8]>,
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
#[derive(Debug)]
struct Mmap {
    addr: *const c_void,
    len: usize,
}

unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Drop for Mmap {
    fn drop(&mut self) {
        let res = unsafe { munmap(self.addr.cast_mut(), self.len) };
        if res != 0 {
            eprintln!("Unable to drop mmapped region, code : {}", res);
        }
    }
}

impl Mmap {
    #[inline]
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

        Ok(Self { len, addr })
    }

    #[inline]
    pub fn as_byte_slice<'a>(&'a self) -> &'a [u8] {
        unsafe { slice::from_raw_parts::<'a, u8>(self.addr.cast::<u8>(), self.len) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
}

#[derive(Clone)]
struct MmapSlice {
    inner: Arc<Mmap>,
    offset: usize,
    len: usize,
}

impl MmapSlice {
    #[inline]
    pub fn as_byte_slice<'a>(&'a self) -> &'a [u8] {
        unsafe {
            slice::from_raw_parts::<'a, u8>(
                self.inner.addr.cast::<u8>().byte_add(self.offset),
                self.len,
            )
        }
    }
}

/// Splits a slice into multiple chunks with considiration of LF positions.
/// It does not gurantees equal chunk sizes since it looks for nearest LF from chunk offset.
pub struct ChunkedLinesIter<T> {
    inner: T,
    pos: usize,
    chunk_size: usize,
}

impl<T> ChunkedLinesIter<T> {
    #[inline]
    pub fn chunk_by(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }
}

impl From<Arc<Mmap>> for ChunkedLinesIter<Arc<Mmap>> {
    fn from(value: Arc<Mmap>) -> Self {
        Self {
            pos: 0,
            inner: value,
            chunk_size: 65536, // Default is 64KiB
        }
    }
}

impl Iterator for ChunkedLinesIter<Arc<Mmap>> {
    type Item = MmapSlice;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let start = self.pos;
        let end = start + self.chunk_size - 1;
        let len = self.inner.len;

        if end >= len {
            if self.pos < len {
                self.pos = len;

                Some(Self::Item {
                    len: len - start,
                    inner: self.inner.clone(),
                    offset: start,
                })
            } else {
                None
            }
        } else {
            // End position could be middle of the row. We simply look ahead to find nearest LF (0x0A) before slicing.
            let look_ahead_chunk = &self.inner.as_byte_slice()[end..];

            if let Some(lf_idx) = look_ahead_chunk.fast_find(b'\n') {
                let chunk_end = end + lf_idx;

                self.pos = chunk_end + 1;
                Some(Self::Item {
                    len: chunk_end - start + 1,
                    inner: self.inner.clone(),
                    offset: start,
                })
            } else {
                Some(Self::Item {
                    len: len - start,
                    inner: self.inner.clone(),
                    offset: start,
                })
            }
        }
    }
}

type StationHashMap = HashMap<HashKey, StationData, City64HasherBuilder>;
type HashKey = u128;

macro_rules! parse_number {
    ($v:expr) => {{
        let mut val: u64 = $v;
        val = (val & 0x0F0F0F0F0F0F0F0F) * 2561 >> 8;
        val = (val & 0x00FF00FF00FF00FF) * 6553601 >> 16;
        (val & 0x0000FFFF0000FFFF) * 42949672960001 >> 32
    }};
}

// Parses float values between -99.9 to 99.9
#[inline]
fn parse_f64(value: &[u8]) -> f64 {
    match value {
        [b'-', h1, h0, b'.', l] => {
            let val = parse_number!(u64::from_le_bytes([0, 0, 0, 0, 0, *h1, *h0, *l]));
            (val as f64) * -0.1
        }
        [b'-', h0, b'.', l] => {
            let val = parse_number!(u64::from_le_bytes([0, 0, 0, 0, 0, 0, *h0, *l]));
            (val as f64) * -0.1
        }
        [h1, h0, b'.', l] => {
            let val = parse_number!(u64::from_le_bytes([0, 0, 0, 0, 0, *h1, *h0, *l]));
            (val as f64) * 0.1
        }
        [h0, b'.', l] => {
            let val = parse_number!(u64::from_le_bytes([0, 0, 0, 0, 0, 0, *h0, *l]));
            (val as f64) * 0.1
        }
        _ => unreachable!(),
    }
}

/// Truncates station names into u128 (16-byte) value with Big-Endian format. This version assumes
/// first 16 characters of station names are always unique.
///
/// Why Big-Endian?
/// On little-endian systems first char reads as least significant bits which makes ordering with
/// std::collection::BTree problematic. For example; when we read "Melbourne" from memory as u128,
/// we endup with "enruobleM" as value. Applying bswap instruction fixes the ordering in a very cheap way.
#[inline]
fn get_key(input: &[u8]) -> HashKey {
    let ptr: *const HashKey = input.as_ptr().cast();
    let len = input.len();

    match len {
        0 => 0,
        1..=15 => {
            let mask = u128::MAX >> ((16 - len) * 8);
            let a = unsafe { ptr.read() } & mask;
            a.to_be()
        }
        _ => unsafe { ptr.read() }.to_be(),
    }
}

#[inline]
fn process_file(mmap: Mmap, chunk_size: usize) -> Result<(), BrcErrors> {
    let mmap_arc = Arc::new(mmap);

    let job_handles: Vec<JoinHandle<StationHashMap>> = ChunkedLinesIter::from(mmap_arc)
        .chunk_by(chunk_size)
        .into_iter()
        .map(|slice| {
            thread::spawn(move || {
                let chunk = slice.as_byte_slice();
                let mut store: StationHashMap =
                    StationHashMap::with_capacity_and_hasher(512, City64HasherBuilder::default());

                for line in chunk.fast_split_by_byte(b'\n') {
                    // There are very limited cases for ';' position.
                    // For;
                    //      C char
                    //      N number
                    //
                    // 6 5 4 3 2 1    -> Line last position indexes
                    // ------------
                    // C ; N N . N
                    // C ; - N . N
                    // C C ; N . N
                    // ; - N N . N
                    //
                    // So far,';' can only appear at position 6, 5 and 4. We simply check these
                    // position patterns which approximately takes 10 instruction and single
                    // branching.
                    let split_pos =
                        match unsafe { line.get_unchecked((line.len() - 6)..(line.len() - 3)) } {
                            [b';', ..] => line.len() - 6,
                            [_, b';', ..] => line.len() - 5,
                            [_, _, b';', ..] => line.len() - 4,
                            _ => {
                                unreachable!();
                            }
                        };

                    let s_val = parse_f64(&line[(split_pos + 1)..line.len()]);
                    let s_name = &line[0..split_pos];
                    let key = get_key(s_name);

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

                store
            })
        })
        .collect();

    let mut merge_btree: BTreeMap<HashKey, StationData> = BTreeMap::new();

    for result_block in job_handles.into_iter().flat_map(|handle| handle.join()) {
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

    // let mut stdout_writer = BufWriter::new(std::io::stdout());
    // let last_idx = merge_btree.len() - 1;
    //
    // _ = stdout_writer.write(b"{")?;
    //
    // for (idx, (_, value)) in merge_btree.into_iter().enumerate() {
    //     stdout_writer.write_fmt(format_args!(
    //         "{}={:.1}/{:.1}/{:.1}",
    //         unsafe { from_utf8_unchecked(&value.name) },
    //         value.min,
    //         value.total / value.count,
    //         value.max,
    //     ))?;
    //
    //     if idx == last_idx {
    //         stdout_writer.write(b"}")?;
    //     } else {
    //         stdout_writer.write(b", ")?;
    //     }
    // }
    //
    // stdout_writer.flush()?;
    Ok(())
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
    let chunk_size: usize = mmap.len() / worker_count;
    process_file(mmap, chunk_size)?;

    Ok(())
}
