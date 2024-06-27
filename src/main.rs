use algorithms::hash::cityhash::hash_fn::cityhash_64_with_seed;
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
    thread::{available_parallelism, ScopedJoinHandle},
};

struct City64HasherBuilder;
struct City64Hasher {
    inner: u64,
}

impl Hasher for City64Hasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.inner
    }

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
enum BrcErrors {
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
    min: f32,
    max: f32,
    count: u32,
    total: f32,
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

#[derive(Debug, Clone)]
struct Mmap {
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

    pub fn as_byte_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.inner.addr.cast::<u8>(), self.inner.len) }
    }

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

struct ChunkedLinesIter<'a> {
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
        let mut end_pos = self.pos + self.chunk_size;

        if end_pos >= self.bytes.len() {
            if self.pos < self.bytes.len() {
                let start = self.pos;
                self.pos = self.bytes.len();

                Some(&self.bytes[start..])
            } else {
                None
            }
        } else {
            // End position could be middle of the row. We simply look ahead to find nearest line feed (0x0A).
            // And, rustc generates pretty acceptable assembly for `iter.skip().enumerate()`.
            for (idx, val) in self.bytes.iter().enumerate().skip(end_pos - 1) {
                if *val == b'\n' {
                    end_pos = idx;
                    break;
                }
            }

            let start = self.pos;
            self.pos = end_pos + 1;

            Some(&self.bytes[start..=end_pos])
        }
    }
}

type StationHashMap = HashMap<HashKey, StationData, City64HasherBuilder>;
type HashKey = u128;

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

                for line in chunk_str.lines() {
                    let mut split_pos: usize = 0;

                    for (idx, char) in line.as_bytes().iter().enumerate().rev() {
                        if *char == b';' {
                            split_pos = idx;
                            break;
                        }
                    }

                    let s_name = &line[0..split_pos];
                    let key = get_key(s_name.as_bytes());

                    if let Ok(s_val) = (&line[(split_pos + 1)..line.len()]).parse::<f32>() {
                        match store.get_mut(&key) {
                            Some(entry) => {
                                entry.min = entry.min.min(s_val);
                                entry.max = entry.max.max(s_val);
                                entry.total += s_val;
                                entry.count += 1;
                            }
                            None => {
                                _ = store.insert(
                                    key,
                                    StationData {
                                        name: Box::from(s_name),
                                        min: s_val,
                                        max: s_val,
                                        count: 1,
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
                "{}={:.1}/{:.1}/{:.1} }}",
                value.name,
                value.min,
                value.total / value.count as f32,
                value.min,
            ))?;
        } else {
            stdout_writer.write_fmt(format_args!(
                "{}={:.1}/{:.1}/{:.1}, ",
                value.name,
                value.min,
                value.total / value.count as f32,
                value.min,
            ))?;
        }
    }

    stdout_writer.flush()?;

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::ChunkedLinesIter;

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
}
