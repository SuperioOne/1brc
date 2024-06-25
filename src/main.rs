use std::collections::BTreeMap;
use std::env::args;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::num::ParseFloatError;

#[derive(Debug)]
enum BrcErrors {
    InvalidPath,
    NoSemiColonForSomeReason,
    FloatParseFailed,
    IOError(String),
}

impl From<std::io::Error> for BrcErrors {
    fn from(value: std::io::Error) -> Self {
        BrcErrors::IOError(value.to_string())
    }
}

impl From<ParseFloatError> for BrcErrors {
    fn from(_value: ParseFloatError) -> Self {
        BrcErrors::FloatParseFailed
    }
}

struct StationData {
    min: f32,
    max: f32,
    count: usize,
    total: f32,
}

fn main() -> Result<(), BrcErrors> {
    let fd = if let Some(file_path) = args().nth(1) {
        let fd = std::fs::File::open(file_path)?;
        Ok(fd)
    } else {
        Err(BrcErrors::InvalidPath)
    }?;

    let mut reader = BufReader::new(fd);
    let mut line_buf = String::with_capacity(100);
    let mut stations: BTreeMap<Box<str>, StationData> = BTreeMap::new();

    loop {
        match reader.read_line(&mut line_buf) {
            Ok(0) => break,
            Ok(_) => {
                let mut split_pos: usize = 0;

                for (idx, char) in line_buf.as_bytes().iter().enumerate() {
                    if *char == b';' {
                        split_pos = idx;
                        break;
                    }
                }

                if split_pos == 0 {
                    return Err(BrcErrors::NoSemiColonForSomeReason);
                }

                let s_name = &line_buf[0..split_pos];
                let s_val = (&line_buf[(split_pos + 1)..(line_buf.len() - 1)]).parse::<f32>()?;

                match stations.get_mut(s_name).as_deref_mut() {
                    Some(station) => {
                        station.min = if s_val.lt(&station.min) {
                            s_val
                        } else {
                            station.min
                        };

                        station.max = if s_val.gt(&station.max) {
                            s_val
                        } else {
                            station.max
                        };

                        station.count += 1;
                        station.total += s_val;
                    }
                    None => {
                        stations.insert(
                            Box::from(s_name),
                            StationData {
                                max: s_val,
                                min: s_val,
                                total: s_val,
                                count: 1,
                            },
                        );
                    }
                }

                line_buf.clear();
            }
            Err(err) => return Err(BrcErrors::IOError(err.to_string())),
        }
    }

    let mut stdout_writer = BufWriter::new(std::io::stdout());
    stdout_writer.write(b"{")?;
    for (key, value) in stations {
        stdout_writer.write_fmt(format_args!(
            "{}={:.1}/{:.1}/{:.1}, ",
            &key,
            value.min,
            value.total / value.count as f32,
            value.min,
        ))?;
    }
    stdout_writer.write(b"}")?;
    stdout_writer.flush()?;

    Ok(())
}
