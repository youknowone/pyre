//! RPython `rpython/rtyper/tool/gcstat.py`.
//!
//! The upstream tool parses GC allocation logs with `malloc`,
//! `malloc_varsize`, and `free` rows into object lifetime records.  This
//! port keeps the same accounting: `birth` is the cumulative allocated
//! byte count before an allocation, `death` is the cumulative byte count
//! at `free`, and still-live objects are closed at EOF with `death`
//! equal to the final cumulative byte count.

use std::collections::HashMap;
use std::io::BufRead;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LifeTime {
    pub typeid: i64,
    pub address: u64,
    pub size: i64,
    pub varsize: bool,
    pub birth: i64,
    pub death: i64,
}

impl LifeTime {
    pub fn new(typeid: i64, address: u64, size: i64, varsize: bool, birth: i64) -> Self {
        Self {
            typeid,
            address,
            size,
            birth,
            death: -1,
            varsize,
        }
    }
}

/// RPython `parse_file(f, callback)` (`gcstat.py:16-38`).
pub fn parse_file<R, F>(reader: R, mut callback: F) -> Result<(), String>
where
    R: BufRead,
    F: FnMut(LifeTime),
{
    let mut unknown_lifetime: HashMap<u64, LifeTime> = HashMap::new();
    let mut current = 0_i64;

    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("gcstat.py: read line {i}: {e}"))?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        match parts[0] {
            "free" => {
                if parts.len() != 3 {
                    return Err(format!("gcstat.py: malformed free line {i}: {line}"));
                }
                let typeid = parse_i64(parts[1], i, "typeid")?;
                let address = parse_address(parts[2], i)?;
                let mut unknown = unknown_lifetime.remove(&address).ok_or_else(|| {
                    format!("gcstat.py: free of unknown address {address:#x} on line {i}")
                })?;
                if unknown.typeid != typeid {
                    return Err(format!(
                        "gcstat.py: free typeid mismatch on line {i}: got {typeid}, expected {}",
                        unknown.typeid
                    ));
                }
                unknown.death = current;
                callback(unknown);
            }
            "malloc" | "malloc_varsize" => {
                if parts.len() != 4 {
                    return Err(format!("gcstat.py: malformed malloc line {i}: {line}"));
                }
                let varsize = parts[0] == "malloc_varsize";
                let typeid = parse_i64(parts[1], i, "typeid")?;
                let size = parse_i64(parts[2], i, "size")?;
                let address = parse_address(parts[3], i)?;
                let new = LifeTime::new(typeid, address, size, varsize, current);
                unknown_lifetime.insert(address, new);
                current += size;
            }
            other => {
                return Err(format!(
                    "gcstat.py: unknown operation {other:?} on line {i}"
                ));
            }
        }
    }

    for mut unknown in unknown_lifetime.into_values() {
        unknown.death = current;
        callback(unknown);
    }
    Ok(())
}

/// RPython `collect_all(f)` (`gcstat.py:40-44`).
pub fn collect_all<R>(reader: R) -> Result<Vec<LifeTime>, String>
where
    R: BufRead,
{
    let mut all = Vec::new();
    parse_file(reader, |obj| all.push(obj))?;
    Ok(all)
}

fn parse_i64(s: &str, line: usize, field: &str) -> Result<i64, String> {
    s.parse::<i64>()
        .map_err(|e| format!("gcstat.py: invalid {field} on line {line}: {e}"))
}

fn parse_address(s: &str, line: usize) -> Result<u64, String> {
    let hex = s.strip_prefix("0x").unwrap_or(s);
    u64::from_str_radix(hex, 16)
        .map_err(|e| format!("gcstat.py: invalid address on line {line}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn collect_all_matches_gcstat_lifetime_accounting() {
        let log = "\
malloc 1 10 0x100
malloc_varsize 2 7 0x200
free 1 0x100
malloc 3 5 0x300
";
        let mut all = collect_all(Cursor::new(log)).unwrap();
        all.sort_by_key(|lt| lt.address);

        assert_eq!(
            all,
            vec![
                LifeTime {
                    typeid: 1,
                    address: 0x100,
                    size: 10,
                    varsize: false,
                    birth: 0,
                    death: 17,
                },
                LifeTime {
                    typeid: 2,
                    address: 0x200,
                    size: 7,
                    varsize: true,
                    birth: 10,
                    death: 22,
                },
                LifeTime {
                    typeid: 3,
                    address: 0x300,
                    size: 5,
                    varsize: false,
                    birth: 17,
                    death: 22,
                },
            ]
        );
    }

    #[test]
    fn parse_file_reports_free_of_unknown_address() {
        let err = collect_all(Cursor::new("free 1 0x123\n")).unwrap_err();
        assert!(err.contains("free of unknown address"), "got {err}");
    }
}
