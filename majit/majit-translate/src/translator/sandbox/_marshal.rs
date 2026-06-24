//! RPython `rpython/translator/sandbox/_marshal.py`.
//!
//! The upstream file is a Python-2-compatible marshal implementation for the
//! sandbox protocol. This Rust slice keeps the same `dump`/`load`/
//! `dumps`/`loads` entry names over the value kinds currently needed by the
//! local sandbox surface.

use std::fmt;
use std::io::{Read, Write};

#[allow(non_upper_case_globals)]
pub const version: i32 = 0;

const TYPE_NONE: u8 = b'N';
const TYPE_FALSE: u8 = b'F';
const TYPE_TRUE: u8 = b'T';
const TYPE_INT: u8 = b'i';
const TYPE_INT64: u8 = b'I';
const TYPE_STRING: u8 = b's';
const TYPE_UNICODE: u8 = b'u';
const TYPE_TUPLE: u8 = b'(';

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MarshalValue {
    None,
    Bool(bool),
    Int(i64),
    String(String),
    Tuple(Vec<MarshalValue>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MarshalError {
    pub message: String,
}

impl MarshalError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for MarshalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for MarshalError {}

pub struct _Marshaller<W: Write> {
    writer: W,
}

impl<W: Write> _Marshaller<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    pub fn dump(&mut self, value: &MarshalValue) -> Result<(), MarshalError> {
        write_value(&mut self.writer, value)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct _NULL;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct _StringBuffer {
    pub buf: Vec<u8>,
}

impl _StringBuffer {
    pub fn write(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }
}

pub struct _Unmarshaller<R: Read> {
    reader: R,
}

impl<R: Read> _Unmarshaller<R> {
    pub fn new(reader: R) -> Self {
        Self { reader }
    }

    pub fn load(&mut self) -> Result<MarshalValue, MarshalError> {
        read_value(&mut self.reader)
    }
}

pub struct _FastUnmarshaller<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> _FastUnmarshaller<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn load(&mut self) -> Result<MarshalValue, MarshalError> {
        read_value(self)
    }
}

impl Read for _FastUnmarshaller<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let remaining = self.data.len().saturating_sub(self.pos);
        let count = remaining.min(buf.len());
        buf[..count].copy_from_slice(&self.data[self.pos..self.pos + count]);
        self.pos += count;
        Ok(count)
    }
}

pub fn dump<W: Write>(x: &MarshalValue, f: &mut W, _version: i32) -> Result<(), MarshalError> {
    write_value(f, x)
}

pub fn load<R: Read>(f: &mut R) -> Result<MarshalValue, MarshalError> {
    read_value(f)
}

pub fn dumps(x: &MarshalValue, _version: i32) -> Result<Vec<u8>, MarshalError> {
    let mut out = Vec::new();
    dump(x, &mut out, _version)?;
    Ok(out)
}

pub fn loads(s: &[u8]) -> Result<MarshalValue, MarshalError> {
    let mut reader = _FastUnmarshaller::new(s);
    reader.load()
}

fn write_value<W: Write>(writer: &mut W, value: &MarshalValue) -> Result<(), MarshalError> {
    match value {
        MarshalValue::None => writer.write_all(&[TYPE_NONE]).map_err(io_error),
        MarshalValue::Bool(false) => writer.write_all(&[TYPE_FALSE]).map_err(io_error),
        MarshalValue::Bool(true) => writer.write_all(&[TYPE_TRUE]).map_err(io_error),
        MarshalValue::Int(value) => dump_int(writer, *value),
        MarshalValue::String(value) => {
            writer.write_all(&[TYPE_STRING]).map_err(io_error)?;
            w_long(writer, value.len() as i64)?;
            writer.write_all(value.as_bytes()).map_err(io_error)
        }
        MarshalValue::Tuple(items) => {
            writer.write_all(&[TYPE_TUPLE]).map_err(io_error)?;
            w_long(writer, items.len() as i64)?;
            for item in items {
                write_value(writer, item)?;
            }
            Ok(())
        }
    }
}

fn dump_int<W: Write>(writer: &mut W, x: i64) -> Result<(), MarshalError> {
    let y = x >> 31;
    if y != 0 && y != -1 {
        writer.write_all(&[TYPE_INT64]).map_err(io_error)?;
        w_long64(writer, x)
    } else {
        writer.write_all(&[TYPE_INT]).map_err(io_error)?;
        w_long(writer, x)
    }
}

fn read_value<R: Read>(reader: &mut R) -> Result<MarshalValue, MarshalError> {
    let mut tag = [0_u8; 1];
    reader.read_exact(&mut tag).map_err(io_error)?;
    match tag[0] {
        TYPE_NONE => Ok(MarshalValue::None),
        TYPE_FALSE => Ok(MarshalValue::Bool(false)),
        TYPE_TRUE => Ok(MarshalValue::Bool(true)),
        TYPE_INT => Ok(MarshalValue::Int(r_long(reader)?)),
        TYPE_INT64 => Ok(MarshalValue::Int(r_long64(reader)?)),
        TYPE_STRING | TYPE_UNICODE => {
            let n = read_size(r_long(reader)?)?;
            let mut bytes = vec![0_u8; n];
            reader.read_exact(&mut bytes).map_err(io_error)?;
            String::from_utf8(bytes)
                .map(MarshalValue::String)
                .map_err(|e| MarshalError::new(e.to_string()))
        }
        TYPE_TUPLE => {
            let n = read_size(r_long(reader)?)?;
            let mut items = Vec::with_capacity(n);
            for _ in 0..n {
                items.push(read_value(reader)?);
            }
            Ok(MarshalValue::Tuple(items))
        }
        other => Err(MarshalError::new(format!(
            "_marshal.py: bad marshal code: {other:?}"
        ))),
    }
}

fn read_size(n: i64) -> Result<usize, MarshalError> {
    usize::try_from(n).map_err(|_| MarshalError::new("_marshal.py: negative length"))
}

fn w_long<W: Write>(writer: &mut W, x: i64) -> Result<(), MarshalError> {
    writer
        .write_all(&(x as u32).to_le_bytes())
        .map_err(io_error)
}

fn w_long64<W: Write>(writer: &mut W, x: i64) -> Result<(), MarshalError> {
    w_long(writer, x)?;
    w_long(writer, x >> 32)
}

fn r_long<R: Read>(reader: &mut R) -> Result<i64, MarshalError> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes).map_err(io_error)?;
    Ok(i32::from_le_bytes(bytes) as i64)
}

fn r_long64<R: Read>(reader: &mut R) -> Result<i64, MarshalError> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes).map_err(io_error)?;
    Ok(i64::from_le_bytes(bytes))
}

fn io_error(e: std::io::Error) -> MarshalError {
    MarshalError::new(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dumps_loads_round_trip_tuple() {
        let value = MarshalValue::Tuple(vec![
            MarshalValue::String("os_open".to_string()),
            MarshalValue::Int(3),
            MarshalValue::Bool(true),
        ]);
        let encoded = dumps(&value, version).unwrap();
        assert_eq!(loads(&encoded).unwrap(), value);
    }
}
