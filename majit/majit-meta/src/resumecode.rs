//! Resume bytecode encoding/decoding.
//!
//! Direct port of rpython/jit/metainterp/resumecode.py.
//!
//! Format (from resumecode.py docstring):
//! ```text
//!   # ----- resume section
//!   [total size of resume section]
//!   [number of failargs]
//!   [<length> <virtualizable object> <numb> <numb> <numb>]    if vinfo
//!    -OR-
//!   [1 <ginfo object>]                                        if ginfo
//!    -OR-
//!   [0]                                                       if both are None
//!
//!   [<length> <virtual> <vref> <virtual> <vref>]     for virtualrefs
//!
//!   [<pc> <jitcode> <numb> <numb> <numb>]            the frames
//!   [<pc> <jitcode> <numb> <numb>]
//!   ...
//!
//!   until the size of the resume section
//!
//!   # ----- optimization section
//!   <more code>                     further sections according to bridgeopt.py
//! ```
//!
//! Encoding: variable-length integers with zigzag encoding.
//! - 7-bit:  0xxxxxxx
//! - 14-bit: 1xxxxxxx 0xxxxxxx
//! - 21-bit: 1xxxxxxx 1xxxxxxx xxxxxxxx

/// resumecode.py: append_numbering(lst, item)
/// Encode a signed integer using zigzag + variable-length encoding.
pub fn encode_varint(buf: &mut Vec<u8>, value: i32) {
    // Zigzag: value * 2, negative → invert
    let mut item = (value as i64) * 2;
    if item < 0 {
        item = -1 - item;
    }
    debug_assert!(item >= 0);
    let item = item as u32;

    if item < (1 << 7) {
        buf.push(item as u8);
    } else if item < (1 << 14) {
        buf.push((item | 0x80) as u8);
        buf.push((item >> 7) as u8);
    } else {
        debug_assert!(item < (1 << 16), "resumecode item too large: {item}");
        buf.push((item | 0x80) as u8);
        buf.push(((item >> 7) | 0x80) as u8);
        buf.push((item >> 14) as u8);
    }
}

/// resumecode.py: numb_next_item(numb, index)
/// Decode a single varint, returning (value, new_index).
/// Exactly matches RPython's decoding:
///   if value & 1: value = -1 - value
///   value >>= 1
pub fn decode_varint(buf: &[u8], index: usize) -> (i32, usize) {
    let mut value = buf[index] as i64;
    let mut index = index + 1;

    if value & (1 << 7) != 0 {
        value &= (1 << 7) - 1;
        value |= (buf[index] as i64) << 7;
        index += 1;
        if value & (1 << 14) != 0 {
            value &= (1 << 14) - 1;
            value |= (buf[index] as i64) << 14;
            index += 1;
        }
    }

    // RPython: if value & 1: value = -1 - value; value >>= 1
    if value & 1 != 0 {
        value = -1 - value;
    }
    value >>= 1;

    (value as i32, index)
}

/// resumecode.py: unpack_numbering(numb)
pub fn unpack_all(buf: &[u8]) -> Vec<i32> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < buf.len() {
        let (next, new_i) = decode_varint(buf, i);
        result.push(next);
        i = new_i;
    }
    result
}

/// resumecode.py: Writer
pub struct Writer {
    pub current: Vec<i32>,
}

impl Writer {
    pub fn new(size_hint: usize) -> Self {
        Writer {
            current: Vec::with_capacity(size_hint),
        }
    }

    /// resumecode.py: append_short
    pub fn append_short(&mut self, item: i32) {
        self.current.push(item);
    }

    /// resumecode.py: append_int (with i16 range check)
    pub fn append_int(&mut self, item: i32) {
        debug_assert!(
            item >= i16::MIN as i32 && item <= i16::MAX as i32,
            "append_int: value {item} out of i16 range"
        );
        self.append_short(item);
    }

    /// resumecode.py: create_numbering
    pub fn create_numbering(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.current.len() * 3);
        for &item in &self.current {
            encode_varint(&mut buf, item);
        }
        buf
    }

    /// resumecode.py: patch_current_size
    pub fn patch_current_size(&mut self, index: usize) {
        self.current[index] = self.current.len() as i32;
    }

    /// resumecode.py: patch
    pub fn patch(&mut self, index: usize, item: i32) {
        self.current[index] = item;
    }
}

/// resumecode.py: Reader
pub struct Reader<'a> {
    code: &'a [u8],
    pub cur_pos: usize,
    pub items_read: usize,
}

impl<'a> Reader<'a> {
    pub fn new(code: &'a [u8]) -> Self {
        Reader {
            code,
            cur_pos: 0,
            items_read: 0,
        }
    }

    /// resumecode.py: next_item
    pub fn next_item(&mut self) -> i32 {
        let (result, new_pos) = decode_varint(self.code, self.cur_pos);
        self.cur_pos = new_pos;
        self.items_read += 1;
        result
    }

    /// resumecode.py: peek
    pub fn peek(&self) -> i32 {
        let (result, _) = decode_varint(self.code, self.cur_pos);
        result
    }

    /// resumecode.py: jump — skip n items forward
    pub fn jump(&mut self, size: usize) {
        for _ in 0..size {
            let (_, new_pos) = decode_varint(self.code, self.cur_pos);
            self.cur_pos = new_pos;
        }
        self.items_read += size;
    }

    /// resumecode.py: unpack (for debugging)
    pub fn unpack(&self) -> Vec<i32> {
        unpack_all(self.code)
    }

    /// Check if there are more items to read.
    pub fn has_more(&self) -> bool {
        self.cur_pos < self.code.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_small_positive() {
        let mut buf = Vec::new();
        encode_varint(&mut buf, 0);
        encode_varint(&mut buf, 1);
        encode_varint(&mut buf, 63);

        let items = unpack_all(&buf);
        assert_eq!(items, vec![0, 1, 63]);
    }

    #[test]
    fn test_roundtrip_small_negative() {
        let mut buf = Vec::new();
        encode_varint(&mut buf, -1);
        encode_varint(&mut buf, -64);

        let items = unpack_all(&buf);
        assert_eq!(items, vec![-1, -64]);
    }

    #[test]
    fn test_roundtrip_medium() {
        let mut buf = Vec::new();
        encode_varint(&mut buf, 100);
        encode_varint(&mut buf, -100);
        encode_varint(&mut buf, 8191);

        let items = unpack_all(&buf);
        assert_eq!(items, vec![100, -100, 8191]);
    }

    #[test]
    fn test_writer_reader() {
        let mut w = Writer::new(10);
        w.append_int(42);
        w.append_int(-7);
        w.append_int(0);
        w.append_int(1000);

        let numb = w.create_numbering();
        let mut r = Reader::new(&numb);

        assert_eq!(r.next_item(), 42);
        assert_eq!(r.next_item(), -7);
        assert_eq!(r.next_item(), 0);
        assert_eq!(r.next_item(), 1000);
        assert!(!r.has_more());
    }

    #[test]
    fn test_reader_peek_and_jump() {
        let mut w = Writer::new(5);
        w.append_int(10);
        w.append_int(20);
        w.append_int(30);

        let numb = w.create_numbering();
        let mut r = Reader::new(&numb);

        assert_eq!(r.peek(), 10);
        assert_eq!(r.next_item(), 10);
        r.jump(1); // skip 20
        assert_eq!(r.next_item(), 30);
    }

    #[test]
    fn test_writer_patch() {
        let mut w = Writer::new(5);
        let idx = w.current.len();
        w.append_short(0); // placeholder
        w.append_int(42);
        w.append_int(43);
        w.patch_current_size(idx);

        assert_eq!(w.current[idx], 3); // total size = 3
    }
}
