//! RPython's `rpython.rlib.rutf8` code point index storage, ported
//! structurally to Rust.
//!
//! A UTF-8 (here WTF-8) buffer addresses code points by byte offset, so
//! resolving the n-th code point means walking from the start.  The index
//! storage turns that walk into a table lookup: one entry per 64 code points
//! holding the byte offset the group starts at, plus a one-byte delta for every
//! fourth code point inside it.  A lookup reads the entry, adds the delta, and
//! steps at most two code points — `codepoint_position_at_index`.
//!
//! The table costs 24 bytes per 64 code points and is built once per string,
//! on the first non-ASCII index; an ASCII payload never needs one because its
//! code point index is already its byte offset (`W_UnicodeObject.is_ascii`).
//!
//! Names, entry layout, group sizes and the build loop follow
//! `rpython/rlib/rutf8.py:131-566`.  The `_is_64bit` branch of
//! `next_codepoint_pos` (a hand-tuned x86-64 sequence guarded by
//! `not jit.we_are_jitted()`) is left out: it computes the same value as the
//! comparison ladder it sits in front of.

/// `UTF8_INDEX_STORAGE`'s element (`rutf8.py:508-511`) — `baseindex` is the
/// byte offset the 64-code-point group starts at, and `ofs[i]` the byte delta
/// from it to the code point *after* the group's `4 * i`-th.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Utf8LocElem {
    pub baseindex: isize,
    pub ofs: [u8; 16],
}

/// `UTF8_INDEX_STORAGE` (`rutf8.py:508`).  A leaf (no inner `PyObjectRef`), so
/// a mortal string's table lives in a storage box whose drop glue reclaims it,
/// as its WTF-8 buffer does.
pub type Utf8IndexStorage = Vec<Utf8LocElem>;

/// `next_codepoint_pos` (`rutf8.py:131`) — the position of the code point after
/// `pos`.  Assumes valid WTF-8 and `pos` before the end.
#[inline]
pub fn next_codepoint_pos(code: &[u8], pos: usize) -> usize {
    let chr1 = code[pos];
    if chr1 <= 0x7F {
        return pos + 1;
    }
    if chr1 <= 0xDF {
        return pos + 2;
    }
    if chr1 <= 0xEF {
        return pos + 3;
    }
    pos + 4
}

/// `prev_codepoint_pos` (`rutf8.py:152`) — the position of the code point
/// before `pos`, which must not be zero.  A `pos` one past the end reads as the
/// extra `'\x00'` the build loop pretends is there.
#[inline]
pub fn prev_codepoint_pos(code: &[u8], pos: usize) -> usize {
    let mut pos = pos - 1;
    if pos >= code.len() {
        return pos;
    }
    if code[pos] <= 0x7F {
        return pos;
    }
    pos -= 1;
    if code[pos] >= 0xC0 {
        return pos;
    }
    pos -= 1;
    if code[pos] >= 0xC0 {
        return pos;
    }
    pos - 1
}

/// `create_utf8_index_storage` (`rutf8.py:516`) — the table for `utf8`, whose
/// code point count is `utf8len`.
pub fn create_utf8_index_storage(utf8: &[u8], utf8len: usize) -> Utf8IndexStorage {
    let arraysize = utf8len / 64 + 1;
    let mut storage = vec![
        Utf8LocElem {
            baseindex: 0,
            ofs: [0u8; 16],
        };
        arraysize
    ];
    // Signed: the count overshoots the last group by design and the loop stops
    // on the first negative value.
    let mut utf8len = utf8len as isize;
    let mut baseindex = 0usize;
    let mut current = 0usize;
    loop {
        storage[current].baseindex = baseindex as isize;
        let mut next = baseindex;
        // Stands in for the `for ... else` upstream breaks out of.
        let mut group_filled = true;
        for i in 0..16 {
            if utf8len == 0 {
                next += 1; // assume there is an extra '\x00' character
            } else {
                next = next_codepoint_pos(utf8, next);
            }
            storage[current].ofs[i] = (next - baseindex) as u8;
            utf8len -= 4;
            if utf8len < 0 {
                debug_assert_eq!(current + 1, storage.len());
                group_filled = false;
                break;
            }
            next = next_codepoint_pos(utf8, next);
            next = next_codepoint_pos(utf8, next);
            next = next_codepoint_pos(utf8, next);
        }
        if !group_filled {
            break;
        }
        current += 1;
        baseindex = next;
    }
    storage
}

/// `codepoint_position_at_index` (`rutf8.py:548`) — the byte offset of code
/// point `index`, which must not exceed the string's code point count.
#[inline]
pub fn codepoint_position_at_index(utf8: &[u8], storage: &[Utf8LocElem], index: usize) -> usize {
    let elem = &storage[index >> 6];
    let bytepos = elem.baseindex as usize + elem.ofs[(index >> 2) & 0x0F] as usize;
    match index & 0x3 {
        0 => prev_codepoint_pos(utf8, bytepos),
        1 => bytepos,
        2 => next_codepoint_pos(utf8, bytepos),
        _ => next_codepoint_pos(utf8, next_codepoint_pos(utf8, bytepos)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustpython_wtf8::{CodePoint, Wtf8Buf};

    fn sample(kind: &str, repeat: usize) -> Wtf8Buf {
        let mut buf = Wtf8Buf::new();
        for i in 0..repeat {
            match kind {
                "ascii" => buf.push_str("abcd"),
                "latin1" => buf.push_str("éèêë"),
                "wide" => buf.push_str("一二三四"),
                "astral" => buf.push_str("\u{1f600}\u{1f601}\u{1f602}\u{1f603}"),
                "mixed" => buf.push_str("a\u{e9}\u{4e00}\u{1f600}"),
                "surrogate" => {
                    buf.push_str("a");
                    buf.push(CodePoint::from_u32(0xD800 + (i % 0x400) as u32).unwrap());
                    buf.push_str("\u{4e00}");
                    buf.push(CodePoint::from_u32(0xDC80).unwrap());
                }
                other => panic!("unknown sample kind {other}"),
            }
        }
        buf
    }

    /// The table must answer exactly what a walk from the start would.
    fn assert_matches_walk(buf: &Wtf8Buf) {
        let utf8 = buf.as_bytes();
        let positions: Vec<usize> = buf.code_point_indices().map(|(pos, _)| pos).collect();
        let storage = create_utf8_index_storage(utf8, positions.len());
        for (index, &expected) in positions.iter().enumerate() {
            assert_eq!(
                codepoint_position_at_index(utf8, &storage, index),
                expected,
                "index {index} of a {} byte / {} code point payload",
                utf8.len(),
                positions.len(),
            );
        }
    }

    #[test]
    fn test_index_storage_matches_a_walk() {
        // Lengths straddling the 4-code-point delta stride and the 64-code
        // point group, in both directions.
        for repeat in [0, 1, 2, 15, 16, 17, 32, 63, 64, 65, 100] {
            for kind in ["ascii", "latin1", "wide", "astral", "mixed", "surrogate"] {
                assert_matches_walk(&sample(kind, repeat));
            }
        }
    }

    #[test]
    fn test_index_storage_handles_a_partial_final_group() {
        // A code point count that is neither a multiple of 4 nor of 64 leaves
        // the last group half-written; every filled entry must still answer.
        let mut buf = sample("mixed", 3);
        buf.push_str("x\u{4e00}");
        assert_eq!(buf.code_points().count(), 14);
        assert_matches_walk(&buf);
    }

    #[test]
    fn test_index_storage_size_is_one_entry_per_64_code_points() {
        assert_eq!(create_utf8_index_storage(b"", 0).len(), 1);
        assert_eq!(create_utf8_index_storage(&[b'a'; 64], 64).len(), 2);
        assert_eq!(create_utf8_index_storage(&[b'a'; 127], 127).len(), 2);
        assert_eq!(create_utf8_index_storage(&[b'a'; 128], 128).len(), 3);
    }
}
