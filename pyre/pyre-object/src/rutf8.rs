//! RPython's `rpython.rlib.rutf8`, ported structurally to Rust.
//!
//! # Relationship to `rustpython-wtf8`
//!
//! RPython has no string *type*: `W_UnicodeObject._utf8` is a plain `str`
//! (immutable bytes) and every operation on it is a free function in `rutf8`
//! taking `(code, pos)`.  pyre stores the same buffer as a `Wtf8Buf`, a type
//! that already carries the well-formedness invariant, so the two overlap and
//! the split has to be deliberate:
//!
//! * **`rustpython-wtf8` owns the representation and sequential access.** An
//!   `rutf8` function whose whole content is "encode, decode or validate,
//!   scanning forward" has a checked counterpart there and is *not* re-ported:
//!   a second, unchecked implementation beside a checked one would be two
//!   sources of truth for one invariant.  `unichr_as_utf8*` (:40) is
//!   `CodePoint::encode_wtf8` / `Wtf8Buf::push`; `check_utf8` (:351),
//!   `_check_utf8` (:373) and `get_utf8_length` (:364) are `Wtf8::from_bytes`;
//!   `check_ascii` (:242) and `first_non_ascii_char` (:249) are
//!   `Wtf8::is_ascii` and a byte scan; `has_surrogates` (:439) and
//!   `surrogate_in_utf8` (:489) are `Wtf8::as_str().is_err()`; `islinebreak`
//!   (:255), `isspace` (:272) and `utf8_in_chars` (:307) are predicates over a
//!   decoded `CodePoint`; `char_escape_helper` (:647),
//!   `make_utf8_escape_function` (:660) and `decode_latin_1` (:867) belong to
//!   `repr` and `_codecs`.
//!
//! * **This module owns random access**, which the crate deliberately has no
//!   counterpart for — its iterators are sequential, so resolving the n-th code
//!   point through them costs O(n).  `UTF8_INDEX_STORAGE` is PyPy's index over
//!   an *already valid* buffer, and everything written against it lives here:
//!   the table and its build, the three lookups that read it, and the byte
//!   offset stepping (`next_codepoint_pos`, `prev_codepoint_pos`) they are
//!   defined in terms of, which is pure arithmetic the crate exposes no
//!   equivalent for.
//!
//! That is why the `utf8` parameter is `&Wtf8` rather than `&[u8]`: `&Wtf8` is
//! what RPython's `str` payload becomes once the invariant lives in the type,
//! and holding it lets `codepoint_at_pos` decode through the crate instead of
//! carrying a second copy of its decoder.
//!
//! Two members of the family are deliberately absent.  `null_storage` (:513)
//! has no counterpart: an absent table is a null pointer in the
//! `W_UnicodeObject` slot.  `_pos_at_index` (:568), the "Slow!" linear
//! fallback, has no pyre caller — upstream reaches it from `unicodehelper` and
//! `formatting`, neither of which pyre routes this way.
//!
//! Names, entry layout, group sizes and the build loop follow
//! `rpython/rlib/rutf8.py`.  The `_is_64bit` branch of
//! `next_codepoint_pos` (a hand-tuned x86-64 sequence guarded by
//! `not jit.we_are_jitted()`) is left out: it computes the same value as the
//! comparison ladder it sits in front of.

use rustpython_wtf8::{CodePoint, Wtf8};

/// `UTF8_INDEX_STORAGE`'s element (`rutf8.py`) — `baseindex` is the
/// byte offset the 64-code-point group starts at, and `ofs[i]` the byte delta
/// from it to the code point *after* the group's `4 * i`-th.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Utf8LocElem {
    pub baseindex: isize,
    pub ofs: [u8; 16],
}

/// `UTF8_INDEX_STORAGE` (`rutf8.py`).  A leaf (no inner `PyObjectRef`), so
/// a mortal string's table lives in a storage box whose drop glue reclaims it,
/// as its WTF-8 buffer does.
pub type Utf8IndexStorage = Vec<Utf8LocElem>;

/// `next_codepoint_pos` (`rutf8.py`) — the position of the code point after
/// `pos`.  Assumes valid WTF-8 and `pos` before the end.
#[inline]
pub fn next_codepoint_pos(code: &Wtf8, pos: usize) -> usize {
    let chr1 = code.as_bytes()[pos];
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

/// `prev_codepoint_pos` (`rutf8.py`) — the position of the code point
/// before `pos`, which must not be zero.  A `pos` one past the end reads as the
/// extra `'\x00'` the build loop pretends is there.
#[inline]
pub fn prev_codepoint_pos(code: &Wtf8, pos: usize) -> usize {
    let code = code.as_bytes();
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

/// `codepoint_at_pos` (`rutf8.py`) — the code point starting at byte `pos`.
///
/// Assumes valid WTF-8 with `pos` on a boundary before the end, as upstream
/// does ("no checking!").  The decode itself goes through the crate rather than
/// a second copy of its bit arithmetic.
#[inline]
pub fn codepoint_at_pos(code: &Wtf8, pos: usize) -> CodePoint {
    let end = next_codepoint_pos(code, pos);
    code.get(pos..end)
        .and_then(|one| one.code_points().next())
        .expect("codepoint_at_pos: pos is not a code point boundary")
}

/// `codepoint_before_pos` (`rutf8.py`) — the code point immediately before
/// `pos`, which must not be zero.
///
/// Upstream walks the continuation bytes backwards inline; composing the
/// already-ported `prev_codepoint_pos` with `codepoint_at_pos` yields the same
/// code point by construction, and inherits `prev_codepoint_pos`'s handling of
/// a `pos` one past the end.
#[inline]
pub fn codepoint_before_pos(code: &Wtf8, pos: usize) -> CodePoint {
    codepoint_at_pos(code, prev_codepoint_pos(code, pos))
}

/// `codepoints_in_utf8` (`rutf8.py`) — the number of code points in
/// `value[start..end]`.
///
/// Counts the bytes that are *not* continuation bytes, which upstream spells as
/// a signed-char comparison against `-0x40`.
pub fn codepoints_in_utf8(value: &Wtf8, start: usize, end: usize) -> usize {
    let value = value.as_bytes();
    let end = end.min(value.len());
    debug_assert!(start <= end);
    value[start..end]
        .iter()
        .filter(|&&ch| (ch as i8) >= -0x40)
        .count()
}

/// `create_utf8_index_storage` (`rutf8.py`) — the table for `utf8`, whose
/// code point count is `utf8len`.
pub fn create_utf8_index_storage(utf8: &Wtf8, utf8len: usize) -> Utf8IndexStorage {
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

/// `codepoint_position_at_index` (`rutf8.py`) — the byte offset of code
/// point `index`, which must not exceed the string's code point count.
#[inline]
pub fn codepoint_position_at_index(utf8: &Wtf8, storage: &[Utf8LocElem], index: usize) -> usize {
    let elem = &storage[index >> 6];
    let bytepos = elem.baseindex as usize + elem.ofs[(index >> 2) & 0x0F] as usize;
    match index & 0x3 {
        0 => prev_codepoint_pos(utf8, bytepos),
        1 => bytepos,
        2 => next_codepoint_pos(utf8, bytepos),
        _ => next_codepoint_pos(utf8, next_codepoint_pos(utf8, bytepos)),
    }
}

/// `codepoint_at_index` (`rutf8.py`) — the code point at code point
/// `index`, which must be below the string's code point count.
///
/// Upstream fuses the position lookup with the decode so the JIT sees one
/// elidable call; both spellings walk to the same byte offset, so this is the
/// composition of the two.
#[inline]
pub fn codepoint_at_index(utf8: &Wtf8, storage: &[Utf8LocElem], index: usize) -> CodePoint {
    codepoint_at_pos(utf8, codepoint_position_at_index(utf8, storage, index))
}

/// `codepoint_index_at_byte_position` (`rutf8.py`) — the code point index
/// for which `codepoint_position_at_index` is `bytepos`.
///
/// Logarithmic in the string length, plus a constant that is not tiny either.
/// Upstream's leading `if bytepos < 0: return bytepos` guard carries `str.find`
/// misses through; pyre's search returns `Option`, so the caller handles a miss
/// and this takes an offset that is always real.
pub fn codepoint_index_at_byte_position(
    utf8: &Wtf8,
    storage: &[Utf8LocElem],
    bytepos: usize,
    num_codepoints: usize,
) -> usize {
    let bytes = utf8.as_bytes();
    // binary search on elements of storage
    let bytes_remaining = bytes.len() - bytepos;
    // the fact that one codepoint is encoded in 1-4 bytes constrains the
    // result. pick good min and max indexes based on this observation. saves a
    // few bisection steps.  Both `max` candidates are clamped at zero, which
    // the `max`/`min` against a non-negative bound would have discarded anyway.
    let mut index_min = std::cmp::max(
        bytepos / 4,
        num_codepoints.saturating_sub(bytes_remaining + 1),
    ) >> 6;
    let mut index_max =
        std::cmp::min(bytepos, num_codepoints.saturating_sub(bytes_remaining / 4)) >> 6;
    while index_min < index_max {
        // this addition can't overflow because storage has a length that is
        // 1/64 of the length of a string
        let index_middle = (index_min + index_max).div_ceil(2);
        let base_bytepos = storage[index_middle].baseindex as usize;
        if bytepos < base_bytepos {
            index_max = index_middle - 1;
        } else {
            index_min = index_middle;
        }
    }

    let baseindex = storage[index_min].baseindex as usize;
    if baseindex == bytepos {
        return index_min << 6;
    }

    // use ofs to get closer to the correct character index.  Reaching here
    // means `bytepos` is past a group start, so the string is not empty and
    // `num_codepoints - 1` cannot underflow.
    let mut result = index_min << 6;
    let mut bytepos1 = baseindex;
    let maxindex = if index_min == storage.len() - 1 {
        ((num_codepoints - 1) >> 2) & 0x0F
    } else {
        16
    };
    for i in 0..maxindex {
        let x = baseindex + storage[index_min].ofs[i] as usize;
        if x >= bytepos {
            break;
        }
        bytepos1 = x;
        result = (index_min << 6) + (i << 2) + 1;
    }

    // this loop should run at most four times
    while bytepos1 < bytepos {
        bytepos1 = next_codepoint_pos(utf8, bytepos1);
        result += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustpython_wtf8::Wtf8Buf;

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

    const KINDS: [&str; 6] = ["ascii", "latin1", "wide", "astral", "mixed", "surrogate"];
    // Lengths straddling the 4-code-point delta stride and the 64-code point
    // group, in both directions.
    const REPEATS: [usize; 11] = [0, 1, 2, 15, 16, 17, 32, 63, 64, 65, 100];

    /// The table must answer exactly what a walk from the start would.
    fn assert_matches_walk(buf: &Wtf8Buf) {
        let positions: Vec<usize> = buf.code_point_indices().map(|(pos, _)| pos).collect();
        let storage = create_utf8_index_storage(buf, positions.len());
        for (index, &expected) in positions.iter().enumerate() {
            assert_eq!(
                codepoint_position_at_index(buf, &storage, index),
                expected,
                "index {index} of a {} byte / {} code point payload",
                buf.len(),
                positions.len(),
            );
        }
        // "smaller than or equal to the utf8 length" — the build loop's
        // padding slot exists so the count itself resolves to the end, which
        // is what a `start`/`end` bound equal to the length asks for.
        assert_eq!(
            codepoint_position_at_index(buf, &storage, positions.len()),
            buf.len(),
            "the code point count of a {} byte payload",
            buf.len(),
        );
    }

    #[test]
    fn test_index_storage_matches_a_walk() {
        for repeat in REPEATS {
            for kind in KINDS {
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
        assert_eq!(create_utf8_index_storage(Wtf8::new(""), 0).len(), 1);
        let ascii = "a".repeat(128);
        assert_eq!(
            create_utf8_index_storage(Wtf8::new(&ascii[..64]), 64).len(),
            2
        );
        assert_eq!(
            create_utf8_index_storage(Wtf8::new(&ascii[..127]), 127).len(),
            2
        );
        assert_eq!(
            create_utf8_index_storage(Wtf8::new(&ascii[..]), 128).len(),
            3
        );
    }

    /// `codepoint_at_index` must yield what iteration yields, at every index.
    #[test]
    fn test_codepoint_at_index_matches_iteration() {
        for repeat in REPEATS {
            for kind in KINDS {
                let buf = sample(kind, repeat);
                let expected: Vec<CodePoint> = buf.code_points().collect();
                let storage = create_utf8_index_storage(&buf, expected.len());
                for (index, &cp) in expected.iter().enumerate() {
                    assert_eq!(
                        codepoint_at_index(&buf, &storage, index).to_u32(),
                        cp.to_u32(),
                        "{kind} * {repeat}, index {index}",
                    );
                }
            }
        }
    }

    /// `codepoint_index_at_byte_position` must invert
    /// `codepoint_position_at_index` at every code point boundary.
    #[test]
    fn test_byte_position_inverts_index_lookup() {
        for repeat in REPEATS {
            for kind in KINDS {
                let buf = sample(kind, repeat);
                let positions: Vec<usize> = buf.code_point_indices().map(|(pos, _)| pos).collect();
                let n = positions.len();
                let storage = create_utf8_index_storage(&buf, n);
                for (index, &bytepos) in positions.iter().enumerate() {
                    assert_eq!(
                        codepoint_index_at_byte_position(&buf, &storage, bytepos, n),
                        index,
                        "{kind} * {repeat}, byte {bytepos}",
                    );
                }
                // The end of the string maps to the code point count.
                assert_eq!(
                    codepoint_index_at_byte_position(&buf, &storage, buf.len(), n),
                    n,
                    "{kind} * {repeat}, end",
                );
            }
        }
    }

    #[test]
    fn test_codepoints_in_utf8_counts_a_byte_window() {
        for repeat in REPEATS {
            for kind in KINDS {
                let buf = sample(kind, repeat);
                let positions: Vec<usize> = buf.code_point_indices().map(|(pos, _)| pos).collect();
                let n = positions.len();
                assert_eq!(codepoints_in_utf8(&buf, 0, buf.len()), n);
                // Every code point-aligned window counts its own span, and an
                // `end` past the buffer clamps.
                for (index, &start) in positions.iter().enumerate() {
                    assert_eq!(codepoints_in_utf8(&buf, start, buf.len()), n - index);
                    assert_eq!(codepoints_in_utf8(&buf, 0, start), index);
                }
                assert_eq!(codepoints_in_utf8(&buf, 0, buf.len() + 10), n);
            }
        }
    }
}
