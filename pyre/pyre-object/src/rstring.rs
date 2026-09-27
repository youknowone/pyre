//! `rpython/rlib/rstring.py`: the forward/reverse byte search
//! (`_search_normal`), shared by `str` and `bytes`, and the `StringBuilder`
//! value.

use crate::rbuilder::rbuilder_runtime;
use crate::unicodeobject::UnicodeValueStorage;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    Count,
    Find,
    RFind,
}

const TWOWAY_MAX_SHIFT: usize = 255;
const TWOWAY_TABLE_SIZE: usize = 64;
const TWOWAY_TABLE_MASK: usize = TWOWAY_TABLE_SIZE - 1;

#[inline]
fn bloom_add(mask: u64, c: u8) -> u64 {
    // RPython `rstring.py:bloom_add`, with LONG_BIT = 64 on this target.
    mask | (1u64 << (c & 63))
}

#[inline]
fn bloom(mask: u64, c: u8) -> bool {
    // RPython `rstring.py:bloom`.
    (mask & (1u64 << (c & 63))) != 0
}

fn lex_search(needle: &[u8], len_needle: usize, invert_alphabet: bool) -> (usize, usize) {
    // RPython `rstring.py:_lex_search`.
    let mut max_suffix = 0usize;
    let mut candidate = 1usize;
    let mut k = 0usize;
    let mut period = 1usize;
    while candidate + k < len_needle {
        let a = needle[candidate + k];
        let b = needle[max_suffix + k];
        if if invert_alphabet { b < a } else { a < b } {
            candidate += k + 1;
            k = 0;
            period = candidate - max_suffix;
        } else if a == b {
            if k + 1 != period {
                k += 1;
            } else {
                candidate += period;
                k = 0;
            }
        } else {
            max_suffix = candidate;
            candidate += 1;
            k = 0;
            period = 1;
        }
    }
    (max_suffix, period)
}

fn factorize(needle: &[u8], len_needle: usize) -> (usize, usize) {
    // RPython `rstring.py:_factorize`.
    let (cut1, period1) = lex_search(needle, len_needle, false);
    let (cut2, period2) = lex_search(needle, len_needle, true);
    if cut1 > cut2 {
        (cut1, period1)
    } else {
        (cut2, period2)
    }
}

fn twoway_preprocess(
    needle: &[u8],
    len_needle: usize,
) -> (usize, usize, usize, bool, [u8; TWOWAY_TABLE_SIZE]) {
    // RPython `rstring.py:_twoway_preprocess`.
    let (cut, mut period) = factorize(needle, len_needle);
    let mut is_periodic = true;
    let mut i = 0usize;
    while i < cut {
        if needle[i] != needle[period + i] {
            is_periodic = false;
            break;
        }
        i += 1;
    }
    let gap = if is_periodic {
        0
    } else {
        period = cut.max(len_needle - cut) + 1;
        let mut gap = len_needle;
        let last = (needle[len_needle - 1] as usize) & TWOWAY_TABLE_MASK;
        let mut i = len_needle - 1;
        while i > 0 {
            i -= 1;
            if ((needle[i] as usize) & TWOWAY_TABLE_MASK) == last {
                gap = len_needle - 1 - i;
                break;
            }
        }
        gap
    };
    let not_found_shift = len_needle.min(TWOWAY_MAX_SHIFT) as u8;
    let mut table = [not_found_shift; TWOWAY_TABLE_SIZE];
    let mut i = len_needle - not_found_shift as usize;
    while i < len_needle {
        table[(needle[i] as usize) & TWOWAY_TABLE_MASK] = (len_needle - 1 - i) as u8;
        i += 1;
    }
    (cut, period, gap, is_periodic, table)
}

fn two_way(
    value: &[u8],
    base: usize,
    n: usize,
    needle: &[u8],
    m: usize,
    cut: usize,
    mut period: usize,
    gap: usize,
    is_periodic: bool,
    table: &[u8; TWOWAY_TABLE_SIZE],
) -> isize {
    // RPython `rstring.py:_two_way`.
    let haystack_end = base + n;
    let mut window_last = base + m - 1;
    if is_periodic {
        let mut memory = 0usize;
        let mut skip_horspool = false;
        while window_last < haystack_end {
            if !skip_horspool {
                loop {
                    let shift = table[(value[window_last] as usize) & TWOWAY_TABLE_MASK] as usize;
                    window_last += shift;
                    if shift == 0 {
                        break;
                    }
                    if window_last >= haystack_end {
                        return -1;
                    }
                }
            }
            skip_horspool = false;
            let window = window_last + 1 - m;
            let mut i = cut.max(memory);
            let mut mismatch = false;
            while i < m {
                if needle[i] != value[window + i] {
                    window_last += i - cut + 1;
                    memory = 0;
                    mismatch = true;
                    break;
                }
                i += 1;
            }
            if mismatch {
                continue;
            }
            i = memory;
            while i < cut {
                if needle[i] != value[window + i] {
                    window_last += period;
                    memory = m - period;
                    if window_last >= haystack_end {
                        return -1;
                    }
                    let shift = table[(value[window_last] as usize) & TWOWAY_TABLE_MASK] as usize;
                    if shift != 0 {
                        let mem_jump = cut.max(memory) - cut + 1;
                        memory = 0;
                        window_last += shift.max(mem_jump);
                    } else {
                        skip_horspool = true;
                    }
                    mismatch = true;
                    break;
                }
                i += 1;
            }
            if mismatch {
                continue;
            }
            return (window - base) as isize;
        }
        -1
    } else {
        if period < gap {
            period = gap;
        }
        let gap_jump_end = (cut + gap).min(m);
        while window_last < haystack_end {
            loop {
                let shift = table[(value[window_last] as usize) & TWOWAY_TABLE_MASK] as usize;
                window_last += shift;
                if shift == 0 {
                    break;
                }
                if window_last >= haystack_end {
                    return -1;
                }
            }
            let window = window_last + 1 - m;
            let mut mismatch = false;
            let mut i = cut;
            while i < gap_jump_end {
                if needle[i] != value[window + i] {
                    window_last += gap;
                    mismatch = true;
                    break;
                }
                i += 1;
            }
            if mismatch {
                continue;
            }
            i = gap_jump_end;
            while i < m {
                if needle[i] != value[window + i] {
                    window_last += i - cut + 1;
                    mismatch = true;
                    break;
                }
                i += 1;
            }
            if mismatch {
                continue;
            }
            i = 0;
            while i < cut {
                if needle[i] != value[window + i] {
                    window_last += period;
                    mismatch = true;
                    break;
                }
                i += 1;
            }
            if mismatch {
                continue;
            }
            return (window - base) as isize;
        }
        -1
    }
}

fn two_way_count(
    value: &[u8],
    base: usize,
    n: usize,
    needle: &[u8],
    m: usize,
    cut: usize,
    period: usize,
    gap: usize,
    is_periodic: bool,
    table: &[u8; TWOWAY_TABLE_SIZE],
) -> usize {
    // RPython `rstring.py:_two_way_count`.
    let mut index = 0usize;
    let mut count = 0usize;
    loop {
        let result = two_way(
            value,
            base + index,
            n - index,
            needle,
            m,
            cut,
            period,
            gap,
            is_periodic,
            table,
        );
        if result == -1 {
            return count;
        }
        count += 1;
        index += result as usize + m;
    }
}

fn default_find(
    value: &[u8],
    base: usize,
    n: usize,
    needle: &[u8],
    m: usize,
    mode: SearchMode,
) -> isize {
    // RPython `rstring.py:_default_find`.
    let w = n - m;
    let mlast = m - 1;
    let mut count = 0usize;
    let mut gap = mlast;
    let last = needle[mlast];
    let mut mask = 0u64;
    let mut j = 0usize;
    while j < mlast {
        mask = bloom_add(mask, needle[j]);
        if needle[j] == last {
            gap = mlast - j - 1;
        }
        j += 1;
    }
    mask = bloom_add(mask, last);
    let mut i = 0usize;
    while i <= w {
        if value[base + mlast + i] == last {
            j = 0;
            while j < mlast {
                if value[base + i + j] != needle[j] {
                    break;
                }
                j += 1;
            }
            if j == mlast {
                if mode != SearchMode::Count {
                    return i as isize;
                }
                count += 1;
                i += mlast;
            } else {
                let la = base + mlast + i + 1;
                let c = if la < value.len() { value[la] } else { 0 };
                if !bloom(mask, c) {
                    i += m;
                } else {
                    i += gap;
                }
            }
        } else {
            let la = base + mlast + i + 1;
            let c = if la < value.len() { value[la] } else { 0 };
            if !bloom(mask, c) {
                i += m;
            }
        }
        i += 1;
    }
    if mode != SearchMode::Count {
        -1
    } else {
        count as isize
    }
}

fn adaptive_find(
    value: &[u8],
    base: usize,
    n: usize,
    needle: &[u8],
    m: usize,
    mode: SearchMode,
) -> isize {
    // RPython `rstring.py:_adaptive_find`.
    let w = n - m;
    let mlast = m - 1;
    let mut count = 0usize;
    let mut gap = mlast;
    let mut hits = 0usize;
    let last = needle[mlast];
    let mut mask = 0u64;
    let mut j = 0usize;
    while j < mlast {
        mask = bloom_add(mask, needle[j]);
        if needle[j] == last {
            gap = mlast - j - 1;
        }
        j += 1;
    }
    mask = bloom_add(mask, last);
    let mut i = 0usize;
    while i <= w {
        if value[base + mlast + i] == last {
            j = 0;
            while j < mlast {
                if value[base + i + j] != needle[j] {
                    break;
                }
                j += 1;
            }
            if j == mlast {
                if mode != SearchMode::Count {
                    return i as isize;
                }
                count += 1;
                i += mlast;
            } else {
                hits += j + 1;
                if hits > m / 4 && w - i > 2000 {
                    let (cut, period, gap, is_periodic, table) = twoway_preprocess(needle, m);
                    if mode != SearchMode::Count {
                        let res = two_way(
                            value,
                            base + i,
                            n - i,
                            needle,
                            m,
                            cut,
                            period,
                            gap,
                            is_periodic,
                            &table,
                        );
                        return if res == -1 { -1 } else { res + i as isize };
                    }
                    let res = two_way_count(
                        value,
                        base + i,
                        n - i,
                        needle,
                        m,
                        cut,
                        period,
                        gap,
                        is_periodic,
                        &table,
                    );
                    return (res + count) as isize;
                }
                let la = base + mlast + i + 1;
                let c = if la < value.len() { value[la] } else { 0 };
                if !bloom(mask, c) {
                    i += m;
                } else {
                    i += gap;
                }
            }
        } else {
            let la = base + mlast + i + 1;
            let c = if la < value.len() { value[la] } else { 0 };
            if !bloom(mask, c) {
                i += m;
            }
        }
        i += 1;
    }
    if mode != SearchMode::Count {
        -1
    } else {
        count as isize
    }
}

pub fn search_normal(
    value: &[u8],
    other: &[u8],
    mut start: usize,
    mut end: usize,
    mode: SearchMode,
) -> isize {
    // RPython `rstring.py:_search_normal`, specialized to byte-backed
    // PyPy unicode `_utf8` / pyre WTF-8 storage.
    end = end.min(value.len());
    start = start.min(end);
    let n = end - start;
    let m = other.len();
    if m == 0 {
        return match mode {
            SearchMode::Count => (end - start + 1) as isize,
            SearchMode::RFind => end as isize,
            SearchMode::Find => start as isize,
        };
    }
    let Some(w) = n.checked_sub(m) else {
        return if mode == SearchMode::Count { 0 } else { -1 };
    };
    if mode != SearchMode::RFind {
        let res = if n < 2500 || (m < 100 && n < 30000) || m < 6 {
            default_find(value, start, n, other, m, mode)
        } else if (m >> 2) * 3 < (n >> 2) {
            let (cut, period, gap, is_periodic, table) = twoway_preprocess(other, m);
            if mode == SearchMode::Count {
                return two_way_count(
                    value,
                    start,
                    n,
                    other,
                    m,
                    cut,
                    period,
                    gap,
                    is_periodic,
                    &table,
                ) as isize;
            }
            two_way(
                value,
                start,
                n,
                other,
                m,
                cut,
                period,
                gap,
                is_periodic,
                &table,
            )
        } else {
            adaptive_find(value, start, n, other, m, mode)
        };
        if mode == SearchMode::Count {
            res
        } else if res == -1 {
            -1
        } else {
            start as isize + res
        }
    } else {
        // RPython `rstring.py:_search_normal` reverse-find branch.
        let mlast = m - 1;
        let mut skip = mlast;
        let mut mask = bloom_add(0, other[0]);
        let mut i = mlast;
        while i > 0 {
            mask = bloom_add(mask, other[i]);
            if other[i] == other[0] {
                skip = i - 1;
            }
            i -= 1;
        }
        let mut i = start + w + 1;
        while i > start {
            i -= 1;
            if value[i] == other[0] {
                let mut matched = true;
                let mut j = mlast;
                while j > 0 {
                    if value[i + j] != other[j] {
                        matched = false;
                        break;
                    }
                    j -= 1;
                }
                if matched {
                    return i as isize;
                }
                if i > 0 && !bloom(mask, value[i - 1]) {
                    i = i.saturating_sub(m);
                } else {
                    i = i.saturating_sub(skip);
                }
            } else if i > 0 && !bloom(mask, value[i - 1]) {
                i = i.saturating_sub(m);
            }
        }
        -1
    }
}

/// `rstring.py` `StringBuilder`.
///
/// Holds the GC reference of the rtyped `STRINGBUILDER` GcStruct
/// (`rbuilder.py`). The translator annotates a value of this type as
/// `SomeStringBuilder` and lowers `new` / `append` / `build` to
/// `StringBuilderRepr`'s `ll_new` / `ll_append` / `ll_build`; the method
/// bodies are the untranslated implementation, so they are `not_rpython`.
pub struct StringBuilder(i64);

impl StringBuilder {
    /// `StringBuilder(init_size)`.
    #[inline]
    #[majit_macros::not_rpython]
    pub fn new(init_size: i64) -> Self {
        Self(rbuilder_runtime::ll_new(
            init_size,
            rbuilder_runtime::STR_ITEM_SIZE,
        ))
    }

    /// `StringBuilder.append(s)`.
    #[inline]
    #[majit_macros::not_rpython]
    pub fn append(&mut self, s: *mut UnicodeValueStorage) {
        rbuilder_runtime::ll_append(self.0, s as i64);
    }

    /// `StringBuilder.build()` — the rstr `STR` payload.
    #[inline]
    #[majit_macros::not_rpython]
    pub fn build(&mut self) -> *mut UnicodeValueStorage {
        rbuilder_runtime::ll_build(self.0, rbuilder_runtime::STR_ITEM_SIZE)
            as *mut UnicodeValueStorage
    }
}

#[cfg(test)]
mod tests {
    use super::{SearchMode, search_normal};

    fn naive(hay: &[u8], needle: &[u8], lo: usize, hi: usize, mode: SearchMode) -> isize {
        let hi = hi.min(hay.len());
        let lo = lo.min(hi);
        let n = hi - lo;
        let m = needle.len();
        if m == 0 {
            return match mode {
                SearchMode::Count => (hi - lo + 1) as isize,
                SearchMode::RFind => hi as isize,
                SearchMode::Find => lo as isize,
            };
        }
        if n < m {
            return if mode == SearchMode::Count { 0 } else { -1 };
        }
        let window = &hay[lo..hi];
        match mode {
            SearchMode::Find => window
                .windows(m)
                .position(|w| w == needle)
                .map(|p| (lo + p) as isize)
                .unwrap_or(-1),
            SearchMode::RFind => window
                .windows(m)
                .rposition(|w| w == needle)
                .map(|p| (lo + p) as isize)
                .unwrap_or(-1),
            SearchMode::Count => {
                let mut count = 0isize;
                let mut i = 0usize;
                while i + m <= window.len() {
                    if &window[i..i + m] == needle {
                        count += 1;
                        i += m;
                    } else {
                        i += 1;
                    }
                }
                count
            }
        }
    }

    fn assert_agree(hay: &[u8], needle: &[u8], lo: usize, hi: usize) {
        for mode in [SearchMode::Find, SearchMode::RFind, SearchMode::Count] {
            let got = search_normal(hay, needle, lo, hi, mode);
            let expect = naive(hay, needle, lo, hi, mode);
            assert_eq!(got, expect, "lo={lo} hi={hi}");
        }
    }

    #[test]
    fn find_rfind_count_match_naive() {
        let hay = b"abracadabra";
        let cases: &[(&[u8], usize, usize)] = &[
            (b"a", 0, hay.len()),
            (b"bra", 0, hay.len()),
            (b"", 0, hay.len()),
            (b"abracadabraX", 0, hay.len()),
            (b"ra", 2, 8),
            (b"", 3, 7),
            (b"cad", 1, 6),
            (b"zzz", 0, 4),
            (b"ab", 0, 1),
        ];
        for &(needle, lo, hi) in cases {
            assert_agree(hay, needle, lo, hi);
        }
        let long = b"xxabracadabrayy";
        assert_agree(long, b"bra", 2, 13);
        assert_agree(long, b"", 4, 9);
        assert_agree(long, b"yyyy", 2, 13);
    }

    #[test]
    fn adaptive_find_is_linear() {
        let n = 200_000usize;
        let a = "a".repeat(n);
        let b = "b".repeat(n);
        let haystack = format!("{a}{a}{b}{a}{a}");
        let needle = format!("{a}{b}{b}{a}");
        let hay_len = haystack.len();
        let ned = needle.as_bytes();
        assert_eq!(
            search_normal(haystack.as_bytes(), ned, 0, hay_len, SearchMode::Find),
            -1
        );
        assert_eq!(
            search_normal(haystack.as_bytes(), ned, 0, hay_len, SearchMode::Count),
            0
        );
        let both = haystack + &needle;
        let both_b = both.as_bytes();
        assert_eq!(
            search_normal(both_b, ned, 0, both_b.len(), SearchMode::Find),
            hay_len as isize
        );
        assert_eq!(
            search_normal(both_b, ned, 0, both_b.len(), SearchMode::Count),
            1
        );
    }

    /// `search_normal` calls `two_way` / `two_way_count` when `mode != RFind`
    /// and the `default_find` guard is false:
    /// `n < 2500 || (m < 100 && n < 30000) || m < 6`.
    /// Haystack windows here have `n >= 30000` and needles have `6 <= m <= 99`,
    /// so that guard is false, and `(m >> 2) * 3 < (n >> 2)` selects two-way
    /// (for `m == 99`, `72 < 7500`). `RFind` stays on the reverse scan and is
    /// still checked against `naive`.
    #[test]
    fn two_way_find_rfind_count_match_naive() {
        const HAY_LEN: usize = 30_000;
        let periodic = b"abcabcabcabcabcX";
        let non_periodic = b"abXcdefX";
        let absent = b"zzzzzz";
        assert!((6..=99).contains(&periodic.len()));
        assert!((6..=99).contains(&non_periodic.len()));
        assert!((6..=99).contains(&absent.len()));

        let mut planted = lcg_hay(1, HAY_LEN);
        planted[1_000..1_000 + periodic.len()].copy_from_slice(periodic);
        planted[4_000..4_000 + periodic.len()].copy_from_slice(periodic);
        planted[8_000..8_000 + non_periodic.len()].copy_from_slice(non_periodic);
        assert_agree(&planted, periodic, 0, planted.len());
        assert_agree(&planted, non_periodic, 0, planted.len());
        assert_agree(&planted, absent, 0, planted.len());

        // `lo != 0` and `hi < len`, with the searched window still `n >= 30000`.
        let mut wide = lcg_hay(9, HAY_LEN + 200);
        let lo = 100usize;
        let hi = lo + HAY_LEN;
        assert!(hi < wide.len());
        wide[lo + 500..lo + 500 + periodic.len()].copy_from_slice(periodic);
        wide[lo - 16..lo - 16 + periodic.len()].copy_from_slice(periodic);
        assert_agree(&wide, periodic, lo, hi);
        assert_agree(&wide, absent, lo, hi);
        assert_agree(&wide, non_periodic, lo, hi);

        for seed in [2u64, 3, 4] {
            let hay = lcg_hay(seed, HAY_LEN);
            assert_agree(&hay, periodic, 0, hay.len());
            assert_agree(&hay, non_periodic, 0, hay.len());
            assert_agree(&hay, absent, 0, hay.len());
        }
    }

    fn lcg_hay(seed: u64, len: usize) -> Vec<u8> {
        let alphabet = b"abcd";
        let mut state = seed;
        let mut hay = Vec::with_capacity(len);
        for _ in 0..len {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            hay.push(alphabet[((state >> 16) as usize) & 3]);
        }
        hay
    }
}
