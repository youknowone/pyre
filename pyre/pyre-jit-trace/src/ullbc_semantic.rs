//! Semantic identity of a Charon ULLBC artefact for cache keys.
//!
//! The raw JSON embeds `files[].contents` (the whole source, comments
//! included), span `beg`/`end` line/col, and `dest_file` (the absolute
//! output path). None of those is an input to generated JIT tables.
//! `source_text` is kept: the translator uses it to tell `static mut`
//! from `static`.

/// Emit `data` with source dumps, dest_file, and span locs removed.
pub fn feed_semantic_ullbc(data: &[u8], emit: &mut dyn FnMut(&[u8])) {
    const CONTENTS: &[u8] = br#""contents":"#;
    const DEST_FILE: &[u8] = br#""dest_file":"#;
    const BEG: &[u8] = br#""beg":"#;
    const END: &[u8] = br#""end":"#;
    let mut index = 0;
    while index < data.len() {
        let found_contents = find_from(data, CONTENTS, index);
        let found_dest = find_from(data, DEST_FILE, index);
        let found_beg = find_from(data, BEG, index);
        let found_end = find_from(data, END, index);
        let next = [found_contents, found_dest, found_beg, found_end]
            .into_iter()
            .flatten()
            .min();
        let Some(next_hit) = next else {
            emit(&data[index..]);
            return;
        };
        emit(&data[index..next_hit]);
        if Some(next_hit) == found_contents {
            emit(CONTENTS);
            emit(br#""""#);
            index = skip_json_value(data, next_hit + CONTENTS.len());
        } else if Some(next_hit) == found_dest {
            emit(DEST_FILE);
            emit(br#""""#);
            index = skip_json_value(data, next_hit + DEST_FILE.len());
        } else {
            let key = if Some(next_hit) == found_beg {
                BEG
            } else {
                END
            };
            let after = skip_ws(data, next_hit + key.len());
            if after < data.len() && data[after] == b'{' {
                let obj_end = skip_json_value(data, after);
                if is_span_loc(&data[after..obj_end]) {
                    emit(key);
                    emit(br#"{"line":0,"col":0}"#);
                    index = obj_end;
                    continue;
                }
            }
            emit(key);
            index = next_hit + key.len();
        }
    }
}

fn find_from(data: &[u8], needle: &[u8], start: usize) -> Option<usize> {
    data[start..]
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|rel| start + rel)
}

fn skip_ws(data: &[u8], mut index: usize) -> usize {
    while index < data.len() && matches!(data[index], b' ' | b'\t' | b'\n' | b'\r') {
        index += 1;
    }
    index
}

fn skip_json_string(data: &[u8], mut index: usize) -> usize {
    index += 1;
    while index < data.len() {
        match data[index] {
            b'\\' => index += 2,
            b'"' => return index + 1,
            _ => index += 1,
        }
    }
    data.len()
}

fn skip_json_container(data: &[u8], mut index: usize) -> usize {
    let mut depth = 0;
    while index < data.len() {
        match data[index] {
            b'"' => index = skip_json_string(data, index),
            b'{' | b'[' => {
                depth += 1;
                index += 1;
            }
            b'}' | b']' => {
                depth -= 1;
                index += 1;
                if depth == 0 {
                    return index;
                }
            }
            _ => index += 1,
        }
    }
    data.len()
}

fn skip_json_value(data: &[u8], mut index: usize) -> usize {
    index = skip_ws(data, index);
    if index >= data.len() {
        return index;
    }
    match data[index] {
        b'"' => skip_json_string(data, index),
        b'{' | b'[' => skip_json_container(data, index),
        b'n' if data[index..].starts_with(b"null") => index + 4,
        b't' if data[index..].starts_with(b"true") => index + 4,
        b'f' if data[index..].starts_with(b"false") => index + 5,
        b'-' | b'0'..=b'9' => {
            if data[index] == b'-' {
                index += 1;
            }
            while index < data.len() && data[index].is_ascii_digit() {
                index += 1;
            }
            if index < data.len() && data[index] == b'.' {
                index += 1;
                while index < data.len() && data[index].is_ascii_digit() {
                    index += 1;
                }
            }
            if index < data.len() && matches!(data[index], b'e' | b'E') {
                index += 1;
                if index < data.len() && matches!(data[index], b'+' | b'-') {
                    index += 1;
                }
                while index < data.len() && data[index].is_ascii_digit() {
                    index += 1;
                }
            }
            index
        }
        _ => index,
    }
}

fn is_span_loc(obj: &[u8]) -> bool {
    let mut index = 0;
    if index >= obj.len() || obj[index] != b'{' {
        return false;
    }
    index += 1;
    index = skip_ws(obj, index);
    if !obj[index..].starts_with(br#""line""#) {
        return false;
    }
    index += 6;
    index = skip_ws(obj, index);
    if index >= obj.len() || obj[index] != b':' {
        return false;
    }
    index = skip_ws(obj, index + 1);
    if index >= obj.len() || !obj[index].is_ascii_digit() {
        return false;
    }
    while index < obj.len() && obj[index].is_ascii_digit() {
        index += 1;
    }
    index = skip_ws(obj, index);
    if index >= obj.len() || obj[index] != b',' {
        return false;
    }
    index = skip_ws(obj, index + 1);
    if !obj[index..].starts_with(br#""col""#) {
        return false;
    }
    index += 5;
    index = skip_ws(obj, index);
    if index >= obj.len() || obj[index] != b':' {
        return false;
    }
    index = skip_ws(obj, index + 1);
    if index >= obj.len() || !obj[index].is_ascii_digit() {
        return false;
    }
    while index < obj.len() && obj[index].is_ascii_digit() {
        index += 1;
    }
    index = skip_ws(obj, index);
    index < obj.len() && obj[index] == b'}' && skip_ws(obj, index + 1) == obj.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    fn digest(data: &[u8]) -> u64 {
        let mut hasher = DefaultHasher::new();
        feed_semantic_ullbc(data, &mut |chunk| hasher.write(chunk));
        hasher.finish()
    }

    const BASE: &[u8] = br#"{"charon_version":"t","options":{"dest_file":"/abs/a.ullbc"},"translated":{"files":[{"id":0,"name":{"Local":"a.rs"},"contents":"fn a() {\n  // c1\n}"}],"fun_decls":[{"item_meta":{"source_text":"fn a() {}","span":{"data":{"file_id":0,"beg":{"line":10,"col":4},"end":{"line":12,"col":1}}}}}]}}"#;

    #[test]
    fn span_line_col_move_keeps_the_digest() {
        let replaced = std::str::from_utf8(BASE)
            .unwrap()
            .replace("\"line\":10", "\"line\":99")
            .replace("\"line\":12", "\"line\":101");
        assert_eq!(digest(BASE), digest(replaced.as_bytes()));
    }

    #[test]
    fn contents_and_dest_file_move_keep_the_digest() {
        let contents = std::str::from_utf8(BASE).unwrap().replace("// c1", "// c2");
        let dest = std::str::from_utf8(BASE)
            .unwrap()
            .replace("/abs/a.ullbc", "/other/b.ullbc");
        assert_eq!(digest(BASE), digest(contents.as_bytes()));
        assert_eq!(digest(BASE), digest(dest.as_bytes()));
    }

    #[test]
    fn source_text_or_fun_decls_move_changes_the_digest() {
        let source_text = std::str::from_utf8(BASE)
            .unwrap()
            .replace("fn a() {}", "static mut X: u8 = 0;");
        let body = std::str::from_utf8(BASE)
            .unwrap()
            .replace("\"fun_decls\":[", "\"fun_decls\":[{\"changed\":true},");
        assert_ne!(digest(BASE), digest(source_text.as_bytes()));
        assert_ne!(digest(BASE), digest(body.as_bytes()));
    }

    #[test]
    fn non_span_end_is_not_zeroed() {
        let a = br#"{"end":{"Unsigned":["U64","0"]},"beg":{"line":1,"col":2}}"#;
        let b = br#"{"end":{"Unsigned":["U64","1"]},"beg":{"line":1,"col":2}}"#;
        let c = br#"{"end":{"Unsigned":["U64","0"]},"beg":{"line":9,"col":8}}"#;
        assert_ne!(digest(a), digest(b));
        assert_eq!(digest(a), digest(c));
    }
}
