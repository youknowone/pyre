//! Runtime decompression for the embedded wasm standard-library image.
//!
//! Blob construction and decoding share this non-LLBC crate.  The interpreter
//! sees only the finished bytes and its Python-facing import adapter.

pub static STDLIB_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/stdlib_vfs.lz4"));

// The generated closure is currently well below one MiB.  Keep enough room
// for stdlib growth while refusing an untrusted size prefix before lz4_flex
// allocates the output buffer.
const MAX_UNCOMPRESSED_SIZE: usize = 16 * 1024 * 1024;

pub fn unpack(blob: &[u8]) -> Result<Vec<(String, String)>, String> {
    let size_prefix: [u8; 4] = blob
        .get(..4)
        .ok_or_else(|| "wasm_vfs: truncated compressed stdlib blob".to_owned())?
        .try_into()
        .expect("four-byte slice has u32 width");
    let uncompressed_size = u32::from_le_bytes(size_prefix) as usize;
    if uncompressed_size > MAX_UNCOMPRESSED_SIZE {
        return Err("wasm_vfs: embedded stdlib blob is too large".to_owned());
    }
    let raw = lz4_flex::block::decompress_size_prepended(blob).map_err(|err| err.to_string())?;
    let mut pos = 0usize;

    fn take<'a>(raw: &'a [u8], pos: &mut usize, len: usize) -> Result<&'a [u8], String> {
        let end = pos
            .checked_add(len)
            .filter(|&end| end <= raw.len())
            .ok_or_else(|| "wasm_vfs: truncated embedded stdlib blob".to_owned())?;
        let value = &raw[*pos..end];
        *pos = end;
        Ok(value)
    }

    fn read_u32(raw: &[u8], pos: &mut usize) -> Result<usize, String> {
        let bytes: [u8; 4] = take(raw, pos, 4)?
            .try_into()
            .expect("four-byte slice has u32 width");
        Ok(u32::from_le_bytes(bytes) as usize)
    }

    let count = read_u32(&raw, &mut pos)?;
    // Every record needs at least its name/source length words.  Reject an
    // impossible count before reserving storage; variable payload bytes only
    // make the bound tighter during the ordinary `take` walk below.
    if count > raw.len().saturating_sub(pos) / 8 {
        return Err("wasm_vfs: record count exceeds remaining header space".to_owned());
    }
    let mut files = Vec::new();
    files
        .try_reserve_exact(count)
        .map_err(|_| "wasm_vfs: cannot reserve embedded stdlib records".to_owned())?;
    for _ in 0..count {
        let name_len = read_u32(&raw, &mut pos)?;
        let name = std::str::from_utf8(take(&raw, &mut pos, name_len)?)
            .map_err(|_| "wasm_vfs: non-utf8 module name".to_owned())?
            .to_owned();
        let source_len = read_u32(&raw, &mut pos)?;
        let source = std::str::from_utf8(take(&raw, &mut pos, source_len)?)
            .map_err(|_| "wasm_vfs: non-utf8 module source".to_owned())?
            .to_owned();
        files.push((name, source));
    }
    if pos != raw.len() {
        return Err("wasm_vfs: trailing data in embedded stdlib blob".to_owned());
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_oversized_size_prefix_before_decompression() {
        let blob = ((MAX_UNCOMPRESSED_SIZE + 1) as u32).to_le_bytes();
        assert_eq!(
            unpack(&blob).unwrap_err(),
            "wasm_vfs: embedded stdlib blob is too large"
        );
    }

    #[test]
    fn rejects_impossible_record_count_before_reserving() {
        let raw = u32::MAX.to_le_bytes();
        let blob = lz4_flex::block::compress_prepend_size(&raw);
        assert_eq!(
            unpack(&blob).unwrap_err(),
            "wasm_vfs: record count exceeds remaining header space"
        );
    }

    #[test]
    fn generated_stdlib_blob_round_trips() {
        assert!(!unpack(STDLIB_BLOB).unwrap().is_empty());
    }
}
