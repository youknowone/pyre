//! Runtime decompression for the embedded wasm standard-library image.
//!
//! Blob construction and decoding share this non-LLBC crate.  The interpreter
//! sees only the finished bytes and its Python-facing import adapter.

pub static STDLIB_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/stdlib_vfs.lz4"));

pub fn unpack(blob: &[u8]) -> Result<Vec<(String, String)>, String> {
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
    let mut files = Vec::with_capacity(count);
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
