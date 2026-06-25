//! Resume bytecode encoding/decoding — re-exported from majit-ir.
//!
//! Keep the public re-export explicit so parity tooling can see the upstream
//! `resumecode.py` helper names.
// This module is an import-path parity surface; callers may use the upstream
// path even when this crate itself does not.
#[allow(unused_imports)]
pub use majit_ir::resumecode::{
    Reader, Writer, append_numbering, create_numbering, numb_next_item, numb_next_n_items,
    unpack_numbering,
};
