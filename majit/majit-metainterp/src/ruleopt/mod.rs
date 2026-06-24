//! Integer optimization rule DSL support.
//!
//! RPython counterpart: `rpython/jit/metainterp/ruleopt/`.
//!
//! PyPy keeps the integer peephole rules in `ruleopt/real.rules` and
//! generates `optimizeopt/autogenintrules.py` with `ruleopt/generate.py`.
//! The Rust optimizer currently carries the generated counterpart in
//! `optimizeopt/autogenintrules.rs`; this module restores the package-level
//! `ruleopt` home for the rule source so the generator/parser/proof pieces can
//! be ported under the same path instead of being hidden behind
//! `optimizeopt`.

pub mod codegen;
pub mod generate;
pub mod parse;
pub mod proof;

/// Contents of PyPy `rpython/jit/metainterp/ruleopt/real.rules`.
///
/// Kept as an include of the vendored PyPy source, not a duplicated copy, so
/// rule edits remain single-sourced until the Rust `generate` pipeline is
/// ported.
pub const REAL_RULES: &str = include_str!("../../../../rpython/jit/metainterp/ruleopt/real.rules");

/// Return the integer peephole rewrite rule source consumed by
/// `ruleopt/generate.py` upstream.
pub fn real_rules() -> &'static str {
    REAL_RULES
}
