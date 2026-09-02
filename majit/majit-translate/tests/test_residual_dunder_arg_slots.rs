//! A residual call's argument must fit one machine word: production-LLBC
//! regression for the `descroperation` override gates.
//!
//! `jit_fnaddr.rs` states the ABI — one word per argument slot, classified
//! `'i'`/`'r'`/`'f'` — so a `&str` parameter is undescribable: it is two
//! words, and the callee would read the second out of whatever the caller
//! happened to leave there.  Each gate below is `dont_look_inside`, so every
//! one of them IS a residual call, and each used to take the forward and
//! reflected dunder NAMES.  They take a fieldless-enum discriminant instead
//! and recover the names inside the residual.
//!
//! The publication side is already checked by the compiler: `jit_fnaddr`'s
//! helpers take the function itself, so `ResidualSlot` rejects a fat pointer
//! before an address can be taken.  What nothing checks is the signature the
//! front actually lowers, which is what this file covers.  `ValueType::Str`
//! is the exact shape that cannot be published, so it — not the argument
//! count — is what these assert on: swapping two `&str`s for one enum also
//! changes the count, but restoring a single `&str` would not.

use majit_charon_reader::Llbc;
use majit_translate::front::mir::lower_function_with_static_addrs;
use majit_translate::model::{OpKind, ValueType};
use majit_translate::{ErrorCarrierSpec, HostStaticAddrs};
use std::sync::OnceLock;

const INTERP: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-interpreter.ullbc",
);

/// Shared parse — the corpus is several GB resident, so one parse behind a
/// `OnceLock` is what keeps concurrent tests off the runner's swap.
fn interp() -> &'static Llbc {
    static L: OnceLock<Llbc> = OnceLock::new();
    L.get_or_init(|| Llbc::load(INTERP).expect("load pyre-interpreter.ullbc"))
}

/// Every parameter of `name`, as the front banks it.
fn parameter_banks(name: &str) -> Vec<(String, ValueType)> {
    let graph = lower_function_with_static_addrs(
        interp(),
        name,
        HostStaticAddrs {
            error_carrier: ErrorCarrierSpec {
                carrier_path: "pyre_interpreter::error::PyError",
                carrier_wrappers: &[],
                to_exc_object: Some(&["pyre_interpreter", "error", "pyerror_to_exc_object"]),
                from_exc_object: Some(("PyError", "from_exc_object")),
            },
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("lower {name}: {e:?}"));
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Input { name, ty, .. } => Some((name.clone(), ty.clone())),
            _ => None,
        })
        .collect()
}

#[test]
fn no_override_gate_takes_a_fat_pointer() {
    for gate in [
        "needs_numeric_binop_dispatch",
        "needs_seq_binop_dispatch",
        "needs_bytes_binop_dispatch",
        "needs_numeric_unaryop_dispatch",
        "needs_set_binop_dispatch",
    ] {
        let banks = parameter_banks(&format!(
            "pyre_interpreter::objspace::descroperation::{gate}"
        ));
        assert!(
            !banks.is_empty(),
            "{gate} lowered with no parameters at all"
        );
        let fat: Vec<_> = banks
            .iter()
            .filter(|(_, ty)| *ty == ValueType::Str)
            .collect();
        assert!(
            fat.is_empty(),
            "{gate} is `dont_look_inside`, so it is a residual call, and a \
             residual argument gets ONE machine word.  A `&str` is two, so \
             this parameter cannot be published through `jit_fnaddr` and the \
             traced caller is left holding an unbound symbolic residual: \
             {fat:?}"
        );
    }
}

#[test]
fn the_dunder_discriminant_is_banked_as_an_integer() {
    // The names live behind `BinopDunder` / `UnaryDunder`, which the front
    // models as the discriminant integer (`tyref_is_fieldless_enum_free`).
    // `SeqBase`'s `base` is the same shape and predates them.
    for (gate, param) in [
        ("needs_numeric_binop_dispatch", "op"),
        ("needs_seq_binop_dispatch", "op"),
        ("needs_seq_binop_dispatch", "base"),
        ("needs_bytes_binop_dispatch", "op"),
        ("needs_numeric_unaryop_dispatch", "op"),
    ] {
        let banks = parameter_banks(&format!(
            "pyre_interpreter::objspace::descroperation::{gate}"
        ));
        let found = banks.iter().find(|(name, _)| name == param);
        assert_eq!(
            found.map(|(_, ty)| ty),
            Some(&ValueType::Int),
            "{gate}'s `{param}` is a fieldless enum, which banks as its \
             discriminant integer.  Parameters: {banks:?}"
        );
    }
}
