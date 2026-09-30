//! Writes `$OUT_DIR/residual_sig_call.rs`: one `extern "C"` trampoline per
//! concrete wasm type the blackhole path `call_indirect`s.
//!
//! Integer mixes cover arity 0..=7 and every i32/i64 mask, with result tags
//! void / i32 / i64 / f64. Uniform f64 covers arity 1..=4 and result tags
//! void / i64 / f64. An i32 argument is truncated; an i32 result is
//! zero-extended; an f64 result is returned as bits.

use std::fmt::Write as _;

fn main() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");
    let path = std::path::Path::new(&out_dir).join("residual_sig_call.rs");
    std::fs::write(&path, generate()).unwrap_or_else(|err| {
        panic!("write {}: {err}", path.display());
    });
}

fn generate() -> String {
    let mut out = String::new();
    out.push_str(
        "use majit_backend_wasm::FuncSigVal;\n\
         \n\
         fn fn_from_slot<T>(index: usize) -> T {\n\
             unsafe { core::mem::transmute_copy(&index) }\n\
         }\n\
         \n",
    );

    for arity in 0..=7 {
        let masks = 1u16 << arity;
        for mask in 0..masks {
            for tag in 0..4 {
                emit_int(&mut out, arity, mask, tag);
            }
        }
    }
    for arity in 1..=4 {
        for tag in [0u8, 2, 3] {
            emit_f64(&mut out, arity, tag);
        }
    }
    emit_int_dispatch(&mut out);
    emit_f64_dispatch(&mut out);
    out
}

fn emit_int(out: &mut String, arity: usize, mask: u16, tag: u8) {
    let params = int_params(arity, mask);
    let args = int_args(arity, mask);
    let _ = writeln!(
        out,
        "unsafe fn c_{arity}_{mask}_{tag}(slot: usize, args: &[i64]) -> i64 {{\n    \
         let f: extern \"C\" fn({params}){ret} = fn_from_slot(slot);\n    \
         {body}\n}}\n",
        ret = ret_ty(tag),
        body = call_body(tag, &format!("f({args})")),
    );
}

fn emit_f64(out: &mut String, arity: usize, tag: u8) {
    let params = std::iter::repeat("f64")
        .take(arity)
        .collect::<Vec<_>>()
        .join(", ");
    let args = (0..arity)
        .map(|i| format!("f64::from_bits(args[{i}] as u64)"))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(
        out,
        "unsafe fn f_{arity}_{tag}(slot: usize, args: &[i64]) -> i64 {{\n    \
         let f: extern \"C\" fn({params}){ret} = fn_from_slot(slot);\n    \
         {body}\n}}\n",
        ret = ret_ty(tag),
        body = call_body(tag, &format!("f({args})")),
    );
}

fn int_params(arity: usize, mask: u16) -> String {
    (0..arity)
        .map(|i| if mask & (1 << i) != 0 { "i32" } else { "i64" })
        .collect::<Vec<_>>()
        .join(", ")
}

fn int_args(arity: usize, mask: u16) -> String {
    (0..arity)
        .map(|i| {
            if mask & (1 << i) != 0 {
                format!("args[{i}] as i32")
            } else {
                format!("args[{i}]")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn ret_ty(tag: u8) -> &'static str {
    match tag {
        0 => "",
        1 => " -> i32",
        2 => " -> i64",
        3 => " -> f64",
        _ => unreachable!(),
    }
}

fn call_body(tag: u8, call: &str) -> String {
    match tag {
        0 => format!("{call};\n    0"),
        1 => format!("{call} as u32 as i64"),
        2 => call.to_string(),
        3 => format!("{call}.to_bits() as i64"),
        _ => unreachable!(),
    }
}

fn emit_int_dispatch(out: &mut String) {
    out.push_str(
        "pub fn call_int_sig(\n    \
         slot: usize,\n    \
         args: &[i64],\n    \
         mask: u16,\n    \
         result: Option<FuncSigVal>,\n\
         ) -> Option<i64> {\n    \
         let tag = match result {\n        \
         None => 0,\n        \
         Some(FuncSigVal::I32) => 1,\n        \
         Some(FuncSigVal::I64) => 2,\n        \
         Some(FuncSigVal::F64) => 3,\n        \
         Some(FuncSigVal::F32) => return None,\n    \
         };\n    \
         Some(unsafe {\n        \
         match (args.len(), mask, tag) {\n",
    );
    for arity in 0..=7 {
        let masks = 1u16 << arity;
        for mask in 0..masks {
            for tag in 0..4 {
                let _ = writeln!(
                    out,
                    "            ({arity}, {mask}, {tag}) => c_{arity}_{mask}_{tag}(slot, args),"
                );
            }
        }
    }
    out.push_str("            _ => return None,\n        }\n    })\n}\n\n");
}

fn emit_f64_dispatch(out: &mut String) {
    out.push_str(
        "pub fn call_uniform_f64(slot: usize, args: &[i64], result: Option<FuncSigVal>) -> Option<i64> {\n    \
         let tag = match result {\n        \
         None => 0,\n        \
         Some(FuncSigVal::I64) => 2,\n        \
         Some(FuncSigVal::F64) => 3,\n        \
         _ => return None,\n    \
         };\n    \
         Some(unsafe {\n        \
         match (args.len(), tag) {\n",
    );
    for arity in 1..=4 {
        for tag in [0u8, 2, 3] {
            let _ = writeln!(
                out,
                "            ({arity}, {tag}) => f_{arity}_{tag}(slot, args),"
            );
        }
    }
    out.push_str("            _ => return None,\n        }\n    })\n}\n");
}
