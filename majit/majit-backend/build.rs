//! Sequences that contain `ArgClass::Single` through arity 5. `Int`/`Float`
//! sequences stay in `call_sig_table`.

use std::fmt::Write as _;

fn main() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");
    let path = std::path::Path::new(&out_dir).join("single_stubs.rs");
    std::fs::write(&path, generate()).unwrap_or_else(|err| {
        panic!("write {}: {err}", path.display());
    });
}

fn generate() -> String {
    let mut seqs = Vec::new();
    for arity in 1..=5 {
        let mut cur = Vec::new();
        walk(arity, &mut cur, false, &mut seqs);
    }
    let mut out = String::new();
    emit_lookup(&mut out, "lookup_single_i", " -> i64", "i64", &seqs);
    emit_lookup(&mut out, "lookup_single_f", " -> f64", "f64", &seqs);
    emit_lookup(&mut out, "lookup_single_v", "", "()", &seqs);
    out
}

fn walk(
    arity: usize,
    cur: &mut Vec<&'static str>,
    seen_single: bool,
    out: &mut Vec<Vec<&'static str>>,
) {
    if cur.len() == arity {
        if seen_single {
            out.push(cur.clone());
        }
        return;
    }
    for name in ["Int", "Float", "Single"] {
        cur.push(name);
        walk(arity, cur, seen_single || name == "Single", out);
        cur.pop();
    }
}

fn emit_lookup(out: &mut String, name: &str, ret: &str, invoke_ret: &str, seqs: &[Vec<&str>]) {
    let _ = writeln!(
        out,
        "fn {name}(classes: &[ArgClass]) -> unsafe fn(usize, &[i64]){ret} {{\n    match classes {{"
    );
    for seq in seqs {
        let pat = seq
            .iter()
            .map(|class| format!("ArgClass::{class}"))
            .collect::<Vec<_>>()
            .join(", ");
        let args = seq.join(", ");
        let _ = writeln!(
            out,
            "        [{pat}] => {{\n            \
             unsafe fn stub(func: usize, args: &[i64]){ret} {{\n                \
             unsafe {{ invoke_stub!(func, args, {invoke_ret}, {args}) }}\n            \
             }}\n            stub\n        }}"
        );
    }
    out.push_str("        classes => unsupported_call_sig(classes),\n    }\n}\n\n");
}
