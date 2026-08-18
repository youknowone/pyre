//! Print the harvested JIT hints and the lowered SemanticFunction entry for
//! function paths matching a substring.
//!
//! usage: hint_probe <leaf-name> <module-path> <file.ullbc> [more.ullbc ...]
use majit_charon_reader::Llbc;

fn main() {
    let mut args = std::env::args().skip(1);
    let needle = args
        .next()
        .expect("usage: hint_probe <leaf> <module> <ullbc>...");
    let module = args
        .next()
        .expect("usage: hint_probe <leaf> <module> <ullbc>...");
    let llbcs: Vec<Llbc> = args.map(|p| Llbc::load(&p).unwrap()).collect();
    let hints = majit_translate::front::llbc_hints::harvest_hints_from_llbcs(&llbcs);
    let mut rows: Vec<_> = hints.iter().filter(|(p, _)| p.contains(&needle)).collect();
    rows.sort();
    println!("--- harvested hints ---");
    for (path, hs) in rows {
        println!("{path} -> {hs:?}");
    }
    println!("--- lowered SemanticFunctions ---");
    match majit_translate::front::mir::build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &llbcs,
        majit_translate::HostStaticAddrs::default(),
        &[module.as_str()],
        &[needle.as_str()],
    ) {
        Ok(prog) => {
            for f in &prog.functions {
                if f.name.contains(&needle) {
                    println!(
                        "fn {}::{} hints={:?} self_ty_root={:?} return_type={:?}",
                        f.module_path, f.name, f.hints, f.self_ty_root, f.return_type
                    );
                }
            }
            println!("(total lowered = {})", prog.functions.len());
        }
        Err(e) => println!("lowering failed: {e:?}"),
    }
}
