//! charon-spike-lower
//!
//! Reads a Charon `.ullbc` JSON file, lowers selected functions into the
//! spike `FunctionGraph` shape, and prints the canonical text form.
//!
//! ```text
//! charon-spike-lower <file.ullbc> <fn_name> [<fn_name> ...]
//! ```
//!
//! Used by `compare.sh` to diff against hand-authored expected fixtures
//! under `expected/`.

mod graph;
mod lower;
mod ullbc;

use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <file.ullbc> <fn_name> [<fn_name> ...]", args[0]);
        return ExitCode::from(2);
    }
    let path = &args[1];
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            return ExitCode::from(1);
        }
    };
    let file: ullbc::LlbcFile = match serde_json::from_slice(&bytes) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot parse {path}: {e}");
            return ExitCode::from(1);
        }
    };
    eprintln!(
        "loaded charon={} crate={} fns={} (has_errors={})",
        file.charon_version,
        file.translated.crate_name,
        file.translated.fun_decls.len(),
        file.has_errors,
    );

    let mut had_err = false;
    for name in &args[2..] {
        match file.local_fn(name) {
            None => {
                eprintln!("error: function '{name}' not found in {path}");
                had_err = true;
            }
            Some(fd) => {
                let g = lower::lower(fd);
                print!("{}", g.canonical());
                println!("---");
            }
        }
    }
    if had_err {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}
