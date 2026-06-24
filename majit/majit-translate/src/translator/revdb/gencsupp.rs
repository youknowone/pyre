//! RPython `rpython/translator/revdb/gencsupp.py`.
//!
//! Reverse-debugger support is mostly C-backend code generation. The local C
//! backend is still a structural port, so this module preserves the upstream
//! helper names and string contracts while keeping runtime registration leaves
//! explicit.

use std::path::{Path, PathBuf};

use crate::translator::c::database::LowLevelDatabase;
use crate::translator::c::support::cdecl;
use crate::translator::rtyper::lltypesystem::lloperation::ll_operations;
use crate::translator::tool::taskengine::TaskError;

pub fn extra_files() -> Vec<PathBuf> {
    vec![PathBuf::from("rpython/translator/revdb/src-revdb/revdb.c")]
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionArg {
    pub lltypename: String,
    pub expr: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FunctionGen {
    pub graph_func_revdb_c_only: bool,
    pub graph_has_gc_stack_bottom: bool,
    pub functionname: String,
    pub args: Vec<FunctionArg>,
    pub revdb_do_next_call: bool,
}

/// RPython `prepare_function(funcgen)`.
pub fn prepare_function(funcgen: &mut FunctionGen) -> (Option<String>, Option<String>) {
    if funcgen.graph_func_revdb_c_only {
        return (
            Some("RPY_REVDB_C_ONLY_ENTER".to_string()),
            Some("RPY_REVDB_C_ONLY_LEAVE".to_string()),
        );
    }
    if funcgen.graph_has_gc_stack_bottom {
        let mut lines = vec![
            "/* this function is a callback */".to_string(),
            format!(
                "RPY_REVDB_CALLBACKLOC(RPY_CALLBACKLOC_{});",
                funcgen.functionname
            ),
        ];
        lines.extend(
            funcgen
                .args
                .iter()
                .map(|arg| format!("\t{}", emit("/*arg*/", &arg.lltypename, &arg.expr))),
        );
        return (
            Some(lines.join("\n")),
            Some("/* RPY_CALLBACK_LEAVE(); */".to_string()),
        );
    }
    (None, None)
}

pub fn emit_void(normal_code: &str) -> String {
    format!("RPY_REVDB_EMIT_VOID({normal_code});")
}

pub fn emit(normal_code: &str, tp: &str, value: &str) -> String {
    if tp == "void @" {
        return emit_void(normal_code);
    }
    format!(
        "RPY_REVDB_EMIT({}, {}, {});",
        normal_code,
        cdecl(tp, "_e", false),
        value
    )
}

pub fn emit_residual_call(
    funcgen: &mut FunctionGen,
    call_code: &str,
    result_lltypename: &str,
    expr_result: &str,
) -> String {
    if funcgen.revdb_do_next_call {
        funcgen.revdb_do_next_call = false;
        return call_code.to_string();
    }
    if call_code == "RPyGilAcquire();" {
        return "RPY_REVDB_CALL_GIL_ACQUIRE();".to_string();
    }
    if call_code == "RPyGilRelease();" {
        return "RPY_REVDB_CALL_GIL_RELEASE();".to_string();
    }
    if result_lltypename == "void @" {
        return format!("RPY_REVDB_CALL_VOID({call_code});");
    }
    format!(
        "RPY_REVDB_CALL({}, {}, {});",
        call_code,
        cdecl(result_lltypename, "_e", false),
        expr_result
    )
}

pub fn record_malloc_uid(expr: &str) -> String {
    format!(" RPY_REVDB_REC_UID({expr});")
}

pub fn boehm_register_finalizer(obj_expr: &str, finalizer_expr: &str) -> String {
    format!("rpy_reverse_db_register_destructor({obj_expr}, {finalizer_expr});")
}

pub fn cast_gcptr_to_int(result_expr: &str, arg_expr: &str) -> String {
    format!("{result_expr} = RPY_REVDB_CAST_PTR_TO_INT({arg_expr});")
}

pub fn set_revdb_protected() -> Vec<String> {
    ll_operations()
        .iter()
        .filter_map(|(opname, opdesc)| {
            if opdesc.revdb_protect {
                Some((*opname).to_string())
            } else {
                None
            }
        })
        .collect()
}

/// RPython `prepare_database(db)`.
pub fn prepare_database(db: &LowLevelDatabase) -> Result<(), TaskError> {
    db.stack_bottom_funcnames.borrow_mut().clear();
    Ok(())
}

pub fn revdb_def_contents(funcnames: &[String]) -> String {
    let mut out = String::new();
    let mut sorted = funcnames.to_vec();
    sorted.sort();
    for (i, name) in sorted.iter().enumerate() {
        out.push_str(&format!("#define RPY_CALLBACKLOC_{name} {i}\n"));
    }
    out.push('\n');
    out.push_str("#define RPY_CALLBACKLOCS \\\n");
    let names = if sorted.is_empty() {
        vec!["NULL".to_string()]
    } else {
        sorted
    };
    for (i, name) in names.iter().enumerate() {
        let tail = if i == names.len() - 1 { "" } else { ", \\" };
        out.push_str(&format!("\t(void *){name}{tail}\n"));
    }
    out
}

pub fn write_revdb_def_file(db: &LowLevelDatabase, target_path: &Path) -> Result<(), TaskError> {
    let contents = revdb_def_contents(&db.stack_bottom_funcnames.borrow());
    std::fs::write(target_path, contents).map_err(|e| TaskError {
        message: format!(
            "revdb/gencsupp.py:164 write_revdb_def_file failed for {}: {e}",
            target_path.display()
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emit_helpers_match_upstream_strings() {
        assert_eq!(emit_void("x = 1"), "RPY_REVDB_EMIT_VOID(x = 1);");
        assert_eq!(
            emit("x = y", "long @", "value"),
            "RPY_REVDB_EMIT(x = y, long _e, value);"
        );
        assert_eq!(
            emit("ignored", "void @", "value"),
            "RPY_REVDB_EMIT_VOID(ignored);"
        );
    }

    #[test]
    fn emit_residual_call_handles_special_calls() {
        let mut funcgen = FunctionGen::default();
        assert_eq!(
            emit_residual_call(&mut funcgen, "RPyGilAcquire();", "void @", ""),
            "RPY_REVDB_CALL_GIL_ACQUIRE();"
        );
        assert_eq!(
            emit_residual_call(&mut funcgen, "call();", "long @", "res"),
            "RPY_REVDB_CALL(call();, long _e, res);"
        );
        funcgen.revdb_do_next_call = true;
        assert_eq!(
            emit_residual_call(&mut funcgen, "really_call();", "void @", ""),
            "really_call();"
        );
        assert!(!funcgen.revdb_do_next_call);
    }

    #[test]
    fn revdb_def_contents_sorts_callback_names() {
        let contents = revdb_def_contents(&["b".to_string(), "a".to_string()]);
        assert!(contents.contains("#define RPY_CALLBACKLOC_a 0"));
        assert!(contents.contains("#define RPY_CALLBACKLOC_b 1"));
        assert!(contents.contains("\t(void *)a, \\"));
        assert!(contents.contains("\t(void *)b\n"));
    }

    #[test]
    fn revdb_def_contents_uses_null_when_empty() {
        let contents = revdb_def_contents(&[]);
        assert!(contents.contains("#define RPY_CALLBACKLOCS \\"));
        assert!(contents.contains("\t(void *)NULL\n"));
    }
}
