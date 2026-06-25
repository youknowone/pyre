//! RPython `rpython/translator/revdb/gencsupp.py`.
//!
//! Reverse-debugger support in upstream mostly hangs off C-backend code
//! generation.  Pyre keeps only the backend-shape hooks needed by the
//! driver, so this module preserves the upstream helper names and string
//! contracts while keeping runtime registration leaves explicit.

use std::path::{Path, PathBuf};
use std::rc::Rc;

use crate::flowspace::model::GraphRef;
use crate::translator::backend::database::{LowLevelDatabase, RevdbCommands};
use crate::translator::backend::support::cdecl;
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

#[derive(Clone, Default)]
pub struct FunctionGen {
    pub graph: Option<GraphRef>,
    pub db: Option<Rc<LowLevelDatabase>>,
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
    let stack_bottom = funcgen
        .graph
        .as_ref()
        .map_or(funcgen.graph_has_gc_stack_bottom, graph_has_gc_stack_bottom);
    if stack_bottom {
        let mut lines = vec![
            "/* this function is a callback */".to_string(),
            format!(
                "RPY_REVDB_CALLBACKLOC(RPY_CALLBACKLOC_{});",
                funcgen.functionname
            ),
        ];
        if let Some(db) = &funcgen.db {
            db.stack_bottom_funcnames
                .borrow_mut()
                .push(funcgen.functionname.clone());
        }
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

fn graph_has_gc_stack_bottom(graph: &GraphRef) -> bool {
    graph.borrow().iterblocks().into_iter().any(|block| {
        block
            .borrow()
            .operations
            .iter()
            .any(|op| op.opname == "gc_stack_bottom")
    })
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

/// RPython `prepare_database(db)` (`revdb/gencsupp.py:127-162`).
///
/// Deferred port. Upstream reads `db.translator.revdb_commands` (the dict
/// populated by `rlib/revdb.py register_debug_command`), resolves each
/// command's function pointer via
/// `getfunctionptr(bk.getdesc(func).getuniquegraph())`, packs them into a raw
/// `RPY_REVDB_COMMANDS` struct (`int` tag -> `names`/`funcs`, `"ALLOCATING"` ->
/// `alloc`), registers the struct with
/// `exports.EXPORTS_obj2name[s._as_obj()] = 'rpy_revdb_commands'` and `db.get(s)`,
/// then resets `stack_bottom_funcnames`.
///
/// Both ends of that chain are absent here: `register_debug_command` /
/// `translator.revdb_commands` is not ported (the command set is therefore
/// always empty), and the raw-struct low-level node factory that `db.get`/the
/// exports table consume is not materialized yet (see `LowLevelDatabase::get`).
/// Until they land this records the command metadata (empty, carrying the
/// export name) and resets `stack_bottom_funcnames`, matching upstream's
/// observable result for the empty-command case. Converge by porting
/// `rlib/revdb.py` registration and the raw-struct node factory, then iterate
/// `revdb_commands` and call `db.get(s)`.
pub fn prepare_database(db: &LowLevelDatabase) -> Result<(), TaskError> {
    *db.revdb_commands.borrow_mut() = Some(RevdbCommands {
        names: Vec::new(),
        funcs: Vec::new(),
        alloc: None,
        exported_name: Some("rpy_revdb_commands".to_string()),
    });
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
    use crate::flowspace::model::{
        Block, FunctionGraph, GraphRef, Hlvalue, SpaceOperation, Variable,
    };
    use crate::translator::backend::database::GcPolicyClass;
    use std::cell::RefCell;

    fn lowlevel_database() -> Rc<LowLevelDatabase> {
        Rc::new(LowLevelDatabase::new(
            None,
            false,
            GcPolicyClass::None,
            None,
            None,
            false,
            false,
            false,
            true,
            false,
        ))
    }

    fn graph_with_operation(opname: &str) -> GraphRef {
        let result = Hlvalue::Variable(Variable::named("result"));
        let start = Block::shared(vec![]);
        start
            .borrow_mut()
            .operations
            .push(SpaceOperation::new(opname, vec![], result));
        Rc::new(RefCell::new(FunctionGraph::new("callback", start)))
    }

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

    #[test]
    fn prepare_function_scans_graph_for_gc_stack_bottom() {
        let db = lowlevel_database();
        let mut funcgen = FunctionGen {
            graph: Some(graph_with_operation("gc_stack_bottom")),
            db: Some(db.clone()),
            functionname: "cb".to_string(),
            args: vec![FunctionArg {
                lltypename: "long @".to_string(),
                expr: "arg0".to_string(),
            }],
            ..FunctionGen::default()
        };

        let (enter, leave) = prepare_function(&mut funcgen);

        let enter = enter.expect("callback enter macro should be generated");
        assert!(enter.contains("RPY_REVDB_CALLBACKLOC(RPY_CALLBACKLOC_cb);"));
        assert!(enter.contains("RPY_REVDB_EMIT(/*arg*/, long _e, arg0);"));
        assert_eq!(leave, Some("/* RPY_CALLBACK_LEAVE(); */".to_string()));
        assert_eq!(*db.stack_bottom_funcnames.borrow(), vec!["cb".to_string()]);
    }

    #[test]
    fn prepare_database_exports_revdb_commands_slot() {
        let db = lowlevel_database();
        db.stack_bottom_funcnames
            .borrow_mut()
            .push("old".to_string());

        prepare_database(&db).unwrap();

        assert!(db.stack_bottom_funcnames.borrow().is_empty());
        let commands = db.revdb_commands.borrow();
        let commands = commands
            .as_ref()
            .expect("prepare_database should allocate command metadata");
        assert_eq!(
            commands.exported_name.as_deref(),
            Some("rpy_revdb_commands")
        );
        assert!(commands.names.is_empty());
        assert!(commands.funcs.is_empty());
    }
}
