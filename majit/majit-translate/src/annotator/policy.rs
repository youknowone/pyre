//! Base annotation policy for specialization.
//!
//! RPython upstream: `rpython/annotator/policy.py` (100 LOC).
//!
//! Ports `AnnotatorPolicy` and the `get_specializer(directive)`
//! dispatch loop that maps `specialize:arg(N)` / `specialize:memo` /
//! `specialize:argtype(N)` / … directive strings to the per-policy
//! specializer methods.
//!
//! ## Phase 5 P5.2+ dependency-blocked paths
//!
//! * Specializer bodies (`default_specialize`, `memo`,
//!   `specialize_argvalue`, `specialize_argtype`,
//!   `specialize_arglistitemtype`, `specialize_arg_or_var`,
//!   `specialize_call_location`) come from
//!   `rpython/annotator/specialize.py` — **not yet ported**. The
//!   [`Specializer`] enum variants name each one so callers can
//!   dispatch once specialize.py lands.
//! * `specialize__ll` / `specialize__ll_and_arg` — forward to
//!   `rpython/rtyper/annlowlevel.py:LowLevelAnnotatorPolicy` —
//!   deferred until the rtyper's annlowlevel lands.
//! * `no_more_blocks_to_annotate(annotator)` (policy.py:69-100) —
//!   structurally ported. Drains `bk.pending_specializations` and
//!   iterates `added_blocks` / `annotated`, matching upstream
//!   line-by-line. The inner sandbox-trampoline rewrite branch
//!   (`needs_sandboxing` check) walks to `continue` because no
//!   `SomeValue` variant carries that attribute yet; the `s_func.args_s
//!   / s_result / const` rewrite activates when the sandbox lattice
//!   type (RPython `SomeSandboxedCallable`) lands.

use crate::annotator::bookkeeper::Bookkeeper;

/// Enumerates the specializer entry points referenced by upstream
/// `AnnotatorPolicy` class-level attributes (policy.py:53-67).
///
/// Each variant names one function in
/// `rpython/annotator/specialize.py` (or
/// `rpython/rtyper/annlowlevel.py` for the `Ll` / `LlAndArg`
/// variants). Dispatch is currently resolved at lookup time —
/// [`AnnotatorPolicy::get_specializer`] returns a `Specializer`
/// tag, and the annotator-driver port will invoke the real
/// specializer body once specialize.py lands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Specializer {
    /// RPython `default_specialize` (specialize.py) — the no-op
    /// fallback used when `directive` is `None`.
    Default,
    /// `specialize:memo` → `specialize.memo`.
    Memo,
    /// `specialize:arg(N)` → `specialize.specialize_argvalue`. The
    /// `parms` tuple carries the argument indices.
    Arg { parms: Vec<String> },
    /// `specialize:arg_or_var(N)` → `specialize.specialize_arg_or_var`.
    ArgOrVar { parms: Vec<String> },
    /// `specialize:argtype(N)` → `specialize.specialize_argtype`.
    Argtype { parms: Vec<String> },
    /// `specialize:arglistitemtype(N)` →
    /// `specialize.specialize_arglistitemtype`.
    Arglistitemtype { parms: Vec<String> },
    /// `specialize:call_location` →
    /// `specialize.specialize_call_location`.
    CallLocation,
    /// `specialize:ll` → `LowLevelAnnotatorPolicy.default_specialize`.
    Ll { parms: Vec<String> },
    /// `specialize:ll_and_arg` →
    /// `LowLevelAnnotatorPolicy.specialize__ll_and_arg`.
    LlAndArg { parms: Vec<String> },
}

/// RPython `class AnnotatorPolicy` (policy.py:11-100).
///
/// The upstream class is subclass-hook-heavy: `event`,
/// `get_specializer`, and `no_more_blocks_to_annotate` all accept a
/// `pol` (policy) instance as their first argument, letting subclasses
/// override individual pieces. The Rust port currently provides only
/// the default implementation; subclass behaviour lands as traits when
/// real policy overrides (e.g. `rpython/rlib/jit.py:JitPolicy`) are
/// ported.
#[derive(Clone, Debug, Default)]
pub struct AnnotatorPolicy;

impl AnnotatorPolicy {
    pub fn new() -> Self {
        AnnotatorPolicy
    }

    /// RPython `AnnotatorPolicy.event(pol, bookkeeper, what, *args)`
    /// (policy.py:17-18). Base implementation is a no-op; subclasses
    /// (e.g. `translator.test.snippet.StrictAnnotatorPolicy`) use
    /// it as a hook.
    pub fn event(&self, _bookkeeper: &Bookkeeper, _what: &str, _args: &[&str]) {}

    /// RPython `AnnotatorPolicy.get_specializer(pol, directive)`
    /// (policy.py:20-49).
    ///
    /// Parses a `specialize:...` directive string produced by
    /// `@specialize(...)` decorators (see
    /// `rpython/rlib/objectmodel.py:specialize`). Upstream splits the
    /// directive at `(`, `eval`s the parameter tuple, then dispatches
    /// to `pol.specialize__<name>`. The Rust port mirrors the parse,
    /// splitting the parameter string on commas instead of using
    /// `eval` — specialize directives carry only `int` / `str` parm
    /// literals (see `specialize:arg(0, 1)`,
    /// `specialize:argtype(0)`), so a comma split covers upstream's
    /// inputs without a full Python expression parser.
    pub fn get_specializer(&self, directive: Option<&str>) -> Result<Specializer, PolicyError> {
        let Some(directive) = directive else {
            // upstream: `if directive is None: return pol.default_specialize`.
            return Ok(Specializer::Default);
        };

        // upstream: `directive_parts = directive.split('(', 1)`.
        let (name, parms_src) = match directive.split_once('(') {
            None => (directive.to_string(), None),
            Some((n, rest)) => {
                // Upstream strips the trailing `)` via `eval("...)" +
                // rest)` — the Rust version drops the trailing `)`
                // explicitly.
                let parms = rest.strip_suffix(')').ok_or_else(|| {
                    PolicyError(format!("broken specialize directive parms: {directive}"))
                })?;
                (n.to_string(), Some(parms.to_string()))
            }
        };
        let parms: Vec<String> = match parms_src {
            None => Vec::new(),
            Some(s) if s.trim().is_empty() => Vec::new(),
            Some(s) => s.split(',').map(|p| p.trim().to_string()).collect(),
        };

        // upstream: `name = name.replace(':', '__')`. The directive
        // namespace uses `:` (e.g. `specialize:arg`); upstream maps
        // that to the `specialize__arg` attribute lookup. The Rust
        // match below uses the post-replace form.
        let normalized = name.replace(':', "__");

        match normalized.as_str() {
            "default_specialize" => Ok(Specializer::Default),
            "specialize__memo" => Ok(Specializer::Memo),
            "specialize__arg" => Ok(Specializer::Arg { parms }),
            "specialize__arg_or_var" => Ok(Specializer::ArgOrVar { parms }),
            "specialize__argtype" => Ok(Specializer::Argtype { parms }),
            "specialize__arglistitemtype" => Ok(Specializer::Arglistitemtype { parms }),
            "specialize__call_location" => Ok(Specializer::CallLocation),
            "specialize__ll" => Ok(Specializer::Ll { parms }),
            "specialize__ll_and_arg" => Ok(Specializer::LlAndArg { parms }),
            other => Err(PolicyError(format!(
                "{other:?} specialize tag not defined in annotation policy AnnotatorPolicy"
            ))),
        }
    }

    /// RPython `AnnotatorPolicy.no_more_blocks_to_annotate(pol, annotator)`
    /// (policy.py:69-100).
    ///
    /// Drains pending specialization callbacks, then walks every
    /// `op.simple_call` / `op.call_args` instruction in the annotator's
    /// added-or-annotated block set. Call sites whose callee annotation
    /// carries `needs_sandboxing` are rewritten to invoke a sandbox
    /// trampoline instead.
    ///
    /// ```python
    /// key = ('sandboxing', s_func.const)
    /// if key not in bk.emulated_pbc_calls:
    ///     params_s = s_func.args_s
    ///     s_result = s_func.s_result
    ///     sandbox_trampoline = make_sandbox_trampoline(s_func.name, params_s, s_result)
    ///     sandbox_trampoline._signature_ = [SomeTuple(items=params_s)], s_result
    ///     bk.emulate_pbc_call(key, bk.immutablevalue(sandbox_trampoline), params_s)
    /// else:
    ///     s_trampoline = bk.emulated_pbc_calls[key][0]
    ///     sandbox_trampoline = s_trampoline.const
    /// new = instr.replace({instr.args[0]: Constant(sandbox_trampoline)})
    /// block.operations[i] = new
    /// ```
    ///
    /// Rust port: `needs_sandboxing` lives as an `Option<SandboxingPayload>`
    /// on `SomeBuiltin` (matches upstream's `setattr` on
    /// SomeExternalFunction). `make_sandbox_trampoline` is deferred —
    /// its body lives in `rpython/translator/sandbox/rsandbox.py` and
    /// depends on rtyper infrastructure — so the Rust branch emits a
    /// synthesized `HostObject::new_builtin_callable` placeholder and
    /// performs the `emulate_pbc_call` bookkeeping + instruction
    /// rewrite. Sandbox-mode translation is not yet end-to-end usable
    /// regardless, so the placeholder trampoline does not reach a
    /// backend.
    pub fn no_more_blocks_to_annotate(&self, ann: &super::annrpython::RPythonAnnotator) {
        use super::bookkeeper::EmulatedPbcCallKey;
        use super::model::{SomePBC, SomeValue};
        use crate::flowspace::model::{BlockKey, ConstValue, Constant, Hlvalue, HostObject};
        use crate::flowspace::operation::OpKind;

        let bk = &ann.bookkeeper;
        // upstream: for callback in bk.pending_specializations: callback()
        let callbacks: Vec<Box<dyn Fn()>> =
            bk.pending_specializations.borrow_mut().drain(..).collect();
        for callback in &callbacks {
            callback();
        }
        // upstream: del bk.pending_specializations[:]  (already drained)

        // upstream:
        //   if annotator.added_blocks is not None:
        //       all_blocks = annotator.added_blocks
        //   else:
        //       all_blocks = annotator.annotated
        let all_blocks: Vec<BlockKey> = {
            let added = ann.added_blocks.borrow();
            match added.as_ref() {
                Some(added_set) => added_set.keys().cloned().collect(),
                None => ann.annotated.borrow().keys().cloned().collect(),
            }
        };

        // upstream: for block in list(all_blocks):
        for block_key in all_blocks {
            let block_ref = {
                let all_blocks_index = ann.all_blocks.borrow();
                match all_blocks_index.get(&block_key).cloned() {
                    Some(b) => b,
                    None => continue,
                }
            };

            // Snapshot length / reify HLOperations so we can mutate the
            // block.operations slice in place without borrow conflicts.
            let op_count = block_ref.borrow().operations.len();
            for i in 0..op_count {
                let sp = block_ref.borrow().operations[i].clone();
                //   if not isinstance(instr, (op.simple_call, op.call_args)):
                //       continue
                let kind = OpKind::from_opname(&sp.opname);
                if !matches!(kind, Some(OpKind::SimpleCall | OpKind::CallArgs)) {
                    continue;
                }
                //   v_func = instr.args[0]
                let Some(v_func) = sp.args.first().cloned() else {
                    continue;
                };
                //   s_func = annotator.annotation(v_func)
                let Some(s_func) = ann.annotation(&v_func) else {
                    continue;
                };
                //   if not hasattr(s_func, 'needs_sandboxing'): continue
                let SomeValue::Builtin(ref builtin) = s_func else {
                    continue;
                };
                let Some(payload) = builtin.needs_sandboxing.clone() else {
                    continue;
                };
                let Some(func_const) = s_func.const_().cloned() else {
                    continue;
                };

                // upstream: key = ('sandboxing', s_func.const)
                let key = EmulatedPbcCallKey::sandboxing(&func_const);
                // upstream: if key not in bk.emulated_pbc_calls: ... else: ...
                let trampoline_host: HostObject = {
                    let existing = bk
                        .emulated_pbc_calls
                        .borrow()
                        .get(&key)
                        .map(|(pbc, _)| pbc.clone());
                    match existing {
                        Some(pbc) => {
                            //   s_trampoline = bk.emulated_pbc_calls[key][0]
                            //   sandbox_trampoline = s_trampoline.const
                            match pbc.base.const_box.as_ref().map(|c| c.value.clone()) {
                                Some(ConstValue::HostObject(obj)) => obj,
                                _ => continue,
                            }
                        }
                        None => {
                            //   sandbox_trampoline = make_sandbox_trampoline(...)
                            //   bk.emulate_pbc_call(key, bk.immutablevalue(sandbox_trampoline), params_s)
                            //
                            // `make_sandbox_trampoline` (rsandbox.py) produces
                            // a real Python function; deferred — we materialize
                            // a `HostObject::new_user_function` stand-in so
                            // `bk.immutablevalue` routes it through the
                            // user-function path that builds a real SomePBC,
                            // matching the shape `emulate_pbc_call` expects.
                            use crate::flowspace::model::GraphFunc;
                            let trampoline_name = format!("sandbox_trampoline({})", payload.name);
                            let globals = Constant::new(ConstValue::Dict(Default::default()));
                            let trampoline = HostObject::new_user_function(GraphFunc::new(
                                trampoline_name.clone(),
                                globals,
                            ));
                            let pbc_value = match bk
                                .immutablevalue(&ConstValue::HostObject(trampoline.clone()))
                            {
                                Ok(v) => v,
                                Err(_) => continue,
                            };
                            let _ = bk.emulate_pbc_call(key, &pbc_value, &payload.args_s, &[]);
                            trampoline
                        }
                    }
                };

                // upstream:
                //   new = instr.replace({instr.args[0]: Constant(sandbox_trampoline)})
                //   block.operations[i] = new
                let new_arg =
                    Hlvalue::Constant(Constant::new(ConstValue::HostObject(trampoline_host)));
                let mut blk_mut = block_ref.borrow_mut();
                if let Some(op_mut) = blk_mut.operations.get_mut(i) {
                    if !op_mut.args.is_empty() {
                        op_mut.args[0] = new_arg;
                    }
                }

                // Suppress unused-warning when the SomePBC import above
                // is structurally expected by upstream but not used on
                // this path.
                let _ = std::marker::PhantomData::<SomePBC>;
            }
        }
    }
}

/// RPython `raise AttributeError(...)` / `raise Exception(...)` from
/// `AnnotatorPolicy.get_specializer` (policy.py:36, :41-42).
///
/// The upstream class hierarchy mixes `AttributeError` and a bare
/// `Exception` on the same code path; Rust collapses both into a
/// single typed error with the upstream message verbatim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyError(pub String);

impl std::fmt::Display for PolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PolicyError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_when_no_directive() {
        let pol = AnnotatorPolicy::new();
        assert_eq!(pol.get_specializer(None).unwrap(), Specializer::Default);
    }

    #[test]
    fn specialize_arg_parses_parms() {
        let pol = AnnotatorPolicy::new();
        let sp = pol.get_specializer(Some("specialize:arg(0)")).unwrap();
        assert_eq!(
            sp,
            Specializer::Arg {
                parms: vec!["0".to_string()]
            }
        );
    }

    #[test]
    fn specialize_arg_multi_parm() {
        let pol = AnnotatorPolicy::new();
        let sp = pol.get_specializer(Some("specialize:arg(0, 1)")).unwrap();
        assert_eq!(
            sp,
            Specializer::Arg {
                parms: vec!["0".to_string(), "1".to_string()]
            }
        );
    }

    #[test]
    fn specialize_memo_no_parm() {
        let pol = AnnotatorPolicy::new();
        assert_eq!(
            pol.get_specializer(Some("specialize:memo")).unwrap(),
            Specializer::Memo
        );
    }

    #[test]
    fn specialize_argtype_call_location_argor_var() {
        let pol = AnnotatorPolicy::new();
        assert!(matches!(
            pol.get_specializer(Some("specialize:argtype(0)")).unwrap(),
            Specializer::Argtype { .. }
        ));
        assert_eq!(
            pol.get_specializer(Some("specialize:call_location"))
                .unwrap(),
            Specializer::CallLocation
        );
        assert!(matches!(
            pol.get_specializer(Some("specialize:arg_or_var(0)"))
                .unwrap(),
            Specializer::ArgOrVar { .. }
        ));
    }

    #[test]
    fn specialize_ll_variants() {
        let pol = AnnotatorPolicy::new();
        assert!(matches!(
            pol.get_specializer(Some("specialize:ll")).unwrap(),
            Specializer::Ll { .. }
        ));
        assert!(matches!(
            pol.get_specializer(Some("specialize:ll_and_arg")).unwrap(),
            Specializer::LlAndArg { .. }
        ));
    }

    #[test]
    fn unknown_directive_errors() {
        let pol = AnnotatorPolicy::new();
        let err = pol
            .get_specializer(Some("specialize:nonexistent"))
            .expect_err("unknown directive must error");
        assert!(err.0.contains("specialize tag not defined"));
    }

    #[test]
    fn broken_parms_errors() {
        let pol = AnnotatorPolicy::new();
        let err = pol
            .get_specializer(Some("specialize:arg(0"))
            .expect_err("missing closing paren must error");
        assert!(err.0.contains("broken specialize directive parms"));
    }

    #[test]
    fn empty_parms_list() {
        // "specialize:arg()" — parses as Arg with empty parms.
        let pol = AnnotatorPolicy::new();
        let sp = pol.get_specializer(Some("specialize:arg()")).unwrap();
        assert_eq!(sp, Specializer::Arg { parms: Vec::new() });
    }

    #[test]
    fn event_is_noop() {
        let pol = AnnotatorPolicy::new();
        let bk = Bookkeeper::new();
        pol.event(&bk, "some_event", &["arg1"]);
    }

    #[test]
    fn no_more_blocks_to_annotate_rewrites_sandboxed_call_site() {
        // Stand up a minimal annotator + graph + block containing a
        // `simple_call` whose callee annotation is a SomeBuiltin with
        // `needs_sandboxing = Some(payload)`. The policy walker must
        // rewrite `instr.args[0]` to a Constant(HostObject) carrying
        // the synthesized sandbox trampoline.
        use super::super::annrpython::RPythonAnnotator;
        use super::super::model::{SandboxingPayload, SomeBuiltin, SomeInteger, SomeValue};
        use crate::flowspace::model::{
            Block, BlockKey, ConstValue, Constant, FunctionGraph, Hlvalue, HostObject,
            SpaceOperation, Variable,
        };
        use std::cell::RefCell;
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);

        // Build a callable HostObject and bind it on a variable as a
        // sandboxed SomeBuiltin.
        let callable = HostObject::new_builtin_callable("pkg.extfn");
        let mut builtin = SomeBuiltin::new("pkg.extfn", None, None);
        builtin.needs_sandboxing = Some(SandboxingPayload {
            name: "pkg.extfn".into(),
            args_s: vec![SomeValue::Integer(SomeInteger::default())],
            s_result: Box::new(SomeValue::Integer(SomeInteger::default())),
        });
        builtin.base.const_box = Some(Constant::new(ConstValue::HostObject(callable.clone())));
        let s_func = SomeValue::Builtin(builtin);

        let mut v_func = Variable::named("f");
        ann.setbinding(&mut v_func, s_func);
        let v_arg = Variable::named("x");
        let result_v = Variable::named("r");

        // Build a block with one simple_call op.
        let block = Rc::new(RefCell::new(Block::new(vec![])));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "simple_call",
            vec![Hlvalue::Variable(v_func), Hlvalue::Variable(v_arg)],
            Hlvalue::Variable(result_v),
        ));
        let graph = Rc::new(RefCell::new(FunctionGraph::new("caller", block.clone())));
        let bkey = BlockKey::of(&block);
        ann.annotated.borrow_mut().insert(bkey.clone(), None);
        ann.all_blocks.borrow_mut().insert(bkey, block.clone());
        let _ = graph; // keep the Rc alive for the duration of the test

        // Run the policy walker.
        ann.policy.borrow().no_more_blocks_to_annotate(&ann);

        // Verify the callee arg was rewritten to a Constant(HostObject).
        let args0 = block.borrow().operations[0].args[0].clone();
        match args0 {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(h) => {
                    assert!(h.qualname().starts_with("sandbox_trampoline("));
                }
                other => panic!("expected HostObject, got {other:?}"),
            },
            other => panic!("expected Hlvalue::Constant, got {other:?}"),
        }

        // Verify the sandbox entry landed in emulated_pbc_calls.
        let key = super::super::bookkeeper::EmulatedPbcCallKey::sandboxing(
            &ConstValue::HostObject(callable),
        );
        assert!(
            ann.bookkeeper
                .emulated_pbc_calls
                .borrow()
                .contains_key(&key)
        );
    }

    #[test]
    fn no_more_blocks_to_annotate_drains_pending_specializations() {
        use std::rc::Rc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let ann = super::super::annrpython::RPythonAnnotator::new(None, None, None, false);
        let counter = Rc::new(AtomicUsize::new(0));
        {
            let c = Rc::clone(&counter);
            ann.bookkeeper
                .pending_specializations
                .borrow_mut()
                .push(Box::new(move || {
                    c.fetch_add(1, Ordering::SeqCst);
                }));
        }
        {
            let c = Rc::clone(&counter);
            ann.bookkeeper
                .pending_specializations
                .borrow_mut()
                .push(Box::new(move || {
                    c.fetch_add(1, Ordering::SeqCst);
                }));
        }
        ann.policy.borrow().no_more_blocks_to_annotate(&ann);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
        assert!(ann.bookkeeper.pending_specializations.borrow().is_empty());
    }
}
