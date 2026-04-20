//! Arguments objects — annotator layer.
//!
//! RPython upstream: `rpython/annotator/argument.py` (254 LOC).
//!
//! Upstream extends `rpython/flowspace/argument.py:CallSpec` with a
//! subclass that understands `SomeTuple` unpacking. The Rust
//! [`flowspace::argument::CallSpec`] is hard-typed to carry
//! [`flowspace::model::Hlvalue`] items, so the annotator-layer
//! counterpart duplicates CallSpec's field shape over
//! [`SomeValue`] instead of extending CallSpec directly.
//!
//! ## PRE-EXISTING-ADAPTATION: CallSpec ↔ ArgumentsForTranslation
//!
//! Upstream Python:
//! ```python
//! class ArgumentsForTranslation(CallSpec):
//!     ...
//! ```
//!
//! Python's duck typing lets `CallSpec.__init__` populate
//! `arguments_w` with either `Hlvalue` (flowspace) or `SomeValue`
//! (annotator) instances. Rust requires a single concrete type per
//! field, so the field shape is mirrored verbatim rather than
//! inherited. Every CallSpec method that `ArgumentsForTranslation`
//! invokes via `self.…` (`_rawshape`, `flatten`, `fromshape`) is
//! re-implemented in this file against `SomeValue`. The duplication
//! scope is limited to the methods upstream actually exercises.

use std::collections::HashMap;

use super::model::{SomeTuple, SomeValue};
use crate::flowspace::argument::{CallShape, Signature};

/// Generalised `rpython/flowspace/argument.py:CallSpec` storage over
/// [`SomeValue`].
///
/// The companion flowspace CallSpec is hard-typed against `Hlvalue`;
/// this struct carries the same field shape for the annotator layer
/// (see module doc).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArgumentsForTranslation {
    /// RPython `CallSpec.arguments_w`.
    pub arguments_w: Vec<SomeValue>,
    /// RPython `CallSpec.keywords`.
    pub keywords: HashMap<String, SomeValue>,
    /// RPython `CallSpec.w_stararg`.
    pub w_stararg: Option<SomeValue>,
}

/// RPython `class ArgErr(Exception)` and its three subclasses
/// (argument.py:174-254). The Rust port collapses the subclass
/// hierarchy into a single enum carrying the upstream message
/// components — each variant mirrors one RPython class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArgErr {
    /// RPython `class ArgErrCount(ArgErr)` (argument.py:179-224).
    Count {
        signature: Signature,
        num_defaults: usize,
        missing_args: usize,
        num_args: usize,
        num_kwds: usize,
    },
    /// RPython `class ArgErrMultipleValues(ArgErr)` (argument.py:227-234).
    MultipleValues { argname: String },
    /// RPython `class ArgErrUnknownKwds(ArgErr)` (argument.py:237-254).
    UnknownKwds { num_kwds: usize, kwd_name: String },
}

impl ArgErr {
    /// RPython `ArgErr.getmsg()` dispatch — upstream overrides per
    /// subclass (argument.py:189-254).
    pub fn getmsg(&self) -> String {
        match self {
            ArgErr::Count {
                signature,
                num_defaults,
                missing_args,
                num_args,
                num_kwds,
            } => {
                // upstream argument.py:189-224.
                let n = signature.num_argnames();
                if n == 0 {
                    return format!("takes no arguments ({} given)", num_args + num_kwds);
                }
                let mut defcount = *num_defaults;
                let mut has_kwarg = signature.has_kwarg();
                let mut n = n;
                let mut num_args = *num_args;
                let mut num_kwds = *num_kwds;
                let msg1 = if defcount == 0 && !signature.has_vararg() {
                    if !has_kwarg {
                        num_args += num_kwds;
                        num_kwds = 0;
                    }
                    "exactly"
                } else if *missing_args == 0 {
                    "at most"
                } else {
                    has_kwarg = false;
                    n -= defcount;
                    defcount = 0;
                    "at least"
                };
                let _ = defcount;
                let plural = if n == 1 { "" } else { "s" };
                let msg2 = if has_kwarg || num_kwds > 0 {
                    " non-keyword"
                } else {
                    ""
                };
                format!("takes {msg1} {n}{msg2} argument{plural} ({num_args} given)")
            }
            ArgErr::MultipleValues { argname } => {
                format!("got multiple values for keyword argument '{argname}'")
            }
            ArgErr::UnknownKwds { num_kwds, kwd_name } => {
                if *num_kwds == 1 {
                    format!("got an unexpected keyword argument '{kwd_name}'")
                } else {
                    format!("got {num_kwds} unexpected keyword arguments")
                }
            }
        }
    }
}

impl std::fmt::Display for ArgErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.getmsg())
    }
}

impl std::error::Error for ArgErr {}

impl ArgumentsForTranslation {
    /// RPython `CallSpec.__init__(args_w, keywords=None, w_stararg=None)`
    /// (flowspace/argument.py) inherited by
    /// `ArgumentsForTranslation`.
    pub fn new(
        arguments_w: Vec<SomeValue>,
        keywords: Option<HashMap<String, SomeValue>>,
        w_stararg: Option<SomeValue>,
    ) -> Self {
        ArgumentsForTranslation {
            arguments_w,
            keywords: keywords.unwrap_or_default(),
            w_stararg,
        }
    }

    /// RPython `ArgumentsForTranslation.positional_args` property
    /// (argument.py:8-14).
    pub fn positional_args(&self) -> Result<Vec<SomeValue>, ArgErr> {
        if let Some(stararg) = &self.w_stararg {
            let args_w = Self::unpackiterable(stararg)?;
            let mut out = self.arguments_w.clone();
            out.extend(args_w);
            Ok(out)
        } else {
            Ok(self.arguments_w.clone())
        }
    }

    /// RPython `ArgumentsForTranslation.newtuple(items_s)`
    /// (argument.py:16-17).
    pub fn newtuple(items_s: Vec<SomeValue>) -> SomeValue {
        SomeValue::Tuple(SomeTuple::new(items_s))
    }

    /// RPython `ArgumentsForTranslation.unpackiterable(s_obj)`
    /// (argument.py:19-21). Upstream asserts the arg is a SomeTuple;
    /// we return `ArgErr::Count` instead of panicking on a mismatch
    /// so callers can surface the error up a controlled path.
    pub fn unpackiterable(s_obj: &SomeValue) -> Result<Vec<SomeValue>, ArgErr> {
        match s_obj {
            SomeValue::Tuple(t) => Ok(t.items.clone()),
            _ => Err(ArgErr::Count {
                signature: Signature::new(Vec::new(), None, None),
                num_defaults: 0,
                missing_args: 0,
                num_args: 0,
                num_kwds: 0,
            }),
        }
    }

    /// RPython `ArgumentsForTranslation.fixedunpack(argcount)`
    /// (argument.py:23-32).
    pub fn fixedunpack(&self, argcount: usize) -> Result<Vec<SomeValue>, String> {
        if !self.keywords.is_empty() {
            return Err("no keyword arguments expected".into());
        }
        match self.arguments_w.len().cmp(&argcount) {
            std::cmp::Ordering::Greater => Err(format!("too many arguments ({argcount} expected)")),
            std::cmp::Ordering::Less => Err(format!("not enough arguments ({argcount} expected)")),
            std::cmp::Ordering::Equal => Ok(self.arguments_w.clone()),
        }
    }

    /// RPython `ArgumentsForTranslation.prepend(w_firstarg)`
    /// (argument.py:34-37).
    pub fn prepend(&self, w_firstarg: SomeValue) -> Self {
        let mut args = Vec::with_capacity(self.arguments_w.len() + 1);
        args.push(w_firstarg);
        args.extend(self.arguments_w.iter().cloned());
        ArgumentsForTranslation {
            arguments_w: args,
            keywords: self.keywords.clone(),
            w_stararg: self.w_stararg.clone(),
        }
    }

    /// RPython `ArgumentsForTranslation.copy()` (argument.py:39-41).
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// RPython `_match_signature(scope_w, signature, defaults_w=None)`
    /// (argument.py:43-120).
    fn match_signature_into(
        &self,
        scope_w: &mut [Option<SomeValue>],
        signature: &Signature,
        defaults_w: Option<&[SomeValue]>,
    ) -> Result<(), ArgErr> {
        let co_argcount = signature.num_argnames();

        let args_w = self.positional_args()?;
        let num_args = args_w.len();
        let num_kwds = self.keywords.len();

        // `take = min(num_args, co_argcount)`. scope_w[:take] = args_w[:take].
        let take = num_args.min(co_argcount);
        for i in 0..take {
            scope_w[i] = Some(args_w[i].clone());
        }
        let input_argcount = take;

        // `if signature.has_vararg(): ...`
        if signature.has_vararg() {
            let starargs_w: Vec<SomeValue> = if num_args > co_argcount {
                args_w[co_argcount..].to_vec()
            } else {
                Vec::new()
            };
            scope_w[co_argcount] = Some(Self::newtuple(starargs_w));
        } else if num_args > co_argcount {
            return Err(ArgErr::Count {
                signature: signature.clone(),
                num_defaults: defaults_w.map_or(0, |d| d.len()),
                missing_args: 0,
                num_args,
                num_kwds,
            });
        }

        // upstream: `assert not signature.has_kwarg() # XXX should not happen?`
        assert!(
            !signature.has_kwarg(),
            "signature.has_kwarg() not supported"
        );

        // upstream: keyword-argument matching (argument.py:74-101).
        if num_kwds > 0 {
            let mut mapping = Vec::<String>::new();
            let mut num_remainingkwds = self.keywords.len();
            for name in self.keywords.keys() {
                let j = signature.find_argname(name);
                if j < input_argcount as i32 {
                    if j >= 0 {
                        return Err(ArgErr::MultipleValues {
                            argname: name.clone(),
                        });
                    }
                } else {
                    mapping.push(name.clone());
                    num_remainingkwds -= 1;
                }
            }

            if num_remainingkwds > 0 {
                if co_argcount == 0 {
                    return Err(ArgErr::Count {
                        signature: signature.clone(),
                        num_defaults: defaults_w.map_or(0, |d| d.len()),
                        missing_args: 0,
                        num_args,
                        num_kwds,
                    });
                }
                // upstream `ArgErrUnknownKwds.__init__` (argument.py:
                // 237-245): when exactly one unmatched keyword remains,
                // iterate the caller-supplied keyword names and pick
                // the first one that is NOT in `kwds_mapping` (the set
                // of successfully-bound names) — that's the "unknown"
                // keyword. For num_remainingkwds > 1, upstream leaves
                // `name = ''` and the message omits the keyword name.
                let kwd_name = if num_remainingkwds == 1 {
                    self.keywords
                        .keys()
                        .find(|k| !mapping.contains(*k))
                        .cloned()
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                return Err(ArgErr::UnknownKwds {
                    num_kwds: num_remainingkwds,
                    kwd_name,
                });
            }
            // upstream `kwds_mapping` is consumed only by the error
            // branches above; once we reach this point it has done its
            // job, so no further use.
            drop(mapping);
        }

        // upstream: defaults-fill (argument.py:103-120).
        let mut missing = 0usize;
        if input_argcount < co_argcount {
            let def_first = co_argcount - defaults_w.map_or(0, |d| d.len());
            for i in input_argcount..co_argcount {
                let name = &signature.argnames[i];
                if let Some(v) = self.keywords.get(name) {
                    scope_w[i] = Some(v.clone());
                    continue;
                }
                let defnum = i as isize - def_first as isize;
                if defnum >= 0 {
                    if let Some(defaults) = defaults_w {
                        scope_w[i] = Some(defaults[defnum as usize].clone());
                    } else {
                        missing += 1;
                    }
                } else {
                    missing += 1;
                }
            }
            if missing > 0 {
                return Err(ArgErr::Count {
                    signature: signature.clone(),
                    num_defaults: defaults_w.map_or(0, |d| d.len()),
                    missing_args: missing,
                    num_args,
                    num_kwds,
                });
            }
        }
        Ok(())
    }

    /// RPython `unpack()` (argument.py:122-124).
    pub fn unpack(&self) -> Result<(Vec<SomeValue>, HashMap<String, SomeValue>), ArgErr> {
        Ok((self.positional_args()?, self.keywords.clone()))
    }

    /// RPython `match_signature(signature, defaults_w)`
    /// (argument.py:126-133).
    pub fn match_signature(
        &self,
        signature: &Signature,
        defaults_w: Option<&[SomeValue]>,
    ) -> Result<Vec<SomeValue>, ArgErr> {
        let scopelen = signature.scope_length();
        let mut scope_w: Vec<Option<SomeValue>> = vec![None; scopelen];
        self.match_signature_into(&mut scope_w, signature, defaults_w)?;
        // Upstream returns the scope_w list possibly containing `None`
        // slots for unmatched-default args; since Rust callers consume
        // a `Vec<SomeValue>` and upstream's next step unconditionally
        // overwrites None slots from defaults, flatten with Impossible
        // as a no-default sentinel.
        Ok(scope_w
            .into_iter()
            .map(|opt| opt.unwrap_or(SomeValue::Impossible))
            .collect())
    }

    /// RPython `unmatch_signature(signature, data_w)`
    /// (argument.py:135-156).
    pub fn unmatch_signature(
        &self,
        signature: &Signature,
        data_w: &[SomeValue],
    ) -> Result<Self, ArgErr> {
        let argnames = &signature.argnames;
        let varargname = signature.varargname.as_deref();
        let kwargname = signature.kwargname.as_deref();
        assert!(kwargname.is_none());
        let cnt = argnames.len();
        let need_cnt = self.positional_args()?.len();

        let (args_source_slice, has_stararg) = if varargname.is_some() {
            assert_eq!(data_w.len(), cnt + 1);
            let stararg_w = Self::unpackiterable(&data_w[cnt])?;
            if !stararg_w.is_empty() {
                let mut args_w = data_w[..cnt].to_vec();
                args_w.extend(stararg_w);
                assert_eq!(args_w.len(), need_cnt);
                assert!(self.keywords.is_empty());
                return Ok(ArgumentsForTranslation::new(args_w, None, None));
            } else {
                (&data_w[..data_w.len() - 1], true)
            }
        } else {
            (data_w, false)
        };
        let _ = has_stararg;

        assert_eq!(args_source_slice.len(), cnt);
        assert!(args_source_slice.len() >= need_cnt);
        let args_w = args_source_slice[..need_cnt].to_vec();
        // upstream: `_kwds_w = dict(zip(argnames[need_cnt:], data_w[need_cnt:]))`
        let mut kwds_w: HashMap<String, SomeValue> = HashMap::new();
        for (name, value) in argnames[need_cnt..]
            .iter()
            .zip(args_source_slice[need_cnt..].iter())
        {
            kwds_w.insert(name.clone(), value.clone());
        }
        // upstream: `keywords_w = [_kwds_w[key] for key in self.keywords]`.
        // Rust uses HashMap by name, so the reverse mapping is simply a
        // clone of kwds_w filtered by self.keywords.keys().
        let mut reverse = HashMap::new();
        for key in self.keywords.keys() {
            if let Some(v) = kwds_w.get(key) {
                reverse.insert(key.clone(), v.clone());
            }
        }
        Ok(ArgumentsForTranslation::new(args_w, Some(reverse), None))
    }

    /// RPython `CallSpec._rawshape()` (flowspace/argument.py). Needed
    /// by the free [`rawshape`] function below.
    pub fn rawshape(&self) -> CallShape {
        let mut shape_keys: Vec<String> = self.keywords.keys().cloned().collect();
        shape_keys.sort();
        CallShape {
            shape_cnt: self.arguments_w.len(),
            shape_keys,
            shape_star: self.w_stararg.is_some(),
        }
    }

    /// RPython `CallSpec.fromshape(cls, shape, data_w)`
    /// (flowspace/argument.py), re-implemented over `SomeValue` for
    /// [`complex_args`].
    pub fn fromshape(shape: &CallShape, data_w: Vec<SomeValue>) -> Self {
        let shape_cnt = shape.shape_cnt;
        let end_keys = shape_cnt + shape.shape_keys.len();
        let args_w = data_w[..shape_cnt].to_vec();
        let keyword_slice = &data_w[shape_cnt..end_keys];
        let mut keywords: HashMap<String, SomeValue> = HashMap::new();
        for (name, value) in shape.shape_keys.iter().zip(keyword_slice.iter()) {
            keywords.insert(name.clone(), value.clone());
        }
        let w_stararg = if shape.shape_star {
            Some(data_w[end_keys].clone())
        } else {
            None
        };
        ArgumentsForTranslation {
            arguments_w: args_w,
            keywords,
            w_stararg,
        }
    }
}

/// RPython `rawshape(args)` (argument.py:159-160).
pub fn rawshape(args: &ArgumentsForTranslation) -> CallShape {
    args.rawshape()
}

/// RPython `simple_args(args_s)` (argument.py:162-163).
pub fn simple_args(args_s: Vec<SomeValue>) -> ArgumentsForTranslation {
    ArgumentsForTranslation::new(args_s, None, None)
}

/// RPython `complex_args(args_s)` (argument.py:165-167).
///
/// Upstream reads the shape out of `args_s[0].const`. The Rust port
/// accepts an explicit [`CallShape`] alongside the `args_s` tail so
/// callers do not need to round-trip a [`CallShape`] through a
/// `SomeValue::Constant` payload just for this call site.
pub fn complex_args(shape: &CallShape, args_s: Vec<SomeValue>) -> ArgumentsForTranslation {
    ArgumentsForTranslation::fromshape(shape, args_s)
}

#[cfg(test)]
mod tests {
    use super::super::model::{SomeInteger, SomeValue};
    use super::*;

    fn s_int() -> SomeValue {
        SomeValue::Integer(SomeInteger::default())
    }

    fn s_tuple(items: Vec<SomeValue>) -> SomeValue {
        SomeValue::Tuple(SomeTuple::new(items))
    }

    #[test]
    fn positional_args_without_stararg() {
        let args = ArgumentsForTranslation::new(vec![s_int(), s_int()], None, None);
        assert_eq!(args.positional_args().unwrap().len(), 2);
    }

    #[test]
    fn positional_args_with_stararg_tuple() {
        let args = ArgumentsForTranslation::new(
            vec![s_int()],
            None,
            Some(s_tuple(vec![s_int(), s_int()])),
        );
        assert_eq!(args.positional_args().unwrap().len(), 3);
    }

    #[test]
    fn fixedunpack_rejects_too_few() {
        let args = ArgumentsForTranslation::new(vec![s_int()], None, None);
        let err = args.fixedunpack(2).expect_err("too few must error");
        assert!(err.contains("not enough"));
    }

    #[test]
    fn fixedunpack_rejects_kwargs() {
        let mut kws = HashMap::new();
        kws.insert("x".into(), s_int());
        let args = ArgumentsForTranslation::new(vec![s_int(), s_int()], Some(kws), None);
        let err = args.fixedunpack(2).expect_err("kwargs must error");
        assert!(err.contains("no keyword arguments"));
    }

    #[test]
    fn prepend_inserts_first() {
        let args = ArgumentsForTranslation::new(vec![s_int()], None, None);
        let ten = SomeValue::Integer(SomeInteger::new(true, false));
        let out = args.prepend(ten.clone());
        assert_eq!(out.arguments_w.len(), 2);
        assert_eq!(out.arguments_w[0], ten);
    }

    #[test]
    fn match_signature_fills_defaults() {
        // def f(a, b=10): ...
        let sig = Signature::new(vec!["a".into(), "b".into()], None, None);
        let defaults = [SomeValue::Integer(SomeInteger::new(true, false))];
        let args = ArgumentsForTranslation::new(vec![s_int()], None, None);
        let scope = args.match_signature(&sig, Some(&defaults)).unwrap();
        assert_eq!(scope.len(), 2);
    }

    #[test]
    fn match_signature_errors_on_missing_arg() {
        // def f(a, b): ...   called as f()
        let sig = Signature::new(vec!["a".into(), "b".into()], None, None);
        let args = ArgumentsForTranslation::new(vec![], None, None);
        let err = args
            .match_signature(&sig, None)
            .expect_err("missing args must error");
        assert!(matches!(
            err,
            ArgErr::Count {
                missing_args: 2,
                ..
            }
        ));
    }

    #[test]
    fn match_signature_errors_on_too_many_without_vararg() {
        // def f(a): ...   called as f(1, 2)
        let sig = Signature::new(vec!["a".into()], None, None);
        let args = ArgumentsForTranslation::new(vec![s_int(), s_int()], None, None);
        let err = args
            .match_signature(&sig, None)
            .expect_err("too many args must error");
        assert!(matches!(err, ArgErr::Count { .. }));
    }

    #[test]
    fn match_signature_collects_stararg_into_tuple() {
        // def f(a, *args): ...   called as f(1, 2, 3)
        let sig = Signature::new(vec!["a".into()], Some("args".into()), None);
        let args = ArgumentsForTranslation::new(vec![s_int(), s_int(), s_int()], None, None);
        let scope = args.match_signature(&sig, None).unwrap();
        assert_eq!(scope.len(), 2);
        match &scope[1] {
            SomeValue::Tuple(t) => assert_eq!(t.items.len(), 2),
            other => panic!("expected stararg SomeTuple, got {other:?}"),
        }
    }

    #[test]
    fn match_signature_rejects_duplicate_keyword_and_positional() {
        // def f(a): ...   called as f(1, a=2)
        let sig = Signature::new(vec!["a".into()], None, None);
        let mut kws = HashMap::new();
        kws.insert("a".into(), s_int());
        let args = ArgumentsForTranslation::new(vec![s_int()], Some(kws), None);
        let err = args
            .match_signature(&sig, None)
            .expect_err("duplicate kwarg must error");
        assert!(matches!(err, ArgErr::MultipleValues { .. }));
    }

    #[test]
    fn argerr_unknown_kwds_picks_actually_unknown_name() {
        // upstream argument.py:237-245 — when exactly one keyword is
        // unmatched, `kwd_name` must be that unmatched name, not one
        // of the successfully-bound names. def f(a): ... called as
        // f(a=1, zzz=2) → `a` binds, `zzz` is the unknown kwarg.
        let sig = Signature::new(vec!["a".into()], None, None);
        let mut kws = HashMap::new();
        kws.insert("a".into(), s_int());
        kws.insert("zzz".into(), s_int());
        let args = ArgumentsForTranslation::new(Vec::new(), Some(kws), None);
        let err = args
            .match_signature(&sig, None)
            .expect_err("unknown kwarg must error");
        match err {
            ArgErr::UnknownKwds {
                num_kwds, kwd_name, ..
            } => {
                assert_eq!(num_kwds, 1);
                assert_eq!(kwd_name, "zzz");
            }
            other => panic!("expected UnknownKwds, got {other:?}"),
        }
    }

    #[test]
    fn argerr_count_getmsg_shapes() {
        // upstream argument.py:199-202: defcount == 0 and
        // !has_vararg() → msg1 = "exactly". Covers the common
        // "wrong number of positional arguments" case.
        let sig = Signature::new(vec!["a".into(), "b".into()], None, None);
        let err = ArgErr::Count {
            signature: sig,
            num_defaults: 0,
            missing_args: 1,
            num_args: 1,
            num_kwds: 0,
        };
        let msg = err.getmsg();
        assert!(msg.contains("exactly 2"));
        assert!(msg.contains("arguments"));
    }

    #[test]
    fn argerr_count_getmsg_at_least_with_defaults() {
        // def f(a, b=10): ...   called as f() — defcount=1,
        // missing_args=1 → "at least 1 non-keyword argument".
        let sig = Signature::new(vec!["a".into(), "b".into()], None, None);
        let err = ArgErr::Count {
            signature: sig,
            num_defaults: 1,
            missing_args: 1,
            num_args: 0,
            num_kwds: 0,
        };
        let msg = err.getmsg();
        assert!(msg.contains("at least"), "{msg}");
    }

    #[test]
    fn rawshape_free_fn_delegates() {
        let mut kws = HashMap::new();
        kws.insert("k".into(), s_int());
        let args = ArgumentsForTranslation::new(vec![s_int()], Some(kws), None);
        let shape = rawshape(&args);
        assert_eq!(shape.shape_cnt, 1);
        assert_eq!(shape.shape_keys, vec!["k".to_string()]);
        assert!(!shape.shape_star);
    }

    #[test]
    fn simple_args_builds_positional_only() {
        let args = simple_args(vec![s_int(), s_int()]);
        assert!(args.keywords.is_empty());
        assert!(args.w_stararg.is_none());
        assert_eq!(args.arguments_w.len(), 2);
    }

    #[test]
    fn fromshape_round_trip() {
        let mut kws = HashMap::new();
        kws.insert("k".into(), s_int());
        let args = ArgumentsForTranslation::new(vec![s_int()], Some(kws), None);
        let shape = args.rawshape();
        let data_w = {
            let mut v = args.arguments_w.clone();
            // Sort keys like flatten() does and append kw values in
            // that order.
            for k in &shape.shape_keys {
                v.push(args.keywords.get(k).unwrap().clone());
            }
            v
        };
        let recovered = ArgumentsForTranslation::fromshape(&shape, data_w);
        assert_eq!(recovered.arguments_w.len(), 1);
        assert_eq!(recovered.keywords.len(), 1);
    }
}
