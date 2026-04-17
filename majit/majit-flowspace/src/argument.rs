//! Arguments objects.
//!
//! RPython basis: `rpython/flowspace/argument.py`.
//!
//! Partial port landed as part of Phase 2 F2.2 because
//! `bytecode.HostCode::signature` requires `Signature`. The sibling
//! `CallSpec` class, the `fromshape` classmethod, and the call-site
//! helpers land in Phase 3 F3.2 when the flow-space operation layer
//! needs them. Order matches upstream `argument.py`: `Signature` first,
//! then `CallSpec`.

/// Descriptor for a function's formal parameter list.
///
/// RPython basis: `rpython/flowspace/argument.py:Signature`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    pub argnames: Vec<String>,
    pub varargname: Option<String>,
    pub kwargname: Option<String>,
}

impl Signature {
    /// RPython: `Signature.__init__`.
    pub fn new(
        argnames: Vec<String>,
        varargname: Option<String>,
        kwargname: Option<String>,
    ) -> Self {
        Self {
            argnames,
            varargname,
            kwargname,
        }
    }

    /// RPython: `Signature.find_argname`.
    pub fn find_argname(&self, name: &str) -> i32 {
        match self.argnames.iter().position(|n| n == name) {
            Some(idx) => idx as i32,
            None => -1,
        }
    }

    /// RPython: `Signature.num_argnames`.
    pub fn num_argnames(&self) -> usize {
        self.argnames.len()
    }

    /// RPython: `Signature.has_vararg`.
    pub fn has_vararg(&self) -> bool {
        self.varargname.is_some()
    }

    /// RPython: `Signature.has_kwarg`.
    pub fn has_kwarg(&self) -> bool {
        self.kwargname.is_some()
    }

    /// RPython: `Signature.scope_length`.
    pub fn scope_length(&self) -> usize {
        self.argnames.len() + usize::from(self.has_vararg()) + usize::from(self.has_kwarg())
    }

    /// RPython: `Signature.getallvarnames`.
    pub fn getallvarnames(&self) -> Vec<String> {
        let mut out = self.argnames.clone();
        if let Some(ref v) = self.varargname {
            out.push(v.clone());
        }
        if let Some(ref k) = self.kwargname {
            out.push(k.clone());
        }
        out
    }
}

// RPython's `Signature` additionally implements `__len__` / `__getitem__`
// so it looks tuply for the annotator (argument.py:62-73). Rust's type
// system cannot index heterogeneous tuple-like access; the annotator port
// in Phase 5 handles this by pattern-matching on `Signature` directly
// rather than `s[0] / s[1] / s[2]`, so no Rust equivalent is emitted.

#[cfg(test)]
mod test {
    use super::*;

    // RPython basis: no upstream test for Signature alone — `test_argument.py`
    // exercises `CallSpec`. These smoke tests pin the behaviours that
    // `bytecode::HostCode::signature` consumes.

    #[test]
    fn signature_basic_arity() {
        let sig = Signature::new(vec!["x".into(), "y".into()], None, None);
        assert_eq!(sig.num_argnames(), 2);
        assert!(!sig.has_vararg());
        assert!(!sig.has_kwarg());
        assert_eq!(sig.scope_length(), 2);
        assert_eq!(sig.find_argname("y"), 1);
        assert_eq!(sig.find_argname("z"), -1);
    }

    #[test]
    fn signature_with_varargs_and_kwargs() {
        let sig = Signature::new(vec!["a".into()], Some("args".into()), Some("kw".into()));
        assert_eq!(sig.scope_length(), 3);
        assert_eq!(
            sig.getallvarnames(),
            vec!["a".to_string(), "args".into(), "kw".into()]
        );
    }
}
