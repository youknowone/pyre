//! Green field support for virtualizables.
//!
//! Mirrors RPython's `greenfield.py` GreenFieldInfo: tracks which fields
//! of a virtualizable are "green" (loop-invariant) and can be promoted
//! to constants during tracing.

/// greenfield.py:1-21 `class GreenFieldInfo(object)`.
///
/// ```python
/// class GreenFieldInfo(object):
///     def __init__(self, cpu, jd):
///         self.cpu = cpu
///         self.jitdriver_sd = jd
///         seen = set()
///         for name in jd.jitdriver.greens:
///             if '.' in name:
///                 objname, fieldname = name.split('.')
///                 seen.add(objname)
///         assert len(seen) == 1, ...
///         self.red_index = jd.jitdriver.reds.index(objname)
///         self.green_fields = jd.jitdriver.ll_greenfields.values()
///         self.green_field_descrs = [cpu.fielddescrof(GTYPE, fieldname)
///                                    for GTYPE, fieldname in self.green_fields]
/// ```
///
/// PRE-EXISTING-ADAPTATION: pyre has no `cpu` / `lltype` GTYPE handle,
/// so `green_fields` carries `(gtype_name: String, fieldname: String)`
/// and `green_field_descrs` carries the host-resolved descriptor
/// indices.  `jitdriver_sd` is omitted on the metainterp side because
/// the codewriter/metainterp jitdrivers_sd split (see
/// `CallControl::make_virtualizable_infos`) means this struct is built
/// without a back-pointer to the runtime jitdriver record.
#[derive(Debug, Clone)]
pub struct GreenFieldInfo {
    /// greenfield.py:14 `self.red_index = jd.jitdriver.reds.index(objname)`
    /// — slot in `jitdriver.reds` that holds the unique green-field
    /// owning object instance.
    pub red_index: usize,
    /// greenfield.py:18 `self.green_fields = jd.jitdriver.ll_greenfields.values()`.
    pub green_fields: Vec<(String, String)>,
    /// greenfield.py:19-20 `self.green_field_descrs = [cpu.fielddescrof(...)]`.
    pub green_field_descrs: Vec<u32>,
}

impl majit_codewriter::call::GreenFieldInfoHandle for GreenFieldInfo {
    /// `(GTYPE, fieldname) in self.green_fields` (call.py:391).
    fn contains_green_field(&self, gtype: &str, fieldname: &str) -> bool {
        self.green_fields
            .iter()
            .any(|(g, f)| g == gtype && f == fieldname)
    }
}

impl GreenFieldInfo {
    /// greenfield.py:2-19 — direct constructor analog.  Pyre supplies
    /// `red_index` + `green_fields` precomputed (the matching
    /// `objname` consistency check happens at the caller, mirroring
    /// the `assert len(seen) == 1` upstream).
    pub fn new(red_index: usize, green_fields: Vec<(String, String)>) -> Self {
        GreenFieldInfo {
            red_index,
            green_fields,
            green_field_descrs: Vec::new(),
        }
    }

    /// Empty constructor for sites that build the table incrementally
    /// (legacy callers from before the warmspot port).
    pub fn empty() -> Self {
        GreenFieldInfo {
            red_index: 0,
            green_fields: Vec::new(),
            green_field_descrs: Vec::new(),
        }
    }

    /// Register a field as green by its descriptor index.
    pub fn add_green_field(&mut self, descr_index: u32) {
        if !self.green_field_descrs.contains(&descr_index) {
            self.green_field_descrs.push(descr_index);
        }
    }

    /// Register a field as green by `(gtype, fieldname)` symbolic key
    /// — mirrors the upstream `ll_greenfields` population in
    /// `greenfield.py:18`.
    pub fn add_green_field_symbolic(&mut self, gtype: &str, fieldname: &str) {
        let pair = (gtype.to_string(), fieldname.to_string());
        if !self.green_fields.contains(&pair) {
            self.green_fields.push(pair);
        }
    }

    /// Check if a field is green.
    pub fn is_green_field(&self, descr_index: u32) -> bool {
        self.green_field_descrs.contains(&descr_index)
    }

    /// Number of green fields.
    pub fn num_green_fields(&self) -> usize {
        self.green_field_descrs.len()
    }
}
