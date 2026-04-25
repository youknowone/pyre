//! Port of `rpython/config/config.py`.
//!
//! Upstream is a 634-LOC dynamic configuration framework: an
//! `OptionDescription` schema tree + a `Config` runtime that holds
//! one value per option and supports dotted-path `.set(...)` /
//! `.override(...)` mutation keyed on the schema. Used by
//! `translator.driver.TranslationDriver.__init__` at `driver.py:70,
//! :82, :85` to hold translation-time options (`translation.verbose`
//! / `translation.backend` / …).
//!
//! Scope of this module:
//!
//! - Exception types (`ConfigError`, `ConflictConfigError`,
//!   `NoMatchingOptionFound`, `AmbigousOptionError`).
//! - [`Option`] trait + concrete [`BoolOption`], [`IntOption`],
//!   [`FloatOption`], [`StrOption`], [`ChoiceOption`],
//!   [`ArbitraryOption`].
//! - [`_getnegation`] helper.
//! - [`OptionDescription`] schema tree node.
//! - [`Config`] runtime state with `setoption` / `set` / `override` /
//!   `suggest` / `copy` / `getkey` / `getpaths` / `_freeze_` / iter.
//!
//! **DEFERRED** (not needed by `TranslationDriver.__init__` path —
//! all optparse-dependent):
//!
//! - `OptHelpFormatter`, `ConfigUpdate`, `BoolConfigUpdate` — the
//!   `optparse` integration path. Port when CLI driver code lands.
//! - `to_optparse()`, `make_dict()` — build an optparse parser from a
//!   config + dump config as dict.
//!
//! **Rust adaptations** (documented PRE-EXISTING-ADAPTATIONs):
//!
//! 1. **Dynamic attribute access**. Upstream `config.translation.verbose`
//!    routes through Python's `__getattr__` / `__setattr__` protocol
//!    (`config.py:64-102`). Rust has no such protocol. The port
//!    exposes the equivalent via [`Config::get`] / [`Config::set_value`]
//!    taking dotted-path `&str`s, plus a [`ConfigValue`] enum for the
//!    dynamic return type. Typed getters (`bool_value`, `int_value`,
//!    …) return `Result<T, ConfigError>` with a mismatched-type error
//!    rather than Python's runtime `TypeError`.
//!
//! 2. **Dict-keyed values**. Upstream stores option values in
//!    `_cfgimpl_values: dict[str, Any]`. The port uses
//!    `RefCell<HashMap<String, ValueOrSubConfig>>` with an
//!    [`OptionValue`] enum covering every concrete option type's
//!    value. Identity with upstream's `Any`-typed storage is preserved
//!    because every option can declare its expected value type
//!    through the enum discriminant.
//!
//! 3. **Parent back-reference**. Upstream `self._cfgimpl_parent` at
//!    `config.py:26` is a raw Python reference. The port uses a
//!    `Weak<Config>` to avoid a Rc-cycle between a nested Config and
//!    its parent.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt;
use std::rc::{Rc, Weak};

/// Upstream `config.py:8 AmbigousOptionError`. Thrown by
/// [`Config::set`] when a short-name (e.g. `gc`) matches multiple
/// full paths (e.g. `translation.gc` and `target.gc`). Name typo
/// `Ambigous` preserved for grep-parity.
#[derive(Debug, Clone)]
pub struct AmbigousOptionError {
    pub message: String,
}

impl fmt::Display for AmbigousOptionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for AmbigousOptionError {}

/// Upstream `config.py:11 NoMatchingOptionFound(AttributeError)`.
#[derive(Debug, Clone)]
pub struct NoMatchingOptionFound {
    pub message: String,
}

impl fmt::Display for NoMatchingOptionFound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for NoMatchingOptionFound {}

/// Upstream `config.py:14 ConfigError`. Generic config-layer error.
#[derive(Debug, Clone)]
pub enum ConfigError {
    /// Generic config error (upstream's `ConfigError` base class raise
    /// sites at `:232`, `:367`, `:388`, `:405`).
    Generic(String),
    /// Upstream `config.py:17 ConflictConfigError(ConfigError)`.
    /// Raised by [`Config::setoption`] when a user mutation would
    /// overwrite a non-default-non-suggested value.
    Conflict(String),
    /// Upstream `AmbigousOptionError` thrown from `Config.set`.
    Ambigous(String),
    /// Upstream `NoMatchingOptionFound` thrown from `Config.set`.
    NoMatch(String),
    /// Upstream raises `AttributeError` when accessing an unknown
    /// option (`config.py:78-79, :81-82, :105`).
    UnknownOption(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Generic(m)
            | ConfigError::Conflict(m)
            | ConfigError::Ambigous(m)
            | ConfigError::NoMatch(m)
            | ConfigError::UnknownOption(m) => write!(f, "{}", m),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Owner of a value, upstream's `config.py:35, :43, :70, :97, :115`
/// string: `"default"`, `"suggested"`, `"user"`, `"required"`,
/// `"cmdline"`. Kept as a typed enum; the upstream string is recovered
/// by [`Owner::as_str`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Owner {
    Default,
    Suggested,
    User,
    Required,
    Cmdline,
}

impl Owner {
    pub fn as_str(self) -> &'static str {
        match self {
            Owner::Default => "default",
            Owner::Suggested => "suggested",
            Owner::User => "user",
            Owner::Required => "required",
            Owner::Cmdline => "cmdline",
        }
    }
}

/// Typed value carried by a [`Config`] for one option. Mirrors
/// upstream's `_cfgimpl_values: dict[str, Any]` storage at
/// `config.py:27`; the enum discriminant carries the type information
/// Python's duck typing encodes implicitly.
#[derive(Debug, Clone)]
pub enum OptionValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Choice(String),
    /// Upstream `ArbitraryOption` holds any Python object. The port
    /// wraps it in `Rc<dyn Any>` so callers can downcast.
    Arbitrary(Rc<dyn Any>),
    /// Upstream `None` sentinel (the default of `BoolOption(default=None)`
    /// and `ChoiceOption(default=None)` slots). Distinguished from a
    /// missing entry in `_cfgimpl_values` — upstream's `None` is an
    /// observable default.
    None,
}

impl OptionValue {
    /// Upstream `config.py:110` compares the stored option value with
    /// the new value via Python's `==`. Used by [`Config::setoption`]
    /// to short-circuit a re-set with the same value (skipping the
    /// `ConflictConfigError` raise path).
    ///
    /// Mirrors Python's numeric / string equality across heterogeneous
    /// types: `True == 1`, `1 == 1.0`, `"abc" == "abc"` (Str↔Choice).
    /// `None` is only equal to `None`. `Arbitrary` falls back to
    /// `Rc::ptr_eq` because no `PartialEq` is required of the boxed
    /// payload.
    fn shallow_eq(&self, other: &OptionValue) -> bool {
        // Python coercion: `True == 1` is True, `False == 0` is True,
        // `True == 1.0` is True, `1 == 1.0` is True. Map every
        // numeric-or-bool pair through `f64` for the equality check
        // since `f64` covers the mixed-type lattice exactly the way
        // Python's numeric tower does. The fully qualified
        // `std::option::Option` avoids shadowing by the `Option`
        // trait at module scope.
        fn to_f64(v: &OptionValue) -> std::option::Option<f64> {
            match v {
                OptionValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
                OptionValue::Int(i) => Some(*i as f64),
                OptionValue::Float(f) => Some(*f),
                _ => None,
            }
        }
        if let (Some(a), Some(b)) = (to_f64(self), to_f64(other)) {
            return a == b;
        }
        match (self, other) {
            (OptionValue::Str(a), OptionValue::Str(b))
            | (OptionValue::Choice(a), OptionValue::Choice(b))
            | (OptionValue::Str(a), OptionValue::Choice(b))
            | (OptionValue::Choice(a), OptionValue::Str(b)) => a == b,
            (OptionValue::None, OptionValue::None) => true,
            (OptionValue::Arbitrary(a), OptionValue::Arbitrary(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// Result of [`Config::get`] — matches upstream's `__getattr__` that
/// returns either the option's value or a nested `Config` for a
/// subgroup.
#[derive(Clone)]
pub enum ConfigValue {
    Value(OptionValue),
    SubConfig(Rc<Config>),
}

impl fmt::Debug for ConfigValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigValue::Value(v) => f.debug_tuple("Value").field(v).finish(),
            ConfigValue::SubConfig(c) => f
                .debug_tuple("SubConfig")
                .field(&format_args!("<Config {}>", c._cfgimpl_descr._name))
                .finish(),
        }
    }
}

/// Contents of `_cfgimpl_values[name]` — an option value, or a
/// nested Config for a subgroup.
enum ValueOrSubConfig {
    Value(OptionValue),
    SubConfig(Rc<Config>),
}

/// A (path, value) pair used by `ChoiceOption._requires` / `_suggests`
/// and `BoolOption._requires` / `_suggests`. Upstream stores these as
/// Python tuples; the port names them for clarity.
#[derive(Clone, Debug)]
pub struct DependencyEdge {
    pub path: String,
    pub value: OptionValue,
}

impl DependencyEdge {
    pub fn new(path: impl Into<String>, value: OptionValue) -> Self {
        DependencyEdge {
            path: path.into(),
            value,
        }
    }
}

/// Upstream `config.py:213 Option` base class. The port uses a trait
/// instead of inheritance because the override points
/// (`setoption`, `validate`, `add_optparse_option`) naturally map to
/// trait methods. `getkey` / `convert_from_cmdline` retain upstream's
/// default bodies here.
pub trait Option {
    /// Upstream `self._name` at `config.py:217`.
    fn name(&self) -> &str;

    /// Upstream `self.doc` at `config.py:218`.
    fn doc(&self) -> &str;

    /// Upstream `self.cmdline`. `None` ↔ upstream `cmdline=None` (the
    /// option is NEVER exposed on the command line — e.g. `ArbitraryOption`
    /// at `:410`). `Some("")` ↔ upstream `DEFAULT_OPTION_NAME` sentinel
    /// (`:220`) which `to_optparse` translates into `--<name>`.
    /// `Some(other)` ↔ explicit override string.
    fn cmdline(&self) -> Option_<&str>;

    /// Upstream `Option.validate(self, value)` at `:221-222`. Default
    /// `NotImplementedError`; concrete types override.
    fn validate(&self, value: &OptionValue) -> bool;

    /// Upstream `Option.getdefault(self)` at `:224-225`. Override on
    /// `ArbitraryOption` to consult `defaultfactory`.
    fn getdefault(&self) -> OptionValue;

    /// Upstream `Option.setoption(self, config, value, who)` at
    /// `:227-233`. Validates + writes into
    /// `config._cfgimpl_values[name]`. Overrides:
    ///
    /// - `ChoiceOption.setoption` propagates `_requires` / `_suggests`.
    /// - `BoolOption.setoption` runs `_validator` + propagates
    ///   `_requires` / `_suggests` (boolean truthy side).
    /// - `IntOption.setoption` / `FloatOption.setoption` / `StrOption.setoption`
    ///   wrap `super().setoption(self, int(value)/float(value)/value, who)`
    ///   with `TypeError → ConfigError` remap.
    ///
    /// The `config: &Rc<Config>` receiver mirrors upstream's bound
    /// `self.config` argument threading — propagation helpers
    /// (`_cfgimpl_get_toplevel`, `_cfgimpl_get_home_by_path`) require
    /// owned `Rc<Config>`, so the trait passes it through.
    fn setoption(
        &self,
        config: &Rc<Config>,
        value: OptionValue,
        who: Owner,
    ) -> Result<(), ConfigError> {
        self.default_setoption(config, value, who)
    }

    /// Shared base-setoption path, equivalent to calling
    /// `Option.setoption(self, config, value, who)` in upstream. Used
    /// by concrete `*Option::setoption` overrides that want to preserve
    /// the base validation + write step after doing their propagation
    /// work.
    fn default_setoption(
        &self,
        config: &Rc<Config>,
        value: OptionValue,
        who: Owner,
    ) -> Result<(), ConfigError> {
        let name = self.name();
        // Upstream `:229-230`: `if who == "default" and value is None: pass`.
        let is_none_default = matches!(value, OptionValue::None) && who == Owner::Default;
        if !is_none_default && !self.validate(&value) {
            return Err(ConfigError::Generic(format!(
                "invalid value {:?} for option {}",
                value, name
            )));
        }
        config
            ._cfgimpl_values
            .borrow_mut()
            .insert(name.to_string(), ValueOrSubConfig::Value(value));
        Ok(())
    }

    /// Upstream `Option.getkey(self, value)` at `:235-236`.
    fn getkey(&self, value: OptionValue) -> OptionValue {
        value
    }

    /// Upstream `Option.convert_from_cmdline(self, value)` at `:238-239`.
    /// Default = identity; `ChoiceOption` override strips whitespace.
    fn convert_from_cmdline(&self, value: String) -> OptionValue {
        OptionValue::Str(value)
    }
}

/// Local alias so the module-level `Option` trait name doesn't clash
/// with `std::option::Option` in the doc-comment signature above.
/// Only used inside this module.
#[allow(non_camel_case_types)]
type Option_<T> = std::option::Option<T>;

/// Upstream `BoolOption(Option)` at `config.py:294-346`.
#[derive(Clone)]
pub struct BoolOption {
    _name: String,
    doc: String,
    cmdline: Option_<String>,
    default: Option_<bool>,
    pub _requires: Option_<Vec<DependencyEdge>>,
    pub _suggests: Option_<Vec<DependencyEdge>>,
    pub negation: bool,
    pub _validator: Option_<Rc<dyn Fn(&Config)>>,
}

impl fmt::Debug for BoolOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BoolOption")
            .field("_name", &self._name)
            .field("doc", &self.doc)
            .field("cmdline", &self.cmdline)
            .field("default", &self.default)
            .field("_requires", &self._requires)
            .field("_suggests", &self._suggests)
            .field("negation", &self.negation)
            .field("_validator", &self._validator.as_ref().map(|_| "<fn>"))
            .finish()
    }
}

impl BoolOption {
    /// Upstream `BoolOption.__init__` at `config.py:295-303`. The
    /// `cmdline` default matches upstream's `DEFAULT_OPTION_NAME`
    /// sentinel (`Some(String::new())` ↔ "use `--<name>"` auto-derived).
    pub fn new(name: impl Into<String>, doc: impl Into<String>) -> Self {
        BoolOption {
            _name: name.into(),
            doc: doc.into(),
            cmdline: Some(String::new()),
            default: None,
            _requires: None,
            _suggests: None,
            negation: true,
            _validator: None,
        }
    }

    pub fn with_default(mut self, default: bool) -> Self {
        self.default = Some(default);
        self
    }

    pub fn with_requires(mut self, requires: Vec<DependencyEdge>) -> Self {
        self._requires = Some(requires);
        self
    }

    pub fn with_suggests(mut self, suggests: Vec<DependencyEdge>) -> Self {
        self._suggests = Some(suggests);
        self
    }

    pub fn with_negation(mut self, negation: bool) -> Self {
        self.negation = negation;
        self
    }

    pub fn with_cmdline(mut self, cmdline: Option_<String>) -> Self {
        self.cmdline = cmdline;
        self
    }
}

impl Option for BoolOption {
    fn name(&self) -> &str {
        &self._name
    }
    fn doc(&self) -> &str {
        &self.doc
    }
    fn cmdline(&self) -> Option_<&str> {
        self.cmdline.as_deref()
    }

    /// Upstream `BoolOption.validate` at `:305-306`:
    /// `isinstance(value, bool)`.
    fn validate(&self, value: &OptionValue) -> bool {
        matches!(value, OptionValue::Bool(_))
    }

    fn getdefault(&self) -> OptionValue {
        match self.default {
            Some(v) => OptionValue::Bool(v),
            None => OptionValue::None,
        }
    }

    /// Upstream `BoolOption.setoption` at `:308-328`.
    fn setoption(
        &self,
        config: &Rc<Config>,
        value: OptionValue,
        who: Owner,
    ) -> Result<(), ConfigError> {
        // Upstream `:310-312`: run `self._validator(toplevel)` when
        // `value` is truthy.
        let truthy = matches!(value, OptionValue::Bool(true));
        if truthy {
            if let Some(validator) = &self._validator {
                let toplevel = config._cfgimpl_get_toplevel();
                validator(&toplevel);
            }
        }
        // Upstream `:313-321`: `_requires` propagation (truthy-only).
        if truthy {
            if let Some(requires) = &self._requires {
                for edge in requires {
                    let toplevel = config._cfgimpl_get_toplevel();
                    let (home, name) = toplevel._cfgimpl_get_home_by_path(&edge.path)?;
                    let who2 = if who == Owner::Default {
                        Owner::Default
                    } else {
                        Owner::Required
                    };
                    home.setoption(&name, edge.value.clone(), who2)?;
                }
            }
        }
        // Upstream `:322-326`: `_suggests` propagation (truthy-only).
        if truthy {
            if let Some(suggests) = &self._suggests {
                for edge in suggests {
                    let toplevel = config._cfgimpl_get_toplevel();
                    let (home, name) = toplevel._cfgimpl_get_home_by_path(&edge.path)?;
                    home.suggestoption(&name, edge.value.clone());
                }
            }
        }
        self.default_setoption(config, value, who)
    }
}

/// Upstream `IntOption(Option)` at `config.py:349-367`.
#[derive(Debug, Clone)]
pub struct IntOption {
    _name: String,
    doc: String,
    cmdline: Option_<String>,
    default: Option_<i64>,
}

impl IntOption {
    pub fn new(name: impl Into<String>, doc: impl Into<String>) -> Self {
        IntOption {
            _name: name.into(),
            doc: doc.into(),
            cmdline: Some(String::new()),
            default: None,
        }
    }
    pub fn with_default(mut self, default: i64) -> Self {
        self.default = Some(default);
        self
    }
    pub fn with_cmdline(mut self, cmdline: Option_<String>) -> Self {
        self.cmdline = cmdline;
        self
    }
}

impl Option for IntOption {
    fn name(&self) -> &str {
        &self._name
    }
    fn doc(&self) -> &str {
        &self.doc
    }
    fn cmdline(&self) -> Option_<&str> {
        self.cmdline.as_deref()
    }

    /// Upstream `IntOption.validate` at `:356-361`: `try: int(value);
    /// except TypeError: return False`. Python's `int(...)` accepts
    /// `bool`, `int`, `float` (truncating), and numeric strings (e.g.
    /// `int("42")`); raises `ValueError` for non-numeric strings (e.g.
    /// `int("abc")`).
    ///
    /// `ValueError` is NOT caught by upstream's `except TypeError:`,
    /// so it bubbles out of `validate` as a `ValueError`. The Rust
    /// port mirrors that lift point by parsing the `Str` here — a
    /// non-numeric string causes `validate` to return `false` rather
    /// than letting the failure surface later inside `setoption`.
    fn validate(&self, value: &OptionValue) -> bool {
        match value {
            OptionValue::Int(_) | OptionValue::Bool(_) | OptionValue::Float(_) => true,
            OptionValue::Str(s) => s.trim().parse::<i64>().is_ok(),
            _ => false,
        }
    }

    fn getdefault(&self) -> OptionValue {
        match self.default {
            Some(v) => OptionValue::Int(v),
            None => OptionValue::None,
        }
    }

    /// Upstream `IntOption.setoption` at `:363-367`: coerces via
    /// `int(value)` before delegating to the base `setoption`. Keeps
    /// storage normalised at `OptionValue::Int`.
    fn setoption(
        &self,
        config: &Rc<Config>,
        value: OptionValue,
        who: Owner,
    ) -> Result<(), ConfigError> {
        let coerced = match &value {
            OptionValue::Int(v) => OptionValue::Int(*v),
            OptionValue::Bool(v) => OptionValue::Int(if *v { 1 } else { 0 }),
            OptionValue::Float(v) => OptionValue::Int(*v as i64),
            OptionValue::Str(s) => match s.trim().parse::<i64>() {
                Ok(v) => OptionValue::Int(v),
                Err(_) => {
                    return Err(ConfigError::Generic(format!(
                        "invalid value {:?}, expected integer",
                        value
                    )));
                }
            },
            OptionValue::None if who == Owner::Default => OptionValue::None,
            _ => {
                return Err(ConfigError::Generic(format!(
                    "invalid value {:?}, expected integer",
                    value
                )));
            }
        };
        self.default_setoption(config, coerced, who)
    }
}

/// Upstream `FloatOption(Option)` at `config.py:370-388`.
#[derive(Debug, Clone)]
pub struct FloatOption {
    _name: String,
    doc: String,
    cmdline: Option_<String>,
    default: Option_<f64>,
}

impl FloatOption {
    pub fn new(name: impl Into<String>, doc: impl Into<String>) -> Self {
        FloatOption {
            _name: name.into(),
            doc: doc.into(),
            cmdline: Some(String::new()),
            default: None,
        }
    }
    pub fn with_default(mut self, default: f64) -> Self {
        self.default = Some(default);
        self
    }
    pub fn with_cmdline(mut self, cmdline: Option_<String>) -> Self {
        self.cmdline = cmdline;
        self
    }
}

impl Option for FloatOption {
    fn name(&self) -> &str {
        &self._name
    }
    fn doc(&self) -> &str {
        &self.doc
    }
    fn cmdline(&self) -> Option_<&str> {
        self.cmdline.as_deref()
    }

    /// Upstream `FloatOption.validate` at `:377-382`: `try: float(value);
    /// except TypeError: return False`. Python's `float(...)` accepts
    /// `bool`, `int`, `float`, and numeric strings; raises
    /// `ValueError` (not `TypeError`) on non-numeric strings, which
    /// bubbles out of `validate` unwrapped. The `Str` arm parses
    /// inline so the same lift point is preserved.
    fn validate(&self, value: &OptionValue) -> bool {
        match value {
            OptionValue::Float(_) | OptionValue::Int(_) | OptionValue::Bool(_) => true,
            OptionValue::Str(s) => s.trim().parse::<f64>().is_ok(),
            _ => false,
        }
    }

    fn getdefault(&self) -> OptionValue {
        match self.default {
            Some(v) => OptionValue::Float(v),
            None => OptionValue::None,
        }
    }

    /// Upstream `FloatOption.setoption` at `:384-388`: coerces via
    /// `float(value)` before delegating to the base `setoption`.
    fn setoption(
        &self,
        config: &Rc<Config>,
        value: OptionValue,
        who: Owner,
    ) -> Result<(), ConfigError> {
        let coerced = match &value {
            OptionValue::Float(v) => OptionValue::Float(*v),
            OptionValue::Int(v) => OptionValue::Float(*v as f64),
            OptionValue::Bool(v) => OptionValue::Float(if *v { 1.0 } else { 0.0 }),
            OptionValue::Str(s) => match s.trim().parse::<f64>() {
                Ok(v) => OptionValue::Float(v),
                Err(_) => {
                    return Err(ConfigError::Generic(format!(
                        "invalid value {:?}, expected float",
                        value
                    )));
                }
            },
            OptionValue::None if who == Owner::Default => OptionValue::None,
            _ => {
                return Err(ConfigError::Generic(format!(
                    "invalid value {:?}, expected float",
                    value
                )));
            }
        };
        self.default_setoption(config, coerced, who)
    }
}

/// Upstream `StrOption(Option)` at `config.py:391-405`.
#[derive(Debug, Clone)]
pub struct StrOption {
    _name: String,
    doc: String,
    cmdline: Option_<String>,
    default: Option_<String>,
}

impl StrOption {
    pub fn new(name: impl Into<String>, doc: impl Into<String>) -> Self {
        StrOption {
            _name: name.into(),
            doc: doc.into(),
            cmdline: Some(String::new()),
            default: None,
        }
    }
    pub fn with_default(mut self, default: impl Into<String>) -> Self {
        self.default = Some(default.into());
        self
    }
    pub fn with_cmdline(mut self, cmdline: Option_<String>) -> Self {
        self.cmdline = cmdline;
        self
    }
}

impl Option for StrOption {
    fn name(&self) -> &str {
        &self._name
    }
    fn doc(&self) -> &str {
        &self.doc
    }
    fn cmdline(&self) -> Option_<&str> {
        self.cmdline.as_deref()
    }

    /// Upstream `StrOption.validate` at `:398-399`:
    /// `isinstance(value, str)`.
    fn validate(&self, value: &OptionValue) -> bool {
        matches!(value, OptionValue::Str(_))
    }

    fn getdefault(&self) -> OptionValue {
        match &self.default {
            Some(v) => OptionValue::Str(v.clone()),
            None => OptionValue::None,
        }
    }

    /// Upstream `StrOption.setoption` at `:401-405`: base `setoption`
    /// guarded by `try: ... except TypeError`. No coercion — upstream
    /// passes `value` through unchanged and relies on `validate` to
    /// reject non-str.
    fn setoption(
        &self,
        config: &Rc<Config>,
        value: OptionValue,
        who: Owner,
    ) -> Result<(), ConfigError> {
        self.default_setoption(config, value, who)
    }
}

/// Upstream `ChoiceOption(Option)` at `config.py:249-284`.
#[derive(Debug, Clone)]
pub struct ChoiceOption {
    _name: String,
    doc: String,
    cmdline: Option_<String>,
    pub values: Vec<String>,
    default: Option_<String>,
    pub _requires: HashMap<String, Vec<DependencyEdge>>,
    pub _suggests: HashMap<String, Vec<DependencyEdge>>,
}

impl ChoiceOption {
    /// Upstream `ChoiceOption.__init__` at `:252-262`.
    pub fn new(name: impl Into<String>, doc: impl Into<String>, values: Vec<String>) -> Self {
        ChoiceOption {
            _name: name.into(),
            doc: doc.into(),
            cmdline: Some(String::new()),
            values,
            default: None,
            _requires: HashMap::new(),
            _suggests: HashMap::new(),
        }
    }
    pub fn with_default(mut self, default: impl Into<String>) -> Self {
        self.default = Some(default.into());
        self
    }
    pub fn with_requires(mut self, requires: HashMap<String, Vec<DependencyEdge>>) -> Self {
        self._requires = requires;
        self
    }
    pub fn with_suggests(mut self, suggests: HashMap<String, Vec<DependencyEdge>>) -> Self {
        self._suggests = suggests;
        self
    }
    pub fn with_cmdline(mut self, cmdline: Option_<String>) -> Self {
        self.cmdline = cmdline;
        self
    }
}

impl Option for ChoiceOption {
    fn name(&self) -> &str {
        &self._name
    }
    fn doc(&self) -> &str {
        &self.doc
    }
    fn cmdline(&self) -> Option_<&str> {
        self.cmdline.as_deref()
    }

    /// Upstream `ChoiceOption.validate` at `:280-281`:
    /// `value is None or value in self.values`.
    fn validate(&self, value: &OptionValue) -> bool {
        match value {
            OptionValue::None => true,
            OptionValue::Choice(s) | OptionValue::Str(s) => self.values.iter().any(|v| v == s),
            _ => false,
        }
    }

    fn getdefault(&self) -> OptionValue {
        match &self.default {
            Some(v) => OptionValue::Choice(v.clone()),
            None => OptionValue::None,
        }
    }

    /// Upstream `ChoiceOption.setoption` at `:264-278`.
    fn setoption(
        &self,
        config: &Rc<Config>,
        value: OptionValue,
        who: Owner,
    ) -> Result<(), ConfigError> {
        // Normalise `Str` to `Choice` for consistency before storage.
        let normalized = match &value {
            OptionValue::Str(s) => OptionValue::Choice(s.clone()),
            _ => value.clone(),
        };
        let value_key: Option_<&str> = match &normalized {
            OptionValue::Choice(s) => Some(s.as_str()),
            _ => None,
        };
        // Upstream `:266-273`: iterate `self._requires.get(value, [])`.
        if let Some(key) = value_key {
            if let Some(edges) = self._requires.get(key) {
                for edge in edges {
                    let toplevel = config._cfgimpl_get_toplevel();
                    let (home, name) = toplevel._cfgimpl_get_home_by_path(&edge.path)?;
                    let who2 = if who == Owner::Default {
                        Owner::Default
                    } else {
                        Owner::Required
                    };
                    home.setoption(&name, edge.value.clone(), who2)?;
                }
            }
            if let Some(edges) = self._suggests.get(key) {
                for edge in edges {
                    let toplevel = config._cfgimpl_get_toplevel();
                    let (home, name) = toplevel._cfgimpl_get_home_by_path(&edge.path)?;
                    home.suggestoption(&name, edge.value.clone());
                }
            }
        }
        self.default_setoption(config, normalized, who)
    }

    /// Upstream `ChoiceOption.convert_from_cmdline` at `:283-284`:
    /// `value.strip()`.
    fn convert_from_cmdline(&self, value: String) -> OptionValue {
        OptionValue::Choice(value.trim().to_string())
    }
}

/// Upstream `ArbitraryOption(Option)` at `config.py:408-425`.
///
/// `defaultfactory` is `dyn Fn() -> OptionValue` so any dynamically-
/// produced default is supported. The constructor asserts upstream's
/// invariant that `default is None` when `defaultfactory` is set.
#[derive(Clone)]
pub struct ArbitraryOption {
    _name: String,
    doc: String,
    default: Option_<OptionValue>,
    defaultfactory: Option_<Rc<dyn Fn() -> OptionValue>>,
}

impl fmt::Debug for ArbitraryOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArbitraryOption")
            .field("_name", &self._name)
            .field("doc", &self.doc)
            .field("default", &self.default)
            .field(
                "defaultfactory",
                &self.defaultfactory.as_ref().map(|_| "<fn>"),
            )
            .finish()
    }
}

impl ArbitraryOption {
    pub fn new(name: impl Into<String>, doc: impl Into<String>) -> Self {
        ArbitraryOption {
            _name: name.into(),
            doc: doc.into(),
            default: None,
            defaultfactory: None,
        }
    }
    pub fn with_default(mut self, default: OptionValue) -> Self {
        assert!(
            self.defaultfactory.is_none(),
            "upstream `config.py:414`: cannot set default when defaultfactory present"
        );
        self.default = Some(default);
        self
    }
    pub fn with_defaultfactory<F: Fn() -> OptionValue + 'static>(mut self, factory: F) -> Self {
        assert!(
            self.default.is_none(),
            "upstream `config.py:414`: cannot set defaultfactory when default present"
        );
        self.defaultfactory = Some(Rc::new(factory));
        self
    }
}

impl Option for ArbitraryOption {
    fn name(&self) -> &str {
        &self._name
    }
    fn doc(&self) -> &str {
        &self.doc
    }

    /// Upstream `cmdline=None` at `:410` — never exposed on the CLI.
    fn cmdline(&self) -> Option_<&str> {
        None
    }

    /// Upstream `ArbitraryOption.validate` at `:416-417`: always `True`.
    fn validate(&self, _value: &OptionValue) -> bool {
        true
    }

    /// Upstream `ArbitraryOption.getdefault` at `:422-425`.
    fn getdefault(&self) -> OptionValue {
        if let Some(factory) = &self.defaultfactory {
            return factory();
        }
        self.default.clone().unwrap_or(OptionValue::None)
    }
}

/// Upstream `config.py:287-292 _getnegation(optname)`. Maps CLI
/// flag names to their "off" counterpart:
///
/// - `without-foo` → `with-foo`
/// - `with-foo` → `without-foo`
/// - anything else → `no-<foo>`
pub fn _getnegation(optname: &str) -> String {
    if let Some(rest) = optname.strip_prefix("without") {
        return format!("with{}", rest);
    }
    if let Some(rest) = optname.strip_prefix("with") {
        return format!("without{}", rest);
    }
    format!("no-{}", optname)
}

/// A schema tree child — either a leaf [`Option`] or a nested
/// [`OptionDescription`].
#[derive(Clone)]
pub enum Child {
    Option(Rc<dyn Option>),
    Description(Rc<OptionDescription>),
}

impl fmt::Debug for Child {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Child::Option(o) => write!(f, "Option({})", o.name()),
            Child::Description(d) => write!(f, "Description({})", d._name),
        }
    }
}

impl Child {
    pub fn name(&self) -> &str {
        match self {
            Child::Option(o) => o.name(),
            Child::Description(d) => &d._name,
        }
    }
}

/// Upstream `OptionDescription` at `config.py:428-472`.
pub struct OptionDescription {
    pub _name: String,
    pub doc: String,
    pub _children: Vec<Child>,
    _children_by_name: HashMap<String, usize>,
}

impl fmt::Debug for OptionDescription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OptionDescription")
            .field("_name", &self._name)
            .field("doc", &self.doc)
            .field("_children", &self._children)
            .finish()
    }
}

impl OptionDescription {
    /// Upstream `OptionDescription.__init__(self, name, doc, children)`
    /// at `:433-437`. `_build()` at `:439-441` populates
    /// `setattr(self, child._name, child)`; the Rust port materialises
    /// the same by-name lookup as a precomputed `HashMap`.
    pub fn new(name: impl Into<String>, doc: impl Into<String>, children: Vec<Child>) -> Self {
        let mut by_name = HashMap::new();
        for (idx, child) in children.iter().enumerate() {
            by_name.insert(child.name().to_string(), idx);
        }
        OptionDescription {
            _name: name.into(),
            doc: doc.into(),
            _children: children,
            _children_by_name: by_name,
        }
    }

    /// Upstream `getattr(self, child._name)` resolution. Returns the
    /// matching child if any. Used by [`Config::setoption`] to find
    /// the Option or nested description under a given name.
    pub fn child(&self, name: &str) -> Option_<&Child> {
        self._children_by_name
            .get(name)
            .map(|&idx| &self._children[idx])
    }

    /// Upstream `OptionDescription.getkey(self, config)` at `:443-445`.
    pub fn getkey(&self, config: &Config) -> Vec<OptionValue> {
        self._children
            .iter()
            .map(|child| {
                let name = child.name();
                match child {
                    Child::Option(opt) => {
                        let value = config._get_stored_value(name);
                        opt.getkey(value)
                    }
                    Child::Description(_) => {
                        // Upstream recurses into the nested Config's
                        // getkey, which itself returns a tuple.
                        let sub = config._get_stored_sub(name);
                        OptionValue::Arbitrary(Rc::new(sub._cfgimpl_descr.getkey(&sub)))
                    }
                }
            })
            .collect()
    }

    /// Upstream `OptionDescription.getpaths(self, include_groups=False,
    /// currpath=None)` at `:450-472`.
    pub fn getpaths(&self, include_groups: bool) -> Vec<String> {
        let mut out = Vec::new();
        let mut currpath: Vec<String> = Vec::new();
        self.getpaths_inner(include_groups, &mut currpath, &mut out);
        out
    }

    fn getpaths_inner(
        &self,
        include_groups: bool,
        currpath: &mut Vec<String>,
        out: &mut Vec<String>,
    ) {
        for child in &self._children {
            let attr = child.name();
            if attr.starts_with("_cfgimpl") {
                continue;
            }
            match child {
                Child::Description(desc) => {
                    if include_groups {
                        let mut full = currpath.clone();
                        full.push(attr.to_string());
                        out.push(full.join("."));
                    }
                    currpath.push(attr.to_string());
                    desc.getpaths_inner(include_groups, currpath, out);
                    currpath.pop();
                }
                Child::Option(_) => {
                    let mut full = currpath.clone();
                    full.push(attr.to_string());
                    out.push(full.join("."));
                }
            }
        }
    }
}

/// Upstream `config.py:20-207 Config`.
///
/// Holds one value per leaf `Option` under a shared [`OptionDescription`]
/// schema. Nested `OptionDescription` children are materialised as
/// child `Config` instances (tree structure). `_cfgimpl_parent` is a
/// `Weak` backref so nested configs can walk to the toplevel without
/// introducing an Rc cycle.
pub struct Config {
    /// Upstream `self._cfgimpl_descr` at `config.py:24`.
    pub _cfgimpl_descr: Rc<OptionDescription>,
    /// Upstream `self._cfgimpl_value_owners` at `:25`.
    _cfgimpl_value_owners: RefCell<HashMap<String, Owner>>,
    /// Upstream `self._cfgimpl_parent` at `:26`.
    _cfgimpl_parent: Option_<Weak<Config>>,
    /// Upstream `self._cfgimpl_values` at `:27`.
    _cfgimpl_values: RefCell<HashMap<String, ValueOrSubConfig>>,
    /// Upstream `self._cfgimpl_warnings` at `:28`. Held on the
    /// toplevel only (nested configs route via
    /// `_cfgimpl_get_toplevel`).
    _cfgimpl_warnings: RefCell<Vec<String>>,
    /// Upstream `_cfgimpl_frozen` class attribute at `:21`, flipped by
    /// `_freeze_` at `:163-165`.
    _cfgimpl_frozen: Cell<bool>,
    /// Self-Rc set post-construction so descendants can hold a
    /// `Weak<Config>` to this node. See [`Config::new`].
    self_rc: RefCell<Weak<Config>>,
}

impl Config {
    /// Upstream `Config.__init__(self, descr, parent=None, **overrides)`
    /// at `config.py:23-29`.
    ///
    /// Two-stage construction: first build the `Rc<Config>`, then call
    /// `_cfgimpl_build` which recurses into nested descriptions (each
    /// nested `Config` needs a `Weak` backref to `self`, which only
    /// exists after the Rc is minted).
    pub fn new(
        descr: Rc<OptionDescription>,
        overrides: HashMap<String, OptionValue>,
    ) -> Result<Rc<Self>, ConfigError> {
        Self::_new_with_parent(descr, None, overrides)
    }

    fn _new_with_parent(
        descr: Rc<OptionDescription>,
        parent: Option_<Weak<Config>>,
        overrides: HashMap<String, OptionValue>,
    ) -> Result<Rc<Self>, ConfigError> {
        let this = Rc::new(Config {
            _cfgimpl_descr: descr,
            _cfgimpl_value_owners: RefCell::new(HashMap::new()),
            _cfgimpl_parent: parent,
            _cfgimpl_values: RefCell::new(HashMap::new()),
            _cfgimpl_warnings: RefCell::new(Vec::new()),
            _cfgimpl_frozen: Cell::new(false),
            self_rc: RefCell::new(Weak::new()),
        });
        *this.self_rc.borrow_mut() = Rc::downgrade(&this);
        this._cfgimpl_build(overrides)?;
        Ok(this)
    }

    fn _cfgimpl_build(
        self: &Rc<Self>,
        overrides: HashMap<String, OptionValue>,
    ) -> Result<(), ConfigError> {
        // Upstream `config.py:31-38`: walk children, seed defaults for
        // Option leaves, recurse into OptionDescription children as
        // nested Configs.
        let children: Vec<Child> = self._cfgimpl_descr._children.clone();
        for child in &children {
            match child {
                Child::Option(opt) => {
                    let default = opt.getdefault();
                    self._cfgimpl_values
                        .borrow_mut()
                        .insert(opt.name().to_string(), ValueOrSubConfig::Value(default));
                    self._cfgimpl_value_owners
                        .borrow_mut()
                        .insert(opt.name().to_string(), Owner::Default);
                }
                Child::Description(desc) => {
                    let nested = Config::_new_with_parent(
                        Rc::clone(desc),
                        Some(Rc::downgrade(self)),
                        HashMap::new(),
                    )?;
                    self._cfgimpl_values
                        .borrow_mut()
                        .insert(desc._name.clone(), ValueOrSubConfig::SubConfig(nested));
                }
            }
        }
        self.override_(overrides)?;
        Ok(())
    }

    /// Upstream `Config.override(self, overrides)` at `config.py:40-43`.
    /// Name suffixed with `_` to avoid the Rust `override` keyword
    /// (structural parity — same identifier stripped of the reserved-
    /// word collision, documented here).
    pub fn override_(
        self: &Rc<Self>,
        overrides: HashMap<String, OptionValue>,
    ) -> Result<(), ConfigError> {
        for (name, value) in overrides {
            let (home, final_name) = self._cfgimpl_get_home_by_path(&name)?;
            home.setoption(&final_name, value, Owner::Default)?;
        }
        Ok(())
    }

    /// Upstream `Config.setoption(self, name, value, who)` at
    /// `config.py:103-115`.
    pub fn setoption(
        self: &Rc<Self>,
        name: &str,
        value: OptionValue,
        who: Owner,
    ) -> Result<(), ConfigError> {
        if !self._cfgimpl_values.borrow().contains_key(name) {
            return Err(ConfigError::UnknownOption(format!(
                "unknown option {}",
                name
            )));
        }
        let child = self
            ._cfgimpl_descr
            .child(name)
            .ok_or_else(|| ConfigError::UnknownOption(format!("unknown option {}", name)))?;
        let opt = match child {
            Child::Option(o) => Rc::clone(o),
            Child::Description(_) => {
                return Err(ConfigError::Generic(format!(
                    "cannot setoption on a subgroup: {}",
                    name
                )));
            }
        };
        let oldowner = *self
            ._cfgimpl_value_owners
            .borrow()
            .get(name)
            .unwrap_or(&Owner::Default);
        if !matches!(oldowner, Owner::Default | Owner::Suggested) {
            let oldvalue = self._get_stored_value(name);
            if oldvalue.shallow_eq(&value) || matches!(who, Owner::Default | Owner::Suggested) {
                return Ok(());
            }
            return Err(ConfigError::Conflict(format!(
                "cannot override value to {:?} for option {}",
                value, name
            )));
        }
        let self_rc: Rc<Config> = Rc::clone(self);
        opt.setoption(&self_rc, value, who)?;
        self._cfgimpl_value_owners
            .borrow_mut()
            .insert(name.to_string(), who);
        Ok(())
    }

    /// Upstream `Config.suggest(self, **kwargs)` at `config.py:117-119`.
    pub fn suggest(
        self: &Rc<Self>,
        kwargs: HashMap<String, OptionValue>,
    ) -> Result<(), ConfigError> {
        for (name, value) in kwargs {
            self.suggestoption(&name, value);
        }
        Ok(())
    }

    /// Upstream `Config.suggestoption(self, name, value)` at `:121-127`.
    pub fn suggestoption(self: &Rc<Self>, name: &str, value: OptionValue) {
        // Upstream silently swallows `ConflictConfigError`.
        let _ = self.setoption(name, value, Owner::Suggested);
    }

    /// Upstream `Config.set(self, **kwargs)` at `config.py:129-143`.
    /// Resolves short names by matching the suffix against the full
    /// dotted paths returned by `getpaths()`.
    pub fn set(self: &Rc<Self>, kwargs: HashMap<String, OptionValue>) -> Result<(), ConfigError> {
        let all_paths: Vec<Vec<String>> = self
            .getpaths(false)
            .into_iter()
            .map(|p| p.split('.').map(|s| s.to_string()).collect())
            .collect();
        for (key, value) in kwargs {
            let key_parts: Vec<String> = key.split('.').map(|s| s.to_string()).collect();
            let candidates: Vec<&Vec<String>> = all_paths
                .iter()
                .filter(|p| {
                    p.len() >= key_parts.len() && p[p.len() - key_parts.len()..] == key_parts[..]
                })
                .collect();
            match candidates.len() {
                1 => {
                    let name = candidates[0].join(".");
                    let (home, final_name) = self._cfgimpl_get_home_by_path(&name)?;
                    home.setoption(&final_name, value, Owner::User)?;
                }
                n if n > 1 => {
                    return Err(ConfigError::Ambigous(format!(
                        "more than one option that ends with {}",
                        key
                    )));
                }
                _ => {
                    return Err(ConfigError::NoMatch(format!(
                        "there is no option that matches {}",
                        key
                    )));
                }
            }
        }
        Ok(())
    }

    /// Upstream `Config._cfgimpl_get_home_by_path(self, path)` at
    /// `config.py:145-150`. Returns `(subconfig, final_name)`.
    pub fn _cfgimpl_get_home_by_path(
        self: &Rc<Self>,
        path: &str,
    ) -> Result<(Rc<Config>, String), ConfigError> {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = Rc::clone(self);
        for step in &parts[..parts.len() - 1] {
            let values = current._cfgimpl_values.borrow();
            let entry = values.get(*step).ok_or_else(|| {
                ConfigError::UnknownOption(format!("unknown path segment {} in {}", step, path))
            })?;
            match entry {
                ValueOrSubConfig::SubConfig(sub) => {
                    let sub = Rc::clone(sub);
                    drop(values);
                    current = sub;
                }
                ValueOrSubConfig::Value(_) => {
                    return Err(ConfigError::UnknownOption(format!(
                        "{} is not a subgroup in path {}",
                        step, path
                    )));
                }
            }
        }
        Ok((current, parts[parts.len() - 1].to_string()))
    }

    /// Upstream `Config._cfgimpl_get_toplevel(self)` at `:152-155`.
    pub fn _cfgimpl_get_toplevel(self: &Rc<Self>) -> Rc<Config> {
        let mut current = Rc::clone(self);
        while let Some(parent_weak) = &current._cfgimpl_parent.clone() {
            match parent_weak.upgrade() {
                Some(parent) => {
                    current = parent;
                }
                None => break,
            }
        }
        current
    }

    /// Upstream `Config.add_warning(self, warning)` at `:157-158`.
    pub fn add_warning(self: &Rc<Self>, warning: impl Into<String>) {
        self._cfgimpl_get_toplevel()
            ._cfgimpl_warnings
            .borrow_mut()
            .push(warning.into());
    }

    /// Upstream `Config.get_warnings(self)` at `:160-161`.
    pub fn get_warnings(self: &Rc<Self>) -> Vec<String> {
        self._cfgimpl_get_toplevel()
            ._cfgimpl_warnings
            .borrow()
            .clone()
    }

    /// Upstream `Config._freeze_(self)` at `:163-165`.
    pub fn _freeze_(&self) -> bool {
        self._cfgimpl_frozen.set(true);
        true
    }

    /// Upstream `Config.getkey(self)` at `:167-168`.
    pub fn getkey(self: &Rc<Self>) -> Vec<OptionValue> {
        self._cfgimpl_descr.getkey(self)
    }

    /// Upstream `config.translation._cfgimpl_value_owners` access at
    /// e.g. `translationoption.py:355`. Returns a shallow clone of the
    /// current owners map — upstream's raw dict read is a fresh
    /// Python reference with no aliasing contract, so cloning preserves
    /// the observable semantics.
    pub fn _cfgimpl_value_owners(&self) -> HashMap<String, Owner> {
        self._cfgimpl_value_owners.borrow().clone()
    }

    /// Upstream `Config.getpaths(self, include_groups=False)` at
    /// `:204-207`.
    pub fn getpaths(&self, include_groups: bool) -> Vec<String> {
        self._cfgimpl_descr.getpaths(include_groups)
    }

    /// Upstream-equivalent of `config.foo` attribute access.
    /// Supports dotted paths (`config.get("translation.verbose")`).
    ///
    /// Returns [`ConfigValue::Value`] for leaf options and
    /// [`ConfigValue::SubConfig`] for `OptionDescription` children.
    pub fn get(self: &Rc<Self>, name: &str) -> Result<ConfigValue, ConfigError> {
        if name.contains('.') {
            let (home, final_name) = self._cfgimpl_get_home_by_path(name)?;
            return home.get(&final_name);
        }
        let values = self._cfgimpl_values.borrow();
        let entry = values.get(name).ok_or_else(|| {
            ConfigError::UnknownOption(format!("{} object has no attribute {}", "Config", name))
        })?;
        Ok(match entry {
            ValueOrSubConfig::Value(v) => ConfigValue::Value(v.clone()),
            ValueOrSubConfig::SubConfig(c) => ConfigValue::SubConfig(Rc::clone(c)),
        })
    }

    /// Upstream-equivalent of `config.foo = value`. Dotted paths
    /// supported. Internally dispatches through [`Config::setoption`]
    /// with `Owner::User` (upstream `__setattr__` at `:70`).
    pub fn set_value(self: &Rc<Self>, name: &str, value: OptionValue) -> Result<(), ConfigError> {
        // Upstream `__setattr__` at `:64-70`: frozen check, then
        // `setoption(name, value, 'user')`.
        if self._cfgimpl_frozen.get() {
            let stored = self.get(name)?;
            let equal = match (&stored, &value) {
                (ConfigValue::Value(a), b) => a.shallow_eq(b),
                _ => false,
            };
            if !equal {
                return Err(ConfigError::Generic(
                    "trying to change a frozen option object".to_string(),
                ));
            }
        }
        if name.contains('.') {
            let (home, final_name) = self._cfgimpl_get_home_by_path(name)?;
            return home.setoption(&final_name, value, Owner::User);
        }
        self.setoption(name, value, Owner::User)
    }

    /// Upstream `config._cfgimpl_values[child._name] = subgroup`
    /// at `translationoption.py:313` for the case where `subgroup`
    /// is itself a nested `Config`. Rebinds the slot to the supplied
    /// sub-config. Upstream's bare dict write does NOT update
    /// `subgroup._cfgimpl_parent` either — the grafted sub-Config
    /// keeps its original parent weak-ref, which only matters for
    /// `_cfgimpl_get_toplevel` (warnings routing). The Rust port
    /// reproduces the same shape: insert into the values map without
    /// touching `_cfgimpl_parent` (which is by-value, not RefCell —
    /// matching upstream's "no fix-up" semantics structurally).
    /// Owner is set to `Owner::User` because the upstream caller
    /// (`translationoption.py:308-313`) is splicing in a config value
    /// that came from outside the schema's defaults.
    pub fn set_subconfig(self: &Rc<Self>, name: &str, sub: Rc<Config>) -> Result<(), ConfigError> {
        if self._cfgimpl_frozen.get() {
            return Err(ConfigError::Generic(
                "trying to change a frozen option object".to_string(),
            ));
        }
        // Refuse if the slot doesn't exist or is a leaf option —
        // upstream's dict write would silently corrupt the schema in
        // the same scenario, so the Rust port surfaces it as an
        // error instead of papering over a structural mismatch.
        let already_subgroup = matches!(
            self._cfgimpl_values.borrow().get(name),
            Some(ValueOrSubConfig::SubConfig(_))
        );
        if !already_subgroup {
            return Err(ConfigError::Generic(format!(
                "set_subconfig: {} is not a subgroup slot in this Config",
                name
            )));
        }
        self._cfgimpl_values
            .borrow_mut()
            .insert(name.to_string(), ValueOrSubConfig::SubConfig(sub));
        self._cfgimpl_value_owners
            .borrow_mut()
            .insert(name.to_string(), Owner::User);
        Ok(())
    }

    /// Internal helper — read the raw `OptionValue` stored for `name`.
    /// Panics if the entry is missing or is a sub-config (caller bug;
    /// the public `get` covers both shapes).
    fn _get_stored_value(&self, name: &str) -> OptionValue {
        let values = self._cfgimpl_values.borrow();
        match values.get(name) {
            Some(ValueOrSubConfig::Value(v)) => v.clone(),
            Some(ValueOrSubConfig::SubConfig(_)) => {
                panic!("_get_stored_value: {} is a subgroup", name)
            }
            None => panic!("_get_stored_value: no entry for {}", name),
        }
    }

    /// Internal helper — read the nested `Config` for `name`.
    fn _get_stored_sub(&self, name: &str) -> Rc<Config> {
        let values = self._cfgimpl_values.borrow();
        match values.get(name) {
            Some(ValueOrSubConfig::SubConfig(c)) => Rc::clone(c),
            Some(ValueOrSubConfig::Value(_)) => {
                panic!("_get_stored_sub: {} is not a subgroup", name)
            }
            None => panic!("_get_stored_sub: no entry for {}", name),
        }
    }

    /// Upstream `Config.copy(self, as_default=False, parent=None)` at
    /// `:45-62`.
    pub fn copy(self: &Rc<Self>, as_default: bool) -> Rc<Self> {
        Self::_copy_with_parent(self, as_default, None)
    }

    fn _copy_with_parent(
        self: &Rc<Self>,
        as_default: bool,
        parent: Option_<Weak<Config>>,
    ) -> Rc<Self> {
        let new_values: RefCell<HashMap<String, ValueOrSubConfig>> = RefCell::new(HashMap::new());
        let new_owners: RefCell<HashMap<String, Owner>> = RefCell::new(HashMap::new());
        let result = Rc::new(Config {
            _cfgimpl_descr: Rc::clone(&self._cfgimpl_descr),
            _cfgimpl_value_owners: new_owners,
            _cfgimpl_parent: parent,
            _cfgimpl_values: new_values,
            _cfgimpl_warnings: RefCell::new(self._cfgimpl_warnings.borrow().clone()),
            _cfgimpl_frozen: Cell::new(self._cfgimpl_frozen.get()),
            self_rc: RefCell::new(Weak::new()),
        });
        *result.self_rc.borrow_mut() = Rc::downgrade(&result);
        for child in &self._cfgimpl_descr._children {
            match child {
                Child::Option(opt) => {
                    let value = self._get_stored_value(opt.name());
                    result
                        ._cfgimpl_values
                        .borrow_mut()
                        .insert(opt.name().to_string(), ValueOrSubConfig::Value(value));
                    let owner = if as_default {
                        Owner::Default
                    } else {
                        *self
                            ._cfgimpl_value_owners
                            .borrow()
                            .get(opt.name())
                            .unwrap_or(&Owner::Default)
                    };
                    result
                        ._cfgimpl_value_owners
                        .borrow_mut()
                        .insert(opt.name().to_string(), owner);
                }
                Child::Description(desc) => {
                    let sub = self._get_stored_sub(&desc._name);
                    let copied = sub._copy_with_parent(as_default, Some(Rc::downgrade(&result)));
                    result
                        ._cfgimpl_values
                        .borrow_mut()
                        .insert(desc._name.clone(), ValueOrSubConfig::SubConfig(copied));
                }
            }
        }
        result
    }

    /// Upstream `Config.__iter__(self)` at `:179-182`. Yields only leaf
    /// Option names + values, not subgroups.
    pub fn iter(self: &Rc<Self>) -> impl Iterator<Item = (String, OptionValue)> + '_ {
        let mut out: Vec<(String, OptionValue)> = Vec::new();
        for child in &self._cfgimpl_descr._children {
            if let Child::Option(opt) = child {
                let value = self._get_stored_value(opt.name());
                out.push((opt.name().to_string(), value));
            }
        }
        out.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build the canonical `translation.backend / type_system /
    /// verbose` schema subset used by `translator.driver` at
    /// `driver.py:70-82`.
    fn translation_descr() -> Rc<OptionDescription> {
        let backend = ChoiceOption::new(
            "backend",
            "backend name",
            vec!["c".to_string(), "cl".to_string()],
        );
        let type_system =
            ChoiceOption::new("type_system", "type system", vec!["lltype".to_string()]);
        let verbose = BoolOption::new("verbose", "verbose output").with_default(true);
        let gc = ChoiceOption::new(
            "gc",
            "garbage collector",
            vec![
                "boehm".to_string(),
                "ref".to_string(),
                "incminimark".to_string(),
            ],
        )
        .with_default("incminimark");
        let translation = OptionDescription::new(
            "translation",
            "translation options",
            vec![
                Child::Option(Rc::new(backend)),
                Child::Option(Rc::new(type_system)),
                Child::Option(Rc::new(verbose)),
                Child::Option(Rc::new(gc)),
            ],
        );
        Rc::new(OptionDescription::new(
            "root",
            "root",
            vec![Child::Description(Rc::new(translation))],
        ))
    }

    #[test]
    fn config_new_seeds_defaults() {
        // Upstream `config.py:31-38 _cfgimpl_build`: every Option child
        // receives its `.getdefault()` seed with owner `"default"`.
        let c = Config::new(translation_descr(), HashMap::new()).expect("config");
        let verbose = c.get("translation.verbose").expect("path");
        assert!(matches!(
            verbose,
            ConfigValue::Value(OptionValue::Bool(true))
        ));
    }

    #[test]
    fn config_set_short_name_resolves_unique_suffix() {
        // Upstream `config.py:129-143 set(**kwargs)`: `gc=boehm` matches
        // the full path `translation.gc` (the only path ending with
        // `gc`).
        let c = Config::new(translation_descr(), HashMap::new()).expect("config");
        let mut kwargs = HashMap::new();
        kwargs.insert("gc".to_string(), OptionValue::Choice("boehm".to_string()));
        c.set(kwargs).expect("set");
        let gc = c.get("translation.gc").expect("path");
        assert!(matches!(
            gc,
            ConfigValue::Value(OptionValue::Choice(ref s)) if s == "boehm"
        ));
    }

    #[test]
    fn config_override_sets_default_owner() {
        // Upstream `config.py:40-43 override`: writes with
        // `who='default'` so subsequent `setoption("user")` is NOT
        // blocked by ConflictConfigError (override is replacing the
        // default itself).
        let mut overrides = HashMap::new();
        overrides.insert("translation.verbose".to_string(), OptionValue::Bool(false));
        let c = Config::new(translation_descr(), overrides).expect("config");
        let v = c.get("translation.verbose").expect("path");
        assert!(matches!(v, ConfigValue::Value(OptionValue::Bool(false))));
    }

    #[test]
    fn setoption_user_then_conflicting_user_raises() {
        // Upstream `config.py:112-113` — raising `ConflictConfigError`.
        let c = Config::new(translation_descr(), HashMap::new()).expect("config");
        c.set_value("translation.verbose", OptionValue::Bool(false))
            .expect("first user set");
        let err = c
            .set_value("translation.verbose", OptionValue::Bool(true))
            .unwrap_err();
        assert!(matches!(err, ConfigError::Conflict(_)));
    }

    #[test]
    fn setoption_equal_value_is_idempotent() {
        // Upstream `:110-111`: `if oldvalue == value: return` — a
        // re-assignment of the same value is silently ignored.
        let c = Config::new(translation_descr(), HashMap::new()).expect("config");
        c.set_value("translation.verbose", OptionValue::Bool(false))
            .expect("first user");
        c.set_value("translation.verbose", OptionValue::Bool(false))
            .expect("repeat is OK");
    }

    #[test]
    fn suggestoption_is_silently_overridden_by_user() {
        // Upstream `:121-127 suggestoption`: `setoption(name, value,
        // "suggested")` swallows `ConflictConfigError`. A subsequent
        // user write must succeed because the prior suggestion did
        // not "lock" the value.
        let c = Config::new(translation_descr(), HashMap::new()).expect("config");
        c.suggestoption("translation.verbose", OptionValue::Bool(false));
        c.set_value("translation.verbose", OptionValue::Bool(true))
            .expect("user write after suggest");
    }

    #[test]
    fn getpaths_returns_leaf_options_in_declaration_order() {
        // Upstream `config.py:204-207 getpaths` + `:450-472`.
        let c = Config::new(translation_descr(), HashMap::new()).expect("config");
        let paths = c.getpaths(false);
        assert_eq!(
            paths,
            vec![
                "translation.backend".to_string(),
                "translation.type_system".to_string(),
                "translation.verbose".to_string(),
                "translation.gc".to_string(),
            ]
        );
    }

    #[test]
    fn getpaths_include_groups_lists_descriptions_too() {
        let c = Config::new(translation_descr(), HashMap::new()).expect("config");
        let paths = c.getpaths(true);
        assert!(paths.iter().any(|p| p == "translation"));
        assert!(paths.iter().any(|p| p == "translation.verbose"));
    }

    #[test]
    fn copy_as_default_resets_owners() {
        // Upstream `config.py:45-62 copy(as_default=True)`: every
        // owner resets to `"default"` so the copy accepts the first
        // user write without conflict.
        let c = Config::new(translation_descr(), HashMap::new()).expect("config");
        c.set_value("translation.verbose", OptionValue::Bool(false))
            .expect("user");
        let copy = c.copy(true);
        // The value carries over…
        let v = copy.get("translation.verbose").expect("path");
        assert!(matches!(v, ConfigValue::Value(OptionValue::Bool(false))));
        // …but the owner reset means we can user-write again.
        copy.set_value("translation.verbose", OptionValue::Bool(true))
            .expect("copy takes user write");
    }

    #[test]
    fn copy_preserves_owners_when_not_as_default() {
        let c = Config::new(translation_descr(), HashMap::new()).expect("config");
        c.set_value("translation.verbose", OptionValue::Bool(false))
            .expect("user");
        let copy = c.copy(false);
        // Second user write should still conflict — owner was
        // preserved.
        let err = copy
            .set_value("translation.verbose", OptionValue::Bool(true))
            .unwrap_err();
        assert!(matches!(err, ConfigError::Conflict(_)));
    }

    #[test]
    fn freeze_rejects_value_change() {
        // Upstream `config.py:64-66 __setattr__` frozen check.
        let c = Config::new(translation_descr(), HashMap::new()).expect("config");
        c._freeze_();
        let err = c
            .set_value("translation.verbose", OptionValue::Bool(false))
            .unwrap_err();
        assert!(matches!(err, ConfigError::Generic(ref m) if m.contains("frozen")));
    }

    #[test]
    fn choice_option_rejects_unknown_value() {
        // Upstream `:280-281 ChoiceOption.validate`.
        let c = Config::new(translation_descr(), HashMap::new()).expect("config");
        let err = c
            .set_value(
                "translation.gc",
                OptionValue::Choice("nosuchgc".to_string()),
            )
            .unwrap_err();
        assert!(matches!(err, ConfigError::Generic(_)));
    }

    #[test]
    fn getnegation_rewrites_cli_prefixes() {
        // Upstream `config.py:287-292 _getnegation`.
        assert_eq!(_getnegation("withthread"), "withoutthread");
        assert_eq!(_getnegation("withoutthread"), "withthread");
        assert_eq!(_getnegation("check"), "no-check");
    }

    #[test]
    fn bool_option_requires_fire_on_truthy_set() {
        // Upstream `config.py:313-321 BoolOption.setoption`: truthy
        // `_requires` propagate under owner "required" (not "default").
        let flag = BoolOption::new("flag", "switch").with_requires(vec![DependencyEdge::new(
            "backend",
            OptionValue::Choice("cl".to_string()),
        )]);
        let backend = ChoiceOption::new(
            "backend",
            "backend",
            vec!["c".to_string(), "cl".to_string()],
        );
        let root = Rc::new(OptionDescription::new(
            "root",
            "root",
            vec![
                Child::Option(Rc::new(flag)),
                Child::Option(Rc::new(backend)),
            ],
        ));
        let c = Config::new(root, HashMap::new()).expect("config");
        c.set_value("flag", OptionValue::Bool(true))
            .expect("flag=true");
        let b = c.get("backend").expect("backend");
        assert!(matches!(
            b,
            ConfigValue::Value(OptionValue::Choice(ref s)) if s == "cl"
        ));
    }

    #[test]
    fn config_set_ambigous_on_duplicate_suffix() {
        // Upstream `config.py:138-140 AmbigousOptionError`. Build a
        // schema with two paths ending in `gc` and probe.
        let a_gc = ChoiceOption::new("gc", "a gc", vec!["x".to_string()]).with_default("x");
        let b_gc = ChoiceOption::new("gc", "b gc", vec!["x".to_string()]).with_default("x");
        let a = OptionDescription::new("a", "", vec![Child::Option(Rc::new(a_gc))]);
        let b = OptionDescription::new("b", "", vec![Child::Option(Rc::new(b_gc))]);
        let root = Rc::new(OptionDescription::new(
            "root",
            "",
            vec![
                Child::Description(Rc::new(a)),
                Child::Description(Rc::new(b)),
            ],
        ));
        let c = Config::new(root, HashMap::new()).expect("config");
        let mut kwargs = HashMap::new();
        kwargs.insert("gc".to_string(), OptionValue::Choice("x".to_string()));
        let err = c.set(kwargs).unwrap_err();
        assert!(matches!(err, ConfigError::Ambigous(_)));
    }

    #[test]
    fn config_set_no_match_on_unknown_suffix() {
        // Upstream `:141-143 NoMatchingOptionFound`.
        let c = Config::new(translation_descr(), HashMap::new()).expect("config");
        let mut kwargs = HashMap::new();
        kwargs.insert("nosuch".to_string(), OptionValue::Bool(true));
        let err = c.set(kwargs).unwrap_err();
        assert!(matches!(err, ConfigError::NoMatch(_)));
    }

    #[test]
    fn arbitrary_option_defaultfactory_invoked_per_getdefault() {
        // Upstream `:422-425 ArbitraryOption.getdefault`: if
        // `defaultfactory` is set, call it every time.
        let opt =
            ArbitraryOption::new("data", "arbitrary").with_defaultfactory(|| OptionValue::Int(7));
        assert!(matches!(opt.getdefault(), OptionValue::Int(7)));
    }
}
