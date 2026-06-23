//! RPython parity module for `rpython/jit/metainterp/gc.py`.
//!
//! This is the metainterp-level GC *description* module, not the backend GC
//! implementation. Allocation rewriting and collectors live in `majit-gc` and
//! the backend crates; this module mirrors PyPy's small `get_description`
//! settings surface.

use std::fmt;
use std::rc::Rc;

use majit_translate::config::config::{Config, ConfigError, ConfigValue, OptionValue};

/// gc.py:5-7 `GcDescription`.
#[derive(Clone)]
pub struct GcDescription {
    pub config: Rc<Config>,
    pub name: &'static str,
    pub malloc_zero_filled: bool,
}

impl fmt::Debug for GcDescription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GcDescription")
            .field("name", &self.name)
            .field("malloc_zero_filled", &self.malloc_zero_filled)
            .finish_non_exhaustive()
    }
}

impl GcDescription {
    fn new(config: &Rc<Config>, name: &'static str, malloc_zero_filled: bool) -> Self {
        Self {
            config: Rc::clone(config),
            name,
            malloc_zero_filled,
        }
    }
}

#[derive(Debug, Clone)]
pub enum GcDescriptionError {
    Config(ConfigError),
    Type(String),
    NotImplemented(String),
}

impl fmt::Display for GcDescriptionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(err) => err.fmt(f),
            Self::Type(message) | Self::NotImplemented(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for GcDescriptionError {}

impl From<ConfigError> for GcDescriptionError {
    fn from(value: ConfigError) -> Self {
        Self::Config(value)
    }
}

macro_rules! gc_class {
    ($name:ident, $gc_name:literal, $malloc_zero_filled:literal) => {
        #[allow(non_camel_case_types)]
        pub struct $name;

        impl $name {
            pub const NAME: &'static str = $gc_name;
            pub const MALLOC_ZERO_FILLED: bool = $malloc_zero_filled;

            pub fn describe(config: &Rc<Config>) -> GcDescription {
                GcDescription::new(config, Self::NAME, Self::MALLOC_ZERO_FILLED)
            }
        }
    };
}

gc_class!(GC_none, "none", true);
gc_class!(GC_boehm, "boehm", true);
gc_class!(GC_semispace, "semispace", true);
gc_class!(GC_generation, "generation", true);
gc_class!(GC_hybrid, "hybrid", true);
gc_class!(GC_minimark, "minimark", true);
gc_class!(GC_incminimark, "incminimark", false);

/// gc.py:25-31 `get_description`.
pub fn get_description(config: &Rc<Config>) -> Result<GcDescription, GcDescriptionError> {
    let name = match config.get("translation.gc")? {
        ConfigValue::Value(OptionValue::Choice(name))
        | ConfigValue::Value(OptionValue::Str(name)) => name,
        other => {
            return Err(GcDescriptionError::Type(format!(
                "translation.gc must be a string choice, got {other:?}"
            )));
        }
    };

    match name.as_str() {
        GC_none::NAME => Ok(GC_none::describe(config)),
        GC_boehm::NAME => Ok(GC_boehm::describe(config)),
        GC_semispace::NAME => Ok(GC_semispace::describe(config)),
        GC_generation::NAME => Ok(GC_generation::describe(config)),
        GC_hybrid::NAME => Ok(GC_hybrid::describe(config)),
        GC_minimark::NAME => Ok(GC_minimark::describe(config)),
        GC_incminimark::NAME => Ok(GC_incminimark::describe(config)),
        name => Err(GcDescriptionError::NotImplemented(format!(
            "GC {name:?} not supported by the JIT"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{GC_boehm, GC_incminimark, get_description};
    use majit_translate::config::config::OptionValue;
    use majit_translate::config::translationoption::get_combined_translation_config;

    fn translation_config_with_gc(
        gc: &str,
    ) -> std::rc::Rc<majit_translate::config::config::Config> {
        let mut overrides = HashMap::new();
        overrides.insert(
            "translation.gc".to_string(),
            OptionValue::Choice(gc.to_string()),
        );
        get_combined_translation_config(None, None, Some(overrides), false).expect("config")
    }

    #[test]
    fn get_description_reads_incminimark_zero_fill_policy() {
        let config = translation_config_with_gc("incminimark");
        let descr = get_description(&config).expect("description");
        assert_eq!(descr.name, GC_incminimark::NAME);
        assert!(!descr.malloc_zero_filled);
    }

    #[test]
    fn get_description_reads_boehm_zero_fill_policy() {
        let config = translation_config_with_gc("boehm");
        let descr = get_description(&config).expect("description");
        assert_eq!(descr.name, GC_boehm::NAME);
        assert!(descr.malloc_zero_filled);
    }

    #[test]
    fn get_description_rejects_default_ref_gc_name() {
        let config = get_combined_translation_config(None, None, None, false).expect("config");
        let err = get_description(&config).expect_err("ref GC must fail");
        assert_eq!(err.to_string(), "GC \"ref\" not supported by the JIT");
    }
}
