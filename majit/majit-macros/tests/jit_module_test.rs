use majit_macros::jit_module;

#[jit_module]
mod basic_module {
    use majit_macros::{dont_look_inside, elidable};

    #[elidable]
    pub fn helper_square(x: i64) -> i64 {
        x * x
    }

    #[dont_look_inside]
    pub fn helper_opaque(a: i64, b: i64) -> i64 {
        a + b
    }

    pub fn not_jit(x: i64) -> i64 {
        x + 1
    }
}

#[test]
fn test_discovered_helpers_names() {
    let helpers = basic_module::__MAJIT_DISCOVERED_HELPERS;
    assert_eq!(helpers.len(), 2);
    assert!(helpers.contains(&"helper_square"));
    assert!(helpers.contains(&"helper_opaque"));
    // not_jit should not be discovered
    assert!(!helpers.contains(&"not_jit"));
}

#[test]
fn test_discovered_helper_policies() {
    let policies = basic_module::__MAJIT_HELPER_POLICIES;
    assert_eq!(policies.len(), 2);
    assert!(policies.contains(&("helper_square", "elidable")));
    assert!(policies.contains(&("helper_opaque", "dont_look_inside")));
}

#[test]
fn test_functions_still_callable() {
    assert_eq!(basic_module::helper_square(5), 25);
    assert_eq!(basic_module::helper_opaque(2, 3), 5);
    assert_eq!(basic_module::not_jit(10), 11);
}

#[jit_module]
mod empty_module {
    pub fn plain_fn() -> i64 {
        42
    }
}

#[test]
fn test_empty_discovery() {
    let helpers = empty_module::__MAJIT_DISCOVERED_HELPERS;
    assert!(helpers.is_empty());
    let policies = empty_module::__MAJIT_HELPER_POLICIES;
    assert!(policies.is_empty());
}

#[jit_module]
mod multi_attr_module {
    use majit_macros::{
        dont_look_inside, elidable, jit_loop_invariant, jit_may_force, jit_release_gil,
    };

    #[elidable]
    pub fn pure_fn(x: i64) -> i64 {
        x * 2
    }

    #[dont_look_inside]
    pub fn opaque_fn(x: i64) -> i64 {
        x + 1
    }

    #[jit_may_force]
    pub fn force_fn(x: i64) -> i64 {
        x - 1
    }

    #[jit_release_gil]
    pub fn gil_fn(x: i64) -> i64 {
        x * 3
    }

    #[jit_loop_invariant]
    pub fn invariant_fn(x: i64) -> i64 {
        x / 2
    }
}

#[test]
fn test_all_attribute_types_discovered() {
    let helpers = multi_attr_module::__MAJIT_DISCOVERED_HELPERS;
    assert_eq!(helpers.len(), 5);
    assert!(helpers.contains(&"pure_fn"));
    assert!(helpers.contains(&"opaque_fn"));
    assert!(helpers.contains(&"force_fn"));
    assert!(helpers.contains(&"gil_fn"));
    assert!(helpers.contains(&"invariant_fn"));
}

#[test]
fn test_all_attribute_policies() {
    let policies = multi_attr_module::__MAJIT_HELPER_POLICIES;
    assert_eq!(policies.len(), 5);
    assert!(policies.contains(&("pure_fn", "elidable")));
    assert!(policies.contains(&("opaque_fn", "dont_look_inside")));
    assert!(policies.contains(&("force_fn", "jit_may_force")));
    assert!(policies.contains(&("gil_fn", "jit_release_gil")));
    assert!(policies.contains(&("invariant_fn", "jit_loop_invariant")));
}

#[test]
fn test_multi_attr_functions_callable() {
    assert_eq!(multi_attr_module::pure_fn(5), 10);
    assert_eq!(multi_attr_module::opaque_fn(5), 6);
    assert_eq!(multi_attr_module::force_fn(5), 4);
    assert_eq!(multi_attr_module::gil_fn(5), 15);
    assert_eq!(multi_attr_module::invariant_fn(10), 5);
}
