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
fn test_helper_trace_fnaddrs_registry() {
    let trace_fnaddrs = basic_module::__majit_helper_trace_fnaddrs();
    assert_eq!(trace_fnaddrs.len(), 2);
    assert!(trace_fnaddrs.iter().any(|(path, addr)| {
        *path == concat!(module_path!(), "::basic_module::helper_square")
            && *addr == basic_module::__majit_call_target_helper_square as *const () as usize as i64
    }));
    assert!(trace_fnaddrs.iter().any(|(path, addr)| {
        *path == concat!(module_path!(), "::basic_module::helper_opaque")
            && *addr == basic_module::__majit_call_target_helper_opaque as *const () as usize as i64
    }));
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
    let trace_fnaddrs = empty_module::__majit_helper_trace_fnaddrs();
    assert!(trace_fnaddrs.is_empty());
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

// `#[jit_module]` discovers JIT helpers inside `impl` blocks. Both
// inherent (`impl Foo { fn ... }`) and trait-impl
// (`impl Trait for Foo { fn ... }`) methods land in the same registry
// via the structured `__majit_helper_impl_trace_fnaddrs()` registry,
// keyed by `impl_type_joined / method` that matches the parser's
// `self_ty_root` canonicalization (parse.rs:702, lib.rs:406-433) —
// RPython `call.py:174-187 getfunctionptr(graph)` parity.
//
// `#[unroll_safe]` is used here because it is one of the JIT attribute
// macros that does not generate out-of-scope trampolines; applying it
// on an impl method simply re-emits the method body.  Instance methods
// (`&self` / `&mut self` / `self`) are also exercised — Rust allows
// `<Type>::method as fn(&Type)` coercion, and RPython upstream treats
// `getfunctionptr(graph)` uniformly across free fns and methods.
#[jit_module]
mod impl_walk_module {
    use majit_macros::unroll_safe;

    pub struct Adder {
        pub value: i64,
    }

    impl Adder {
        #[unroll_safe]
        pub fn add(a: i64, b: i64) -> i64 {
            a + b
        }

        #[unroll_safe]
        pub fn bump(&self, x: i64) -> i64 {
            self.value + x
        }

        // No JIT attribute — must be skipped by discovery.
        pub fn ignore_me(_x: i64) -> i64 {
            0
        }
    }
}

#[test]
fn test_jit_module_discovers_impl_methods_including_receivers() {
    let helpers = impl_walk_module::__MAJIT_DISCOVERED_HELPERS;
    // Both the associated fn and the `&self` method must land in the
    // registry — discovery no longer excludes receiver methods.
    assert!(
        helpers.contains(&"Adder::add"),
        "free-signature impl method must be discovered, got {helpers:?}"
    );
    assert!(
        helpers.contains(&"Adder::bump"),
        "&self instance method must be discovered, got {helpers:?}"
    );
    assert!(!helpers.contains(&"Adder::ignore_me"));
    assert_eq!(helpers.len(), 2);

    let policies = impl_walk_module::__MAJIT_HELPER_POLICIES;
    assert!(policies.contains(&("Adder::add", "unroll_safe")));
    assert!(policies.contains(&("Adder::bump", "unroll_safe")));
}

#[test]
fn test_jit_module_emits_structured_impl_trace_fnaddrs() {
    let entries = impl_walk_module::__majit_helper_impl_trace_fnaddrs();
    // Entries are `(impl_type_joined, method, fnaddr)` triples.
    assert_eq!(entries.len(), 2);

    let add_entry = entries
        .iter()
        .find(|(ty, m, _)| *ty == "Adder" && *m == "add")
        .expect("Adder::add entry");
    assert_eq!(
        add_entry.2,
        impl_walk_module::Adder::add as *const () as usize as i64,
    );

    let bump_entry = entries
        .iter()
        .find(|(ty, m, _)| *ty == "Adder" && *m == "bump")
        .expect("Adder::bump entry");
    // Casting a `&self` associated fn to a plain `*const ()` works —
    // confirms Rust allows the coercion the reviewer called out.
    assert_eq!(
        bump_entry.2,
        impl_walk_module::Adder::bump as *const () as usize as i64,
    );
}
