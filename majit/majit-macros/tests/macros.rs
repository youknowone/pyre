mod driver {
    use majit_macros::{
        dont_look_inside, elidable, elidable_cannot_raise, elidable_or_memerror, jit_driver,
    };

    #[jit_driver(greens = [pc, code], reds = [frame])]
    struct MyDriver;

    #[test]
    fn test_driver_greens() {
        assert_eq!(MyDriver::GREENS, &["pc", "code"]);
    }

    #[test]
    fn test_driver_reds() {
        assert_eq!(MyDriver::REDS, &["frame"]);
    }

    #[test]
    fn test_driver_num_greens() {
        assert_eq!(MyDriver::NUM_GREENS, 2);
    }

    #[test]
    fn test_driver_num_reds() {
        assert_eq!(MyDriver::NUM_REDS, 1);
    }

    #[test]
    fn test_driver_num_vars() {
        assert_eq!(MyDriver::NUM_VARS, 3);
    }

    #[jit_driver(greens = [pc], reds = [frame, stack])]
    struct SingleGreenDriver;

    #[test]
    fn test_single_green_driver() {
        assert_eq!(SingleGreenDriver::GREENS, &["pc"]);
        assert_eq!(SingleGreenDriver::REDS, &["frame", "stack"]);
        assert_eq!(SingleGreenDriver::NUM_GREENS, 1);
        assert_eq!(SingleGreenDriver::NUM_REDS, 2);
        assert_eq!(SingleGreenDriver::NUM_VARS, 3);
    }

    #[elidable]
    fn compute(x: i64) -> i64 {
        x * x + 1
    }

    #[test]
    fn test_elidable_function() {
        assert_eq!(compute(5), 26);
        assert_eq!(compute(0), 1);
        assert_eq!(compute(-3), 10);
        // EF_ELIDABLE_CAN_RAISE — call.py:297 `elif cr:` branch.
        let (policy, _, _, _, _, _) = __majit_call_policy_compute();
        assert_eq!(policy, 3u8);
    }

    #[elidable_cannot_raise]
    fn compute_pure(x: i64) -> i64 {
        x * 2
    }

    #[test]
    fn test_elidable_cannot_raise_function() {
        assert_eq!(compute_pure(7), 14);
        // EF_ELIDABLE_CANNOT_RAISE — call.py:299 `else` branch.
        let (policy, _, _, _, _, _) = __majit_call_policy_compute_pure();
        assert_eq!(policy, 19u8);
    }

    #[elidable_or_memerror]
    fn compute_memerror(x: i64) -> i64 {
        x + 100
    }

    #[test]
    fn test_elidable_or_memerror_function() {
        assert_eq!(compute_memerror(7), 107);
        // EF_ELIDABLE_OR_MEMORYERROR — call.py:295 `if cr == "mem":`.
        let (policy, _, _, _, _, _) = __majit_call_policy_compute_memerror();
        assert_eq!(policy, 20u8);
    }

    #[dont_look_inside]
    fn opaque_call(x: i64, y: i64) -> i64 {
        x + y
    }

    #[test]
    fn test_opaque_function() {
        assert_eq!(opaque_call(2, 3), 5);
        assert_eq!(opaque_call(-1, 1), 0);
    }
}

mod jit_module {
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
                && *addr
                    == basic_module::__majit_call_target_helper_square as *const () as usize as i64
        }));
        assert!(trace_fnaddrs.iter().any(|(path, addr)| {
            *path == concat!(module_path!(), "::basic_module::helper_opaque")
                && *addr
                    == basic_module::__majit_call_target_helper_opaque as *const () as usize as i64
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
        #[expect(
            dead_code,
            reason = "plain function is intentionally skipped by jit_module discovery"
        )]
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
            dont_look_inside, elidable, jit_may_force, jit_release_gil, loop_invariant,
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

        // `#[loop_invariant]` — RPython canonical `rlib/jit.py:162
        // @loop_invariant`.  `#[jit_loop_invariant]` is a pyre-prefixed
        // alias kept for back-compat (exercised separately in
        // `rpython_attribute_name_module`).
        #[loop_invariant]
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
        assert!(policies.contains(&("invariant_fn", "loop_invariant")));
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
            #[expect(
                dead_code,
                reason = "plain method is intentionally skipped by jit_module discovery"
            )]
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
        // Entries are
        //   `(module_path_with_crate, impl_type_as_written, method, fnaddr)`
        // 4-tuples. The codewriter applies
        // `CallControl::register_macro_impl_helper_trace_fnaddr`'s
        // module-prefix-qualification rule using `module_path_with_crate` to
        // decide whether to prepend the module prefix.
        assert_eq!(entries.len(), 2);

        let expected_module_path = concat!(module_path!(), "::impl_walk_module");

        let add_entry = entries
            .iter()
            .find(|(_, ty, m, _)| *ty == "Adder" && *m == "add")
            .expect("Adder::add entry");
        assert_eq!(add_entry.0, expected_module_path);
        // Rust does not guarantee that two `<Type>::method as *const ()`
        // casts at different call sites yield the same numeric address —
        // release-mode LLVM may instantiate the fn item per cast site
        // (debug mode incidentally produces a single instance). Verify the
        // recorded address is callable as the right function instead of
        // comparing it against an independent cast.
        let add_fn: fn(i64, i64) -> i64 = unsafe { std::mem::transmute(add_entry.3 as *const ()) };
        assert_eq!(add_fn(3, 4), impl_walk_module::Adder::add(3, 4));

        let bump_entry = entries
            .iter()
            .find(|(_, ty, m, _)| *ty == "Adder" && *m == "bump")
            .expect("Adder::bump entry");
        assert_eq!(bump_entry.0, expected_module_path);
        // `&self` associated fn lowers to `fn(&Adder, i64) -> i64` — the
        // cast through `*const ()` is still valid (the reviewer's
        // structural concern); verify functionally for release-mode
        // address stability.
        let adder = impl_walk_module::Adder { value: 10 };
        let bump_fn: fn(&impl_walk_module::Adder, i64) -> i64 =
            unsafe { std::mem::transmute(bump_entry.3 as *const ()) };
        assert_eq!(bump_fn(&adder, 5), adder.bump(5));
    }

    // Trait-impl disambiguation: when a type implements a trait method
    // whose name could collide with an inherent method (or with another
    // trait's method), the macro must emit `<Type as Trait>::method` to
    // keep the fnaddr cast unambiguous. RPython `getfunctionptr(graph)`
    // (call.py:174) does not need this because it uses graph identity;
    // pyre's Rust-layer registry does need it.
    pub trait NameCollider {
        fn conflict(&self) -> i64;
    }

    #[jit_module]
    mod trait_impl_module {
        use super::NameCollider;
        use majit_macros::unroll_safe;

        pub struct Widget {
            pub value: i64,
        }

        // Inherent method with the same name as the trait's — without
        // `<Widget as NameCollider>::conflict` disambiguation the fnaddr
        // cast would be rejected by rustc.
        impl Widget {
            #[unroll_safe]
            pub fn conflict(&self) -> i64 {
                self.value * 10
            }
        }

        impl NameCollider for Widget {
            #[unroll_safe]
            fn conflict(&self) -> i64 {
                self.value + 1
            }
        }
    }

    #[test]
    fn test_jit_module_disambiguates_trait_impl_with_as_trait_cast() {
        let entries = trait_impl_module::__majit_helper_impl_trace_fnaddrs();
        assert_eq!(entries.len(), 2, "both methods discovered: {entries:?}");

        let widget = trait_impl_module::Widget { value: 3 };
        // Find both entries.
        let mut seen_inherent = false;
        let mut seen_trait = false;
        for (_module_path, ty, method, fnaddr) in &entries {
            assert_eq!(*ty, "Widget");
            assert_eq!(*method, "conflict");
            // The fnaddr is one of the two method addresses; call through
            // the address to verify which.
            let fp: fn(&trait_impl_module::Widget) -> i64 =
                unsafe { std::mem::transmute(*fnaddr as usize) };
            let result = fp(&widget);
            if result == 30 {
                seen_inherent = true;
            } else if result == 4 {
                seen_trait = true;
            } else {
                panic!("unexpected conflict() result {result}");
            }
        }
        assert!(seen_inherent, "inherent Widget::conflict entry missing");
        assert!(
            seen_trait,
            "<Widget as NameCollider>::conflict entry missing"
        );
    }

    // ── `#[jit_elidable]` impl-method attachment ──────────────────────
    //
    // `#[elidable]` / `#[elidable_cannot_raise]` and friends emit a
    // module-level trampoline (`__majit_call_target_*`), so they cannot be
    // attached inside an `impl` block (probe result: `not found in this
    // scope`).  `#[jit_elidable]` (lib.rs:993) is a pure pass-through and
    // only relies on `front::llbc_hints`'s hint flip (the
    // `_elidable_function_` marker const → `elidable`), so it can safely
    // sit on impl methods.  This fixture verifies that live wire.
    pub trait PureTrait {
        fn trait_elidable(&self) -> i64;
    }

    #[jit_module]
    mod elidable_method_module {
        use super::PureTrait;
        use majit_macros::jit_elidable;

        pub struct PureCalc {
            pub seed: i64,
        }

        impl PureCalc {
            // Free-style impl method (no receiver).
            #[jit_elidable]
            pub fn compute_xor(a: i64, b: i64) -> i64 {
                a ^ b
            }

            // `&self` instance method.
            #[jit_elidable]
            pub fn shifted(&self, x: i64) -> i64 {
                self.seed.wrapping_add(x)
            }
        }

        impl PureTrait for PureCalc {
            #[jit_elidable]
            fn trait_elidable(&self) -> i64 {
                self.seed ^ 0x55
            }
        }
    }

    #[test]
    fn test_jit_elidable_on_impl_methods_is_discovered() {
        let helpers = elidable_method_module::__MAJIT_DISCOVERED_HELPERS;
        assert!(
            helpers.contains(&"PureCalc::compute_xor"),
            "free-style elidable method must be discovered, got {helpers:?}"
        );
        assert!(
            helpers.contains(&"PureCalc::shifted"),
            "&self elidable method must be discovered, got {helpers:?}"
        );
        assert!(
            helpers.contains(&"PureCalc::trait_elidable"),
            "trait-impl elidable method must be discovered, got {helpers:?}"
        );

        let policies = elidable_method_module::__MAJIT_HELPER_POLICIES;
        // jit_module records the raw attribute name; normalisation happens
        // in `front::llbc_hints`, which flips the harvested
        // `_elidable_function_` marker const to the canonical "elidable"
        // hint before mark_elidable consumes it.
        assert!(policies.contains(&("PureCalc::compute_xor", "jit_elidable")));
        assert!(policies.contains(&("PureCalc::shifted", "jit_elidable")));
        assert!(policies.contains(&("PureCalc::trait_elidable", "jit_elidable")));
    }

    #[test]
    fn test_jit_elidable_method_runtime_callable() {
        // `#[jit_elidable]` is pass-through, so runtime behaviour is unchanged.
        assert_eq!(
            elidable_method_module::PureCalc::compute_xor(0xff, 0x0f),
            0xf0
        );
        let calc = elidable_method_module::PureCalc { seed: 100 };
        assert_eq!(calc.shifted(7), 107);
        assert_eq!(PureTrait::trait_elidable(&calc), 49);
    }

    // ── pass-through attribute on a *free* fn under `#[jit_module]` ───
    //
    // `#[jit_elidable]` (lib.rs:1008) is a pure pass-through and emits no
    // `__majit_call_policy_<name>()` trampoline.  `__majit_helper_trace_fnaddrs()`
    // must therefore record the function's direct address (impl_addr_expr's
    // pass-through branch) instead of routing through the missing policy fn —
    // without that branch the `#[jit_module]` expansion fails to compile with
    // `cannot find function __majit_call_policy_*`.  Other pass-through attrs
    // (`#[unroll_safe]`, `#[not_in_trace]`) must take the same direct path; the
    // fixture exercises one of each to pin the static surface in tests.
    #[jit_module]
    mod passthrough_free_fn_module {
        use majit_macros::{jit_elidable, not_in_trace, unroll_safe};

        #[jit_elidable]
        pub fn pure_xor(a: i64, b: i64) -> i64 {
            a ^ b
        }

        #[unroll_safe]
        pub fn unrolled(x: i64) -> i64 {
            x.wrapping_add(1)
        }

        #[not_in_trace]
        pub fn out_of_trace(x: i64) -> i64 {
            x.wrapping_mul(2)
        }
    }

    #[test]
    fn test_passthrough_free_fn_discovery_uses_direct_fn_address() {
        let helpers = passthrough_free_fn_module::__MAJIT_DISCOVERED_HELPERS;
        assert!(helpers.contains(&"pure_xor"));
        assert!(helpers.contains(&"unrolled"));
        assert!(helpers.contains(&"out_of_trace"));

        let policies = passthrough_free_fn_module::__MAJIT_HELPER_POLICIES;
        // jit_module records the raw attribute name; normalisation lives
        // in `front::llbc_hints`.
        assert!(policies.contains(&("pure_xor", "jit_elidable")));
        assert!(policies.contains(&("unrolled", "unroll_safe")));
        assert!(policies.contains(&("out_of_trace", "not_in_trace")));

        let trace_fnaddrs = passthrough_free_fn_module::__majit_helper_trace_fnaddrs();
        assert_eq!(trace_fnaddrs.len(), 3);
        // No `__majit_call_policy_*` exists for any of these, so the only
        // legal address is the function's direct cast. Verify functionally
        // by invoking through the recorded address — Rust does not
        // guarantee numeric equality across independent fn-item casts in
        // release mode (LLVM may instantiate the fn item per cast site),
        // so a structural assertion via fn pointer equality is unreliable.
        let pure_xor_entry = trace_fnaddrs
            .iter()
            .find(|(p, _)| *p == concat!(module_path!(), "::passthrough_free_fn_module::pure_xor"))
            .expect("pure_xor entry");
        let pure_xor_fn: fn(i64, i64) -> i64 =
            unsafe { std::mem::transmute(pure_xor_entry.1 as *const ()) };
        assert_eq!(
            pure_xor_fn(0b1010, 0b0110),
            passthrough_free_fn_module::pure_xor(0b1010, 0b0110),
        );

        let unrolled_entry = trace_fnaddrs
            .iter()
            .find(|(p, _)| *p == concat!(module_path!(), "::passthrough_free_fn_module::unrolled"))
            .expect("unrolled entry");
        let unrolled_fn: fn(i64) -> i64 =
            unsafe { std::mem::transmute(unrolled_entry.1 as *const ()) };
        assert_eq!(unrolled_fn(7), passthrough_free_fn_module::unrolled(7));

        let out_of_trace_entry = trace_fnaddrs
            .iter()
            .find(|(p, _)| {
                *p == concat!(module_path!(), "::passthrough_free_fn_module::out_of_trace")
            })
            .expect("out_of_trace entry");
        let out_of_trace_fn: fn(i64) -> i64 =
            unsafe { std::mem::transmute(out_of_trace_entry.1 as *const ()) };
        assert_eq!(
            out_of_trace_fn(5),
            passthrough_free_fn_module::out_of_trace(5),
        );
    }

    #[test]
    fn test_passthrough_free_fn_callable() {
        assert_eq!(passthrough_free_fn_module::pure_xor(0b1010, 0b0110), 0b1100);
        assert_eq!(passthrough_free_fn_module::unrolled(7), 8);
        assert_eq!(passthrough_free_fn_module::out_of_trace(5), 10);
    }

    mod release_gil_aroundstate_module {
        use majit_macros::jit_release_gil;

        #[jit_release_gil]
        pub fn release_default(x: i64) -> i64 {
            x + 1
        }

        #[jit_release_gil(save_err = 5)]
        pub fn release_with_errno(x: i64) -> i64 {
            x + 2
        }
    }

    mod rpython_attribute_name_module {
        use majit_macros::{
            dont_look_inside, dont_look_inside_cannot_raise, elidable, elidable_cannot_raise,
            elidable_or_memerror, jit_loop_invariant, loop_invariant, unroll_safe,
        };

        #[elidable]
        pub fn pure_plain(x: i64) -> i64 {
            x + 1
        }

        #[elidable_cannot_raise]
        pub fn pure_cannot_raise(x: i64) -> i64 {
            x + 2
        }

        #[elidable_or_memerror]
        pub fn pure_or_memerror(x: i64) -> i64 {
            x + 3
        }

        #[dont_look_inside]
        pub fn opaque_plain(x: i64) -> i64 {
            x + 4
        }

        #[dont_look_inside_cannot_raise]
        pub fn opaque_cannot_raise(x: i64) -> i64 {
            x + 5
        }

        #[jit_loop_invariant]
        pub fn invariant_jit(x: i64) -> i64 {
            x + 6
        }

        #[loop_invariant]
        pub fn invariant_plain(x: i64) -> i64 {
            x + 7
        }

        #[unroll_safe]
        pub fn unrolled_helper(x: i64) -> i64 {
            x + 9
        }
    }

    /// RPython attribute-name parity: every annotated function carries a
    /// `pub const <attribute_name>_<NAME>: <ty> = <value>` next to it,
    /// matching upstream's wrapper-attached attribute model.  `rg
    /// <attribute>` must find both pyre and PyPy callsites under the same
    /// identifier.
    #[test]
    fn test_rpython_attribute_name_parity() {
        use rpython_attribute_name_module::*;

        // `rlib/jit.py:72 _elidable_function_ = True` — all three elidable
        // variants normalise to the same upstream attribute name (the
        // `_cannot_raise` / `_or_memerror` distinctions are codewriter-
        // derived `EffectInfo` classes per `call.py:292-299`).
        assert!(_elidable_function_pure_plain);
        assert!(_elidable_function_pure_cannot_raise);
        assert!(_elidable_function_pure_or_memerror);
        // Methods (`self`-receiver) skip module-level const emission —
        // `rpython_attribute_const_for`'s receiver guard avoids
        // trait-impl associated-item conflicts.  RPython's
        // `func._elidable_function_` parity at method scope is left for
        // a follow-up slice that knows the surrounding `impl` context.

        // `rlib/jit.py:139 _jit_look_inside_ = False` — both opaque
        // variants share the upstream attribute.
        assert!(!_jit_look_inside_opaque_plain);
        assert!(!_jit_look_inside_opaque_cannot_raise);

        // `rlib/jit.py:169 _jit_loop_invariant_ = True` — both
        // `loop_invariant` and `jit_loop_invariant` (the latter is a pyre
        // alias for the unprefixed name) share the upstream attribute.
        assert!(_jit_loop_invariant_invariant_jit);
        assert!(_jit_loop_invariant_invariant_plain);

        // `rlib/jit.py:159 _jit_unroll_safe_ = True`.
        assert!(_jit_unroll_safe_unrolled_helper);
    }

    mod look_inside_alias_module {
        use majit_macros::{look_inside, purefunction, purefunction_promote};

        // `rlib/jit.py:142-150 @look_inside` sets `_jit_look_inside_ =
        // True` (line 148) — the opposite of `@dont_look_inside`
        // (`_jit_look_inside_ = False`).
        #[look_inside]
        pub fn force_traced(x: i64) -> i64 {
            x + 1
        }

        // `rlib/jit.py:75-78 @purefunction` is a deprecated alias for
        // `@elidable`; pyre's `#[purefunction]` forwards verbatim so the
        // emitted attribute identifier is `_elidable_function_<NAME>` like
        // canonical `#[elidable]`.
        #[purefunction]
        pub fn purefunction_helper(x: i64) -> i64 {
            x * 2
        }

        // `rlib/jit.py:203-205 @purefunction_promote` — deprecated alias
        // for `@elidable_promote`; `#[purefunction_promote]` forwards.
        #[purefunction_promote]
        pub fn purefunction_promote_helper(x: i64) -> i64 {
            x * 3
        }
    }

    /// RPython parity for the alias / override decorators added in this
    /// slice: `@look_inside` (`jit.py:148`) flips `_jit_look_inside_` to
    /// `True`, while `@purefunction` (`jit.py:75`) and
    /// `@purefunction_promote` (`jit.py:203`) are deprecated aliases for
    /// `@elidable` / `@elidable_promote`.
    #[test]
    fn test_look_inside_and_purefunction_aliases() {
        use look_inside_alias_module::*;

        // `@look_inside` sets `_jit_look_inside_ = True`.
        assert!(_jit_look_inside_force_traced);

        // `@purefunction` is the `@elidable` alias — emits
        // `_elidable_function_<NAME>` exactly like canonical `@elidable`.
        assert!(_elidable_function_purefunction_helper);

        // `@purefunction_promote` (`jit.py:203`) is the `@elidable_promote`
        // alias.  `jit.py:185 elidable(func)` puts `_elidable_function_ =
        // True` on the ORIGINAL `func` — which pyre stores as the hidden
        // `_orig_<NAME>_unlikely_name` — and NOT on the wrapper `result`
        // returned at `jit.py:201`.
        assert!(_elidable_function__orig_purefunction_promote_helper_unlikely_name);
    }

    mod oopspec_attribute_module {
        use majit_macros::{not_in_trace, oopspec};

        #[oopspec("jit.isconstant(value)")]
        pub fn marked_isconstant(value: i64) -> bool {
            value == 0
        }

        #[not_in_trace]
        pub fn marked_not_in_trace(x: i64) -> i64 {
            x + 1
        }
    }

    /// RPython attribute-name parity for `oopspec`.  `rlib/jit.py:255
    /// func.oopspec = spec` (set by `@oopspec(spec)`) and
    /// `rlib/jit.py:261 func.oopspec = "jit.not_in_trace()"` (set by
    /// `@not_in_trace`) both write the same attribute name (`oopspec`)
    /// with a string value.  Pyre's `#[oopspec(...)]` and
    /// `#[not_in_trace]` proc-macros emit `pub const oopspec_<NAME>:
    /// &'static str` matching the upstream attribute identifier.
    #[test]
    fn test_oopspec_attribute_name_parity() {
        use oopspec_attribute_module::*;

        assert_eq!(oopspec_marked_isconstant, "jit.isconstant(value)");
        assert_eq!(oopspec_marked_not_in_trace, "jit.not_in_trace()");
    }

    /// RPython attribute-name parity: `#[jit_release_gil(save_err = N)]`
    /// emits a named static `_call_aroundstate_target_<NAME>` next to the
    /// wrapper, mirroring `rffi.py:228
    /// call_external_function._call_aroundstate_target_ = funcptr,
    /// save_err`.  Both halves (concrete target + save_err) must be
    /// reachable under this upstream-named identifier so `rg
    /// _call_aroundstate_target_` finds the parity counterpart in both
    /// pyre and PyPy repositories.
    #[test]
    fn test_release_gil_emits_call_aroundstate_target_static() {
        let (default_ptr, default_save_err) =
            release_gil_aroundstate_module::_call_aroundstate_target_release_default;
        assert!(
            !default_ptr.is_null(),
            "_call_aroundstate_target_release_default[0] must point at the concrete wrapper",
        );
        assert_eq!(
            default_save_err, 0,
            "default save_err = 0 (RFFI_ERR_NONE per rffi.py:80)",
        );

        let (errno_ptr, errno_save_err) =
            release_gil_aroundstate_module::_call_aroundstate_target_release_with_errno;
        assert!(
            !errno_ptr.is_null(),
            "_call_aroundstate_target_release_with_errno[0] must point at the concrete wrapper",
        );
        assert_eq!(
            errno_save_err, 5,
            "save_err = 5 must flow through the proc-macro tuple verbatim",
        );

        assert_ne!(
            default_ptr, errno_ptr,
            "per-function `_call_aroundstate_target_` consts must not alias",
        );
    }
}

mod jit_struct {
    //! Smoke tests for `#[jit_struct]` — descr.py:105-127 / :218-239 auto-discovery.

    use majit_ir::descr::{Descr, GcCache, LLType};
    use majit_ir::value::Type;
    use majit_macros::jit_struct;

    #[jit_struct]
    struct Node {
        value: i64,
        next: Option<Box<Node>>,
    }

    #[jit_struct]
    struct Stack {
        #[expect(
            dead_code,
            reason = "field is inspected by jit_struct descriptor generation"
        )]
        head: Option<Box<Node>>,
        #[expect(
            dead_code,
            reason = "field is inspected by jit_struct descriptor generation"
        )]
        size: usize,
    }

    #[test]
    fn field_names_are_declaration_order() {
        assert_eq!(Node::__MAJIT_FIELD_NAMES, &["value", "next"]);
        assert_eq!(Stack::__MAJIT_FIELD_NAMES, &["head", "size"]);
    }

    #[test]
    fn type_id_is_stable_within_process() {
        let a = Node::__majit_type_id();
        let b = Node::__majit_type_id();
        assert_eq!(a, b);
        assert_ne!(Node::__majit_type_id(), Stack::__majit_type_id());
    }

    #[test]
    fn register_descrs_populates_gc_cache() {
        let mut gc = GcCache::new();
        let node_size = Node::__majit_register_descrs(&mut gc);
        let size_sd = node_size.as_size_descr().expect("SizeDescr");
        assert_eq!(size_sd.size(), std::mem::size_of::<Node>());

        // Field cache populated with the two named fields.
        let key = LLType::struct_key(Node::__majit_type_id());
        let field_cache = gc._cache_field.get(&key).expect("field cache entry");
        assert_eq!(field_cache.len(), 2);
        assert!(field_cache.contains_key("value"));
        assert!(field_cache.contains_key("next"));

        let value_fd = field_cache.get("value").unwrap().as_field_descr().unwrap();
        assert_eq!(value_fd.offset(), std::mem::offset_of!(Node, value));
        assert_eq!(value_fd.field_size(), std::mem::size_of::<i64>());
        assert_eq!(value_fd.field_type(), Type::Int);
        assert_eq!(value_fd.index_in_parent(), 0);

        let next_fd = field_cache.get("next").unwrap().as_field_descr().unwrap();
        assert_eq!(next_fd.offset(), std::mem::offset_of!(Node, next));
        assert_eq!(next_fd.field_type(), Type::Ref);
        assert_eq!(next_fd.index_in_parent(), 1);
    }

    #[test]
    fn register_descrs_is_idempotent() {
        let mut gc = GcCache::new();
        let a = Node::__majit_register_descrs(&mut gc);
        let b = Node::__majit_register_descrs(&mut gc);
        // Second call must return the cached SizeDescr, not a new one.
        assert!(std::sync::Arc::ptr_eq(&a, &b));
        let key = LLType::struct_key(Node::__majit_type_id());
        assert_eq!(gc._cache_field.get(&key).unwrap().len(), 2);
    }

    #[test]
    fn parent_descr_backref_wired() {
        let mut gc = GcCache::new();
        let node_size = Node::__majit_register_descrs(&mut gc);
        let key = LLType::struct_key(Node::__majit_type_id());
        let value_fd = gc
            ._cache_field
            .get(&key)
            .unwrap()
            .get("value")
            .unwrap()
            .as_field_descr()
            .unwrap();
        // descr.py:238: fielddescr.parent_descr = get_size_descr(...)
        let parent = value_fd.get_parent_descr().expect("parent_descr wired");
        assert!(std::sync::Arc::ptr_eq(&parent, &node_size));
    }

    // ─────────────────────────────────────────────────────────────────────
    // rpaheui shape parity: `LinkedList`/`Stack`/`Queue` / `Port` structures
    // from `rpaheui/aheui/storage/linkedlist.py` recoded as `#[jit_struct]`.
    // These tests document the end-state of the generic-storage migration:
    // once the tracer consumes descrs through GcCache lookup, the existing
    // `linked_list_*` trait methods on `JitCodeSym` become redundant.
    // ─────────────────────────────────────────────────────────────────────

    /// rpaheui/aheui/storage/linkedlist.py:4-11 (`class Node`).
    #[jit_struct]
    struct AheuiNode {
        value: i64,
        next: Option<Box<AheuiNode>>,
    }

    /// rpaheui/aheui/storage/linkedlist.py:67-91 (`class Stack(LinkedList)`).
    #[jit_struct]
    struct AheuiStack {
        head: Option<Box<AheuiNode>>,
        size: usize,
    }

    /// rpaheui/aheui/storage/linkedlist.py:94-122 (`class Queue(LinkedList)`).
    #[jit_struct]
    struct AheuiQueue {
        head: Option<Box<AheuiNode>>,
        tail: Option<Box<AheuiNode>>,
        size: usize,
    }

    /// rpaheui/aheui/storage/linkedlist.py:125-148 (`class Port(LinkedList)`).
    #[jit_struct]
    struct AheuiPort {
        head: Option<Box<AheuiNode>>,
        size: usize,
        last_push: i64,
    }

    #[test]
    fn aheui_shapes_register_descrs() {
        let mut gc = GcCache::new();
        let node = AheuiNode::__majit_register_descrs(&mut gc);
        let stack = AheuiStack::__majit_register_descrs(&mut gc);
        let queue = AheuiQueue::__majit_register_descrs(&mut gc);
        let port = AheuiPort::__majit_register_descrs(&mut gc);

        // Each shape gets a distinct SizeDescr.
        for (a, b) in [
            (&node, &stack),
            (&node, &queue),
            (&node, &port),
            (&stack, &queue),
            (&stack, &port),
            (&queue, &port),
        ] {
            assert!(!std::sync::Arc::ptr_eq(a, b));
        }

        assert_eq!(AheuiNode::__MAJIT_FIELD_NAMES, &["value", "next"]);
        assert_eq!(AheuiStack::__MAJIT_FIELD_NAMES, &["head", "size"]);
        assert_eq!(AheuiQueue::__MAJIT_FIELD_NAMES, &["head", "tail", "size"]);
        assert_eq!(
            AheuiPort::__MAJIT_FIELD_NAMES,
            &["head", "size", "last_push"]
        );
    }

    #[test]
    fn aheui_node_field_types_match_rpaheui() {
        let mut gc = GcCache::new();
        let _ = AheuiNode::__majit_register_descrs(&mut gc);
        let key = LLType::struct_key(AheuiNode::__majit_type_id());
        let fields = gc._cache_field.get(&key).unwrap();

        // `value: i64` → Int (rpaheui bigint lowered to i64 for the integer trace).
        let value_fd = fields.get("value").unwrap().as_field_descr().unwrap();
        assert_eq!(value_fd.field_type(), Type::Int);

        // `next: Option<Box<Node>>` → Ref (the Node* in RPython).
        let next_fd = fields.get("next").unwrap().as_field_descr().unwrap();
        assert_eq!(next_fd.field_type(), Type::Ref);
    }

    #[test]
    fn aheui_queue_tail_descr_distinct_from_head() {
        let mut gc = GcCache::new();
        let _ = AheuiQueue::__majit_register_descrs(&mut gc);
        let key = LLType::struct_key(AheuiQueue::__majit_type_id());
        let fields = gc._cache_field.get(&key).unwrap();
        let head = fields.get("head").unwrap();
        let tail = fields.get("tail").unwrap();
        assert!(!std::sync::Arc::ptr_eq(head, tail));
        let head_fd = head.as_field_descr().unwrap();
        let tail_fd = tail.as_field_descr().unwrap();
        assert_ne!(head_fd.offset(), tail_fd.offset());
        assert_eq!(head_fd.index_in_parent(), 0);
        assert_eq!(tail_fd.index_in_parent(), 1);
    }
}
