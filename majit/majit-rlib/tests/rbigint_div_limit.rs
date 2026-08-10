//! Tests which temporarily tune PyPy's process-global `HOLDER`.
//!
//! `test_rbigint.py` runs these tests sequentially. Keep the mutation in its
//! own Rust test binary so libtest's parallel unit-test threads cannot observe
//! a transient algorithm cutoff halfway through another bigint operation.

use majit_rlib::rbigint::{HOLDER, RBigInt, divmod_big};

#[test]
fn test_upstream_divmod_big_bug() {
    // Literal regression from test_rbigint.py::test_divmod_big_bug.
    let a = RBigInt::fromstr(
        "-0x13131313131313131313cfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcfd0",
        0,
        false,
    )
    .unwrap();
    let b = RBigInt::fromstr("-0x131313131313131313d0", 0, false).unwrap();
    let expected_q = RBigInt::fromstr(
        "0xfffffffffffffffff62250d79435e50d79a516388dd408827b60354d2c7c82a19c0c899f3a5c6ec65bf3dec47ccdf4e958621674f1a87df99ddc2dbe34a9d6f11bdb31f432673b4781ce9f106260fc7ac1a54256d982703b4a4055475b5e13378c559f6ffab632f5dfb640c83426e799c8817c1e5c023130d2bb8afa3b",
        0,
        false,
    )
    .unwrap();
    let expected_r = RBigInt::fromstr("-0xdcaee6551682af11ee0", 0, false).unwrap();

    struct DivLimitGuard(i64);
    impl Drop for DivLimitGuard {
        fn drop(&mut self) {
            HOLDER
                .DIV_LIMIT
                .store(self.0, std::sync::atomic::Ordering::Relaxed);
        }
    }
    let old_limit = HOLDER
        .DIV_LIMIT
        .swap(2, std::sync::atomic::Ordering::Relaxed);
    let _guard = DivLimitGuard(old_limit);

    let (q, r) = divmod_big(&a, &b).unwrap();
    assert!(q.eq(&expected_q));
    assert!(r.eq(&expected_r));
    assert!(q.mul(&b).add(&r).eq(&a));
}
