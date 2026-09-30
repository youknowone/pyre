//! Blackhole residual calls whose wasm type the function table published.
//!
//! Each trampoline is one concrete `extern "C"` type. The slot is
//! called only after `residual_target_sig` reported that exact type,
//! so the `call_indirect` type-check matches the callee. An i32 result
//! is zero-extended into the word the reflective host writes. A void
//! callee returns 0.

#![allow(unused_variables, clippy::missing_safety_doc)]

use majit_backend_wasm::FuncSigVal;

fn fn_from_slot<T>(index: usize) -> T {
    unsafe { core::mem::transmute_copy(&index) }
}

unsafe fn c_0_0_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn() = fn_from_slot(slot);
    f();
    0
}

unsafe fn c_0_0_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn() -> i32 = fn_from_slot(slot);
    f() as u32 as i64
}

unsafe fn c_0_0_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn() -> i64 = fn_from_slot(slot);
    f()
}

unsafe fn c_0_0_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn() -> f64 = fn_from_slot(slot);
    f().to_bits() as i64
}

unsafe fn c_1_0_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64) = fn_from_slot(slot);
    f(args[0]);
    0
}

unsafe fn c_1_0_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64) -> i32 = fn_from_slot(slot);
    f(args[0]) as u32 as i64
}

unsafe fn c_1_0_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64) -> i64 = fn_from_slot(slot);
    f(args[0])
}

unsafe fn c_1_0_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64) -> f64 = fn_from_slot(slot);
    f(args[0]).to_bits() as i64
}

unsafe fn c_1_1_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32) = fn_from_slot(slot);
    f(args[0] as i32);
    0
}

unsafe fn c_1_1_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32) -> i32 = fn_from_slot(slot);
    f(args[0] as i32) as u32 as i64
}

unsafe fn c_1_1_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32) -> i64 = fn_from_slot(slot);
    f(args[0] as i32)
}

unsafe fn c_1_1_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32) -> f64 = fn_from_slot(slot);
    f(args[0] as i32).to_bits() as i64
}

unsafe fn c_2_0_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64) = fn_from_slot(slot);
    f(args[0], args[1]);
    0
}

unsafe fn c_2_0_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64) -> i32 = fn_from_slot(slot);
    f(args[0], args[1]) as u32 as i64
}

unsafe fn c_2_0_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64) -> i64 = fn_from_slot(slot);
    f(args[0], args[1])
}

unsafe fn c_2_0_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64) -> f64 = fn_from_slot(slot);
    f(args[0], args[1]).to_bits() as i64
}

unsafe fn c_2_1_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64) = fn_from_slot(slot);
    f(args[0] as i32, args[1]);
    0
}

unsafe fn c_2_1_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64) -> i32 = fn_from_slot(slot);
    f(args[0] as i32, args[1]) as u32 as i64
}

unsafe fn c_2_1_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64) -> i64 = fn_from_slot(slot);
    f(args[0] as i32, args[1])
}

unsafe fn c_2_1_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64) -> f64 = fn_from_slot(slot);
    f(args[0] as i32, args[1]).to_bits() as i64
}

unsafe fn c_2_2_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32) = fn_from_slot(slot);
    f(args[0], args[1] as i32);
    0
}

unsafe fn c_2_2_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32) -> i32 = fn_from_slot(slot);
    f(args[0], args[1] as i32) as u32 as i64
}

unsafe fn c_2_2_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32) -> i64 = fn_from_slot(slot);
    f(args[0], args[1] as i32)
}

unsafe fn c_2_2_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32) -> f64 = fn_from_slot(slot);
    f(args[0], args[1] as i32).to_bits() as i64
}

unsafe fn c_2_3_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32) = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32);
    0
}

unsafe fn c_2_3_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32) -> i32 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32) as u32 as i64
}

unsafe fn c_2_3_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32) -> i64 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32)
}

unsafe fn c_2_3_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32) -> f64 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32).to_bits() as i64
}

unsafe fn c_3_0_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64) = fn_from_slot(slot);
    f(args[0], args[1], args[2]);
    0
}

unsafe fn c_3_0_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(args[0], args[1], args[2]) as u32 as i64
}

unsafe fn c_3_0_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(args[0], args[1], args[2])
}

unsafe fn c_3_0_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(args[0], args[1], args[2]).to_bits() as i64
}

unsafe fn c_3_1_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64) = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2]);
    0
}

unsafe fn c_3_1_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2]) as u32 as i64
}

unsafe fn c_3_1_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2])
}

unsafe fn c_3_1_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2]).to_bits() as i64
}

unsafe fn c_3_2_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64) = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2]);
    0
}

unsafe fn c_3_2_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2]) as u32 as i64
}

unsafe fn c_3_2_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2])
}

unsafe fn c_3_2_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2]).to_bits() as i64
}

unsafe fn c_3_3_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64) = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2]);
    0
}

unsafe fn c_3_3_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2]) as u32 as i64
}

unsafe fn c_3_3_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2])
}

unsafe fn c_3_3_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2]).to_bits() as i64
}

unsafe fn c_3_4_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32) = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32);
    0
}

unsafe fn c_3_4_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32) as u32 as i64
}

unsafe fn c_3_4_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32)
}

unsafe fn c_3_4_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32).to_bits() as i64
}

unsafe fn c_3_5_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32) = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2] as i32);
    0
}

unsafe fn c_3_5_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2] as i32) as u32 as i64
}

unsafe fn c_3_5_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2] as i32)
}

unsafe fn c_3_5_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2] as i32).to_bits() as i64
}

unsafe fn c_3_6_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32) = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2] as i32);
    0
}

unsafe fn c_3_6_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2] as i32) as u32 as i64
}

unsafe fn c_3_6_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2] as i32)
}

unsafe fn c_3_6_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2] as i32).to_bits() as i64
}

unsafe fn c_3_7_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32) = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2] as i32);
    0
}

unsafe fn c_3_7_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2] as i32) as u32 as i64
}

unsafe fn c_3_7_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2] as i32)
}

unsafe fn c_3_7_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2] as i32).to_bits() as i64
}

unsafe fn c_4_0_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64) = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3]);
    0
}

unsafe fn c_4_0_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3]) as u32 as i64
}

unsafe fn c_4_0_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3])
}

unsafe fn c_4_0_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3]).to_bits() as i64
}

unsafe fn c_4_1_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64) = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3]);
    0
}

unsafe fn c_4_1_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3]) as u32 as i64
}

unsafe fn c_4_1_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3])
}

unsafe fn c_4_1_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3]).to_bits() as i64
}

unsafe fn c_4_2_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64) = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3]);
    0
}

unsafe fn c_4_2_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3]) as u32 as i64
}

unsafe fn c_4_2_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3])
}

unsafe fn c_4_2_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3]).to_bits() as i64
}

unsafe fn c_4_3_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64) = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2], args[3]);
    0
}

unsafe fn c_4_3_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2], args[3]) as u32 as i64
}

unsafe fn c_4_3_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2], args[3])
}

unsafe fn c_4_3_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2], args[3]).to_bits() as i64
}

unsafe fn c_4_4_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64) = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3]);
    0
}

unsafe fn c_4_4_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3]) as u32 as i64
}

unsafe fn c_4_4_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3])
}

unsafe fn c_4_4_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3]).to_bits() as i64
}

unsafe fn c_4_5_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64) = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2] as i32, args[3]);
    0
}

unsafe fn c_4_5_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2] as i32, args[3]) as u32 as i64
}

unsafe fn c_4_5_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2] as i32, args[3])
}

unsafe fn c_4_5_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2] as i32, args[3]).to_bits() as i64
}

unsafe fn c_4_6_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64) = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2] as i32, args[3]);
    0
}

unsafe fn c_4_6_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2] as i32, args[3]) as u32 as i64
}

unsafe fn c_4_6_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2] as i32, args[3])
}

unsafe fn c_4_6_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2] as i32, args[3]).to_bits() as i64
}

unsafe fn c_4_7_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64) = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2] as i32, args[3]);
    0
}

unsafe fn c_4_7_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2] as i32, args[3]) as u32 as i64
}

unsafe fn c_4_7_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2] as i32, args[3])
}

unsafe fn c_4_7_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2] as i32, args[3]).to_bits() as i64
}

unsafe fn c_4_8_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32) = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3] as i32);
    0
}

unsafe fn c_4_8_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3] as i32) as u32 as i64
}

unsafe fn c_4_8_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3] as i32)
}

unsafe fn c_4_8_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3] as i32).to_bits() as i64
}

unsafe fn c_4_9_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32) = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3] as i32);
    0
}

unsafe fn c_4_9_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3] as i32) as u32 as i64
}

unsafe fn c_4_9_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3] as i32)
}

unsafe fn c_4_9_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3] as i32).to_bits() as i64
}

unsafe fn c_4_10_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32) = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3] as i32);
    0
}

unsafe fn c_4_10_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3] as i32) as u32 as i64
}

unsafe fn c_4_10_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3] as i32)
}

unsafe fn c_4_10_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3] as i32).to_bits() as i64
}

unsafe fn c_4_11_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32) = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2], args[3] as i32);
    0
}

unsafe fn c_4_11_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2], args[3] as i32) as u32 as i64
}

unsafe fn c_4_11_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2], args[3] as i32)
}

unsafe fn c_4_11_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2], args[3] as i32).to_bits() as i64
}

unsafe fn c_4_12_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32) = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3] as i32);
    0
}

unsafe fn c_4_12_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3] as i32) as u32 as i64
}

unsafe fn c_4_12_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3] as i32)
}

unsafe fn c_4_12_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3] as i32).to_bits() as i64
}

unsafe fn c_4_13_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32) = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2] as i32, args[3] as i32);
    0
}

unsafe fn c_4_13_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2] as i32, args[3] as i32) as u32 as i64
}

unsafe fn c_4_13_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2] as i32, args[3] as i32)
}

unsafe fn c_4_13_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2] as i32, args[3] as i32).to_bits() as i64
}

unsafe fn c_4_14_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32) = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2] as i32, args[3] as i32);
    0
}

unsafe fn c_4_14_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2] as i32, args[3] as i32) as u32 as i64
}

unsafe fn c_4_14_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2] as i32, args[3] as i32)
}

unsafe fn c_4_14_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2] as i32, args[3] as i32).to_bits() as i64
}

unsafe fn c_4_15_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
    );
    0
}

unsafe fn c_4_15_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
    ) as u32 as i64
}

unsafe fn c_4_15_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
    )
}

unsafe fn c_4_15_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_5_0_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64) = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3], args[4]);
    0
}

unsafe fn c_5_0_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3], args[4]) as u32 as i64
}

unsafe fn c_5_0_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3], args[4])
}

unsafe fn c_5_0_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3], args[4]).to_bits() as i64
}

unsafe fn c_5_1_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64) = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3], args[4]);
    0
}

unsafe fn c_5_1_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3], args[4]) as u32 as i64
}

unsafe fn c_5_1_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3], args[4])
}

unsafe fn c_5_1_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3], args[4]).to_bits() as i64
}

unsafe fn c_5_2_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64) = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3], args[4]);
    0
}

unsafe fn c_5_2_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3], args[4]) as u32 as i64
}

unsafe fn c_5_2_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3], args[4])
}

unsafe fn c_5_2_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3], args[4]).to_bits() as i64
}

unsafe fn c_5_3_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64) = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2], args[3], args[4]);
    0
}

unsafe fn c_5_3_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2], args[3], args[4]) as u32 as i64
}

unsafe fn c_5_3_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2], args[3], args[4])
}

unsafe fn c_5_3_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(args[0] as i32, args[1] as i32, args[2], args[3], args[4]).to_bits() as i64
}

unsafe fn c_5_4_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64) = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3], args[4]);
    0
}

unsafe fn c_5_4_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3], args[4]) as u32 as i64
}

unsafe fn c_5_4_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3], args[4])
}

unsafe fn c_5_4_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3], args[4]).to_bits() as i64
}

unsafe fn c_5_5_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64) = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2] as i32, args[3], args[4]);
    0
}

unsafe fn c_5_5_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2] as i32, args[3], args[4]) as u32 as i64
}

unsafe fn c_5_5_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2] as i32, args[3], args[4])
}

unsafe fn c_5_5_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2] as i32, args[3], args[4]).to_bits() as i64
}

unsafe fn c_5_6_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64) = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2] as i32, args[3], args[4]);
    0
}

unsafe fn c_5_6_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2] as i32, args[3], args[4]) as u32 as i64
}

unsafe fn c_5_6_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2] as i32, args[3], args[4])
}

unsafe fn c_5_6_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2] as i32, args[3], args[4]).to_bits() as i64
}

unsafe fn c_5_7_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
    );
    0
}

unsafe fn c_5_7_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
    ) as u32 as i64
}

unsafe fn c_5_7_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
    )
}

unsafe fn c_5_7_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
    )
    .to_bits() as i64
}

unsafe fn c_5_8_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64) = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3] as i32, args[4]);
    0
}

unsafe fn c_5_8_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3] as i32, args[4]) as u32 as i64
}

unsafe fn c_5_8_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3] as i32, args[4])
}

unsafe fn c_5_8_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3] as i32, args[4]).to_bits() as i64
}

unsafe fn c_5_9_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64) = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3] as i32, args[4]);
    0
}

unsafe fn c_5_9_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3] as i32, args[4]) as u32 as i64
}

unsafe fn c_5_9_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3] as i32, args[4])
}

unsafe fn c_5_9_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3] as i32, args[4]).to_bits() as i64
}

unsafe fn c_5_10_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64) = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3] as i32, args[4]);
    0
}

unsafe fn c_5_10_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3] as i32, args[4]) as u32 as i64
}

unsafe fn c_5_10_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3] as i32, args[4])
}

unsafe fn c_5_10_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3] as i32, args[4]).to_bits() as i64
}

unsafe fn c_5_11_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
    );
    0
}

unsafe fn c_5_11_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
    ) as u32 as i64
}

unsafe fn c_5_11_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
    )
}

unsafe fn c_5_11_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
    )
    .to_bits() as i64
}

unsafe fn c_5_12_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64) = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3] as i32, args[4]);
    0
}

unsafe fn c_5_12_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3] as i32, args[4]) as u32 as i64
}

unsafe fn c_5_12_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3] as i32, args[4])
}

unsafe fn c_5_12_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3] as i32, args[4]).to_bits() as i64
}

unsafe fn c_5_13_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
    );
    0
}

unsafe fn c_5_13_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
    ) as u32 as i64
}

unsafe fn c_5_13_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
    )
}

unsafe fn c_5_13_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
    )
    .to_bits() as i64
}

unsafe fn c_5_14_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
    );
    0
}

unsafe fn c_5_14_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
    ) as u32 as i64
}

unsafe fn c_5_14_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
    )
}

unsafe fn c_5_14_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
    )
    .to_bits() as i64
}

unsafe fn c_5_15_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
    );
    0
}

unsafe fn c_5_15_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
    ) as u32 as i64
}

unsafe fn c_5_15_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
    )
}

unsafe fn c_5_15_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
    )
    .to_bits() as i64
}

unsafe fn c_5_16_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32) = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3], args[4] as i32);
    0
}

unsafe fn c_5_16_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3], args[4] as i32) as u32 as i64
}

unsafe fn c_5_16_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3], args[4] as i32)
}

unsafe fn c_5_16_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3], args[4] as i32).to_bits() as i64
}

unsafe fn c_5_17_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32) = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3], args[4] as i32);
    0
}

unsafe fn c_5_17_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3], args[4] as i32) as u32 as i64
}

unsafe fn c_5_17_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3], args[4] as i32)
}

unsafe fn c_5_17_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3], args[4] as i32).to_bits() as i64
}

unsafe fn c_5_18_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32) = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3], args[4] as i32);
    0
}

unsafe fn c_5_18_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3], args[4] as i32) as u32 as i64
}

unsafe fn c_5_18_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3], args[4] as i32)
}

unsafe fn c_5_18_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3], args[4] as i32).to_bits() as i64
}

unsafe fn c_5_19_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
    );
    0
}

unsafe fn c_5_19_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
    ) as u32 as i64
}

unsafe fn c_5_19_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
    )
}

unsafe fn c_5_19_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_5_20_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32) = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3], args[4] as i32);
    0
}

unsafe fn c_5_20_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3], args[4] as i32) as u32 as i64
}

unsafe fn c_5_20_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3], args[4] as i32)
}

unsafe fn c_5_20_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3], args[4] as i32).to_bits() as i64
}

unsafe fn c_5_21_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
    );
    0
}

unsafe fn c_5_21_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
    ) as u32 as i64
}

unsafe fn c_5_21_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
    )
}

unsafe fn c_5_21_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_5_22_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
    );
    0
}

unsafe fn c_5_22_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
    ) as u32 as i64
}

unsafe fn c_5_22_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
    )
}

unsafe fn c_5_22_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_5_23_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
    );
    0
}

unsafe fn c_5_23_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
    ) as u32 as i64
}

unsafe fn c_5_23_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
    )
}

unsafe fn c_5_23_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_5_24_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32) = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3] as i32, args[4] as i32);
    0
}

unsafe fn c_5_24_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3] as i32, args[4] as i32) as u32 as i64
}

unsafe fn c_5_24_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3] as i32, args[4] as i32)
}

unsafe fn c_5_24_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3] as i32, args[4] as i32).to_bits() as i64
}

unsafe fn c_5_25_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
    );
    0
}

unsafe fn c_5_25_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
    ) as u32 as i64
}

unsafe fn c_5_25_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
    )
}

unsafe fn c_5_25_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_5_26_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
    );
    0
}

unsafe fn c_5_26_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
    ) as u32 as i64
}

unsafe fn c_5_26_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
    )
}

unsafe fn c_5_26_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_5_27_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
    );
    0
}

unsafe fn c_5_27_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
    ) as u32 as i64
}

unsafe fn c_5_27_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
    )
}

unsafe fn c_5_27_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_5_28_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
    );
    0
}

unsafe fn c_5_28_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
    ) as u32 as i64
}

unsafe fn c_5_28_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
    )
}

unsafe fn c_5_28_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_5_29_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
    );
    0
}

unsafe fn c_5_29_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
    ) as u32 as i64
}

unsafe fn c_5_29_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
    )
}

unsafe fn c_5_29_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_5_30_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
    );
    0
}

unsafe fn c_5_30_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
    ) as u32 as i64
}

unsafe fn c_5_30_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
    )
}

unsafe fn c_5_30_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_5_31_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
    );
    0
}

unsafe fn c_5_31_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
    ) as u32 as i64
}

unsafe fn c_5_31_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
    )
}

unsafe fn c_5_31_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_0_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i64) = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3], args[4], args[5]);
    0
}

unsafe fn c_6_0_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3], args[4], args[5]) as u32 as i64
}

unsafe fn c_6_0_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3], args[4], args[5])
}

unsafe fn c_6_0_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3], args[4], args[5]).to_bits() as i64
}

unsafe fn c_6_1_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i64) = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3], args[4], args[5]);
    0
}

unsafe fn c_6_1_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3], args[4], args[5]) as u32 as i64
}

unsafe fn c_6_1_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3], args[4], args[5])
}

unsafe fn c_6_1_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(args[0] as i32, args[1], args[2], args[3], args[4], args[5]).to_bits() as i64
}

unsafe fn c_6_2_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i64) = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3], args[4], args[5]);
    0
}

unsafe fn c_6_2_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3], args[4], args[5]) as u32 as i64
}

unsafe fn c_6_2_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3], args[4], args[5])
}

unsafe fn c_6_2_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(args[0], args[1] as i32, args[2], args[3], args[4], args[5]).to_bits() as i64
}

unsafe fn c_6_3_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5],
    );
    0
}

unsafe fn c_6_3_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_3_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5],
    )
}

unsafe fn c_6_3_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_4_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i64) = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3], args[4], args[5]);
    0
}

unsafe fn c_6_4_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3], args[4], args[5]) as u32 as i64
}

unsafe fn c_6_4_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3], args[4], args[5])
}

unsafe fn c_6_4_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(args[0], args[1], args[2] as i32, args[3], args[4], args[5]).to_bits() as i64
}

unsafe fn c_6_5_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5],
    );
    0
}

unsafe fn c_6_5_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_5_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5],
    )
}

unsafe fn c_6_5_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_6_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
    );
    0
}

unsafe fn c_6_6_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_6_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
    )
}

unsafe fn c_6_6_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_7_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
    );
    0
}

unsafe fn c_6_7_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_7_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
    )
}

unsafe fn c_6_7_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_8_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i64) = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3] as i32, args[4], args[5]);
    0
}

unsafe fn c_6_8_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3] as i32, args[4], args[5]) as u32 as i64
}

unsafe fn c_6_8_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3] as i32, args[4], args[5])
}

unsafe fn c_6_8_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3] as i32, args[4], args[5]).to_bits() as i64
}

unsafe fn c_6_9_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5],
    );
    0
}

unsafe fn c_6_9_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_9_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5],
    )
}

unsafe fn c_6_9_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_10_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
    );
    0
}

unsafe fn c_6_10_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_10_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
    )
}

unsafe fn c_6_10_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_11_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
    );
    0
}

unsafe fn c_6_11_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_11_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
    )
}

unsafe fn c_6_11_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_12_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
    );
    0
}

unsafe fn c_6_12_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_12_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
    )
}

unsafe fn c_6_12_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_13_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
    );
    0
}

unsafe fn c_6_13_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_13_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
    )
}

unsafe fn c_6_13_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_14_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
    );
    0
}

unsafe fn c_6_14_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_14_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
    )
}

unsafe fn c_6_14_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_15_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
    );
    0
}

unsafe fn c_6_15_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_15_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
    )
}

unsafe fn c_6_15_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_16_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i64) = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3], args[4] as i32, args[5]);
    0
}

unsafe fn c_6_16_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3], args[4] as i32, args[5]) as u32 as i64
}

unsafe fn c_6_16_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3], args[4] as i32, args[5])
}

unsafe fn c_6_16_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3], args[4] as i32, args[5]).to_bits() as i64
}

unsafe fn c_6_17_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5],
    );
    0
}

unsafe fn c_6_17_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_17_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5],
    )
}

unsafe fn c_6_17_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_18_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
    );
    0
}

unsafe fn c_6_18_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_18_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
    )
}

unsafe fn c_6_18_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_19_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
    );
    0
}

unsafe fn c_6_19_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_19_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
    )
}

unsafe fn c_6_19_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_20_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
    );
    0
}

unsafe fn c_6_20_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_20_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
    )
}

unsafe fn c_6_20_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_21_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
    );
    0
}

unsafe fn c_6_21_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_21_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
    )
}

unsafe fn c_6_21_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_22_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
    );
    0
}

unsafe fn c_6_22_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_22_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
    )
}

unsafe fn c_6_22_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_23_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
    );
    0
}

unsafe fn c_6_23_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_23_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
    )
}

unsafe fn c_6_23_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_24_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
    );
    0
}

unsafe fn c_6_24_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_24_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
    )
}

unsafe fn c_6_24_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_25_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
    );
    0
}

unsafe fn c_6_25_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_25_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
    )
}

unsafe fn c_6_25_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_26_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
    );
    0
}

unsafe fn c_6_26_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_26_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
    )
}

unsafe fn c_6_26_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_27_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
    );
    0
}

unsafe fn c_6_27_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_27_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
    )
}

unsafe fn c_6_27_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_28_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
    );
    0
}

unsafe fn c_6_28_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_28_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
    )
}

unsafe fn c_6_28_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_29_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
    );
    0
}

unsafe fn c_6_29_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_29_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
    )
}

unsafe fn c_6_29_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_30_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
    );
    0
}

unsafe fn c_6_30_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_30_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
    )
}

unsafe fn c_6_30_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_31_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
    );
    0
}

unsafe fn c_6_31_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
    ) as u32 as i64
}

unsafe fn c_6_31_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
    )
}

unsafe fn c_6_31_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
    )
    .to_bits() as i64
}

unsafe fn c_6_32_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i32) = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3], args[4], args[5] as i32);
    0
}

unsafe fn c_6_32_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3], args[4], args[5] as i32) as u32 as i64
}

unsafe fn c_6_32_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3], args[4], args[5] as i32)
}

unsafe fn c_6_32_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(args[0], args[1], args[2], args[3], args[4], args[5] as i32).to_bits() as i64
}

unsafe fn c_6_33_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5] as i32,
    );
    0
}

unsafe fn c_6_33_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_33_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5] as i32,
    )
}

unsafe fn c_6_33_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_34_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
    );
    0
}

unsafe fn c_6_34_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_34_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
    )
}

unsafe fn c_6_34_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_35_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
    );
    0
}

unsafe fn c_6_35_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_35_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
    )
}

unsafe fn c_6_35_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_36_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
    );
    0
}

unsafe fn c_6_36_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_36_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
    )
}

unsafe fn c_6_36_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_37_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
    );
    0
}

unsafe fn c_6_37_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_37_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
    )
}

unsafe fn c_6_37_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_38_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
    );
    0
}

unsafe fn c_6_38_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_38_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
    )
}

unsafe fn c_6_38_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_39_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
    );
    0
}

unsafe fn c_6_39_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_39_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
    )
}

unsafe fn c_6_39_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_40_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
    );
    0
}

unsafe fn c_6_40_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_40_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
    )
}

unsafe fn c_6_40_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_41_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
    );
    0
}

unsafe fn c_6_41_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_41_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
    )
}

unsafe fn c_6_41_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_42_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
    );
    0
}

unsafe fn c_6_42_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_42_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
    )
}

unsafe fn c_6_42_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_43_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
    );
    0
}

unsafe fn c_6_43_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_43_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
    )
}

unsafe fn c_6_43_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_44_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
    );
    0
}

unsafe fn c_6_44_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_44_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
    )
}

unsafe fn c_6_44_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_45_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
    );
    0
}

unsafe fn c_6_45_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_45_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
    )
}

unsafe fn c_6_45_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_46_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
    );
    0
}

unsafe fn c_6_46_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_46_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
    )
}

unsafe fn c_6_46_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_47_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
    );
    0
}

unsafe fn c_6_47_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_47_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
    )
}

unsafe fn c_6_47_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_48_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
    );
    0
}

unsafe fn c_6_48_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_48_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
    )
}

unsafe fn c_6_48_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_49_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
    );
    0
}

unsafe fn c_6_49_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_49_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
    )
}

unsafe fn c_6_49_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_50_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
    );
    0
}

unsafe fn c_6_50_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_50_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
    )
}

unsafe fn c_6_50_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_51_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
    );
    0
}

unsafe fn c_6_51_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_51_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
    )
}

unsafe fn c_6_51_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_52_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
    );
    0
}

unsafe fn c_6_52_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_52_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
    )
}

unsafe fn c_6_52_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_53_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
    );
    0
}

unsafe fn c_6_53_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_53_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
    )
}

unsafe fn c_6_53_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_54_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
    );
    0
}

unsafe fn c_6_54_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_54_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
    )
}

unsafe fn c_6_54_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_55_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
    );
    0
}

unsafe fn c_6_55_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_55_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
    )
}

unsafe fn c_6_55_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_56_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    );
    0
}

unsafe fn c_6_56_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_56_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    )
}

unsafe fn c_6_56_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_57_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    );
    0
}

unsafe fn c_6_57_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_57_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    )
}

unsafe fn c_6_57_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_58_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    );
    0
}

unsafe fn c_6_58_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_58_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    )
}

unsafe fn c_6_58_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_59_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    );
    0
}

unsafe fn c_6_59_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_59_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    )
}

unsafe fn c_6_59_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_60_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    );
    0
}

unsafe fn c_6_60_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_60_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    )
}

unsafe fn c_6_60_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_61_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    );
    0
}

unsafe fn c_6_61_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_61_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    )
}

unsafe fn c_6_61_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_62_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    );
    0
}

unsafe fn c_6_62_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_62_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    )
}

unsafe fn c_6_62_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_6_63_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    );
    0
}

unsafe fn c_6_63_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    ) as u32 as i64
}

unsafe fn c_6_63_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    )
}

unsafe fn c_6_63_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_0_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) = fn_from_slot(slot);
    f(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6],
    );
    0
}

unsafe fn c_7_0_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6],
    ) as u32 as i64
}

unsafe fn c_7_0_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6],
    )
}

unsafe fn c_7_0_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_1_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_1_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_1_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
    )
}

unsafe fn c_7_1_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_2_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i64, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_2_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_2_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
    )
}

unsafe fn c_7_2_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_3_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_3_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_3_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
    )
}

unsafe fn c_7_3_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_4_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i64, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_4_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_4_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6],
    )
}

unsafe fn c_7_4_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_5_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_5_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_5_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6],
    )
}

unsafe fn c_7_5_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_6_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i64, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_6_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_6_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6],
    )
}

unsafe fn c_7_6_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_7_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_7_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_7_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6],
    )
}

unsafe fn c_7_7_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_8_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i64, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_8_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_8_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    )
}

unsafe fn c_7_8_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_9_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_9_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_9_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    )
}

unsafe fn c_7_9_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_10_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i64, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_10_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_10_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    )
}

unsafe fn c_7_10_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_11_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_11_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_11_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    )
}

unsafe fn c_7_11_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_12_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i64, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_12_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_12_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    )
}

unsafe fn c_7_12_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_13_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_13_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_13_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    )
}

unsafe fn c_7_13_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_14_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i64, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_14_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_14_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    )
}

unsafe fn c_7_14_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_15_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_15_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_15_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    )
}

unsafe fn c_7_15_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_16_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_16_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_16_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    )
}

unsafe fn c_7_16_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_17_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_17_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_17_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    )
}

unsafe fn c_7_17_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_18_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_18_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_18_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    )
}

unsafe fn c_7_18_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_19_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_19_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_19_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    )
}

unsafe fn c_7_19_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_20_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_20_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_20_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    )
}

unsafe fn c_7_20_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_21_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_21_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_21_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    )
}

unsafe fn c_7_21_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_22_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_22_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_22_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    )
}

unsafe fn c_7_22_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_23_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_23_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_23_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    )
}

unsafe fn c_7_23_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_24_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_24_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_24_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    )
}

unsafe fn c_7_24_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_25_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_25_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_25_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    )
}

unsafe fn c_7_25_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_26_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_26_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_26_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    )
}

unsafe fn c_7_26_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_27_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_27_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_27_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    )
}

unsafe fn c_7_27_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_28_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_28_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_28_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    )
}

unsafe fn c_7_28_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_29_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_29_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_29_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    )
}

unsafe fn c_7_29_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_30_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_30_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_30_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    )
}

unsafe fn c_7_30_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_31_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i64, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    );
    0
}

unsafe fn c_7_31_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i64, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_31_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i64, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    )
}

unsafe fn c_7_31_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i64, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_32_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_32_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_32_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_32_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_33_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_33_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_33_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_33_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_34_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_34_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_34_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_34_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_35_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_35_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_35_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_35_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_36_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_36_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_36_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_36_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_37_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_37_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_37_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_37_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_38_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_38_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_38_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_38_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_39_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_39_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_39_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_39_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_40_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_40_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_40_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_40_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_41_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_41_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_41_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_41_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_42_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_42_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_42_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_42_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_43_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_43_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_43_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_43_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_44_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_44_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_44_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_44_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_45_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_45_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_45_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_45_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_46_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_46_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_46_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_46_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_47_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_47_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_47_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_47_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_48_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_48_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_48_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_48_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_49_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_49_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_49_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_49_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_50_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_50_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_50_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_50_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_51_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_51_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_51_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_51_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_52_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_52_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_52_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_52_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_53_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_53_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_53_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_53_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_54_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_54_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_54_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_54_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_55_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_55_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_55_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_55_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_56_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_56_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_56_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_56_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_57_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_57_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_57_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_57_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_58_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_58_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_58_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_58_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_59_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_59_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_59_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_59_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_60_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_60_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_60_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_60_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_61_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_61_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_61_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_61_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_62_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_62_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_62_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_62_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_63_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i32, i64) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    );
    0
}

unsafe fn c_7_63_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i32, i64) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    ) as u32 as i64
}

unsafe fn c_7_63_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i32, i64) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
}

unsafe fn c_7_63_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i32, i64) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6],
    )
    .to_bits() as i64
}

unsafe fn c_7_64_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_64_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_64_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_64_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_65_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_65_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_65_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_65_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_66_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_66_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_66_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_66_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_67_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_67_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_67_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_67_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_68_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_68_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_68_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_68_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_69_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_69_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_69_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_69_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_70_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_70_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_70_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_70_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_71_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_71_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_71_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_71_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_72_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_72_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_72_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_72_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_73_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_73_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_73_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_73_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_74_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_74_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_74_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_74_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_75_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_75_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_75_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_75_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_76_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_76_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_76_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_76_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_77_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_77_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_77_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_77_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_78_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_78_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_78_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_78_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_79_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_79_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_79_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_79_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_80_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_80_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_80_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_80_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_81_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_81_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_81_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_81_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_82_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_82_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_82_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_82_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_83_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_83_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_83_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_83_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_84_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_84_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_84_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_84_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_85_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_85_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_85_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_85_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_86_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_86_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_86_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_86_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_87_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_87_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_87_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_87_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_88_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_88_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_88_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_88_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_89_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_89_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_89_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_89_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_90_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_90_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_90_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_90_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_91_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_91_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_91_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_91_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_92_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_92_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_92_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_92_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_93_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_93_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_93_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_93_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_94_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_94_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_94_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_94_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_95_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i64, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    );
    0
}

unsafe fn c_7_95_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i64, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_95_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i64, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
}

unsafe fn c_7_95_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i64, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5],
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_96_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_96_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_96_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_96_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_97_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_97_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_97_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_97_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_98_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_98_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_98_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_98_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_99_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_99_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_99_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_99_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_100_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_100_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_100_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_100_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_101_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_101_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_101_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_101_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_102_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_102_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_102_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_102_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_103_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_103_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_103_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_103_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_104_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_104_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_104_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_104_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_105_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_105_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_105_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_105_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_106_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_106_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_106_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_106_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_107_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_107_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_107_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_107_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_108_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_108_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_108_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_108_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_109_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_109_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_109_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_109_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_110_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_110_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_110_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_110_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_111_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_111_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_111_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_111_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i64, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4],
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_112_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_112_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_112_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_112_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i64, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_113_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_113_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_113_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_113_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i64, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_114_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_114_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_114_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_114_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i64, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_115_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_115_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_115_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_115_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i64, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_116_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_116_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_116_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_116_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i64, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_117_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_117_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_117_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_117_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i64, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_118_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_118_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_118_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_118_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i64, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_119_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_119_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_119_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_119_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i64, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3],
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_120_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_120_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_120_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_120_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i64, i32, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_121_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_121_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_121_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_121_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i64, i32, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_122_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_122_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_122_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_122_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i64, i32, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_123_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_123_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_123_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_123_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i64, i32, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2],
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_124_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_124_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_124_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_124_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i64, i32, i32, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_125_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_125_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_125_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_125_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i64, i32, i32, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1],
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_126_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_126_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_126_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_126_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i64, i32, i32, i32, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0],
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn c_7_127_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i32, i32) = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    );
    0
}

unsafe fn c_7_127_1(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i32, i32) -> i32 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    ) as u32 as i64
}

unsafe fn c_7_127_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i32, i32) -> i64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
}

unsafe fn c_7_127_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(i32, i32, i32, i32, i32, i32, i32) -> f64 = fn_from_slot(slot);
    f(
        args[0] as i32,
        args[1] as i32,
        args[2] as i32,
        args[3] as i32,
        args[4] as i32,
        args[5] as i32,
        args[6] as i32,
    )
    .to_bits() as i64
}

unsafe fn f_1_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(f64) = fn_from_slot(slot);
    f(f64::from_bits(args[0] as u64));
    0
}

unsafe fn f_1_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(f64) -> i64 = fn_from_slot(slot);
    f(f64::from_bits(args[0] as u64))
}

unsafe fn f_1_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(f64) -> f64 = fn_from_slot(slot);
    f(f64::from_bits(args[0] as u64)).to_bits() as i64
}

unsafe fn f_2_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(f64, f64) = fn_from_slot(slot);
    f(
        f64::from_bits(args[0] as u64),
        f64::from_bits(args[1] as u64),
    );
    0
}

unsafe fn f_2_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(f64, f64) -> i64 = fn_from_slot(slot);
    f(
        f64::from_bits(args[0] as u64),
        f64::from_bits(args[1] as u64),
    )
}

unsafe fn f_2_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(f64, f64) -> f64 = fn_from_slot(slot);
    f(
        f64::from_bits(args[0] as u64),
        f64::from_bits(args[1] as u64),
    )
    .to_bits() as i64
}

unsafe fn f_3_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(f64, f64, f64) = fn_from_slot(slot);
    f(
        f64::from_bits(args[0] as u64),
        f64::from_bits(args[1] as u64),
        f64::from_bits(args[2] as u64),
    );
    0
}

unsafe fn f_3_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(f64, f64, f64) -> i64 = fn_from_slot(slot);
    f(
        f64::from_bits(args[0] as u64),
        f64::from_bits(args[1] as u64),
        f64::from_bits(args[2] as u64),
    )
}

unsafe fn f_3_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(f64, f64, f64) -> f64 = fn_from_slot(slot);
    f(
        f64::from_bits(args[0] as u64),
        f64::from_bits(args[1] as u64),
        f64::from_bits(args[2] as u64),
    )
    .to_bits() as i64
}

unsafe fn f_4_0(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(f64, f64, f64, f64) = fn_from_slot(slot);
    f(
        f64::from_bits(args[0] as u64),
        f64::from_bits(args[1] as u64),
        f64::from_bits(args[2] as u64),
        f64::from_bits(args[3] as u64),
    );
    0
}

unsafe fn f_4_2(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(f64, f64, f64, f64) -> i64 = fn_from_slot(slot);
    f(
        f64::from_bits(args[0] as u64),
        f64::from_bits(args[1] as u64),
        f64::from_bits(args[2] as u64),
        f64::from_bits(args[3] as u64),
    )
}

unsafe fn f_4_3(slot: usize, args: &[i64]) -> i64 {
    let f: extern "C" fn(f64, f64, f64, f64) -> f64 = fn_from_slot(slot);
    f(
        f64::from_bits(args[0] as u64),
        f64::from_bits(args[1] as u64),
        f64::from_bits(args[2] as u64),
        f64::from_bits(args[3] as u64),
    )
    .to_bits() as i64
}

pub fn call_int_sig(
    slot: usize,
    args: &[i64],
    mask: u16,
    result: Option<FuncSigVal>,
) -> Option<i64> {
    let tag = match result {
        None => 0,
        Some(FuncSigVal::I32) => 1,
        Some(FuncSigVal::I64) => 2,
        Some(FuncSigVal::F64) => 3,
        Some(FuncSigVal::F32) => return None,
    };
    Some(unsafe {
        match (args.len(), mask, tag) {
            (0, 0, 0) => c_0_0_0(slot, args),
            (0, 0, 1) => c_0_0_1(slot, args),
            (0, 0, 2) => c_0_0_2(slot, args),
            (0, 0, 3) => c_0_0_3(slot, args),
            (1, 0, 0) => c_1_0_0(slot, args),
            (1, 0, 1) => c_1_0_1(slot, args),
            (1, 0, 2) => c_1_0_2(slot, args),
            (1, 0, 3) => c_1_0_3(slot, args),
            (1, 1, 0) => c_1_1_0(slot, args),
            (1, 1, 1) => c_1_1_1(slot, args),
            (1, 1, 2) => c_1_1_2(slot, args),
            (1, 1, 3) => c_1_1_3(slot, args),
            (2, 0, 0) => c_2_0_0(slot, args),
            (2, 0, 1) => c_2_0_1(slot, args),
            (2, 0, 2) => c_2_0_2(slot, args),
            (2, 0, 3) => c_2_0_3(slot, args),
            (2, 1, 0) => c_2_1_0(slot, args),
            (2, 1, 1) => c_2_1_1(slot, args),
            (2, 1, 2) => c_2_1_2(slot, args),
            (2, 1, 3) => c_2_1_3(slot, args),
            (2, 2, 0) => c_2_2_0(slot, args),
            (2, 2, 1) => c_2_2_1(slot, args),
            (2, 2, 2) => c_2_2_2(slot, args),
            (2, 2, 3) => c_2_2_3(slot, args),
            (2, 3, 0) => c_2_3_0(slot, args),
            (2, 3, 1) => c_2_3_1(slot, args),
            (2, 3, 2) => c_2_3_2(slot, args),
            (2, 3, 3) => c_2_3_3(slot, args),
            (3, 0, 0) => c_3_0_0(slot, args),
            (3, 0, 1) => c_3_0_1(slot, args),
            (3, 0, 2) => c_3_0_2(slot, args),
            (3, 0, 3) => c_3_0_3(slot, args),
            (3, 1, 0) => c_3_1_0(slot, args),
            (3, 1, 1) => c_3_1_1(slot, args),
            (3, 1, 2) => c_3_1_2(slot, args),
            (3, 1, 3) => c_3_1_3(slot, args),
            (3, 2, 0) => c_3_2_0(slot, args),
            (3, 2, 1) => c_3_2_1(slot, args),
            (3, 2, 2) => c_3_2_2(slot, args),
            (3, 2, 3) => c_3_2_3(slot, args),
            (3, 3, 0) => c_3_3_0(slot, args),
            (3, 3, 1) => c_3_3_1(slot, args),
            (3, 3, 2) => c_3_3_2(slot, args),
            (3, 3, 3) => c_3_3_3(slot, args),
            (3, 4, 0) => c_3_4_0(slot, args),
            (3, 4, 1) => c_3_4_1(slot, args),
            (3, 4, 2) => c_3_4_2(slot, args),
            (3, 4, 3) => c_3_4_3(slot, args),
            (3, 5, 0) => c_3_5_0(slot, args),
            (3, 5, 1) => c_3_5_1(slot, args),
            (3, 5, 2) => c_3_5_2(slot, args),
            (3, 5, 3) => c_3_5_3(slot, args),
            (3, 6, 0) => c_3_6_0(slot, args),
            (3, 6, 1) => c_3_6_1(slot, args),
            (3, 6, 2) => c_3_6_2(slot, args),
            (3, 6, 3) => c_3_6_3(slot, args),
            (3, 7, 0) => c_3_7_0(slot, args),
            (3, 7, 1) => c_3_7_1(slot, args),
            (3, 7, 2) => c_3_7_2(slot, args),
            (3, 7, 3) => c_3_7_3(slot, args),
            (4, 0, 0) => c_4_0_0(slot, args),
            (4, 0, 1) => c_4_0_1(slot, args),
            (4, 0, 2) => c_4_0_2(slot, args),
            (4, 0, 3) => c_4_0_3(slot, args),
            (4, 1, 0) => c_4_1_0(slot, args),
            (4, 1, 1) => c_4_1_1(slot, args),
            (4, 1, 2) => c_4_1_2(slot, args),
            (4, 1, 3) => c_4_1_3(slot, args),
            (4, 2, 0) => c_4_2_0(slot, args),
            (4, 2, 1) => c_4_2_1(slot, args),
            (4, 2, 2) => c_4_2_2(slot, args),
            (4, 2, 3) => c_4_2_3(slot, args),
            (4, 3, 0) => c_4_3_0(slot, args),
            (4, 3, 1) => c_4_3_1(slot, args),
            (4, 3, 2) => c_4_3_2(slot, args),
            (4, 3, 3) => c_4_3_3(slot, args),
            (4, 4, 0) => c_4_4_0(slot, args),
            (4, 4, 1) => c_4_4_1(slot, args),
            (4, 4, 2) => c_4_4_2(slot, args),
            (4, 4, 3) => c_4_4_3(slot, args),
            (4, 5, 0) => c_4_5_0(slot, args),
            (4, 5, 1) => c_4_5_1(slot, args),
            (4, 5, 2) => c_4_5_2(slot, args),
            (4, 5, 3) => c_4_5_3(slot, args),
            (4, 6, 0) => c_4_6_0(slot, args),
            (4, 6, 1) => c_4_6_1(slot, args),
            (4, 6, 2) => c_4_6_2(slot, args),
            (4, 6, 3) => c_4_6_3(slot, args),
            (4, 7, 0) => c_4_7_0(slot, args),
            (4, 7, 1) => c_4_7_1(slot, args),
            (4, 7, 2) => c_4_7_2(slot, args),
            (4, 7, 3) => c_4_7_3(slot, args),
            (4, 8, 0) => c_4_8_0(slot, args),
            (4, 8, 1) => c_4_8_1(slot, args),
            (4, 8, 2) => c_4_8_2(slot, args),
            (4, 8, 3) => c_4_8_3(slot, args),
            (4, 9, 0) => c_4_9_0(slot, args),
            (4, 9, 1) => c_4_9_1(slot, args),
            (4, 9, 2) => c_4_9_2(slot, args),
            (4, 9, 3) => c_4_9_3(slot, args),
            (4, 10, 0) => c_4_10_0(slot, args),
            (4, 10, 1) => c_4_10_1(slot, args),
            (4, 10, 2) => c_4_10_2(slot, args),
            (4, 10, 3) => c_4_10_3(slot, args),
            (4, 11, 0) => c_4_11_0(slot, args),
            (4, 11, 1) => c_4_11_1(slot, args),
            (4, 11, 2) => c_4_11_2(slot, args),
            (4, 11, 3) => c_4_11_3(slot, args),
            (4, 12, 0) => c_4_12_0(slot, args),
            (4, 12, 1) => c_4_12_1(slot, args),
            (4, 12, 2) => c_4_12_2(slot, args),
            (4, 12, 3) => c_4_12_3(slot, args),
            (4, 13, 0) => c_4_13_0(slot, args),
            (4, 13, 1) => c_4_13_1(slot, args),
            (4, 13, 2) => c_4_13_2(slot, args),
            (4, 13, 3) => c_4_13_3(slot, args),
            (4, 14, 0) => c_4_14_0(slot, args),
            (4, 14, 1) => c_4_14_1(slot, args),
            (4, 14, 2) => c_4_14_2(slot, args),
            (4, 14, 3) => c_4_14_3(slot, args),
            (4, 15, 0) => c_4_15_0(slot, args),
            (4, 15, 1) => c_4_15_1(slot, args),
            (4, 15, 2) => c_4_15_2(slot, args),
            (4, 15, 3) => c_4_15_3(slot, args),
            (5, 0, 0) => c_5_0_0(slot, args),
            (5, 0, 1) => c_5_0_1(slot, args),
            (5, 0, 2) => c_5_0_2(slot, args),
            (5, 0, 3) => c_5_0_3(slot, args),
            (5, 1, 0) => c_5_1_0(slot, args),
            (5, 1, 1) => c_5_1_1(slot, args),
            (5, 1, 2) => c_5_1_2(slot, args),
            (5, 1, 3) => c_5_1_3(slot, args),
            (5, 2, 0) => c_5_2_0(slot, args),
            (5, 2, 1) => c_5_2_1(slot, args),
            (5, 2, 2) => c_5_2_2(slot, args),
            (5, 2, 3) => c_5_2_3(slot, args),
            (5, 3, 0) => c_5_3_0(slot, args),
            (5, 3, 1) => c_5_3_1(slot, args),
            (5, 3, 2) => c_5_3_2(slot, args),
            (5, 3, 3) => c_5_3_3(slot, args),
            (5, 4, 0) => c_5_4_0(slot, args),
            (5, 4, 1) => c_5_4_1(slot, args),
            (5, 4, 2) => c_5_4_2(slot, args),
            (5, 4, 3) => c_5_4_3(slot, args),
            (5, 5, 0) => c_5_5_0(slot, args),
            (5, 5, 1) => c_5_5_1(slot, args),
            (5, 5, 2) => c_5_5_2(slot, args),
            (5, 5, 3) => c_5_5_3(slot, args),
            (5, 6, 0) => c_5_6_0(slot, args),
            (5, 6, 1) => c_5_6_1(slot, args),
            (5, 6, 2) => c_5_6_2(slot, args),
            (5, 6, 3) => c_5_6_3(slot, args),
            (5, 7, 0) => c_5_7_0(slot, args),
            (5, 7, 1) => c_5_7_1(slot, args),
            (5, 7, 2) => c_5_7_2(slot, args),
            (5, 7, 3) => c_5_7_3(slot, args),
            (5, 8, 0) => c_5_8_0(slot, args),
            (5, 8, 1) => c_5_8_1(slot, args),
            (5, 8, 2) => c_5_8_2(slot, args),
            (5, 8, 3) => c_5_8_3(slot, args),
            (5, 9, 0) => c_5_9_0(slot, args),
            (5, 9, 1) => c_5_9_1(slot, args),
            (5, 9, 2) => c_5_9_2(slot, args),
            (5, 9, 3) => c_5_9_3(slot, args),
            (5, 10, 0) => c_5_10_0(slot, args),
            (5, 10, 1) => c_5_10_1(slot, args),
            (5, 10, 2) => c_5_10_2(slot, args),
            (5, 10, 3) => c_5_10_3(slot, args),
            (5, 11, 0) => c_5_11_0(slot, args),
            (5, 11, 1) => c_5_11_1(slot, args),
            (5, 11, 2) => c_5_11_2(slot, args),
            (5, 11, 3) => c_5_11_3(slot, args),
            (5, 12, 0) => c_5_12_0(slot, args),
            (5, 12, 1) => c_5_12_1(slot, args),
            (5, 12, 2) => c_5_12_2(slot, args),
            (5, 12, 3) => c_5_12_3(slot, args),
            (5, 13, 0) => c_5_13_0(slot, args),
            (5, 13, 1) => c_5_13_1(slot, args),
            (5, 13, 2) => c_5_13_2(slot, args),
            (5, 13, 3) => c_5_13_3(slot, args),
            (5, 14, 0) => c_5_14_0(slot, args),
            (5, 14, 1) => c_5_14_1(slot, args),
            (5, 14, 2) => c_5_14_2(slot, args),
            (5, 14, 3) => c_5_14_3(slot, args),
            (5, 15, 0) => c_5_15_0(slot, args),
            (5, 15, 1) => c_5_15_1(slot, args),
            (5, 15, 2) => c_5_15_2(slot, args),
            (5, 15, 3) => c_5_15_3(slot, args),
            (5, 16, 0) => c_5_16_0(slot, args),
            (5, 16, 1) => c_5_16_1(slot, args),
            (5, 16, 2) => c_5_16_2(slot, args),
            (5, 16, 3) => c_5_16_3(slot, args),
            (5, 17, 0) => c_5_17_0(slot, args),
            (5, 17, 1) => c_5_17_1(slot, args),
            (5, 17, 2) => c_5_17_2(slot, args),
            (5, 17, 3) => c_5_17_3(slot, args),
            (5, 18, 0) => c_5_18_0(slot, args),
            (5, 18, 1) => c_5_18_1(slot, args),
            (5, 18, 2) => c_5_18_2(slot, args),
            (5, 18, 3) => c_5_18_3(slot, args),
            (5, 19, 0) => c_5_19_0(slot, args),
            (5, 19, 1) => c_5_19_1(slot, args),
            (5, 19, 2) => c_5_19_2(slot, args),
            (5, 19, 3) => c_5_19_3(slot, args),
            (5, 20, 0) => c_5_20_0(slot, args),
            (5, 20, 1) => c_5_20_1(slot, args),
            (5, 20, 2) => c_5_20_2(slot, args),
            (5, 20, 3) => c_5_20_3(slot, args),
            (5, 21, 0) => c_5_21_0(slot, args),
            (5, 21, 1) => c_5_21_1(slot, args),
            (5, 21, 2) => c_5_21_2(slot, args),
            (5, 21, 3) => c_5_21_3(slot, args),
            (5, 22, 0) => c_5_22_0(slot, args),
            (5, 22, 1) => c_5_22_1(slot, args),
            (5, 22, 2) => c_5_22_2(slot, args),
            (5, 22, 3) => c_5_22_3(slot, args),
            (5, 23, 0) => c_5_23_0(slot, args),
            (5, 23, 1) => c_5_23_1(slot, args),
            (5, 23, 2) => c_5_23_2(slot, args),
            (5, 23, 3) => c_5_23_3(slot, args),
            (5, 24, 0) => c_5_24_0(slot, args),
            (5, 24, 1) => c_5_24_1(slot, args),
            (5, 24, 2) => c_5_24_2(slot, args),
            (5, 24, 3) => c_5_24_3(slot, args),
            (5, 25, 0) => c_5_25_0(slot, args),
            (5, 25, 1) => c_5_25_1(slot, args),
            (5, 25, 2) => c_5_25_2(slot, args),
            (5, 25, 3) => c_5_25_3(slot, args),
            (5, 26, 0) => c_5_26_0(slot, args),
            (5, 26, 1) => c_5_26_1(slot, args),
            (5, 26, 2) => c_5_26_2(slot, args),
            (5, 26, 3) => c_5_26_3(slot, args),
            (5, 27, 0) => c_5_27_0(slot, args),
            (5, 27, 1) => c_5_27_1(slot, args),
            (5, 27, 2) => c_5_27_2(slot, args),
            (5, 27, 3) => c_5_27_3(slot, args),
            (5, 28, 0) => c_5_28_0(slot, args),
            (5, 28, 1) => c_5_28_1(slot, args),
            (5, 28, 2) => c_5_28_2(slot, args),
            (5, 28, 3) => c_5_28_3(slot, args),
            (5, 29, 0) => c_5_29_0(slot, args),
            (5, 29, 1) => c_5_29_1(slot, args),
            (5, 29, 2) => c_5_29_2(slot, args),
            (5, 29, 3) => c_5_29_3(slot, args),
            (5, 30, 0) => c_5_30_0(slot, args),
            (5, 30, 1) => c_5_30_1(slot, args),
            (5, 30, 2) => c_5_30_2(slot, args),
            (5, 30, 3) => c_5_30_3(slot, args),
            (5, 31, 0) => c_5_31_0(slot, args),
            (5, 31, 1) => c_5_31_1(slot, args),
            (5, 31, 2) => c_5_31_2(slot, args),
            (5, 31, 3) => c_5_31_3(slot, args),
            (6, 0, 0) => c_6_0_0(slot, args),
            (6, 0, 1) => c_6_0_1(slot, args),
            (6, 0, 2) => c_6_0_2(slot, args),
            (6, 0, 3) => c_6_0_3(slot, args),
            (6, 1, 0) => c_6_1_0(slot, args),
            (6, 1, 1) => c_6_1_1(slot, args),
            (6, 1, 2) => c_6_1_2(slot, args),
            (6, 1, 3) => c_6_1_3(slot, args),
            (6, 2, 0) => c_6_2_0(slot, args),
            (6, 2, 1) => c_6_2_1(slot, args),
            (6, 2, 2) => c_6_2_2(slot, args),
            (6, 2, 3) => c_6_2_3(slot, args),
            (6, 3, 0) => c_6_3_0(slot, args),
            (6, 3, 1) => c_6_3_1(slot, args),
            (6, 3, 2) => c_6_3_2(slot, args),
            (6, 3, 3) => c_6_3_3(slot, args),
            (6, 4, 0) => c_6_4_0(slot, args),
            (6, 4, 1) => c_6_4_1(slot, args),
            (6, 4, 2) => c_6_4_2(slot, args),
            (6, 4, 3) => c_6_4_3(slot, args),
            (6, 5, 0) => c_6_5_0(slot, args),
            (6, 5, 1) => c_6_5_1(slot, args),
            (6, 5, 2) => c_6_5_2(slot, args),
            (6, 5, 3) => c_6_5_3(slot, args),
            (6, 6, 0) => c_6_6_0(slot, args),
            (6, 6, 1) => c_6_6_1(slot, args),
            (6, 6, 2) => c_6_6_2(slot, args),
            (6, 6, 3) => c_6_6_3(slot, args),
            (6, 7, 0) => c_6_7_0(slot, args),
            (6, 7, 1) => c_6_7_1(slot, args),
            (6, 7, 2) => c_6_7_2(slot, args),
            (6, 7, 3) => c_6_7_3(slot, args),
            (6, 8, 0) => c_6_8_0(slot, args),
            (6, 8, 1) => c_6_8_1(slot, args),
            (6, 8, 2) => c_6_8_2(slot, args),
            (6, 8, 3) => c_6_8_3(slot, args),
            (6, 9, 0) => c_6_9_0(slot, args),
            (6, 9, 1) => c_6_9_1(slot, args),
            (6, 9, 2) => c_6_9_2(slot, args),
            (6, 9, 3) => c_6_9_3(slot, args),
            (6, 10, 0) => c_6_10_0(slot, args),
            (6, 10, 1) => c_6_10_1(slot, args),
            (6, 10, 2) => c_6_10_2(slot, args),
            (6, 10, 3) => c_6_10_3(slot, args),
            (6, 11, 0) => c_6_11_0(slot, args),
            (6, 11, 1) => c_6_11_1(slot, args),
            (6, 11, 2) => c_6_11_2(slot, args),
            (6, 11, 3) => c_6_11_3(slot, args),
            (6, 12, 0) => c_6_12_0(slot, args),
            (6, 12, 1) => c_6_12_1(slot, args),
            (6, 12, 2) => c_6_12_2(slot, args),
            (6, 12, 3) => c_6_12_3(slot, args),
            (6, 13, 0) => c_6_13_0(slot, args),
            (6, 13, 1) => c_6_13_1(slot, args),
            (6, 13, 2) => c_6_13_2(slot, args),
            (6, 13, 3) => c_6_13_3(slot, args),
            (6, 14, 0) => c_6_14_0(slot, args),
            (6, 14, 1) => c_6_14_1(slot, args),
            (6, 14, 2) => c_6_14_2(slot, args),
            (6, 14, 3) => c_6_14_3(slot, args),
            (6, 15, 0) => c_6_15_0(slot, args),
            (6, 15, 1) => c_6_15_1(slot, args),
            (6, 15, 2) => c_6_15_2(slot, args),
            (6, 15, 3) => c_6_15_3(slot, args),
            (6, 16, 0) => c_6_16_0(slot, args),
            (6, 16, 1) => c_6_16_1(slot, args),
            (6, 16, 2) => c_6_16_2(slot, args),
            (6, 16, 3) => c_6_16_3(slot, args),
            (6, 17, 0) => c_6_17_0(slot, args),
            (6, 17, 1) => c_6_17_1(slot, args),
            (6, 17, 2) => c_6_17_2(slot, args),
            (6, 17, 3) => c_6_17_3(slot, args),
            (6, 18, 0) => c_6_18_0(slot, args),
            (6, 18, 1) => c_6_18_1(slot, args),
            (6, 18, 2) => c_6_18_2(slot, args),
            (6, 18, 3) => c_6_18_3(slot, args),
            (6, 19, 0) => c_6_19_0(slot, args),
            (6, 19, 1) => c_6_19_1(slot, args),
            (6, 19, 2) => c_6_19_2(slot, args),
            (6, 19, 3) => c_6_19_3(slot, args),
            (6, 20, 0) => c_6_20_0(slot, args),
            (6, 20, 1) => c_6_20_1(slot, args),
            (6, 20, 2) => c_6_20_2(slot, args),
            (6, 20, 3) => c_6_20_3(slot, args),
            (6, 21, 0) => c_6_21_0(slot, args),
            (6, 21, 1) => c_6_21_1(slot, args),
            (6, 21, 2) => c_6_21_2(slot, args),
            (6, 21, 3) => c_6_21_3(slot, args),
            (6, 22, 0) => c_6_22_0(slot, args),
            (6, 22, 1) => c_6_22_1(slot, args),
            (6, 22, 2) => c_6_22_2(slot, args),
            (6, 22, 3) => c_6_22_3(slot, args),
            (6, 23, 0) => c_6_23_0(slot, args),
            (6, 23, 1) => c_6_23_1(slot, args),
            (6, 23, 2) => c_6_23_2(slot, args),
            (6, 23, 3) => c_6_23_3(slot, args),
            (6, 24, 0) => c_6_24_0(slot, args),
            (6, 24, 1) => c_6_24_1(slot, args),
            (6, 24, 2) => c_6_24_2(slot, args),
            (6, 24, 3) => c_6_24_3(slot, args),
            (6, 25, 0) => c_6_25_0(slot, args),
            (6, 25, 1) => c_6_25_1(slot, args),
            (6, 25, 2) => c_6_25_2(slot, args),
            (6, 25, 3) => c_6_25_3(slot, args),
            (6, 26, 0) => c_6_26_0(slot, args),
            (6, 26, 1) => c_6_26_1(slot, args),
            (6, 26, 2) => c_6_26_2(slot, args),
            (6, 26, 3) => c_6_26_3(slot, args),
            (6, 27, 0) => c_6_27_0(slot, args),
            (6, 27, 1) => c_6_27_1(slot, args),
            (6, 27, 2) => c_6_27_2(slot, args),
            (6, 27, 3) => c_6_27_3(slot, args),
            (6, 28, 0) => c_6_28_0(slot, args),
            (6, 28, 1) => c_6_28_1(slot, args),
            (6, 28, 2) => c_6_28_2(slot, args),
            (6, 28, 3) => c_6_28_3(slot, args),
            (6, 29, 0) => c_6_29_0(slot, args),
            (6, 29, 1) => c_6_29_1(slot, args),
            (6, 29, 2) => c_6_29_2(slot, args),
            (6, 29, 3) => c_6_29_3(slot, args),
            (6, 30, 0) => c_6_30_0(slot, args),
            (6, 30, 1) => c_6_30_1(slot, args),
            (6, 30, 2) => c_6_30_2(slot, args),
            (6, 30, 3) => c_6_30_3(slot, args),
            (6, 31, 0) => c_6_31_0(slot, args),
            (6, 31, 1) => c_6_31_1(slot, args),
            (6, 31, 2) => c_6_31_2(slot, args),
            (6, 31, 3) => c_6_31_3(slot, args),
            (6, 32, 0) => c_6_32_0(slot, args),
            (6, 32, 1) => c_6_32_1(slot, args),
            (6, 32, 2) => c_6_32_2(slot, args),
            (6, 32, 3) => c_6_32_3(slot, args),
            (6, 33, 0) => c_6_33_0(slot, args),
            (6, 33, 1) => c_6_33_1(slot, args),
            (6, 33, 2) => c_6_33_2(slot, args),
            (6, 33, 3) => c_6_33_3(slot, args),
            (6, 34, 0) => c_6_34_0(slot, args),
            (6, 34, 1) => c_6_34_1(slot, args),
            (6, 34, 2) => c_6_34_2(slot, args),
            (6, 34, 3) => c_6_34_3(slot, args),
            (6, 35, 0) => c_6_35_0(slot, args),
            (6, 35, 1) => c_6_35_1(slot, args),
            (6, 35, 2) => c_6_35_2(slot, args),
            (6, 35, 3) => c_6_35_3(slot, args),
            (6, 36, 0) => c_6_36_0(slot, args),
            (6, 36, 1) => c_6_36_1(slot, args),
            (6, 36, 2) => c_6_36_2(slot, args),
            (6, 36, 3) => c_6_36_3(slot, args),
            (6, 37, 0) => c_6_37_0(slot, args),
            (6, 37, 1) => c_6_37_1(slot, args),
            (6, 37, 2) => c_6_37_2(slot, args),
            (6, 37, 3) => c_6_37_3(slot, args),
            (6, 38, 0) => c_6_38_0(slot, args),
            (6, 38, 1) => c_6_38_1(slot, args),
            (6, 38, 2) => c_6_38_2(slot, args),
            (6, 38, 3) => c_6_38_3(slot, args),
            (6, 39, 0) => c_6_39_0(slot, args),
            (6, 39, 1) => c_6_39_1(slot, args),
            (6, 39, 2) => c_6_39_2(slot, args),
            (6, 39, 3) => c_6_39_3(slot, args),
            (6, 40, 0) => c_6_40_0(slot, args),
            (6, 40, 1) => c_6_40_1(slot, args),
            (6, 40, 2) => c_6_40_2(slot, args),
            (6, 40, 3) => c_6_40_3(slot, args),
            (6, 41, 0) => c_6_41_0(slot, args),
            (6, 41, 1) => c_6_41_1(slot, args),
            (6, 41, 2) => c_6_41_2(slot, args),
            (6, 41, 3) => c_6_41_3(slot, args),
            (6, 42, 0) => c_6_42_0(slot, args),
            (6, 42, 1) => c_6_42_1(slot, args),
            (6, 42, 2) => c_6_42_2(slot, args),
            (6, 42, 3) => c_6_42_3(slot, args),
            (6, 43, 0) => c_6_43_0(slot, args),
            (6, 43, 1) => c_6_43_1(slot, args),
            (6, 43, 2) => c_6_43_2(slot, args),
            (6, 43, 3) => c_6_43_3(slot, args),
            (6, 44, 0) => c_6_44_0(slot, args),
            (6, 44, 1) => c_6_44_1(slot, args),
            (6, 44, 2) => c_6_44_2(slot, args),
            (6, 44, 3) => c_6_44_3(slot, args),
            (6, 45, 0) => c_6_45_0(slot, args),
            (6, 45, 1) => c_6_45_1(slot, args),
            (6, 45, 2) => c_6_45_2(slot, args),
            (6, 45, 3) => c_6_45_3(slot, args),
            (6, 46, 0) => c_6_46_0(slot, args),
            (6, 46, 1) => c_6_46_1(slot, args),
            (6, 46, 2) => c_6_46_2(slot, args),
            (6, 46, 3) => c_6_46_3(slot, args),
            (6, 47, 0) => c_6_47_0(slot, args),
            (6, 47, 1) => c_6_47_1(slot, args),
            (6, 47, 2) => c_6_47_2(slot, args),
            (6, 47, 3) => c_6_47_3(slot, args),
            (6, 48, 0) => c_6_48_0(slot, args),
            (6, 48, 1) => c_6_48_1(slot, args),
            (6, 48, 2) => c_6_48_2(slot, args),
            (6, 48, 3) => c_6_48_3(slot, args),
            (6, 49, 0) => c_6_49_0(slot, args),
            (6, 49, 1) => c_6_49_1(slot, args),
            (6, 49, 2) => c_6_49_2(slot, args),
            (6, 49, 3) => c_6_49_3(slot, args),
            (6, 50, 0) => c_6_50_0(slot, args),
            (6, 50, 1) => c_6_50_1(slot, args),
            (6, 50, 2) => c_6_50_2(slot, args),
            (6, 50, 3) => c_6_50_3(slot, args),
            (6, 51, 0) => c_6_51_0(slot, args),
            (6, 51, 1) => c_6_51_1(slot, args),
            (6, 51, 2) => c_6_51_2(slot, args),
            (6, 51, 3) => c_6_51_3(slot, args),
            (6, 52, 0) => c_6_52_0(slot, args),
            (6, 52, 1) => c_6_52_1(slot, args),
            (6, 52, 2) => c_6_52_2(slot, args),
            (6, 52, 3) => c_6_52_3(slot, args),
            (6, 53, 0) => c_6_53_0(slot, args),
            (6, 53, 1) => c_6_53_1(slot, args),
            (6, 53, 2) => c_6_53_2(slot, args),
            (6, 53, 3) => c_6_53_3(slot, args),
            (6, 54, 0) => c_6_54_0(slot, args),
            (6, 54, 1) => c_6_54_1(slot, args),
            (6, 54, 2) => c_6_54_2(slot, args),
            (6, 54, 3) => c_6_54_3(slot, args),
            (6, 55, 0) => c_6_55_0(slot, args),
            (6, 55, 1) => c_6_55_1(slot, args),
            (6, 55, 2) => c_6_55_2(slot, args),
            (6, 55, 3) => c_6_55_3(slot, args),
            (6, 56, 0) => c_6_56_0(slot, args),
            (6, 56, 1) => c_6_56_1(slot, args),
            (6, 56, 2) => c_6_56_2(slot, args),
            (6, 56, 3) => c_6_56_3(slot, args),
            (6, 57, 0) => c_6_57_0(slot, args),
            (6, 57, 1) => c_6_57_1(slot, args),
            (6, 57, 2) => c_6_57_2(slot, args),
            (6, 57, 3) => c_6_57_3(slot, args),
            (6, 58, 0) => c_6_58_0(slot, args),
            (6, 58, 1) => c_6_58_1(slot, args),
            (6, 58, 2) => c_6_58_2(slot, args),
            (6, 58, 3) => c_6_58_3(slot, args),
            (6, 59, 0) => c_6_59_0(slot, args),
            (6, 59, 1) => c_6_59_1(slot, args),
            (6, 59, 2) => c_6_59_2(slot, args),
            (6, 59, 3) => c_6_59_3(slot, args),
            (6, 60, 0) => c_6_60_0(slot, args),
            (6, 60, 1) => c_6_60_1(slot, args),
            (6, 60, 2) => c_6_60_2(slot, args),
            (6, 60, 3) => c_6_60_3(slot, args),
            (6, 61, 0) => c_6_61_0(slot, args),
            (6, 61, 1) => c_6_61_1(slot, args),
            (6, 61, 2) => c_6_61_2(slot, args),
            (6, 61, 3) => c_6_61_3(slot, args),
            (6, 62, 0) => c_6_62_0(slot, args),
            (6, 62, 1) => c_6_62_1(slot, args),
            (6, 62, 2) => c_6_62_2(slot, args),
            (6, 62, 3) => c_6_62_3(slot, args),
            (6, 63, 0) => c_6_63_0(slot, args),
            (6, 63, 1) => c_6_63_1(slot, args),
            (6, 63, 2) => c_6_63_2(slot, args),
            (6, 63, 3) => c_6_63_3(slot, args),
            (7, 0, 0) => c_7_0_0(slot, args),
            (7, 0, 1) => c_7_0_1(slot, args),
            (7, 0, 2) => c_7_0_2(slot, args),
            (7, 0, 3) => c_7_0_3(slot, args),
            (7, 1, 0) => c_7_1_0(slot, args),
            (7, 1, 1) => c_7_1_1(slot, args),
            (7, 1, 2) => c_7_1_2(slot, args),
            (7, 1, 3) => c_7_1_3(slot, args),
            (7, 2, 0) => c_7_2_0(slot, args),
            (7, 2, 1) => c_7_2_1(slot, args),
            (7, 2, 2) => c_7_2_2(slot, args),
            (7, 2, 3) => c_7_2_3(slot, args),
            (7, 3, 0) => c_7_3_0(slot, args),
            (7, 3, 1) => c_7_3_1(slot, args),
            (7, 3, 2) => c_7_3_2(slot, args),
            (7, 3, 3) => c_7_3_3(slot, args),
            (7, 4, 0) => c_7_4_0(slot, args),
            (7, 4, 1) => c_7_4_1(slot, args),
            (7, 4, 2) => c_7_4_2(slot, args),
            (7, 4, 3) => c_7_4_3(slot, args),
            (7, 5, 0) => c_7_5_0(slot, args),
            (7, 5, 1) => c_7_5_1(slot, args),
            (7, 5, 2) => c_7_5_2(slot, args),
            (7, 5, 3) => c_7_5_3(slot, args),
            (7, 6, 0) => c_7_6_0(slot, args),
            (7, 6, 1) => c_7_6_1(slot, args),
            (7, 6, 2) => c_7_6_2(slot, args),
            (7, 6, 3) => c_7_6_3(slot, args),
            (7, 7, 0) => c_7_7_0(slot, args),
            (7, 7, 1) => c_7_7_1(slot, args),
            (7, 7, 2) => c_7_7_2(slot, args),
            (7, 7, 3) => c_7_7_3(slot, args),
            (7, 8, 0) => c_7_8_0(slot, args),
            (7, 8, 1) => c_7_8_1(slot, args),
            (7, 8, 2) => c_7_8_2(slot, args),
            (7, 8, 3) => c_7_8_3(slot, args),
            (7, 9, 0) => c_7_9_0(slot, args),
            (7, 9, 1) => c_7_9_1(slot, args),
            (7, 9, 2) => c_7_9_2(slot, args),
            (7, 9, 3) => c_7_9_3(slot, args),
            (7, 10, 0) => c_7_10_0(slot, args),
            (7, 10, 1) => c_7_10_1(slot, args),
            (7, 10, 2) => c_7_10_2(slot, args),
            (7, 10, 3) => c_7_10_3(slot, args),
            (7, 11, 0) => c_7_11_0(slot, args),
            (7, 11, 1) => c_7_11_1(slot, args),
            (7, 11, 2) => c_7_11_2(slot, args),
            (7, 11, 3) => c_7_11_3(slot, args),
            (7, 12, 0) => c_7_12_0(slot, args),
            (7, 12, 1) => c_7_12_1(slot, args),
            (7, 12, 2) => c_7_12_2(slot, args),
            (7, 12, 3) => c_7_12_3(slot, args),
            (7, 13, 0) => c_7_13_0(slot, args),
            (7, 13, 1) => c_7_13_1(slot, args),
            (7, 13, 2) => c_7_13_2(slot, args),
            (7, 13, 3) => c_7_13_3(slot, args),
            (7, 14, 0) => c_7_14_0(slot, args),
            (7, 14, 1) => c_7_14_1(slot, args),
            (7, 14, 2) => c_7_14_2(slot, args),
            (7, 14, 3) => c_7_14_3(slot, args),
            (7, 15, 0) => c_7_15_0(slot, args),
            (7, 15, 1) => c_7_15_1(slot, args),
            (7, 15, 2) => c_7_15_2(slot, args),
            (7, 15, 3) => c_7_15_3(slot, args),
            (7, 16, 0) => c_7_16_0(slot, args),
            (7, 16, 1) => c_7_16_1(slot, args),
            (7, 16, 2) => c_7_16_2(slot, args),
            (7, 16, 3) => c_7_16_3(slot, args),
            (7, 17, 0) => c_7_17_0(slot, args),
            (7, 17, 1) => c_7_17_1(slot, args),
            (7, 17, 2) => c_7_17_2(slot, args),
            (7, 17, 3) => c_7_17_3(slot, args),
            (7, 18, 0) => c_7_18_0(slot, args),
            (7, 18, 1) => c_7_18_1(slot, args),
            (7, 18, 2) => c_7_18_2(slot, args),
            (7, 18, 3) => c_7_18_3(slot, args),
            (7, 19, 0) => c_7_19_0(slot, args),
            (7, 19, 1) => c_7_19_1(slot, args),
            (7, 19, 2) => c_7_19_2(slot, args),
            (7, 19, 3) => c_7_19_3(slot, args),
            (7, 20, 0) => c_7_20_0(slot, args),
            (7, 20, 1) => c_7_20_1(slot, args),
            (7, 20, 2) => c_7_20_2(slot, args),
            (7, 20, 3) => c_7_20_3(slot, args),
            (7, 21, 0) => c_7_21_0(slot, args),
            (7, 21, 1) => c_7_21_1(slot, args),
            (7, 21, 2) => c_7_21_2(slot, args),
            (7, 21, 3) => c_7_21_3(slot, args),
            (7, 22, 0) => c_7_22_0(slot, args),
            (7, 22, 1) => c_7_22_1(slot, args),
            (7, 22, 2) => c_7_22_2(slot, args),
            (7, 22, 3) => c_7_22_3(slot, args),
            (7, 23, 0) => c_7_23_0(slot, args),
            (7, 23, 1) => c_7_23_1(slot, args),
            (7, 23, 2) => c_7_23_2(slot, args),
            (7, 23, 3) => c_7_23_3(slot, args),
            (7, 24, 0) => c_7_24_0(slot, args),
            (7, 24, 1) => c_7_24_1(slot, args),
            (7, 24, 2) => c_7_24_2(slot, args),
            (7, 24, 3) => c_7_24_3(slot, args),
            (7, 25, 0) => c_7_25_0(slot, args),
            (7, 25, 1) => c_7_25_1(slot, args),
            (7, 25, 2) => c_7_25_2(slot, args),
            (7, 25, 3) => c_7_25_3(slot, args),
            (7, 26, 0) => c_7_26_0(slot, args),
            (7, 26, 1) => c_7_26_1(slot, args),
            (7, 26, 2) => c_7_26_2(slot, args),
            (7, 26, 3) => c_7_26_3(slot, args),
            (7, 27, 0) => c_7_27_0(slot, args),
            (7, 27, 1) => c_7_27_1(slot, args),
            (7, 27, 2) => c_7_27_2(slot, args),
            (7, 27, 3) => c_7_27_3(slot, args),
            (7, 28, 0) => c_7_28_0(slot, args),
            (7, 28, 1) => c_7_28_1(slot, args),
            (7, 28, 2) => c_7_28_2(slot, args),
            (7, 28, 3) => c_7_28_3(slot, args),
            (7, 29, 0) => c_7_29_0(slot, args),
            (7, 29, 1) => c_7_29_1(slot, args),
            (7, 29, 2) => c_7_29_2(slot, args),
            (7, 29, 3) => c_7_29_3(slot, args),
            (7, 30, 0) => c_7_30_0(slot, args),
            (7, 30, 1) => c_7_30_1(slot, args),
            (7, 30, 2) => c_7_30_2(slot, args),
            (7, 30, 3) => c_7_30_3(slot, args),
            (7, 31, 0) => c_7_31_0(slot, args),
            (7, 31, 1) => c_7_31_1(slot, args),
            (7, 31, 2) => c_7_31_2(slot, args),
            (7, 31, 3) => c_7_31_3(slot, args),
            (7, 32, 0) => c_7_32_0(slot, args),
            (7, 32, 1) => c_7_32_1(slot, args),
            (7, 32, 2) => c_7_32_2(slot, args),
            (7, 32, 3) => c_7_32_3(slot, args),
            (7, 33, 0) => c_7_33_0(slot, args),
            (7, 33, 1) => c_7_33_1(slot, args),
            (7, 33, 2) => c_7_33_2(slot, args),
            (7, 33, 3) => c_7_33_3(slot, args),
            (7, 34, 0) => c_7_34_0(slot, args),
            (7, 34, 1) => c_7_34_1(slot, args),
            (7, 34, 2) => c_7_34_2(slot, args),
            (7, 34, 3) => c_7_34_3(slot, args),
            (7, 35, 0) => c_7_35_0(slot, args),
            (7, 35, 1) => c_7_35_1(slot, args),
            (7, 35, 2) => c_7_35_2(slot, args),
            (7, 35, 3) => c_7_35_3(slot, args),
            (7, 36, 0) => c_7_36_0(slot, args),
            (7, 36, 1) => c_7_36_1(slot, args),
            (7, 36, 2) => c_7_36_2(slot, args),
            (7, 36, 3) => c_7_36_3(slot, args),
            (7, 37, 0) => c_7_37_0(slot, args),
            (7, 37, 1) => c_7_37_1(slot, args),
            (7, 37, 2) => c_7_37_2(slot, args),
            (7, 37, 3) => c_7_37_3(slot, args),
            (7, 38, 0) => c_7_38_0(slot, args),
            (7, 38, 1) => c_7_38_1(slot, args),
            (7, 38, 2) => c_7_38_2(slot, args),
            (7, 38, 3) => c_7_38_3(slot, args),
            (7, 39, 0) => c_7_39_0(slot, args),
            (7, 39, 1) => c_7_39_1(slot, args),
            (7, 39, 2) => c_7_39_2(slot, args),
            (7, 39, 3) => c_7_39_3(slot, args),
            (7, 40, 0) => c_7_40_0(slot, args),
            (7, 40, 1) => c_7_40_1(slot, args),
            (7, 40, 2) => c_7_40_2(slot, args),
            (7, 40, 3) => c_7_40_3(slot, args),
            (7, 41, 0) => c_7_41_0(slot, args),
            (7, 41, 1) => c_7_41_1(slot, args),
            (7, 41, 2) => c_7_41_2(slot, args),
            (7, 41, 3) => c_7_41_3(slot, args),
            (7, 42, 0) => c_7_42_0(slot, args),
            (7, 42, 1) => c_7_42_1(slot, args),
            (7, 42, 2) => c_7_42_2(slot, args),
            (7, 42, 3) => c_7_42_3(slot, args),
            (7, 43, 0) => c_7_43_0(slot, args),
            (7, 43, 1) => c_7_43_1(slot, args),
            (7, 43, 2) => c_7_43_2(slot, args),
            (7, 43, 3) => c_7_43_3(slot, args),
            (7, 44, 0) => c_7_44_0(slot, args),
            (7, 44, 1) => c_7_44_1(slot, args),
            (7, 44, 2) => c_7_44_2(slot, args),
            (7, 44, 3) => c_7_44_3(slot, args),
            (7, 45, 0) => c_7_45_0(slot, args),
            (7, 45, 1) => c_7_45_1(slot, args),
            (7, 45, 2) => c_7_45_2(slot, args),
            (7, 45, 3) => c_7_45_3(slot, args),
            (7, 46, 0) => c_7_46_0(slot, args),
            (7, 46, 1) => c_7_46_1(slot, args),
            (7, 46, 2) => c_7_46_2(slot, args),
            (7, 46, 3) => c_7_46_3(slot, args),
            (7, 47, 0) => c_7_47_0(slot, args),
            (7, 47, 1) => c_7_47_1(slot, args),
            (7, 47, 2) => c_7_47_2(slot, args),
            (7, 47, 3) => c_7_47_3(slot, args),
            (7, 48, 0) => c_7_48_0(slot, args),
            (7, 48, 1) => c_7_48_1(slot, args),
            (7, 48, 2) => c_7_48_2(slot, args),
            (7, 48, 3) => c_7_48_3(slot, args),
            (7, 49, 0) => c_7_49_0(slot, args),
            (7, 49, 1) => c_7_49_1(slot, args),
            (7, 49, 2) => c_7_49_2(slot, args),
            (7, 49, 3) => c_7_49_3(slot, args),
            (7, 50, 0) => c_7_50_0(slot, args),
            (7, 50, 1) => c_7_50_1(slot, args),
            (7, 50, 2) => c_7_50_2(slot, args),
            (7, 50, 3) => c_7_50_3(slot, args),
            (7, 51, 0) => c_7_51_0(slot, args),
            (7, 51, 1) => c_7_51_1(slot, args),
            (7, 51, 2) => c_7_51_2(slot, args),
            (7, 51, 3) => c_7_51_3(slot, args),
            (7, 52, 0) => c_7_52_0(slot, args),
            (7, 52, 1) => c_7_52_1(slot, args),
            (7, 52, 2) => c_7_52_2(slot, args),
            (7, 52, 3) => c_7_52_3(slot, args),
            (7, 53, 0) => c_7_53_0(slot, args),
            (7, 53, 1) => c_7_53_1(slot, args),
            (7, 53, 2) => c_7_53_2(slot, args),
            (7, 53, 3) => c_7_53_3(slot, args),
            (7, 54, 0) => c_7_54_0(slot, args),
            (7, 54, 1) => c_7_54_1(slot, args),
            (7, 54, 2) => c_7_54_2(slot, args),
            (7, 54, 3) => c_7_54_3(slot, args),
            (7, 55, 0) => c_7_55_0(slot, args),
            (7, 55, 1) => c_7_55_1(slot, args),
            (7, 55, 2) => c_7_55_2(slot, args),
            (7, 55, 3) => c_7_55_3(slot, args),
            (7, 56, 0) => c_7_56_0(slot, args),
            (7, 56, 1) => c_7_56_1(slot, args),
            (7, 56, 2) => c_7_56_2(slot, args),
            (7, 56, 3) => c_7_56_3(slot, args),
            (7, 57, 0) => c_7_57_0(slot, args),
            (7, 57, 1) => c_7_57_1(slot, args),
            (7, 57, 2) => c_7_57_2(slot, args),
            (7, 57, 3) => c_7_57_3(slot, args),
            (7, 58, 0) => c_7_58_0(slot, args),
            (7, 58, 1) => c_7_58_1(slot, args),
            (7, 58, 2) => c_7_58_2(slot, args),
            (7, 58, 3) => c_7_58_3(slot, args),
            (7, 59, 0) => c_7_59_0(slot, args),
            (7, 59, 1) => c_7_59_1(slot, args),
            (7, 59, 2) => c_7_59_2(slot, args),
            (7, 59, 3) => c_7_59_3(slot, args),
            (7, 60, 0) => c_7_60_0(slot, args),
            (7, 60, 1) => c_7_60_1(slot, args),
            (7, 60, 2) => c_7_60_2(slot, args),
            (7, 60, 3) => c_7_60_3(slot, args),
            (7, 61, 0) => c_7_61_0(slot, args),
            (7, 61, 1) => c_7_61_1(slot, args),
            (7, 61, 2) => c_7_61_2(slot, args),
            (7, 61, 3) => c_7_61_3(slot, args),
            (7, 62, 0) => c_7_62_0(slot, args),
            (7, 62, 1) => c_7_62_1(slot, args),
            (7, 62, 2) => c_7_62_2(slot, args),
            (7, 62, 3) => c_7_62_3(slot, args),
            (7, 63, 0) => c_7_63_0(slot, args),
            (7, 63, 1) => c_7_63_1(slot, args),
            (7, 63, 2) => c_7_63_2(slot, args),
            (7, 63, 3) => c_7_63_3(slot, args),
            (7, 64, 0) => c_7_64_0(slot, args),
            (7, 64, 1) => c_7_64_1(slot, args),
            (7, 64, 2) => c_7_64_2(slot, args),
            (7, 64, 3) => c_7_64_3(slot, args),
            (7, 65, 0) => c_7_65_0(slot, args),
            (7, 65, 1) => c_7_65_1(slot, args),
            (7, 65, 2) => c_7_65_2(slot, args),
            (7, 65, 3) => c_7_65_3(slot, args),
            (7, 66, 0) => c_7_66_0(slot, args),
            (7, 66, 1) => c_7_66_1(slot, args),
            (7, 66, 2) => c_7_66_2(slot, args),
            (7, 66, 3) => c_7_66_3(slot, args),
            (7, 67, 0) => c_7_67_0(slot, args),
            (7, 67, 1) => c_7_67_1(slot, args),
            (7, 67, 2) => c_7_67_2(slot, args),
            (7, 67, 3) => c_7_67_3(slot, args),
            (7, 68, 0) => c_7_68_0(slot, args),
            (7, 68, 1) => c_7_68_1(slot, args),
            (7, 68, 2) => c_7_68_2(slot, args),
            (7, 68, 3) => c_7_68_3(slot, args),
            (7, 69, 0) => c_7_69_0(slot, args),
            (7, 69, 1) => c_7_69_1(slot, args),
            (7, 69, 2) => c_7_69_2(slot, args),
            (7, 69, 3) => c_7_69_3(slot, args),
            (7, 70, 0) => c_7_70_0(slot, args),
            (7, 70, 1) => c_7_70_1(slot, args),
            (7, 70, 2) => c_7_70_2(slot, args),
            (7, 70, 3) => c_7_70_3(slot, args),
            (7, 71, 0) => c_7_71_0(slot, args),
            (7, 71, 1) => c_7_71_1(slot, args),
            (7, 71, 2) => c_7_71_2(slot, args),
            (7, 71, 3) => c_7_71_3(slot, args),
            (7, 72, 0) => c_7_72_0(slot, args),
            (7, 72, 1) => c_7_72_1(slot, args),
            (7, 72, 2) => c_7_72_2(slot, args),
            (7, 72, 3) => c_7_72_3(slot, args),
            (7, 73, 0) => c_7_73_0(slot, args),
            (7, 73, 1) => c_7_73_1(slot, args),
            (7, 73, 2) => c_7_73_2(slot, args),
            (7, 73, 3) => c_7_73_3(slot, args),
            (7, 74, 0) => c_7_74_0(slot, args),
            (7, 74, 1) => c_7_74_1(slot, args),
            (7, 74, 2) => c_7_74_2(slot, args),
            (7, 74, 3) => c_7_74_3(slot, args),
            (7, 75, 0) => c_7_75_0(slot, args),
            (7, 75, 1) => c_7_75_1(slot, args),
            (7, 75, 2) => c_7_75_2(slot, args),
            (7, 75, 3) => c_7_75_3(slot, args),
            (7, 76, 0) => c_7_76_0(slot, args),
            (7, 76, 1) => c_7_76_1(slot, args),
            (7, 76, 2) => c_7_76_2(slot, args),
            (7, 76, 3) => c_7_76_3(slot, args),
            (7, 77, 0) => c_7_77_0(slot, args),
            (7, 77, 1) => c_7_77_1(slot, args),
            (7, 77, 2) => c_7_77_2(slot, args),
            (7, 77, 3) => c_7_77_3(slot, args),
            (7, 78, 0) => c_7_78_0(slot, args),
            (7, 78, 1) => c_7_78_1(slot, args),
            (7, 78, 2) => c_7_78_2(slot, args),
            (7, 78, 3) => c_7_78_3(slot, args),
            (7, 79, 0) => c_7_79_0(slot, args),
            (7, 79, 1) => c_7_79_1(slot, args),
            (7, 79, 2) => c_7_79_2(slot, args),
            (7, 79, 3) => c_7_79_3(slot, args),
            (7, 80, 0) => c_7_80_0(slot, args),
            (7, 80, 1) => c_7_80_1(slot, args),
            (7, 80, 2) => c_7_80_2(slot, args),
            (7, 80, 3) => c_7_80_3(slot, args),
            (7, 81, 0) => c_7_81_0(slot, args),
            (7, 81, 1) => c_7_81_1(slot, args),
            (7, 81, 2) => c_7_81_2(slot, args),
            (7, 81, 3) => c_7_81_3(slot, args),
            (7, 82, 0) => c_7_82_0(slot, args),
            (7, 82, 1) => c_7_82_1(slot, args),
            (7, 82, 2) => c_7_82_2(slot, args),
            (7, 82, 3) => c_7_82_3(slot, args),
            (7, 83, 0) => c_7_83_0(slot, args),
            (7, 83, 1) => c_7_83_1(slot, args),
            (7, 83, 2) => c_7_83_2(slot, args),
            (7, 83, 3) => c_7_83_3(slot, args),
            (7, 84, 0) => c_7_84_0(slot, args),
            (7, 84, 1) => c_7_84_1(slot, args),
            (7, 84, 2) => c_7_84_2(slot, args),
            (7, 84, 3) => c_7_84_3(slot, args),
            (7, 85, 0) => c_7_85_0(slot, args),
            (7, 85, 1) => c_7_85_1(slot, args),
            (7, 85, 2) => c_7_85_2(slot, args),
            (7, 85, 3) => c_7_85_3(slot, args),
            (7, 86, 0) => c_7_86_0(slot, args),
            (7, 86, 1) => c_7_86_1(slot, args),
            (7, 86, 2) => c_7_86_2(slot, args),
            (7, 86, 3) => c_7_86_3(slot, args),
            (7, 87, 0) => c_7_87_0(slot, args),
            (7, 87, 1) => c_7_87_1(slot, args),
            (7, 87, 2) => c_7_87_2(slot, args),
            (7, 87, 3) => c_7_87_3(slot, args),
            (7, 88, 0) => c_7_88_0(slot, args),
            (7, 88, 1) => c_7_88_1(slot, args),
            (7, 88, 2) => c_7_88_2(slot, args),
            (7, 88, 3) => c_7_88_3(slot, args),
            (7, 89, 0) => c_7_89_0(slot, args),
            (7, 89, 1) => c_7_89_1(slot, args),
            (7, 89, 2) => c_7_89_2(slot, args),
            (7, 89, 3) => c_7_89_3(slot, args),
            (7, 90, 0) => c_7_90_0(slot, args),
            (7, 90, 1) => c_7_90_1(slot, args),
            (7, 90, 2) => c_7_90_2(slot, args),
            (7, 90, 3) => c_7_90_3(slot, args),
            (7, 91, 0) => c_7_91_0(slot, args),
            (7, 91, 1) => c_7_91_1(slot, args),
            (7, 91, 2) => c_7_91_2(slot, args),
            (7, 91, 3) => c_7_91_3(slot, args),
            (7, 92, 0) => c_7_92_0(slot, args),
            (7, 92, 1) => c_7_92_1(slot, args),
            (7, 92, 2) => c_7_92_2(slot, args),
            (7, 92, 3) => c_7_92_3(slot, args),
            (7, 93, 0) => c_7_93_0(slot, args),
            (7, 93, 1) => c_7_93_1(slot, args),
            (7, 93, 2) => c_7_93_2(slot, args),
            (7, 93, 3) => c_7_93_3(slot, args),
            (7, 94, 0) => c_7_94_0(slot, args),
            (7, 94, 1) => c_7_94_1(slot, args),
            (7, 94, 2) => c_7_94_2(slot, args),
            (7, 94, 3) => c_7_94_3(slot, args),
            (7, 95, 0) => c_7_95_0(slot, args),
            (7, 95, 1) => c_7_95_1(slot, args),
            (7, 95, 2) => c_7_95_2(slot, args),
            (7, 95, 3) => c_7_95_3(slot, args),
            (7, 96, 0) => c_7_96_0(slot, args),
            (7, 96, 1) => c_7_96_1(slot, args),
            (7, 96, 2) => c_7_96_2(slot, args),
            (7, 96, 3) => c_7_96_3(slot, args),
            (7, 97, 0) => c_7_97_0(slot, args),
            (7, 97, 1) => c_7_97_1(slot, args),
            (7, 97, 2) => c_7_97_2(slot, args),
            (7, 97, 3) => c_7_97_3(slot, args),
            (7, 98, 0) => c_7_98_0(slot, args),
            (7, 98, 1) => c_7_98_1(slot, args),
            (7, 98, 2) => c_7_98_2(slot, args),
            (7, 98, 3) => c_7_98_3(slot, args),
            (7, 99, 0) => c_7_99_0(slot, args),
            (7, 99, 1) => c_7_99_1(slot, args),
            (7, 99, 2) => c_7_99_2(slot, args),
            (7, 99, 3) => c_7_99_3(slot, args),
            (7, 100, 0) => c_7_100_0(slot, args),
            (7, 100, 1) => c_7_100_1(slot, args),
            (7, 100, 2) => c_7_100_2(slot, args),
            (7, 100, 3) => c_7_100_3(slot, args),
            (7, 101, 0) => c_7_101_0(slot, args),
            (7, 101, 1) => c_7_101_1(slot, args),
            (7, 101, 2) => c_7_101_2(slot, args),
            (7, 101, 3) => c_7_101_3(slot, args),
            (7, 102, 0) => c_7_102_0(slot, args),
            (7, 102, 1) => c_7_102_1(slot, args),
            (7, 102, 2) => c_7_102_2(slot, args),
            (7, 102, 3) => c_7_102_3(slot, args),
            (7, 103, 0) => c_7_103_0(slot, args),
            (7, 103, 1) => c_7_103_1(slot, args),
            (7, 103, 2) => c_7_103_2(slot, args),
            (7, 103, 3) => c_7_103_3(slot, args),
            (7, 104, 0) => c_7_104_0(slot, args),
            (7, 104, 1) => c_7_104_1(slot, args),
            (7, 104, 2) => c_7_104_2(slot, args),
            (7, 104, 3) => c_7_104_3(slot, args),
            (7, 105, 0) => c_7_105_0(slot, args),
            (7, 105, 1) => c_7_105_1(slot, args),
            (7, 105, 2) => c_7_105_2(slot, args),
            (7, 105, 3) => c_7_105_3(slot, args),
            (7, 106, 0) => c_7_106_0(slot, args),
            (7, 106, 1) => c_7_106_1(slot, args),
            (7, 106, 2) => c_7_106_2(slot, args),
            (7, 106, 3) => c_7_106_3(slot, args),
            (7, 107, 0) => c_7_107_0(slot, args),
            (7, 107, 1) => c_7_107_1(slot, args),
            (7, 107, 2) => c_7_107_2(slot, args),
            (7, 107, 3) => c_7_107_3(slot, args),
            (7, 108, 0) => c_7_108_0(slot, args),
            (7, 108, 1) => c_7_108_1(slot, args),
            (7, 108, 2) => c_7_108_2(slot, args),
            (7, 108, 3) => c_7_108_3(slot, args),
            (7, 109, 0) => c_7_109_0(slot, args),
            (7, 109, 1) => c_7_109_1(slot, args),
            (7, 109, 2) => c_7_109_2(slot, args),
            (7, 109, 3) => c_7_109_3(slot, args),
            (7, 110, 0) => c_7_110_0(slot, args),
            (7, 110, 1) => c_7_110_1(slot, args),
            (7, 110, 2) => c_7_110_2(slot, args),
            (7, 110, 3) => c_7_110_3(slot, args),
            (7, 111, 0) => c_7_111_0(slot, args),
            (7, 111, 1) => c_7_111_1(slot, args),
            (7, 111, 2) => c_7_111_2(slot, args),
            (7, 111, 3) => c_7_111_3(slot, args),
            (7, 112, 0) => c_7_112_0(slot, args),
            (7, 112, 1) => c_7_112_1(slot, args),
            (7, 112, 2) => c_7_112_2(slot, args),
            (7, 112, 3) => c_7_112_3(slot, args),
            (7, 113, 0) => c_7_113_0(slot, args),
            (7, 113, 1) => c_7_113_1(slot, args),
            (7, 113, 2) => c_7_113_2(slot, args),
            (7, 113, 3) => c_7_113_3(slot, args),
            (7, 114, 0) => c_7_114_0(slot, args),
            (7, 114, 1) => c_7_114_1(slot, args),
            (7, 114, 2) => c_7_114_2(slot, args),
            (7, 114, 3) => c_7_114_3(slot, args),
            (7, 115, 0) => c_7_115_0(slot, args),
            (7, 115, 1) => c_7_115_1(slot, args),
            (7, 115, 2) => c_7_115_2(slot, args),
            (7, 115, 3) => c_7_115_3(slot, args),
            (7, 116, 0) => c_7_116_0(slot, args),
            (7, 116, 1) => c_7_116_1(slot, args),
            (7, 116, 2) => c_7_116_2(slot, args),
            (7, 116, 3) => c_7_116_3(slot, args),
            (7, 117, 0) => c_7_117_0(slot, args),
            (7, 117, 1) => c_7_117_1(slot, args),
            (7, 117, 2) => c_7_117_2(slot, args),
            (7, 117, 3) => c_7_117_3(slot, args),
            (7, 118, 0) => c_7_118_0(slot, args),
            (7, 118, 1) => c_7_118_1(slot, args),
            (7, 118, 2) => c_7_118_2(slot, args),
            (7, 118, 3) => c_7_118_3(slot, args),
            (7, 119, 0) => c_7_119_0(slot, args),
            (7, 119, 1) => c_7_119_1(slot, args),
            (7, 119, 2) => c_7_119_2(slot, args),
            (7, 119, 3) => c_7_119_3(slot, args),
            (7, 120, 0) => c_7_120_0(slot, args),
            (7, 120, 1) => c_7_120_1(slot, args),
            (7, 120, 2) => c_7_120_2(slot, args),
            (7, 120, 3) => c_7_120_3(slot, args),
            (7, 121, 0) => c_7_121_0(slot, args),
            (7, 121, 1) => c_7_121_1(slot, args),
            (7, 121, 2) => c_7_121_2(slot, args),
            (7, 121, 3) => c_7_121_3(slot, args),
            (7, 122, 0) => c_7_122_0(slot, args),
            (7, 122, 1) => c_7_122_1(slot, args),
            (7, 122, 2) => c_7_122_2(slot, args),
            (7, 122, 3) => c_7_122_3(slot, args),
            (7, 123, 0) => c_7_123_0(slot, args),
            (7, 123, 1) => c_7_123_1(slot, args),
            (7, 123, 2) => c_7_123_2(slot, args),
            (7, 123, 3) => c_7_123_3(slot, args),
            (7, 124, 0) => c_7_124_0(slot, args),
            (7, 124, 1) => c_7_124_1(slot, args),
            (7, 124, 2) => c_7_124_2(slot, args),
            (7, 124, 3) => c_7_124_3(slot, args),
            (7, 125, 0) => c_7_125_0(slot, args),
            (7, 125, 1) => c_7_125_1(slot, args),
            (7, 125, 2) => c_7_125_2(slot, args),
            (7, 125, 3) => c_7_125_3(slot, args),
            (7, 126, 0) => c_7_126_0(slot, args),
            (7, 126, 1) => c_7_126_1(slot, args),
            (7, 126, 2) => c_7_126_2(slot, args),
            (7, 126, 3) => c_7_126_3(slot, args),
            (7, 127, 0) => c_7_127_0(slot, args),
            (7, 127, 1) => c_7_127_1(slot, args),
            (7, 127, 2) => c_7_127_2(slot, args),
            (7, 127, 3) => c_7_127_3(slot, args),
            _ => return None,
        }
    })
}

pub fn call_uniform_f64(slot: usize, args: &[i64], result: Option<FuncSigVal>) -> Option<i64> {
    let tag = match result {
        None => 0,
        Some(FuncSigVal::I64) => 2,
        Some(FuncSigVal::F64) => 3,
        _ => return None,
    };
    Some(unsafe {
        match (args.len(), tag) {
            (1, 0) => f_1_0(slot, args),
            (1, 2) => f_1_2(slot, args),
            (1, 3) => f_1_3(slot, args),
            (2, 0) => f_2_0(slot, args),
            (2, 2) => f_2_2(slot, args),
            (2, 3) => f_2_3(slot, args),
            (3, 0) => f_3_0(slot, args),
            (3, 2) => f_3_2(slot, args),
            (3, 3) => f_3_3(slot, args),
            (4, 0) => f_4_0(slot, args),
            (4, 2) => f_4_2(slot, args),
            (4, 3) => f_4_3(slot, args),
            _ => return None,
        }
    })
}
