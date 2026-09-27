//! `rpython/rtyper/lltypesystem/module/ll_math.py` — the C `llexternal`s.
//!
//! Upstream binds each `math_<name>` to the libm function of that name. The
//! codewriter retargets the `f64` inherent methods onto these (see
//! `F64_METHOD_LLEXTERNALS`), so every float operation of the interpreter
//! that reaches libm — float `**`, float `%`, `math.floor` — calls one of
//! them, and the JIT's residual call needs the address. They are IEEE and do
//! not raise; the raising `ll_math_*` wrappers stay around them.

macro_rules! math_unary {
    ($($name:ident => $method:ident),* $(,)?) => {
        $(
            pub extern "C" fn $name(x: f64) -> f64 {
                x.$method()
            }
        )*
    };
}

macro_rules! math_binary {
    ($($name:ident => $method:ident),* $(,)?) => {
        $(
            pub extern "C" fn $name(x: f64, y: f64) -> f64 {
                x.$method(y)
            }
        )*
    };
}

math_binary! {
    math_hypot => hypot,
    math_atan2 => atan2,
    math_copysign => copysign,
    math_pow => powf,
}

math_unary! {
    math_floor => floor,
    math_ceil => ceil,
    math_log => ln,
    math_log10 => log10,
    math_log1p => ln_1p,
    math_exp => exp,
    math_exp2 => exp2,
    math_expm1 => exp_m1,
    math_sqrt => sqrt,
    math_cbrt => cbrt,
    math_sin => sin,
    math_cos => cos,
    math_tan => tan,
    math_asin => asin,
    math_acos => acos,
    math_atan => atan,
    math_sinh => sinh,
    math_cosh => cosh,
    math_tanh => tanh,
    math_asinh => asinh,
    math_acosh => acosh,
    math_atanh => atanh,
}

/// `math_fmod = llexternal('fmod', ...)`. Also what `%` over two floats
/// lowers to: `lloperation.py` has no `float_mod`, so the codewriter emits a
/// residual call of this llexternal.
pub extern "C" fn math_fmod(x: f64, y: f64) -> f64 {
    x % y
}
