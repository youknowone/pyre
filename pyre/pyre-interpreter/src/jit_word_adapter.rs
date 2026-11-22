//! Word conversions for generated residual-call adapters.
//!
//! A generated adapter takes one register per calldescr argument (`i64`, or
//! `f64` for a float argument), converts each to the callee's parameter type,
//! calls the callee, and converts the result back to one register. Type
//! inference picks the conversion from the callee's own signature, so the
//! generator never spells a Rust type.

use core::ptr::NonNull;

pub trait JitWordArg: Sized {
    type Reg: Copy;
    fn from_reg(reg: Self::Reg) -> Self;
}

pub trait JitWordRet {
    type Reg: Copy;
    fn into_reg(self) -> Self::Reg;
}

macro_rules! word_int {
    ($($t:ty),* $(,)?) => {
        $(
            impl JitWordArg for $t {
                type Reg = i64;
                fn from_reg(reg: Self::Reg) -> Self {
                    reg as $t
                }
            }
            impl JitWordRet for $t {
                type Reg = i64;
                fn into_reg(self) -> Self::Reg {
                    self as i64
                }
            }
        )*
    };
}

word_int!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

impl JitWordArg for bool {
    type Reg = i64;
    fn from_reg(reg: Self::Reg) -> Self {
        reg != 0
    }
}

impl JitWordRet for bool {
    type Reg = i64;
    fn into_reg(self) -> Self::Reg {
        self as i64
    }
}

impl JitWordArg for char {
    type Reg = i64;
    fn from_reg(reg: Self::Reg) -> Self {
        char::from_u32(reg as u32).unwrap_or('\0')
    }
}

impl JitWordRet for char {
    type Reg = i64;
    fn into_reg(self) -> Self::Reg {
        self as i64
    }
}

impl<T> JitWordArg for *const T {
    type Reg = i64;
    fn from_reg(reg: Self::Reg) -> Self {
        reg as usize as *const T
    }
}

impl<T> JitWordRet for *const T {
    type Reg = i64;
    fn into_reg(self) -> Self::Reg {
        self as usize as i64
    }
}

impl<T> JitWordArg for *mut T {
    type Reg = i64;
    fn from_reg(reg: Self::Reg) -> Self {
        reg as usize as *mut T
    }
}

impl<T> JitWordRet for *mut T {
    type Reg = i64;
    fn into_reg(self) -> Self::Reg {
        self as usize as i64
    }
}

impl<T> JitWordArg for NonNull<T> {
    type Reg = i64;
    fn from_reg(reg: Self::Reg) -> Self {
        NonNull::new(reg as usize as *mut T).expect("null NonNull residual argument")
    }
}

impl<T> JitWordRet for NonNull<T> {
    type Reg = i64;
    fn into_reg(self) -> Self::Reg {
        self.as_ptr() as usize as i64
    }
}

impl<T> JitWordArg for Option<NonNull<T>> {
    type Reg = i64;
    fn from_reg(reg: Self::Reg) -> Self {
        NonNull::new(reg as usize as *mut T)
    }
}

impl<T> JitWordRet for Option<NonNull<T>> {
    type Reg = i64;
    fn into_reg(self) -> Self::Reg {
        self.map_or(0, |p| p.as_ptr() as usize as i64)
    }
}

/// Rebuild a shared reference from a residual register.
///
/// The caller owns `reg`'s lifetime: the pointer must stay valid for `'a`.
impl<'a, T: Sized> JitWordArg for &'a T {
    type Reg = i64;
    fn from_reg(reg: Self::Reg) -> Self {
        unsafe { &*(reg as usize as *const T) }
    }
}

/// Rebuild a unique reference from a residual register.
///
/// The caller owns `reg`'s lifetime: the pointer must stay valid for `'a`.
impl<'a, T: Sized> JitWordArg for &'a mut T {
    type Reg = i64;
    fn from_reg(reg: Self::Reg) -> Self {
        unsafe { &mut *(reg as usize as *mut T) }
    }
}

impl<'a, T: Sized> JitWordRet for &'a T {
    type Reg = i64;
    fn into_reg(self) -> Self::Reg {
        self as *const T as usize as i64
    }
}

impl<'a, T: Sized> JitWordRet for &'a mut T {
    type Reg = i64;
    fn into_reg(self) -> Self::Reg {
        self as *mut T as usize as i64
    }
}

impl JitWordArg for f64 {
    type Reg = f64;
    fn from_reg(reg: Self::Reg) -> Self {
        reg
    }
}

impl JitWordRet for f64 {
    type Reg = f64;
    fn into_reg(self) -> Self::Reg {
        self
    }
}

impl JitWordRet for () {
    type Reg = ();
    fn into_reg(self) -> Self::Reg {}
}

/// `PyError` is `#[repr(transparent)]` over `NonNull<PyObject>`: one word.
impl JitWordRet for crate::PyError {
    type Reg = i64;

    fn into_reg(self) -> Self::Reg {
        self.0.as_ptr() as usize as i64
    }
}

/// The calldescr marks the call can-raise; `GUARD_NO_EXCEPTION` tells a
/// published error from a zero result — same contract as
/// `is_true_lookup_jit_abi`.
impl<T> JitWordRet for Result<T, crate::PyError>
where
    T: JitWordRet,
    T::Reg: Default,
{
    type Reg = T::Reg;
    fn into_reg(self) -> Self::Reg {
        self.map_or_else(
            |e| {
                crate::runtime_ops::jit_publish_residual_error(e);
                T::Reg::default()
            },
            JitWordRet::into_reg,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{JitWordArg, JitWordRet};
    use core::ptr::NonNull;

    #[test]
    fn i64_round_trip() {
        assert_eq!(i64::from_reg(42), 42);
        assert_eq!(42i64.into_reg(), 42);
    }

    #[test]
    fn bool_round_trip() {
        assert!(bool::from_reg(1));
        assert!(!bool::from_reg(0));
        assert_eq!(true.into_reg(), 1);
        assert_eq!(false.into_reg(), 0);
    }

    #[test]
    fn option_nonnull_some_and_none() {
        let ptr = NonNull::<u8>::new(0x1000 as *mut u8).expect("nonzero");
        assert_eq!(
            Option::<NonNull<u8>>::from_reg(ptr.as_ptr() as usize as i64),
            Some(ptr)
        );
        assert_eq!(Some(ptr).into_reg(), ptr.as_ptr() as usize as i64);
        assert_eq!(Option::<NonNull<u8>>::from_reg(0), None);
        assert_eq!((None::<NonNull<u8>>).into_reg(), 0);
    }

    #[test]
    fn result_ok_i64() {
        let value: Result<i64, crate::PyError> = Ok(7);
        assert_eq!(value.into_reg(), 7);
    }
}
