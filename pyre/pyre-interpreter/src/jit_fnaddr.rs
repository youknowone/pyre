//! Build-time fnaddr registry for pyre's traced helper surface.
//!
//! `pyre-jit-trace/build.rs` runs the source-only codewriter. Unlike the
//! proc-macro path, it cannot call `#[jit_module]::__majit_helper_trace_fnaddrs()`
//! on the analyzed sources, so pyre publishes the same shape explicitly here.

/// A type that occupies exactly one residual-call argument slot.
///
/// A residual call gives every argument one machine word, classified `'i'` /
/// `'r'` / `'f'` by `type_to_argclass` (`majit-translate` `codewriter::call`).
/// There is no two-register argument and no `sret`, so a parameter wider than
/// a word cannot be described. `T: Sized` is implicit on the reference impls
/// below, and that single fact is what rejects `&str`, `&[u8]`, `&Wtf8` and
/// `&dyn Trait`: a fat pointer is two words and the callee would read the
/// second one out of whatever the caller happened to leave there.
pub trait ResidualSlot {}

/// A type that fits the single residual-call result word.
///
/// `()` is result class `'v'` (`result_size == 0`). Everything else is one
/// word. An `Option<*mut T>` is *not* one word — a raw pointer has no niche,
/// so the option is 16 bytes and is returned in two registers with the
/// discriminant in the first. A caller that reads one result register from
/// such a function receives `1` for `Some(p)` and `0` for `None`, never `p`.
/// That is a silent wrong value rather than a crash, which is why this trait
/// is deliberately narrow.
pub trait ResidualRet {}

impl ResidualRet for () {}

macro_rules! residual_scalar {
    ($($t:ty),* $(,)?) => { $(
        impl ResidualSlot for $t {}
        impl ResidualRet for $t {}
    )* };
}

// `f32` is deliberately absent: `return_type_string_to_value_type` maps it to
// the integer class while the machine ABI returns it in the float bank.
residual_scalar!(
    i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, bool, char, f64
);

/// `OpArg` is `#[repr(transparent)] struct OpArg(u32)` — one word.
impl ResidualSlot for rustpython_compiler_core::bytecode::OpArg {}

/// `Arg<T>` is the zero-sized oparg marker (`struct Arg<T>(PhantomData<T>)`).
/// It consumes no slot at all rather than one: a zero-sized parameter is not
/// passed in the Rust ABI, and the codewriter classifies it `Type::Void`,
/// which `resolve_non_void_arg_types_from_vars` skips. Both sides agree that
/// it is absent, so it is describable — this trait means "the residual ABI can
/// describe this parameter", not "this parameter is exactly one word".
impl<T: rustpython_compiler_core::bytecode::OpArgType> ResidualSlot
    for rustpython_compiler_core::bytecode::Arg<T>
{
}

/// The dunder-pair and builtin-base discriminants the override gates in
/// `descroperation` take.  Each is a fieldless enum, which the front models
/// as its discriminant integer (`tyref_is_fieldless_enum_free`), so it fills
/// exactly one argument slot — the reason those gates carry a discriminant
/// rather than the `&str` names themselves.
impl ResidualSlot for crate::objspace::descroperation::BinopDunder {}
impl ResidualSlot for crate::objspace::descroperation::UnaryDunder {}
impl ResidualSlot for crate::objspace::descroperation::SeqBase {}
impl ResidualSlot for crate::objspace::descroperation::RepeatDunder {}

impl<T> ResidualSlot for &T {}
impl<T> ResidualSlot for &mut T {}
impl<T> ResidualSlot for *const T {}
impl<T> ResidualSlot for *mut T {}
impl<T> ResidualRet for *const T {}
impl<T> ResidualRet for *mut T {}

/// Word-ABI bridges for the three shadow-stack operations `eval::FrameAnchor`
/// reaches.  `majit_ir::GcRef` is `#[repr(transparent)]` over one `usize`, so
/// each of the raw functions is `(usize) -> usize`, `(usize) -> usize` and
/// `(usize) -> ()` — and `usize` is 32-bit on wasm32.  A residual call whose
/// descr types are all words lowers to an in-module `(i64xn) -> i64` (or
/// `(i64xn) -> ()`) `call_indirect`, which type-checks its callee on every
/// call, so the raw functions are a different table type there.
extern "C" fn shadow_stack_push_word(gcref: i64) -> i64 {
    majit_gc::shadow_stack::push(majit_ir::GcRef(gcref as usize)) as i64
}

extern "C" fn shadow_stack_get_word(index: i64) -> i64 {
    majit_gc::shadow_stack::get(index as usize).as_usize() as i64
}

extern "C" fn shadow_stack_try_pop_to_word(depth: i64) {
    majit_gc::shadow_stack::try_pop_to(depth as usize);
}

/// Word-ABI bridge for the scalar bytecode read used by translated residual
/// calls.  The backends call integer helpers uniformly as `(i64, ..) -> i64`;
/// the raw Rust function is `(pointer, usize) -> u16`, which is a different
/// `call_indirect` table type on wasm32.
extern "C" fn bh_code_unit_at(code: i64, index: i64) -> i64 {
    let code = unsafe { &*(code as usize as *const crate::CodeObject) };
    i64::from(crate::pyopcode::code_unit_at(code, index as usize))
}

/// Publication helpers that check the signature instead of erasing it.
///
/// Taking `*const ()` means every caller casts, and a cast accepts any
/// function whatsoever. These take the function itself, so the parameter and
/// result types have to satisfy [`ResidualSlot`] / [`ResidualRet`] before the
/// address can be taken. The bounds sit on the helper's *parameter* rather
/// than on a trait's self type on purpose: a bound of the form
/// `impl<A: ResidualSlot, R> Trait for fn(A) -> R` does not match a signature
/// carrying a lifetime, such as `fn(&PyFrame) -> i64`, and fails with
/// "implementation is not general enough". In parameter position the fn item
/// coerces and the lifetime is inferred.
///
/// The digit is the arity. `p*` publishes a single path, `pa*` publishes the
/// module-qualified path and the crate-root alias. `up*` is `unsafe fn`,
/// `cp*` is `extern "C" fn`, `ucp*` is both. Only the shapes something below
/// actually publishes exist; publishing a new shape means adding its helper.
#[inline]
fn p0<R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    full_path: &'static str,
    f: fn() -> R,
) {
    push_raw_fnaddr(entries, full_path, f as *const ());
}

#[inline]
fn pa0<R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    module_path: &'static str,
    root_path: &'static str,
    f: fn() -> R,
) {
    push_raw_fnaddr(entries, module_path, f as *const ());
    push_raw_fnaddr(entries, root_path, f as *const ());
}

#[inline]
fn cp0<R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    full_path: &'static str,
    f: extern "C" fn() -> R,
) {
    push_raw_fnaddr(entries, full_path, f as *const ());
}

#[inline]
fn cpa0<R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    module_path: &'static str,
    root_path: &'static str,
    f: extern "C" fn() -> R,
) {
    push_raw_fnaddr(entries, module_path, f as *const ());
    push_raw_fnaddr(entries, root_path, f as *const ());
}

#[inline]
fn p1<A1: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    full_path: &'static str,
    f: fn(A1) -> R,
) {
    push_raw_fnaddr(entries, full_path, f as *const ());
}

#[inline]
fn pa1<A1: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    module_path: &'static str,
    root_path: &'static str,
    f: fn(A1) -> R,
) {
    push_raw_fnaddr(entries, module_path, f as *const ());
    push_raw_fnaddr(entries, root_path, f as *const ());
}

#[inline]
fn upa1<A1: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    module_path: &'static str,
    root_path: &'static str,
    f: unsafe fn(A1) -> R,
) {
    push_raw_fnaddr(entries, module_path, f as *const ());
    push_raw_fnaddr(entries, root_path, f as *const ());
}

#[inline]
fn up1<A1: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    path: &'static str,
    f: unsafe fn(A1) -> R,
) {
    push_raw_fnaddr(entries, path, f as *const ());
}

#[inline]
fn cp1<A1: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    full_path: &'static str,
    f: extern "C" fn(A1) -> R,
) {
    push_raw_fnaddr(entries, full_path, f as *const ());
}

#[inline]
fn cpa1<A1: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    module_path: &'static str,
    root_path: &'static str,
    f: extern "C" fn(A1) -> R,
) {
    push_raw_fnaddr(entries, module_path, f as *const ());
    push_raw_fnaddr(entries, root_path, f as *const ());
}

#[inline]
fn p2<A1: ResidualSlot, A2: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    full_path: &'static str,
    f: fn(A1, A2) -> R,
) {
    push_raw_fnaddr(entries, full_path, f as *const ());
}

#[inline]
fn pa2<A1: ResidualSlot, A2: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    module_path: &'static str,
    root_path: &'static str,
    f: fn(A1, A2) -> R,
) {
    push_raw_fnaddr(entries, module_path, f as *const ());
    push_raw_fnaddr(entries, root_path, f as *const ());
}

#[inline]
fn up2<A1: ResidualSlot, A2: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    full_path: &'static str,
    f: unsafe fn(A1, A2) -> R,
) {
    push_raw_fnaddr(entries, full_path, f as *const ());
}

#[inline]
fn upa2<A1: ResidualSlot, A2: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    module_path: &'static str,
    root_path: &'static str,
    f: unsafe fn(A1, A2) -> R,
) {
    push_raw_fnaddr(entries, module_path, f as *const ());
    push_raw_fnaddr(entries, root_path, f as *const ());
}

#[inline]
fn cp2<A1: ResidualSlot, A2: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    full_path: &'static str,
    f: extern "C" fn(A1, A2) -> R,
) {
    push_raw_fnaddr(entries, full_path, f as *const ());
}

#[inline]
fn cpa2<A1: ResidualSlot, A2: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    module_path: &'static str,
    root_path: &'static str,
    f: extern "C" fn(A1, A2) -> R,
) {
    push_raw_fnaddr(entries, module_path, f as *const ());
    push_raw_fnaddr(entries, root_path, f as *const ());
}

#[inline]
fn pa3<A1: ResidualSlot, A2: ResidualSlot, A3: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    module_path: &'static str,
    root_path: &'static str,
    f: fn(A1, A2, A3) -> R,
) {
    push_raw_fnaddr(entries, module_path, f as *const ());
    push_raw_fnaddr(entries, root_path, f as *const ());
}

#[inline]
fn upa3<A1: ResidualSlot, A2: ResidualSlot, A3: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    module_path: &'static str,
    root_path: &'static str,
    f: unsafe fn(A1, A2, A3) -> R,
) {
    push_raw_fnaddr(entries, module_path, f as *const ());
    push_raw_fnaddr(entries, root_path, f as *const ());
}

#[inline]
fn up3<A1: ResidualSlot, A2: ResidualSlot, A3: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    full_path: &'static str,
    f: unsafe fn(A1, A2, A3) -> R,
) {
    push_raw_fnaddr(entries, full_path, f as *const ());
}

#[inline]
fn cp3<A1: ResidualSlot, A2: ResidualSlot, A3: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    full_path: &'static str,
    f: extern "C" fn(A1, A2, A3) -> R,
) {
    push_raw_fnaddr(entries, full_path, f as *const ());
}

#[inline]
fn cpa3<A1: ResidualSlot, A2: ResidualSlot, A3: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    module_path: &'static str,
    root_path: &'static str,
    f: extern "C" fn(A1, A2, A3) -> R,
) {
    push_raw_fnaddr(entries, module_path, f as *const ());
    push_raw_fnaddr(entries, root_path, f as *const ());
}

#[inline]
fn pa4<A1: ResidualSlot, A2: ResidualSlot, A3: ResidualSlot, A4: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    module_path: &'static str,
    root_path: &'static str,
    f: fn(A1, A2, A3, A4) -> R,
) {
    push_raw_fnaddr(entries, module_path, f as *const ());
    push_raw_fnaddr(entries, root_path, f as *const ());
}

#[inline]
fn upa4<A1: ResidualSlot, A2: ResidualSlot, A3: ResidualSlot, A4: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    module_path: &'static str,
    root_path: &'static str,
    f: unsafe fn(A1, A2, A3, A4) -> R,
) {
    push_raw_fnaddr(entries, module_path, f as *const ());
    push_raw_fnaddr(entries, root_path, f as *const ());
}

#[inline]
fn up4<A1: ResidualSlot, A2: ResidualSlot, A3: ResidualSlot, A4: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    full_path: &'static str,
    f: unsafe fn(A1, A2, A3, A4) -> R,
) {
    push_raw_fnaddr(entries, full_path, f as *const ());
}

#[inline]
fn cpa4<A1: ResidualSlot, A2: ResidualSlot, A3: ResidualSlot, A4: ResidualSlot, R: ResidualRet>(
    entries: &mut Vec<(&'static str, i64)>,
    module_path: &'static str,
    root_path: &'static str,
    f: extern "C" fn(A1, A2, A3, A4) -> R,
) {
    push_raw_fnaddr(entries, module_path, f as *const ());
    push_raw_fnaddr(entries, root_path, f as *const ());
}

#[inline]
fn cpa7<
    A1: ResidualSlot,
    A2: ResidualSlot,
    A3: ResidualSlot,
    A4: ResidualSlot,
    A5: ResidualSlot,
    A6: ResidualSlot,
    A7: ResidualSlot,
    R: ResidualRet,
>(
    entries: &mut Vec<(&'static str, i64)>,
    module_path: &'static str,
    root_path: &'static str,
    f: extern "C" fn(A1, A2, A3, A4, A5, A6, A7) -> R,
) {
    push_raw_fnaddr(entries, module_path, f as *const ());
    push_raw_fnaddr(entries, root_path, f as *const ());
}

/// Publish an address whose signature the residual-call ABI **cannot**
/// express.
///
/// Every caller is a known defect kept only because unpublishing changes what
/// the codewriter emits and that needs its own measurement. Each site states
/// which part of the signature is unrepresentable. Do not add callers: use the
/// checked publishers above, and if a helper does not fit, change the helper.
///
/// What "cannot express" costs differs by which half of the signature is
/// unrepresentable. An unrepresentable result reads one register: a function
/// returning `Option<*mut T>` answers `1` for `Some(p)` and `0` for `None` —
/// never `p`, a wrong pointer that passes a null check rather than a crash.
/// An unrepresentable parameter is worse. The executor writes one word per
/// `arg_types()` entry, so the second half of a fat pointer is whatever the
/// caller happened to leave in that register, and a callee that reads it as a
/// length dereferences an address nothing chose. Publish those through
/// [`push_abi_unsound_argument_fnaddr`] instead, which names them for
/// [`is_abi_unsound_argument_residual`].
fn push_abi_unsound_fnaddr(
    entries: &mut Vec<(&'static str, i64)>,
    full_path: &'static str,
    fnptr: *const (),
) {
    push_raw_fnaddr(entries, full_path, fnptr);
}

/// Alias-pair form of [`push_abi_unsound_fnaddr`].
fn push_abi_unsound_alias_pair(
    entries: &mut Vec<(&'static str, i64)>,
    module_path: &'static str,
    root_path: &'static str,
    fnptr: *const (),
) {
    push_raw_fnaddr(entries, module_path, fnptr);
    push_raw_fnaddr(entries, root_path, fnptr);
}

/// Argument half of [`push_abi_unsound_fnaddr`]: the part of the signature the
/// residual ABI cannot express is a parameter rather than the result.
///
/// The address is collected as it is published, so
/// [`is_abi_unsound_argument_residual`] answers from the same site that states
/// the reason. A second list keyed by name would be a copy of this
/// classification that nothing keeps in step with it.
fn push_abi_unsound_argument_fnaddr(
    entries: &mut Vec<(&'static str, i64)>,
    abi_unsound_arguments: &mut Vec<i64>,
    full_path: &'static str,
    fnptr: *const (),
) {
    abi_unsound_arguments.push(fnptr as i64);
    push_raw_fnaddr(entries, full_path, fnptr);
}

/// Alias-pair form of [`push_abi_unsound_argument_fnaddr`].
fn push_abi_unsound_argument_alias_pair(
    entries: &mut Vec<(&'static str, i64)>,
    abi_unsound_arguments: &mut Vec<i64>,
    module_path: &'static str,
    root_path: &'static str,
    fnptr: *const (),
) {
    abi_unsound_arguments.push(fnptr as i64);
    push_raw_fnaddr(entries, module_path, fnptr);
    push_raw_fnaddr(entries, root_path, fnptr);
}

/// Append one `(path, address)` row, dropping a null address.
///
/// Every published address reaches this function through one of three kinds
/// of caller, and that list is the invariant worth keeping: the checked
/// publishers above (`p*` / `pa*` / `cp*` / `cpa*` / `up*` / `upa*`), which
/// read the signature; [`push_word_accessor_alias_pair`], whose address was
/// checked by `runtime_ops`'s `word_fn_addr!` before the arity lookup erased
/// it; and the `push_abi_unsound_*` hatches, which name the part of the
/// signature the residual ABI cannot express. A new caller belongs in the
/// first group — a raw call here publishes an address nothing checked, and a
/// mismatch surfaces as a wrong register count at trace-time call, not as a
/// build error.
fn push_raw_fnaddr(
    entries: &mut Vec<(&'static str, i64)>,
    full_path: &'static str,
    fnptr: *const (),
) {
    let fnaddr = fnptr as usize as i64;
    if fnaddr != 0 {
        entries.push((full_path, fnaddr));
    }
}

/// Alias-pair form for an address a `runtime_ops` arity-to-address accessor
/// already checked.
///
/// The accessor selects the helper by a runtime count, so it must erase the
/// signature before returning, and by the time the address arrives here there
/// is nothing left for [`ResidualSlot`] / [`ResidualRet`] to read. The check
/// is not skipped, only moved: each accessor takes its address through
/// `runtime_ops`'s `word_fn_addr!`, which ascribes the fn item to an explicit
/// `extern "C" fn(i64, ..) -> i64` first. Publishing through this helper
/// asserts that the address came from such an accessor; anything else uses
/// the checked publishers above.
fn push_word_accessor_alias_pair(
    entries: &mut Vec<(&'static str, i64)>,
    module_path: &'static str,
    root_path: &'static str,
    fnptr: *const (),
) {
    push_raw_fnaddr(entries, module_path, fnptr);
    push_raw_fnaddr(entries, root_path, fnptr);
}

const CALLABLE_HELPER_PATHS: &[(&str, &str)] = &[
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_0",
        "pyre_interpreter::jit_call_callable_0",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_1",
        "pyre_interpreter::jit_call_callable_1",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_2",
        "pyre_interpreter::jit_call_callable_2",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_3",
        "pyre_interpreter::jit_call_callable_3",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_4",
        "pyre_interpreter::jit_call_callable_4",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_5",
        "pyre_interpreter::jit_call_callable_5",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_6",
        "pyre_interpreter::jit_call_callable_6",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_7",
        "pyre_interpreter::jit_call_callable_7",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_8",
        "pyre_interpreter::jit_call_callable_8",
    ),
];

const KNOWN_BUILTIN_HELPER_PATHS: &[(&str, &str)] = &[
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_0",
        "pyre_interpreter::jit_call_known_builtin_0",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_1",
        "pyre_interpreter::jit_call_known_builtin_1",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_2",
        "pyre_interpreter::jit_call_known_builtin_2",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_3",
        "pyre_interpreter::jit_call_known_builtin_3",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_4",
        "pyre_interpreter::jit_call_known_builtin_4",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_5",
        "pyre_interpreter::jit_call_known_builtin_5",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_6",
        "pyre_interpreter::jit_call_known_builtin_6",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_7",
        "pyre_interpreter::jit_call_known_builtin_7",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_8",
        "pyre_interpreter::jit_call_known_builtin_8",
    ),
];

const KNOWN_FUNCTION_HELPER_PATHS: &[(&str, &str)] = &[
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_0",
        "pyre_interpreter::jit_call_known_function_0",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_1",
        "pyre_interpreter::jit_call_known_function_1",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_2",
        "pyre_interpreter::jit_call_known_function_2",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_3",
        "pyre_interpreter::jit_call_known_function_3",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_4",
        "pyre_interpreter::jit_call_known_function_4",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_5",
        "pyre_interpreter::jit_call_known_function_5",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_6",
        "pyre_interpreter::jit_call_known_function_6",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_7",
        "pyre_interpreter::jit_call_known_function_7",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_8",
        "pyre_interpreter::jit_call_known_function_8",
    ),
];

const LIST_BUILD_HELPER_PATHS: &[(&str, &str)] = &[
    (
        "pyre_interpreter::runtime_ops::jit_build_list_0",
        "pyre_interpreter::jit_build_list_0",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_1",
        "pyre_interpreter::jit_build_list_1",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_2",
        "pyre_interpreter::jit_build_list_2",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_3",
        "pyre_interpreter::jit_build_list_3",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_4",
        "pyre_interpreter::jit_build_list_4",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_5",
        "pyre_interpreter::jit_build_list_5",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_6",
        "pyre_interpreter::jit_build_list_6",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_7",
        "pyre_interpreter::jit_build_list_7",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_8",
        "pyre_interpreter::jit_build_list_8",
    ),
];

const TUPLE_BUILD_HELPER_PATHS: &[(&str, &str)] = &[
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_0",
        "pyre_interpreter::jit_build_tuple_0",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_1",
        "pyre_interpreter::jit_build_tuple_1",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_2",
        "pyre_interpreter::jit_build_tuple_2",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_3",
        "pyre_interpreter::jit_build_tuple_3",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_4",
        "pyre_interpreter::jit_build_tuple_4",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_5",
        "pyre_interpreter::jit_build_tuple_5",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_6",
        "pyre_interpreter::jit_build_tuple_6",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_7",
        "pyre_interpreter::jit_build_tuple_7",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_8",
        "pyre_interpreter::jit_build_tuple_8",
    ),
];

const MAP_BUILD_HELPER_PATHS: &[(&str, &str)] = &[
    (
        "pyre_interpreter::runtime_ops::jit_build_map_0",
        "pyre_interpreter::jit_build_map_0",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_map_1",
        "pyre_interpreter::jit_build_map_1",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_map_2",
        "pyre_interpreter::jit_build_map_2",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_map_3",
        "pyre_interpreter::jit_build_map_3",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_map_4",
        "pyre_interpreter::jit_build_map_4",
    ),
];

/// Returns `true` when `addr` is the runtime address of a `PyFrame`
/// operand-stack accessor (`pop` / `push` / `peek` / `peek_at`).
///
/// The full-body-walk tracer concretely executes plain residual calls during
/// tracing to fold their results, but a residual targeting one of these
/// accessors reads or mutates the live frame's operand stack — which during a
/// walk is empty, because the walk tracks operand values symbolically in its
/// register banks rather than on the real frame.  Executing one underflows
/// (`pop` asserts `valuestackdepth > stack_base()`).  The walker uses this
/// predicate to leave such a residual symbolic so it runs at runtime against a
/// frame whose operand stack is populated.
///
/// The accessors are `#[inline]`, so a fresh `PyFrame::pop as *const ()` is
/// not address-stable across call sites — it can resolve to a distinct
/// out-of-line copy than the one the codewriter baked into the JitCode
/// constant pool.  Match instead against the exact funcptrs the codewriter
/// bakes: the values [`jit_trace_fnaddrs`] records for the accessor paths,
/// computed through the very same coercion site (cached once, addresses are
/// process-stable).
///
/// Today only `PyFrame::pop` is registered in [`jit_trace_fnaddrs`] (the only
/// accessor a residual call currently reaches — `pop_value`'s sub-jitcode).
/// The `push` / `peek` / `peek_at` arms below are dormant defensive guards:
/// their paths never appear in the registry, so they never match.  They
/// activate (still as a SAFE leave-symbolic decline) only if those accessors
/// are later registered; an unregistered helper is already declined upstream
/// by the funcptr-hash gate, so registering them is unnecessary for soundness.
pub fn is_pyframe_operand_stack_accessor(addr: usize) -> bool {
    use std::sync::OnceLock;
    static ACCESSOR_ADDRS: OnceLock<Vec<i64>> = OnceLock::new();
    let addrs = ACCESSOR_ADDRS.get_or_init(|| {
        jit_trace_fnaddrs()
            .into_iter()
            .filter(|(path, _)| {
                path.ends_with("::PyFrame::pop")
                    || path.ends_with("::PyFrame::push")
                    || path.ends_with("::PyFrame::peek")
                    || path.ends_with("::PyFrame::peek_at")
            })
            .map(|(_, fnaddr)| fnaddr)
            .collect()
    });
    addrs.contains(&(addr as i64))
}

/// True when `addr` is the list write barrier's residual fnaddr, in either the
/// bare [`pyre_object::list_write_barrier`] spelling or the
/// `prepare_list_ref_store` wrapper the Object-strategy store goes through
/// (the wrapper only adds the `push_roots` bracket that keeps the stored value
/// addressable across the barrier's safepoint — the same bookkeeping, so the
/// same exemption).
///
/// The #171 object-append fold descends `w_list_append`; its Object-strategy
/// arm stores a GC ref and runs `list_write_barrier(obj)`
/// (`pyre_object::listobject:::869`/`:872`), which residualizes (registered in
/// [`jit_trace_fnaddrs`]) because it is `#[dont_look_inside]`. The barrier is
/// pure GC bookkeeping — `try_gc_write_barrier` adds `obj` to the remembered
/// set — and is idempotent: re-running it on a body replay only re-adds `obj`,
/// never doubling any user-visible state. It is therefore not a "body effect"
/// in the FBW replay sense, and the full-body walker uses this predicate to
/// keep it out of the in-flight-FOR_ITER body-effect accounting.
///
/// This matches RPython, where the write barrier is not a metatracing
/// operation at all: `COND_CALL_GC_WB` is in the "never executed by pyjitpl"
/// set (`rpython/jit/metainterp/executor.py:446`), is neither can-raise nor a
/// call (`resoperation.py:1124-1125`, outside those ranges), and is inserted
/// only by the backend GC rewrite pass after optimization
/// (`backend/llsupport/rewrite.py`). pyre has no separate backend rewrite
/// pass, so the barrier surfaces as a residual during the walk; exempting it
/// from the FBW body-effect gate restores the parity RPython gets for free.
///
/// Matches the registered fnaddr (not `list_write_barrier as *const ()`) for
/// the same address-stability reason as [`is_pyframe_operand_stack_accessor`]:
/// the codewriter bakes the [`jit_trace_fnaddrs`] value into the JitCode
/// constant pool.
pub fn is_list_write_barrier(addr: usize) -> bool {
    use std::sync::OnceLock;
    static BARRIER_ADDRS: OnceLock<Vec<i64>> = OnceLock::new();
    let addrs = BARRIER_ADDRS.get_or_init(|| {
        jit_trace_fnaddrs()
            .into_iter()
            .filter(|(path, _)| {
                path.ends_with("::listobject::list_write_barrier")
                    || *path == "pyre_object::list_write_barrier"
                    || path.ends_with("::listobject::prepare_list_ref_store")
                    || *path == "pyre_object::prepare_list_ref_store"
                    || path.ends_with("::listobject::current_gc_ref")
                    || *path == "pyre_object::current_gc_ref"
            })
            .map(|(_, fnaddr)| fnaddr)
            .collect()
    });
    addrs.contains(&(addr as i64))
}

/// Void-returning bookkeeping residuals whose re-execution reaches the same
/// state, the [`is_list_write_barrier`] category for helpers that are not GC
/// barriers.
///
/// The FBW effect accounting proxies "writes live heap" by a `Void` result
/// type, because a void residual has no value for the walk to carry and is
/// therefore usually a store.  These two are the void residuals a descent into
/// a translated body meets first, and for both the proxy is wrong:
///
/// * `stack_check` only READS — the recursion depth counter and the stack
///   bounds — and raises on overflow; the counter is bumped around calls, not
///   here.  Its slow path revises a cached stack bound, which recomputes to
///   the same answer.
/// * `ensure_object_subclass_ranges_initialized` is a `OnceLock` lazy init,
///   idempotent by construction, and every production entry point has already
///   run the full initialisation before a trace executes, so the residual is a
///   no-op there.
///
/// Neither leaves anything for a replay to double, so counting them keeps a
/// walk that met only these from taking any of the no-replay walk-end roads —
/// the abort then falls back to the legacy entry replay, which re-applies
/// whatever the walk really did commit.
///
/// This is deliberately NOT an `#[elidable]` annotation: elidability licenses
/// the optimizer to FOLD the call away, and `stack_check` must actually run to
/// raise `RecursionError`.  Re-runnability and foldability are different
/// questions, and only the former is asked here.
///
/// Matches the registered fnaddr for the same address-stability reason as
/// [`is_list_write_barrier`].
pub fn is_rerunnable_bookkeeping_residual(addr: usize) -> bool {
    use std::sync::OnceLock;
    static RERUNNABLE_ADDRS: OnceLock<Vec<i64>> = OnceLock::new();
    let addrs = RERUNNABLE_ADDRS.get_or_init(|| {
        jit_trace_fnaddrs()
            .into_iter()
            .filter(|(path, _)| {
                path.ends_with("::stack_check::stack_check")
                    || path.ends_with("::pyobject::ensure_object_subclass_ranges_initialized")
                    || *path == "pyre_object::ensure_object_subclass_ranges_initialized"
            })
            .map(|(_, fnaddr)| fnaddr)
            .collect()
    });
    addrs.contains(&(addr as i64))
}

/// Build-time equivalent of `#[jit_module]::__majit_helper_trace_fnaddrs()`.
///
/// The registry includes both the module-qualified path produced by the
/// source analyzer (`runtime_ops::foo`) and the crate-root re-export path
/// (`foo`) that pyre's runtime helper code often calls directly.
pub fn jit_trace_fnaddrs() -> Vec<(&'static str, i64)> {
    build_jit_trace_fnaddrs().0
}

/// True for an address published through [`push_abi_unsound_argument_fnaddr`]:
/// a helper at least one of whose parameters is wider than the single machine
/// word a residual argument slot carries.
///
/// An inline sub-walk consults this before executing such a residual. The
/// walk's recorded trace is committed and compiled, and the executor supplies
/// one word per `arg_types()` entry, so running the helper reads a register
/// the model never wrote; declining the descent leaves the call to the
/// interpreter, which passes the argument whole.
pub fn is_abi_unsound_argument_residual(addr: usize) -> bool {
    use std::sync::OnceLock;
    static ADDRS: OnceLock<Vec<i64>> = OnceLock::new();
    ADDRS
        .get_or_init(|| build_jit_trace_fnaddrs().1)
        .contains(&(addr as i64))
}

/// [`jit_trace_fnaddrs`] and the [`is_abi_unsound_argument_residual`] set,
/// which the publication sites fill in one pass.
fn build_jit_trace_fnaddrs() -> (Vec<(&'static str, i64)>, Vec<i64>) {
    let mut entries = Vec::new();
    let mut abi_unsound_arguments = Vec::new();

    // `eval::FrameAnchor` is interpreter runtime rooting, outside the LLBC
    // module set.  `majit-translate` declares these three functions through
    // its annotator-only `register_external` carrier; publish an address for
    // each declared path here so a residual call never falls back to a
    // symbolic hash.  The addresses are the word-ABI bridges above, not the
    // raw functions, for the reason their doc gives.
    cp1(
        &mut entries,
        "majit_gc::shadow_stack::push",
        shadow_stack_push_word,
    );
    cp1(
        &mut entries,
        "majit_gc::shadow_stack::get",
        shadow_stack_get_word,
    );
    cp1(
        &mut entries,
        "majit_gc::shadow_stack::try_pop_to",
        shadow_stack_try_pop_to_word,
    );

    pa1(
        &mut entries,
        "pyre_interpreter::builtins::builtin_kwargs_marker_dict",
        "builtins::builtin_kwargs_marker_dict",
        crate::builtins::builtin_kwargs_marker_dict,
    );
    // `builtin_unexpected_keyword_failure` deliberately remains unpublished:
    // its `&str` and `&Wtf8` arguments are two-word aggregates and its
    // `Result<Vec<PyObjectRef>, PyError>` return is multiword, neither of
    // which the one-word residual-call ABI carries.  `bind_builtin_kwargs` is
    // `unroll_safe`, so the codewriter descends into it and reaches this
    // `#[cold]` `#[dont_look_inside]` call as a residual; without an address
    // it falls back to the symbolic hash instead of passing and returning the
    // wrong number of words.

    // RPython annotator PBC parity for `BuiltinCode.func`: every generated
    // interp2app wrapper is a possible value of the indirect function-pointer
    // field.  `#[pyre_methods]` contributes these process-global descriptors
    // through the same link-time census used for pyre class descriptors.
    #[cfg(not(target_arch = "wasm32"))]
    for wrapper in crate::gateway::BUILTIN_WRAPPER_DESCRIPTORS {
        // ABI-UNSOUND: `Result<*mut PyObject, error::PyError>` does not fit one residual slot.
        push_abi_unsound_fnaddr(&mut entries, wrapper.path, wrapper.func as *const ());
    }
    // `BUILTIN_WRAPPER_DESCRIPTORS` does not exist on wasm32
    // (`linkme::distributed_slice` has no arm for `target_os = "unknown"`),
    // so the loop above registers nothing there and
    // `bytecode_for_address(__majit_wrap_builtin_len)` finds no jitcode: the
    // builtin `len` gateway descent then declines before its spec gate.  The
    // address is used for the jitcode lookup only — the gateway body is
    // descended, never residual-called — so register the one wrapper the
    // descent recognises explicitly, the way this file registers every other
    // wasm-reachable helper.
    #[cfg(target_arch = "wasm32")]
    push_abi_unsound_fnaddr(
        &mut entries,
        "pyre_interpreter::builtins::__majit_wrap_builtin_len",
        crate::builtins::__majit_wrap_builtin_len as *const (),
    );

    // `type_object()` accessors are `dont_look_inside` (`majit-translate`
    // `front::llbc_hints` stamps them: the JIT residualizes the `OnceLock` body
    // rather than lifting its unliftable `CELL` read), so each residual call
    // needs the accessor's runtime address.  Every accessor registers its
    // `(path, fn)` through `register_type_object_fnaddr!`; iterate that
    // registry instead of hand-listing ~46 module-qualified paths (which
    // mis-resolve inline-mod spellings and miss per-module `cfg` gates).  The
    // registry covers every target, because the stamp does: an accessor the
    // front residualizes but this loop never publishes leaves the residual
    // holding a symbolic fnaddr.  Register the crate-stripped alias too, so
    // either spelling of the residual `FunctionPath` resolves.  The descriptor
    // carries the accessor as a `fn() -> PyObjectRef` rather than an address,
    // so `p0` checks it the same way it checks a hand-written publication.
    pyre_object::lltype::for_each_type_object_fnaddr(|path, func| {
        p0(&mut entries, path, func);
        if let Some((_crate_seg, rest)) = path.split_once("::") {
            p0(&mut entries, rest, func);
        }
    });

    cpa2(
        &mut entries,
        "pyre_interpreter::runtime_ops::jit_make_function_from_globals",
        "pyre_interpreter::jit_make_function_from_globals",
        crate::runtime_ops::jit_make_function_from_globals,
    );
    cpa4(
        &mut entries,
        "pyre_interpreter::runtime_ops::jit_load_name_from_namespace",
        "pyre_interpreter::jit_load_name_from_namespace",
        crate::runtime_ops::jit_load_name_from_namespace,
    );
    cpa4(
        &mut entries,
        "pyre_interpreter::runtime_ops::jit_store_name_to_namespace",
        "pyre_interpreter::jit_store_name_to_namespace",
        crate::runtime_ops::jit_store_name_to_namespace,
    );
    cpa2(
        &mut entries,
        "pyre_interpreter::runtime_ops::jit_sequence_getitem",
        "pyre_interpreter::jit_sequence_getitem",
        crate::runtime_ops::jit_sequence_getitem,
    );
    // `rpython/rlib/rrandom.py Random.genrand32` contains the Mersenne
    // Twister refill loops.  `JitPolicy.look_inside_graph` deliberately
    // rejects the loopy graph (it is not `@jit.unroll_safe`), so
    // `Random.random` keeps two ordinary residual calls to the translated
    // native helper.  Publish that helper's address just as RPython's source
    // translation/link step does; otherwise the codewriter can only emit a
    // `symbolic_fnaddr_for_path` hash and an inline sub-walk must abort before
    // reaching the native residual.
    let random_genrand32: fn(&mut crate::module::_random::Random) -> u32 =
        crate::module::_random::Random::genrand32;
    pa1(
        &mut entries,
        "pyre_interpreter::module::_random::Random::genrand32",
        "module::_random::Random::genrand32",
        random_genrand32,
    );
    cpa1(
        &mut entries,
        "pyre_interpreter::runtime_ops::jit_next",
        "pyre_interpreter::jit_next",
        crate::runtime_ops::jit_next,
    );

    #[cfg(all(not(feature = "sandbox"), not(target_arch = "wasm32")))]
    {
        // `_cffi_backend` — the residual leaves of a traced `W_CTypeFunc._call`:
        // the raw exchange-buffer block, the errno swap around the foreign call
        // (`rposix._errno_before` / `_errno_after`), and the libffi call itself.
        p1(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::cdataobj::raw_malloc_varsize_char",
            crate::module::_cffi_backend::cdataobj::raw_malloc_varsize_char,
        );
        p1(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::cdataobj::raw_free",
            crate::module::_cffi_backend::cdataobj::raw_free,
        );
        p0(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::cerrno::errno_before",
            crate::module::_cffi_backend::cerrno::errno_before,
        );
        p0(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::cerrno::errno_after",
            crate::module::_cffi_backend::cerrno::errno_after,
        );
        up1(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::ctypefunc::get_mustfree_flag",
            crate::module::_cffi_backend::ctypefunc::get_mustfree_flag,
        );
        // `jit_libffi` reaches a trace through two families: the `CIF_DESCRIPTION`
        // readers a walked `jit_ffi_call` folds against, and the per-result-kind
        // `libffi_call` oopspec leaves it ends in.  Both spellings of each path
        // are published because the bodies live in the module's cfg-selected
        // inner module and are re-exported from the file.
        upa1(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::exchange_size",
            "pyre_interpreter::module::_cffi_backend::jit_libffi::exchange_size",
            crate::module::_cffi_backend::jit_libffi::exchange_size,
        );
        upa1(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::exchange_result",
            "pyre_interpreter::module::_cffi_backend::jit_libffi::exchange_result",
            crate::module::_cffi_backend::jit_libffi::exchange_result,
        );
        upa2(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::exchange_arg",
            "pyre_interpreter::module::_cffi_backend::jit_libffi::exchange_arg",
            crate::module::_cffi_backend::jit_libffi::exchange_arg,
        );
        upa1(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::rtype",
            "pyre_interpreter::module::_cffi_backend::jit_libffi::rtype",
            crate::module::_cffi_backend::jit_libffi::rtype,
        );
        upa1(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::nargs",
            "pyre_interpreter::module::_cffi_backend::jit_libffi::nargs",
            crate::module::_cffi_backend::jit_libffi::nargs,
        );
        upa1(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::types::getkind",
            "pyre_interpreter::module::_cffi_backend::jit_libffi::types::getkind",
            crate::module::_cffi_backend::jit_libffi::types::getkind,
        );
        upa1(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::types::getsize",
            "pyre_interpreter::module::_cffi_backend::jit_libffi::types::getsize",
            crate::module::_cffi_backend::jit_libffi::types::getsize,
        );
        upa3(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::jit_ffi_call_impl_int",
            "pyre_interpreter::module::_cffi_backend::jit_libffi::jit_ffi_call_impl_int",
            crate::module::_cffi_backend::jit_libffi::jit_ffi_call_impl_int,
        );
        upa3(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::jit_ffi_call_impl_float",
            "pyre_interpreter::module::_cffi_backend::jit_libffi::jit_ffi_call_impl_float",
            crate::module::_cffi_backend::jit_libffi::jit_ffi_call_impl_float,
        );
        upa3(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::jit_ffi_call_impl_singlefloat",
            "pyre_interpreter::module::_cffi_backend::jit_libffi::jit_ffi_call_impl_singlefloat",
            crate::module::_cffi_backend::jit_libffi::jit_ffi_call_impl_singlefloat,
        );
        upa3(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::jit_ffi_call_impl_void",
            "pyre_interpreter::module::_cffi_backend::jit_libffi::jit_ffi_call_impl_void",
            crate::module::_cffi_backend::jit_libffi::jit_ffi_call_impl_void,
        );
        p2(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::cdataobj::raw_ptradd",
            crate::module::_cffi_backend::cdataobj::raw_ptradd,
        );
        p1(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::cdataobj::raw_read_ptr",
            crate::module::_cffi_backend::cdataobj::raw_read_ptr,
        );
        p1(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_i8",
            crate::module::_cffi_backend::misc::raw_read_i8,
        );
        p1(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_u8",
            crate::module::_cffi_backend::misc::raw_read_u8,
        );
        p2(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_i8",
            crate::module::_cffi_backend::misc::raw_write_i8,
        );
        p2(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_u8",
            crate::module::_cffi_backend::misc::raw_write_u8,
        );
        p1(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_i16",
            crate::module::_cffi_backend::misc::raw_read_i16,
        );
        p1(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_u16",
            crate::module::_cffi_backend::misc::raw_read_u16,
        );
        p2(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_i16",
            crate::module::_cffi_backend::misc::raw_write_i16,
        );
        p2(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_u16",
            crate::module::_cffi_backend::misc::raw_write_u16,
        );
        p1(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_i32",
            crate::module::_cffi_backend::misc::raw_read_i32,
        );
        p1(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_u32",
            crate::module::_cffi_backend::misc::raw_read_u32,
        );
        p2(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_i32",
            crate::module::_cffi_backend::misc::raw_write_i32,
        );
        p2(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_u32",
            crate::module::_cffi_backend::misc::raw_write_u32,
        );
        p1(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_i64",
            crate::module::_cffi_backend::misc::raw_read_i64,
        );
        p1(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_u64",
            crate::module::_cffi_backend::misc::raw_read_u64,
        );
        p2(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_i64",
            crate::module::_cffi_backend::misc::raw_write_i64,
        );
        p2(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_u64",
            crate::module::_cffi_backend::misc::raw_write_u64,
        );
        p1(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_f32",
            crate::module::_cffi_backend::misc::raw_read_f32,
        );
        p2(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_f32",
            crate::module::_cffi_backend::misc::raw_write_f32,
        );
        p1(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_f64",
            crate::module::_cffi_backend::misc::raw_read_f64,
        );
        p2(
            &mut entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_f64",
            crate::module::_cffi_backend::misc::raw_write_f64,
        );
    }

    // `unpackiterable_driver` (jd1) portal callees.  Its extracted body
    // (`_unpackiterable_unknown_length`) residual-calls `next(w_iterator)` and
    // `drain_list_append(items, w_item)` directly in source, so the codewriter
    // records the bare source paths; without a runtime binding the funcptr
    // constants fall back to a `symbolic_fnaddr_for_path` hash the residual
    // handler cannot resolve.  `next` returns `Result<PyObjectRef, PyError>`
    // and rides the Ref-returning `bh_next` bridge (publishes StopIteration,
    // unlike the FOR_ITER `jit_next`); `drain_list_append` is a `-> ()`
    // `dont_look_inside` seam over `w_list_append` that collapses append's
    // strategy/grow helper subtree to one registered residual (the global
    // `list.append` stays traced). The registered target is its uniform i64
    // carrier adapter: the raw pointer arguments are wasm i32 values, while
    // residual Int/Ref operands use i64 carriers.
    cpa1(
        &mut entries,
        "pyre_interpreter::baseobjspace::next",
        "pyre_interpreter::next",
        crate::runtime_ops::bh_next,
    );
    cp1(&mut entries, "next", crate::runtime_ops::bh_next);
    cpa2(
        &mut entries,
        "pyre_object::listobject::drain_list_append",
        "pyre_object::drain_list_append",
        pyre_object::listobject::jit_drain_list_append,
    );
    cp2(
        &mut entries,
        "drain_list_append",
        pyre_object::listobject::jit_drain_list_append,
    );

    // The drain's prologue (`w_list_new_object_with_sizehint`) wraps opaque
    // host plumbing and has a one-word return, so publish it as a
    // residual-call target. Keep `w_list_new_empty` registered for its other
    // residual sites.
    // `w_list_new_object` is residualized (`#[dont_look_inside]`) but was
    // unregistered; bind it too so any direct residual site resolves.
    let w_list_new_empty: fn() -> pyre_object::PyObjectRef =
        pyre_object::listobject::w_list_new_empty;
    pa0(
        &mut entries,
        "pyre_object::listobject::w_list_new_empty",
        "pyre_object::w_list_new_empty",
        w_list_new_empty,
    );
    p0(&mut entries, "w_list_new_empty", w_list_new_empty);
    let w_list_new_object_with_sizehint: fn(i64) -> pyre_object::PyObjectRef =
        pyre_object::listobject::w_list_new_object_with_sizehint;
    pa1(
        &mut entries,
        "pyre_object::listobject::w_list_new_object_with_sizehint",
        "pyre_object::w_list_new_object_with_sizehint",
        w_list_new_object_with_sizehint,
    );
    p1(
        &mut entries,
        "w_list_new_object_with_sizehint",
        w_list_new_object_with_sizehint,
    );
    let w_none: fn() -> pyre_object::PyObjectRef = pyre_object::noneobject::w_none;
    pa0(
        &mut entries,
        "pyre_object::noneobject::w_none",
        "pyre_object::w_none",
        w_none,
    );
    let w_list_new_object: fn(Vec<pyre_object::PyObjectRef>) -> pyre_object::PyObjectRef =
        pyre_object::listobject::w_list_new_object;
    // ABI-UNSOUND: `Vec<PyObjectRef>` is three words by value; a residual argument slot is one.
    push_abi_unsound_argument_alias_pair(
        &mut entries,
        &mut abi_unsound_arguments,
        "pyre_object::listobject::w_list_new_object",
        "pyre_object::w_list_new_object",
        w_list_new_object as *const (),
    );
    // `drain_collect_items` deliberately remains unpublished: its multiword
    // `Vec<PyObjectRef>` return has no one-word residual-call ABI.

    cpa1(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_truth_value",
        "pyre_interpreter::jit_truth_value",
        crate::opcode_ops::jit_truth_value,
    );
    cpa1(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_bool_value_from_truth",
        "pyre_interpreter::jit_bool_value_from_truth",
        crate::opcode_ops::jit_bool_value_from_truth,
    );
    cpa3(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_binary_value_from_tag",
        "pyre_interpreter::jit_binary_value_from_tag",
        crate::opcode_ops::jit_binary_value_from_tag,
    );
    cpa3(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_compare_value_from_tag",
        "pyre_interpreter::jit_compare_value_from_tag",
        crate::opcode_ops::jit_compare_value_from_tag,
    );
    cpa1(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_unary_negative_value",
        "pyre_interpreter::jit_unary_negative_value",
        crate::opcode_ops::jit_unary_negative_value,
    );
    cpa1(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_unary_invert_value",
        "pyre_interpreter::jit_unary_invert_value",
        crate::opcode_ops::jit_unary_invert_value,
    );
    cpa1(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_unary_positive_value",
        "pyre_interpreter::jit_unary_positive_value",
        crate::opcode_ops::jit_unary_positive_value,
    );
    // Codewriter `inline_call_r_r` targets the object-space graph names, not
    // the opcode residual wrappers.  Bind each graph to its dedicated
    // one-word C-ABI entry point so both recording-time descent and
    // guard-failure blackholing execute the same interpreter operation.
    cp1(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::neg",
        crate::opcode_ops::jit_descroperation_neg,
    );
    cp1(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::invert",
        crate::opcode_ops::jit_descroperation_invert,
    );
    cp1(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::pos",
        crate::opcode_ops::jit_descroperation_pos,
    );
    cpa2(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_getitem",
        "pyre_interpreter::jit_getitem",
        crate::opcode_ops::jit_getitem,
    );
    cpa3(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_setitem",
        "pyre_interpreter::jit_setitem",
        crate::opcode_ops::jit_setitem,
    );
    cpa3(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_getattr",
        "pyre_interpreter::jit_getattr",
        crate::opcode_ops::jit_getattr,
    );
    cpa4(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_setattr",
        "pyre_interpreter::jit_setattr",
        crate::opcode_ops::jit_setattr,
    );

    // Production walker's `Instruction::StoreSubscr` arm emits a
    // `residual_call_r_r` whose funcptr resolves at codewriter time
    // through the bare path `["execute_store_subscr"]` (`pyopcode.rs`'s
    // `execute_store_subscr`).  Without a runtime fnaddr
    // entry the codewriter mints a `symbolic_fnaddr_for_path` hash
    // that the `runtime_fnaddr_patch` cannot rewrite; the walker rejects
    // the unresolved address and skips the heap mutation, leaving the next
    // read to observe stale container state.  `bh_execute_store_subscr`
    // is the C-ABI bridge over the generic
    // `execute_store_subscr::<PyFrame>` whose `Result<StepResult<_>,
    // PyError>` cannot ride the residual_call's single-register Ref
    // result slot.  Registering the bare path here lets the codewriter
    // bake the wrapper address directly into `JitCode.constants_i`,
    // mirroring PyPy's `cpu.bh_call_*` -> linker-resolved C symbol
    // contract (`pyjitpl.py:1346 _opimpl_residual_call*`).
    cp1(
        &mut entries,
        "execute_store_subscr",
        crate::opcode_ops::bh_execute_store_subscr,
    );

    // `cpu.store_subscr_fn` binding (`pyre-jit/src/jit/cpu.rs`)
    // bound via `pyre_interpreter::opcode_ops::bh_store_subscr_fn`.
    // Registered here so a consumer can recover the runtime address via
    // `jit_trace_fnaddrs()` lookup without a cross-crate dependency edge.
    cpa3(
        &mut entries,
        "pyre_interpreter::opcode_ops::bh_store_subscr_fn",
        "pyre_interpreter::bh_store_subscr_fn",
        crate::opcode_ops::bh_store_subscr_fn,
    );

    // `dont_look_inside` runtime-state accessors residualised at trace
    // time (TLS / per-type atomic the tracer cannot model).  Their
    // residual call resolves its address here by qualified path; a
    // missing entry would fall back to a symbolic hash that SEGVs at
    // trace time.  `shadow_stack_len` carries a JIT-representable
    // `-> int` signature and binds its Rust `fn` directly (the
    // `PyFrame::nlocals` / `get_current_exception` precedent);
    // `w_type_set_uses_object_setattr` rides a C-ABI bridge that
    // normalises its `bool` argument.
    pa0(
        &mut entries,
        "pyre_object::gc_roots::shadow_stack_len",
        "pyre_object::shadow_stack_len",
        pyre_object::gc_roots::shadow_stack_len,
    );
    pa1(
        &mut entries,
        "pyre_object::gc_roots::shadow_stack_get",
        "pyre_object::shadow_stack_get",
        pyre_object::gc_roots::shadow_stack_get,
    );
    pa2(
        &mut entries,
        "pyre_object::gc_roots::shadow_stack_set",
        "pyre_object::shadow_stack_set",
        pyre_object::gc_roots::shadow_stack_set,
    );
    pa0(
        &mut entries,
        "pyre_object::gc_roots::shadow_stack_cell",
        "pyre_object::shadow_stack_cell",
        pyre_object::gc_roots::shadow_stack_cell,
    );
    // The other two thirds of the `push_roots` bracket.  Both take the cell
    // pointer `shadow_stack_cell` returns, so a descent that gets past the
    // resolution lands on these next; all three are one-word scalars in and
    // out, with no fat pointer, `Option`, or sret.
    pa1(
        &mut entries,
        "pyre_object::gc_roots::shadow_stack_cell_len",
        "pyre_object::shadow_stack_cell_len",
        pyre_object::gc_roots::shadow_stack_cell_len,
    );
    pa2(
        &mut entries,
        "pyre_object::gc_roots::shadow_stack_cell_truncate",
        "pyre_object::shadow_stack_cell_truncate",
        pyre_object::gc_roots::shadow_stack_cell_truncate,
    );
    cpa2(
        &mut entries,
        "pyre_object::typeobject::w_type_set_uses_object_setattr",
        "pyre_object::w_type_set_uses_object_setattr",
        crate::opcode_ops::bh_w_type_set_uses_object_setattr,
    );
    cpa2(
        &mut entries,
        "pyre_object::typeobject::w_type_set_uses_object_getattribute",
        "pyre_object::w_type_set_uses_object_getattribute",
        crate::opcode_ops::bh_w_type_set_uses_object_getattribute,
    );
    // `w_type_issubtype` is the MRO membership scan (`_issubtype`,
    // typeobject.py), run under the JIT inside `_pure_issubtype`
    // (`@elidable_promote`, typeobject.py:1657).  Its `#[dont_look_inside]`
    // residualises the call; bind the `-> bool` Rust `fn` directly by
    // qualified path (2-pointer args, JIT-representable, no C-ABI bridge).
    let w_type_issubtype: unsafe fn(pyre_object::PyObjectRef, pyre_object::PyObjectRef) -> bool =
        pyre_object::w_type_issubtype;
    upa2(
        &mut entries,
        "pyre_object::typeobject::w_type_issubtype",
        "pyre_object::w_type_issubtype",
        w_type_issubtype,
    );
    // `lookup_exc_class_for_kind` reads the process-global `EXC_CLASS_BY_KIND`
    // registry the tracer cannot model; its residual call rides a C-ABI
    // bridge that reconstructs the `ExcKind` from the integer arg slot.
    cpa1(
        &mut entries,
        "pyre_object::interp_exceptions::lookup_exc_class_for_kind",
        "pyre_object::lookup_exc_class_for_kind",
        crate::opcode_ops::bh_lookup_exc_class_for_kind,
    );
    // `exc_kind_discriminant` reads the caught exception object's `kind`
    // discriminant; its residual call rides a C-ABI bridge that returns the
    // `ExcKind` discriminant in the integer result slot a residual result
    // register wants.  Emitted by the `try_fuse_drain_match` recognizer
    // (`front::result_exc`) for the drain loop's exception-edge kind test.
    cpa1(
        &mut entries,
        "pyre_object::interp_exceptions::exc_kind_discriminant",
        "pyre_object::exc_kind_discriminant",
        crate::opcode_ops::bh_w_exception_get_kind,
    );
    // `exception_object_matches_stop_iteration` performs the cached
    // StopIteration class lookup and MRO match for the caught exception
    // object. Its residual call rides a C-ABI bridge that returns the boolean
    // in the integer result slot. Emitted by `try_fuse_drain_match` for the
    // drain loop's exception-edge subclass test.
    cpa1(
        &mut entries,
        "pyre_interpreter::error::exception_object_matches_stop_iteration",
        "pyre_interpreter::exception_object_matches_stop_iteration",
        crate::opcode_ops::bh_exception_object_matches_stop_iteration,
    );
    // `pin_root` pushes onto the TLS `SHADOW_STACK` (the `shadow_stack_len`
    // twin), `dereference` reads the weakref `w_obj_weak` slot
    // (`@jit.dont_look_inside` upstream, the `proxy_type` twin), and
    // `_obj_setdict` writes the per-instance `INSTANCE_DICT` side table —
    // all through closures the tracer cannot model.  Their `#[dont_look_inside]`
    // calls bind the Rust `fn` directly by qualified path (pointer / `-> ()`
    // / `-> Result<(), PyError>` signatures are JIT-representable).
    pa1(
        &mut entries,
        "pyre_object::gc_roots::pin_root",
        "pyre_object::pin_root",
        pyre_object::gc_roots::pin_root,
    );
    // `reload_top_root` re-reads the top entry of the `majit_gc` shadow stack
    // (a different structure from `pin_root`'s root stack) after a call that
    // may have moved what was published there.  A trace that kept the pre-move
    // word instead has no other forwarding for it, so the call stays a
    // residual — and a `Ref` result makes it a direct `call_indirect`, hence
    // the word-ABI bridge rather than the raw `fn` its neighbours bind.
    cpa1(
        &mut entries,
        "pyre_object::gc_roots::reload_top_root",
        "pyre_object::reload_top_root",
        pyre_object::gc_roots::reload_top_root_jit_abi,
    );
    // The scope-local pair a bracket body spells as `roots.pin_root(w)` /
    // `roots.get(slot)`: the same pin through the cached cell, and its
    // read-back half.  The codewriter names an inherent method by its
    // crate-stripped path, so that spelling is the alias.
    let scope_pin_root: fn(
        &pyre_object::gc_roots::RootScope,
        pyre_object::PyObjectRef,
    ) -> pyre_object::PyObjectRef = pyre_object::gc_roots::RootScope::pin_root;
    pa2(
        &mut entries,
        "pyre_object::gc_roots::RootScope::pin_root",
        "gc_roots::RootScope::pin_root",
        scope_pin_root,
    );
    let scope_get: fn(&pyre_object::gc_roots::RootScope, usize) -> pyre_object::PyObjectRef =
        pyre_object::gc_roots::RootScope::get;
    pa2(
        &mut entries,
        "pyre_object::gc_roots::RootScope::get",
        "gc_roots::RootScope::get",
        scope_get,
    );
    // `mark_prebuilt_roots_dirty` sets the static `PREBUILT_ROOTS_DIRTY` bit,
    // and `try_gc_add_root` dispatches the TLS `GC_ADD_ROOT_HOOK` — both through
    // state the tracer cannot model (the `pin_root` / `try_gc_write_barrier`
    // twins). Their `#[dont_look_inside]` calls bind the Rust `fn` directly by
    // qualified path (`-> ()` / `-> bool` signatures are JIT-representable).
    pa0(
        &mut entries,
        "pyre_object::gc_roots::mark_prebuilt_roots_dirty",
        "pyre_object::mark_prebuilt_roots_dirty",
        pyre_object::gc_roots::mark_prebuilt_roots_dirty,
    );
    pa1(
        &mut entries,
        "pyre_object::unicodeobject::w_str_from_codepoint",
        "pyre_object::w_str_from_codepoint",
        pyre_object::unicodeobject::w_str_from_codepoint,
    );
    let w_str_slice_codepoints: unsafe fn(
        pyre_object::PyObjectRef,
        i64,
        i64,
        i64,
    ) -> pyre_object::PyObjectRef = pyre_object::unicodeobject::w_str_slice_codepoints;
    upa4(
        &mut entries,
        "pyre_object::unicodeobject::w_str_slice_codepoints",
        "pyre_object::w_str_slice_codepoints",
        w_str_slice_codepoints,
    );
    let w_str_concat: unsafe fn(
        pyre_object::PyObjectRef,
        pyre_object::PyObjectRef,
    ) -> pyre_object::PyObjectRef = pyre_object::unicodeobject::w_str_concat;
    upa2(
        &mut entries,
        "pyre_object::unicodeobject::w_str_concat",
        "pyre_object::w_str_concat",
        w_str_concat,
    );
    let w_str_first_surrogate: unsafe fn(pyre_object::PyObjectRef) -> i64 =
        pyre_object::unicodeobject::w_str_first_surrogate;
    upa1(
        &mut entries,
        "pyre_object::unicodeobject::w_str_first_surrogate",
        "pyre_object::w_str_first_surrogate",
        w_str_first_surrogate,
    );
    // ABI-UNSOUND: `RBigInt` does not fit one residual slot.
    push_abi_unsound_argument_alias_pair(
        &mut entries,
        &mut abi_unsound_arguments,
        "pyre_object::longobject::w_long_new",
        "pyre_object::w_long_new",
        pyre_object::longobject::w_long_new as *const (),
    );
    // ABI-UNSOUND: `RBigInt` does not fit one residual slot.
    push_abi_unsound_argument_alias_pair(
        &mut entries,
        &mut abi_unsound_arguments,
        "pyre_object::longobject::w_long_new_fresh_rbigint_handle",
        "pyre_object::w_long_new_fresh_rbigint_handle",
        pyre_object::longobject::w_long_new_fresh_rbigint_handle as *const (),
    );
    upa1(
        &mut entries,
        "pyre_object::gc_hook::try_gc_add_root",
        "pyre_object::try_gc_add_root",
        pyre_object::gc_hook::try_gc_add_root,
    );
    pa1(
        &mut entries,
        "pyre_object::gc_hook::try_gc_remove_root",
        "pyre_object::try_gc_remove_root",
        pyre_object::gc_hook::try_gc_remove_root,
    );
    // #346: direct allocation roots residualised via `#[dont_look_inside]`;
    // each binds both the qualified module path and the glob-re-exported root
    // alias. `function_new_impl` lives in this crate so it binds through
    // `crate::`. The bytearray constructors allocate a GC-managed storage box
    // (off-GC storage) that is not phaseA-liftable, so they
    // residualise like the `malloc_typed` (`NewWithVtable`) roots below.
    pa1(
        &mut entries,
        "pyre_object::bytearrayobject::w_bytearray_new",
        "pyre_object::w_bytearray_new",
        pyre_object::bytearrayobject::w_bytearray_new,
    );
    // ABI-UNSOUND: `&[u8]` does not fit one residual slot.
    push_abi_unsound_argument_alias_pair(
        &mut entries,
        &mut abi_unsound_arguments,
        "pyre_object::bytearrayobject::w_bytearray_from_bytes",
        "pyre_object::w_bytearray_from_bytes",
        pyre_object::bytearrayobject::w_bytearray_from_bytes as *const (),
    );
    // ABI-UNSOUND: `W_DictObject` does not fit one residual slot.
    push_abi_unsound_argument_alias_pair(
        &mut entries,
        &mut abi_unsound_arguments,
        "pyre_object::dictmultiobject::alloc_dict_object",
        "pyre_object::alloc_dict_object",
        pyre_object::dictmultiobject::alloc_dict_object as *const (),
    );
    // `w_dict_new` is `#[dont_look_inside]` (residualised over the host
    // `IndexMap::new` storage box); bind its zero-arg `fn() -> PyObjectRef`
    // so the residual call resolves, mirroring `w_list_new_empty`.
    let w_dict_new: fn() -> pyre_object::PyObjectRef = pyre_object::dictmultiobject::w_dict_new;
    pa0(
        &mut entries,
        "pyre_object::dictmultiobject::w_dict_new",
        "pyre_object::w_dict_new",
        w_dict_new,
    );
    p0(&mut entries, "w_dict_new", w_dict_new);
    // `w_dict_new_instance` is `#[dont_look_inside]` (it dispatches through the
    // `MAKE_INSTANCE_DICT_HOOK` fn-pointer cell); bind its zero-arg
    // `fn() -> PyObjectRef` so the residual call resolves, mirroring `w_dict_new`.
    let w_dict_new_instance: fn() -> pyre_object::PyObjectRef =
        pyre_object::dictmultiobject::w_dict_new_instance;
    pa0(
        &mut entries,
        "pyre_object::dictmultiobject::w_dict_new_instance",
        "pyre_object::w_dict_new_instance",
        w_dict_new_instance,
    );
    p0(&mut entries, "w_dict_new_instance", w_dict_new_instance);
    // `bool_invert_deprecation_text` is `#[dont_look_inside]` (it hides a
    // `static` prebuilt cell the front-end cannot lift); bind its zero-arg
    // `fn() -> PyObjectRef` so `invert`'s residual call to it resolves.
    let bool_invert_deprecation_text: fn() -> pyre_object::PyObjectRef =
        crate::objspace::descroperation::bool_invert_deprecation_text;
    pa0(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::bool_invert_deprecation_text",
        "pyre_interpreter::bool_invert_deprecation_text",
        bool_invert_deprecation_text,
    );
    p0(
        &mut entries,
        "bool_invert_deprecation_text",
        bool_invert_deprecation_text,
    );
    // `emit_stdout` is `#[dont_look_inside]` (host stdio handle); bind it so
    // the residual call resolves.
    let emit_stdout: fn(&[u8]) = crate::host_seam::emit_stdout;
    // ABI-UNSOUND: `&[u8]` is a fat pointer (ptr+len); a residual argument slot is one word.
    push_abi_unsound_argument_alias_pair(
        &mut entries,
        &mut abi_unsound_arguments,
        "pyre_interpreter::host_seam::emit_stdout",
        "pyre_interpreter::emit_stdout",
        emit_stdout as *const (),
    );
    // ABI-UNSOUND: `&[u8]` is a fat pointer (ptr+len); a residual argument slot is one word.
    push_abi_unsound_argument_fnaddr(
        &mut entries,
        &mut abi_unsound_arguments,
        "emit_stdout",
        emit_stdout as *const (),
    );
    // `w_set_new` / `w_frozenset_new` are `#[dont_look_inside]` for the same
    // host `IndexMap::new` storage-box reason; bind their zero-arg
    // `fn() -> PyObjectRef` so the residual calls resolve.
    let w_set_new: fn() -> pyre_object::PyObjectRef = pyre_object::setobject::w_set_new;
    pa0(
        &mut entries,
        "pyre_object::setobject::w_set_new",
        "pyre_object::w_set_new",
        w_set_new,
    );
    p0(&mut entries, "w_set_new", w_set_new);
    let w_frozenset_new: fn() -> pyre_object::PyObjectRef = pyre_object::setobject::w_frozenset_new;
    pa0(
        &mut entries,
        "pyre_object::setobject::w_frozenset_new",
        "pyre_object::w_frozenset_new",
        w_frozenset_new,
    );
    p0(&mut entries, "w_frozenset_new", w_frozenset_new);
    // `w_set_copy_storage_from` is `#[dont_look_inside]` (its body clones the
    // host `SetItemsStorage` `IndexMap` and boxes it into `d.items`); bind its
    // `unsafe fn(PyObjectRef, PyObjectRef)` so the residual call resolves. The
    // void 2-arg fn registers exactly like the void `w_type_set_abstract`
    // sibling below.
    upa2(
        &mut entries,
        "pyre_object::setobject::w_set_copy_storage_from",
        "pyre_object::w_set_copy_storage_from",
        pyre_object::setobject::w_set_copy_storage_from,
    );
    upa1(
        &mut entries,
        "pyre_object::dictmultiobject::w_dict_len",
        "pyre_object::w_dict_len",
        pyre_object::dictmultiobject::w_dict_len,
    );
    // The wtf8-keyed dict adapters residualise their fallible `Wtf8::as_str`
    // dispatch: `wtf8_key_is_utf8` is the `bool` validity probe, and
    // `wtf8_surrogate_key_str_object` wraps the cold lone-surrogate
    // `to_wtf8_buf` + `w_str_from_wtf8` into one objectptr call.
    // ABI-UNSOUND: `&Wtf8 (a fat pointer)` does not fit one residual slot.
    push_abi_unsound_argument_alias_pair(
        &mut entries,
        &mut abi_unsound_arguments,
        "pyre_object::dictmultiobject::wtf8_key_is_utf8",
        "pyre_object::wtf8_key_is_utf8",
        pyre_object::dictmultiobject::wtf8_key_is_utf8 as *const (),
    );
    // ABI-UNSOUND: `&Wtf8 (a fat pointer)` does not fit one residual slot.
    push_abi_unsound_argument_alias_pair(
        &mut entries,
        &mut abi_unsound_arguments,
        "pyre_object::dictmultiobject::wtf8_surrogate_key_str_object",
        "pyre_object::wtf8_surrogate_key_str_object",
        pyre_object::dictmultiobject::wtf8_surrogate_key_str_object as *const (),
    );
    // The typed int/bytes dict-storage leaves residualise their
    // `IndexMap::{insert,get}` (an external-crate heap store/lookup the tracer
    // cannot model): the stores return `()`, the lookups `Option<PyObjectRef>`.
    upa3(
        &mut entries,
        "pyre_object::dictmultiobject::w_dict_store_int_strategy",
        "pyre_object::w_dict_store_int_strategy",
        pyre_object::dictmultiobject::w_dict_store_int_strategy,
    );
    // ABI-UNSOUND: `Option<*mut PyObject>` does not fit one residual slot.
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_object::dictmultiobject::w_dict_lookup_int_strategy",
        "pyre_object::w_dict_lookup_int_strategy",
        pyre_object::dictmultiobject::w_dict_lookup_int_strategy as *const (),
    );
    // ABI-UNSOUND: `Option<*mut PyObject>` does not fit one residual slot.
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_object::identitydict::w_dict_lookup_identity_strategy",
        "pyre_object::w_dict_lookup_identity_strategy",
        pyre_object::identitydict::w_dict_lookup_identity_strategy as *const (),
    );
    // ABI-UNSOUND: `Option<*mut PyObject>` does not fit one residual slot.
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_object::dictmultiobject::w_module_dict_lookup_object_entries",
        "pyre_object::w_module_dict_lookup_object_entries",
        pyre_object::dictmultiobject::w_module_dict_lookup_object_entries as *const (),
    );
    upa3(
        &mut entries,
        "pyre_object::dictmultiobject::w_dict_store_bytes_strategy",
        "pyre_object::w_dict_store_bytes_strategy",
        pyre_object::dictmultiobject::w_dict_store_bytes_strategy,
    );
    // ABI-UNSOUND: `Option<*mut PyObject>` does not fit one residual slot.
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_object::dictmultiobject::w_dict_lookup_bytes_strategy",
        "pyre_object::w_dict_lookup_bytes_strategy",
        pyre_object::dictmultiobject::w_dict_lookup_bytes_strategy as *const (),
    );
    pa0(
        &mut entries,
        "pyre_object::dictmultiobject::w_module_dict_new",
        "pyre_object::w_module_dict_new",
        pyre_object::dictmultiobject::w_module_dict_new,
    );
    // ABI-UNSOUND: `&str` does not fit one residual slot.
    push_abi_unsound_argument_alias_pair(
        &mut entries,
        &mut abi_unsound_arguments,
        "pyre_object::module::w_module_new_aliasing_dict",
        "pyre_object::w_module_new_aliasing_dict",
        pyre_object::module::w_module_new_aliasing_dict as *const (),
    );
    // ABI-UNSOUND: `FunctionName` does not fit one residual slot.
    push_abi_unsound_argument_alias_pair(
        &mut entries,
        &mut abi_unsound_arguments,
        "pyre_interpreter::function::function_new_impl",
        "pyre_interpreter::function_new_impl",
        crate::function::function_new_impl as *const (),
    );
    let pure_version_tag: extern "C" fn(i64) -> i64 =
        crate::baseobjspace::__majit_call_target__orig__pure_version_tag_unlikely_name;
    cpa1(
        &mut entries,
        "pyre_interpreter::baseobjspace::_pure_version_tag",
        "pyre_interpreter::_pure_version_tag",
        pure_version_tag,
    );
    let pure_lookup_where_with_method_cache: extern "C" fn(i64, i64, i64) -> i64 =
        crate::baseobjspace::__majit_call_target__pure_lookup_where_with_method_cache;
    cpa3(
        &mut entries,
        "pyre_interpreter::baseobjspace::_pure_lookup_where_with_method_cache",
        "pyre_interpreter::_pure_lookup_where_with_method_cache",
        pure_lookup_where_with_method_cache,
    );
    let pure_lookup_class_with_method_cache: extern "C" fn(i64, i64, i64) -> i64 =
        crate::baseobjspace::__majit_call_target__pure_lookup_class_with_method_cache;
    cpa3(
        &mut entries,
        "pyre_interpreter::baseobjspace::_pure_lookup_class_with_method_cache",
        "pyre_interpreter::_pure_lookup_class_with_method_cache",
        pure_lookup_class_with_method_cache,
    );
    // `W_Super.getattribute` walks the MRO itself and reads each class's own
    // namespace, so it needs the single-type elidable rather than the
    // method-cache pair above.
    let pure_getdictvalue_no_unwrapping: extern "C" fn(i64, i64, i64) -> i64 =
        crate::baseobjspace::__majit_call_target__pure_getdictvalue_no_unwrapping;
    cpa3(
        &mut entries,
        "pyre_interpreter::baseobjspace::_pure_getdictvalue_no_unwrapping",
        "pyre_interpreter::_pure_getdictvalue_no_unwrapping",
        pure_getdictvalue_no_unwrapping,
    );
    // The uncached arm's thin-pointer twins (`lookup_where_pair` under the
    // JIT): a boxed name in, a raw pointer (null for `None`) out.
    let lookup_in_type_uncached: unsafe fn(
        *mut pyre_object::PyObject,
        *mut pyre_object::PyObject,
    ) -> *mut pyre_object::PyObject = crate::baseobjspace::_lookup_in_type_uncached;
    up2(
        &mut entries,
        "pyre_interpreter::baseobjspace::_lookup_in_type_uncached",
        lookup_in_type_uncached,
    );
    let lookup_where_class_uncached: unsafe fn(
        *mut pyre_object::PyObject,
        *mut pyre_object::PyObject,
    ) -> *mut pyre_object::PyObject = crate::baseobjspace::_lookup_where_class_uncached;
    up2(
        &mut entries,
        "pyre_interpreter::baseobjspace::_lookup_where_class_uncached",
        lookup_where_class_uncached,
    );
    // #346: null-collapsing stable-alloc primitive residualised via
    // `#[dont_look_inside]`, keeping the thread-local GC hook dispatch out of
    // the trace.
    pa2(
        &mut entries,
        "pyre_object::gc_hook::try_gc_alloc_stable_raw",
        "pyre_object::try_gc_alloc_stable_raw",
        pyre_object::gc_hook::try_gc_alloc_stable_raw,
    );
    // Its nursery twin, same signature and the same reason.  The stable arm has
    // had an entry since #346 and the nursery arm never did, so a body that
    // allocates through it resolves to no address and keeps a symbolic
    // residual, which `descent_decline` counts as an un-lowered helper.  No
    // body reaches it on a walked path today — this is the entry every
    // constructor moved onto the nursery would otherwise need, paid before it
    // is owed rather than after.
    pa2(
        &mut entries,
        "pyre_object::gc_hook::try_gc_alloc_nursery_raw",
        "pyre_object::try_gc_alloc_nursery_raw",
        pyre_object::gc_hook::try_gc_alloc_nursery_raw,
    );
    // `w_int_gc_alloc` is the collector-heap arm of `w_int_new`, reached from
    // inside a descended body whenever a fold boxes an int. Bind the
    // macro-emitted trampoline rather than the raw fn, for the reason
    // `prepare_list_ref_store` documents: the raw `(i64) -> *mut PyObject` is
    // `(i64) -> i32` on wasm32, while the wasm backend types the residual's
    // `call_indirect` `(i64) -> i64` from the descr alone.
    let w_int_gc_alloc: extern "C" fn(i64) -> i64 =
        pyre_object::intobject::__majit_call_target_w_int_gc_alloc;
    cpa1(
        &mut entries,
        "pyre_object::intobject::w_int_gc_alloc",
        "pyre_object::w_int_gc_alloc",
        w_int_gc_alloc,
    );
    // `w_type_set_abstract` stores the runtime-mutable `flag_abstract` atomic — a
    // side effect on per-type state, not a build-time constant, so it carries
    // `#[dont_look_inside]` and binds its `()`-returning `fn` directly by
    // qualified path (sibling of `gc_interp::enabled`).
    upa2(
        &mut entries,
        "pyre_object::typeobject::w_type_set_abstract",
        "pyre_object::w_type_set_abstract",
        pyre_object::w_type_set_abstract,
    );
    p1(
        &mut entries,
        "pyre_interpreter::module::_weakref::interp__weakref::dereference",
        crate::module::_weakref::interp__weakref::dereference,
    );
    // ABI-UNSOUND: `Result<(), error::PyError>` does not fit one residual slot.
    push_abi_unsound_fnaddr(
        &mut entries,
        "pyre_interpreter::objspace::std::mapdict::_obj_setdict",
        crate::objspace::std::mapdict::_obj_setdict as *const (),
    );
    p1(
        &mut entries,
        "pyre_interpreter::objspace::std::mapdict::_obj_getdict",
        crate::objspace::std::mapdict::_obj_getdict,
    );
    // `compute_mro` deliberately remains unpublished: its multiword
    // `Vec<PyObjectRef>` return has no one-word residual-call ABI.
    // `compute_default_mro` deliberately remains unpublished for the same
    // multiword `Vec<PyObjectRef>` return ABI.
    // `memoryview_gather_bytes` deliberately remains unpublished: its
    // multiword `Vec<u8>` return has no one-word residual-call ABI.
    // #346: the `not hasmro` subtype fallback for a partially-initialised type
    // (`_issubtype_slow_and_wrong`, typeobject.py).  Its cold best-base
    // walk bottoms out in an opaque `Vec` iteration and returns a single-word
    // `bool`, so it is `#[dont_look_inside]` — keeping the hot cached-MRO
    // branch of `issubtype_w` a pure typed-slice iteration.  Bind its `fn` by
    // qualified path.
    upa2(
        &mut entries,
        "pyre_interpreter::baseobjspace::issubtype_slow_and_wrong",
        "pyre_interpreter::issubtype_slow_and_wrong",
        crate::baseobjspace::issubtype_slow_and_wrong,
    );
    // Two helpers a descent reaches on an executed path and cannot get past:
    // neither is given a jitcode, so the codewriter residualizes the call, and
    // an unpublished residual carries a symbolic funcbox the walker refuses
    // (`OrthodoxSubWalkTraceUnsupported`).
    //
    // Both bind a word-ABI bridge rather than the Rust `fn`. One machine word
    // per argument is not the contract the direct residual call emits: it is
    // uniformly `(i64xn) -> i64` (`majit-backend-wasm/src/codegen.rs`
    // `residual_call_i64_arity`), and on wasm32 a `PyObjectRef` argument is
    // `i32` and a `bool` result is `i32`, so the raw functions are `(i32) ->
    // i32` and `(i32, i32, i32) -> i64` — a table-entry type the
    // `call_indirect` rejects. The mismatch is invisible on 64-bit targets,
    // where every word agrees. This is the rule `jit_force_vref`
    // (`pyre-jit-trace/src/helpers.rs`) and `enabled` below already follow.
    cpa1(
        &mut entries,
        "pyre_interpreter::builtins::abs_uses_builtin",
        "pyre_interpreter::abs_uses_builtin",
        crate::builtins::bh_abs_uses_builtin,
    );
    cpa3(
        &mut entries,
        "pyre_object::bytearrayobject::w_bytearray_find",
        "pyre_object::w_bytearray_find",
        pyre_object::bytearrayobject::bh_w_bytearray_find,
    );
    // `gc_interp::enabled` reads (and lazily inits) the `STATE` atomic, and
    // `longobject::bigint_gc_type_id` /
    // `dictmultiobject::dict_view_iterator_gc_type_id` read the
    // init-assigned `BIGINT_GC_TYPE_ID` / `W_DICT_VIEW_ITERATOR_GC_TYPE_ID`
    // cells — none is a build-time constant, so all three carry
    // `#[dont_look_inside]` and bind their `-> bool` / `-> u32` Rust `fn`
    // directly by qualified path.  A type-id cell deliberately does NOT get
    // a `jit_static_pytype_addrs` / `jit_static_ref_addrs` row: those carry
    // the address of a value fixed at build time, and folding a
    // runtime-stamped id that way would bake `TypeIdCell::UNASSIGNED`.
    //
    // `enabled` binds the trampoline instead: it gates the GC allocation route
    // of every boxing constructor, so a descended body reaches it, and a raw
    // `-> bool` is `() -> i32` on wasm32 against the descr-derived
    // `() -> i64`.  The two type-id readers keep the direct binding — no
    // descended body reaches them yet — but they are the same latent shape.
    let gc_interp_enabled: extern "C" fn() -> i64 =
        pyre_object::gc_interp::__majit_call_target_enabled;
    cpa0(
        &mut entries,
        "pyre_object::gc_interp::enabled",
        "pyre_object::enabled",
        gc_interp_enabled,
    );
    pa0(
        &mut entries,
        "pyre_object::longobject::bigint_gc_type_id",
        "pyre_object::bigint_gc_type_id",
        pyre_object::longobject::bigint_gc_type_id,
    );
    pa0(
        &mut entries,
        "pyre_object::dictmultiobject::dict_view_iterator_gc_type_id",
        "pyre_object::dict_view_iterator_gc_type_id",
        pyre_object::dictmultiobject::dict_view_iterator_gc_type_id,
    );
    // The same shape over the object space's remaining runtime-mutable
    // globals: `sys_modules_dict` reads the `SYS_MODULES_DICT` pointer
    // `set_sys_modules_dict` stamps, `sys_modules_registry_get` the
    // `SYS_MODULES` name→module registry those stamps mirror,
    // `set_in_flight_exception` writes the
    // `IN_FLIGHT_EXCEPTION` thread-local, `mmap_type` the lazily-installed
    // `mmap` type object, and the two `note_eval_activation_*` twins move the
    // `EVAL_NESTING` thread-local `at_outermost_activation` already reads.
    // None is a build-time constant, so each carries `#[dont_look_inside]`
    // and binds its Rust `fn` directly by qualified path rather than taking a
    // `jit_static_*_addrs` address row.
    let sys_modules_dict: fn() -> pyre_object::PyObjectRef = crate::importing::sys_modules_dict;
    pa0(
        &mut entries,
        "pyre_interpreter::importing::sys_modules_dict",
        "pyre_interpreter::sys_modules_dict",
        sys_modules_dict,
    );
    let sys_modules_registry_get: fn(&str) -> Option<pyre_object::PyObjectRef> =
        crate::importing::sys_modules_registry_get;
    // ABI-UNSOUND: `&str` is a fat pointer and `Option<PyObjectRef>` is two words.
    push_abi_unsound_argument_alias_pair(
        &mut entries,
        &mut abi_unsound_arguments,
        "pyre_interpreter::importing::sys_modules_registry_get",
        "pyre_interpreter::sys_modules_registry_get",
        sys_modules_registry_get as *const (),
    );
    // The same shape over four more runtime-mutable cells:
    // `_io::unsupported_operation_type` reads the `UNSUPPORTED_OPERATION_TYPE`
    // `OnceLock` the `_io` module init stamps with its module-local
    // `UnsupportedOperation` class, `eval::current_frame` the `CURRENT_FRAME`
    // thread-local `install_current_frame` moves, the two `display::repr_*`
    // twins the `REPR_ACTIVE` mid-repr set (the
    // `note_eval_activation_{enter,exit}` twin shape), and `autoflusher_add`
    // the process-global `AUTOFLUSHER` handle table owned by the object space.
    let unsupported_operation_type: fn() -> pyre_object::PyObjectRef =
        crate::module::_io::unsupported_operation_type;
    pa0(
        &mut entries,
        "pyre_interpreter::module::_io::unsupported_operation_type",
        "pyre_interpreter::unsupported_operation_type",
        unsupported_operation_type,
    );
    let current_frame: fn() -> *mut crate::pyframe::PyFrame = crate::eval::current_frame;
    pa0(
        &mut entries,
        "pyre_interpreter::eval::current_frame",
        "pyre_interpreter::current_frame",
        current_frame,
    );
    let repr_enter: fn(pyre_object::PyObjectRef) -> bool = crate::display::repr_enter;
    pa1(
        &mut entries,
        "pyre_interpreter::display::repr_enter",
        "pyre_interpreter::repr_enter",
        repr_enter,
    );
    let repr_leave: fn(pyre_object::PyObjectRef) = crate::display::repr_leave;
    pa1(
        &mut entries,
        "pyre_interpreter::display::repr_leave",
        "pyre_interpreter::repr_leave",
        repr_leave,
    );
    let autoflusher_add: fn(pyre_object::PyObjectRef) -> pyre_object::PyObjectRef =
        crate::module::_io::autoflusher_add;
    pa1(
        &mut entries,
        "pyre_interpreter::module::_io::autoflusher_add",
        "pyre_interpreter::autoflusher_add",
        autoflusher_add,
    );
    let allocate_buffered_lock: fn() -> usize = crate::module::_io::allocate_buffered_lock;
    pa0(
        &mut entries,
        "pyre_interpreter::module::_io::allocate_buffered_lock",
        "pyre_interpreter::allocate_buffered_lock",
        allocate_buffered_lock,
    );
    let acquire_buffered_lock: fn(usize) -> bool = crate::module::_io::acquire_buffered_lock;
    pa1(
        &mut entries,
        "pyre_interpreter::module::_io::acquire_buffered_lock",
        "pyre_interpreter::acquire_buffered_lock",
        acquire_buffered_lock,
    );
    let release_buffered_lock: fn(usize) = crate::module::_io::release_buffered_lock;
    pa1(
        &mut entries,
        "pyre_interpreter::module::_io::release_buffered_lock",
        "pyre_interpreter::release_buffered_lock",
        release_buffered_lock,
    );
    let warnings_state_ns: fn() -> pyre_object::PyObjectRef = crate::module::_warnings::state_ns;
    pa0(
        &mut entries,
        "pyre_interpreter::module::_warnings::state_ns",
        "pyre_interpreter::state_ns",
        warnings_state_ns,
    );
    // The host-boundary seams beside them: the stdout fd writer and the
    // thread-identity read.
    let emit_stdout: fn(&[u8]) = crate::host_seam::emit_stdout;
    // ABI-UNSOUND: `&[u8]` is a fat pointer (ptr+len); a residual argument slot is one word.
    push_abi_unsound_argument_alias_pair(
        &mut entries,
        &mut abi_unsound_arguments,
        "pyre_interpreter::host_seam::emit_stdout",
        "pyre_interpreter::emit_stdout",
        emit_stdout as *const (),
    );
    let current_ident: fn() -> i64 = crate::module::thread::current_ident;
    pa0(
        &mut entries,
        "pyre_interpreter::module::thread::current_ident",
        "pyre_interpreter::current_ident",
        current_ident,
    );
    let finalize_failed_attr_receiver_now: fn(pyre_object::PyObjectRef) -> bool =
        crate::eval::finalize_failed_attr_receiver_now;
    pa1(
        &mut entries,
        "pyre_interpreter::eval::finalize_failed_attr_receiver_now",
        "pyre_interpreter::finalize_failed_attr_receiver_now",
        finalize_failed_attr_receiver_now,
    );
    let set_in_flight_exception: fn(pyre_object::PyObjectRef) =
        crate::eval::set_in_flight_exception;
    pa1(
        &mut entries,
        "pyre_interpreter::eval::set_in_flight_exception",
        "pyre_interpreter::set_in_flight_exception",
        set_in_flight_exception,
    );
    // `mmap_type` is `#[cfg(unix)]` inside `interp_mmap`, and the `mmap`
    // module itself is gated at `module/mod.rs`'s `pub mod mmap`; the row has to carry
    // both or a sandbox build on Linux satisfies `unix` with the module
    // configured out.
    #[cfg(all(
        any(unix, windows),
        not(target_arch = "wasm32"),
        not(feature = "sandbox")
    ))]
    {
        let mmap_type: fn() -> pyre_object::PyObjectRef =
            crate::module::mmap::interp_mmap::mmap_type;
        pa0(
            &mut entries,
            "pyre_interpreter::module::mmap::interp_mmap::mmap_type",
            "pyre_interpreter::mmap_type",
            mmap_type,
        );
    }
    // `cdata_bytes_object` carries the `_ctypes` module's own gate
    // (`module/mod.rs`), so the row repeats it rather than resolving a path
    // configured out of the build.
    #[cfg(all(any(unix, windows), feature = "host_env", not(feature = "sandbox")))]
    {
        let cdata_bytes_object: fn(pyre_object::PyObjectRef) -> Option<pyre_object::PyObjectRef> =
            crate::module::_ctypes::cdata::cdata_bytes_object;
        // ABI-UNSOUND: `Option<PyObjectRef>` is two words: a raw pointer has no niche.
        push_abi_unsound_alias_pair(
            &mut entries,
            "pyre_interpreter::module::_ctypes::cdata::cdata_bytes_object",
            "pyre_interpreter::cdata_bytes_object",
            cdata_bytes_object as *const (),
        );
    }
    pa0(
        &mut entries,
        "pyre_object::gc_interp::note_eval_activation_enter",
        "pyre_object::note_eval_activation_enter",
        pyre_object::gc_interp::note_eval_activation_enter,
    );
    pa0(
        &mut entries,
        "pyre_object::gc_interp::note_eval_activation_exit",
        "pyre_object::note_eval_activation_exit",
        pyre_object::gc_interp::note_eval_activation_exit,
    );
    // The dispatch-loop safepoint's five toucher residuals plus the frame-entry
    // odometer bump and the items-block strategy gate: each reads a
    // runtime-mutable global (`COLLECT_STATE` atomic, `EVAL_NESTING` / `POLL_TICK`
    // TLS, the two GC hook fn-pointer cells, `FRAME_ENTRY_COUNT` TLS, the
    // `MAJIT_GC_ITEMSBLOCK` `OnceLock`) — none a build-time constant — so all carry
    // `#[dont_look_inside]` and bind their `-> bool` / `()` Rust `fn` directly by
    // qualified path (siblings of `gc_interp::enabled`).
    pa0(
        &mut entries,
        "pyre_object::gc_interp::collect_enabled",
        "pyre_object::collect_enabled",
        pyre_object::gc_interp::collect_enabled,
    );
    pa0(
        &mut entries,
        "pyre_object::gc_interp::poll_due",
        "pyre_object::poll_due",
        pyre_object::gc_interp::poll_due,
    );
    pa0(
        &mut entries,
        "pyre_object::gc_interp::at_outermost_activation",
        "pyre_object::at_outermost_activation",
        pyre_object::gc_interp::at_outermost_activation,
    );
    pa0(
        &mut entries,
        "pyre_object::gc_hook::try_gc_major_threshold_reached",
        "pyre_object::try_gc_major_threshold_reached",
        pyre_object::gc_hook::try_gc_major_threshold_reached,
    );
    pa0(
        &mut entries,
        "pyre_object::gc_hook::try_gc_collect_oldgen",
        "pyre_object::try_gc_collect_oldgen",
        pyre_object::gc_hook::try_gc_collect_oldgen,
    );
    // `rgc.may_ignore_finalizer` is itself `@jit.dont_look_inside`; publish
    // the matching interpreter helper so its opaque graph becomes a real
    // residual call rather than a symbolic placeholder.
    pa1(
        &mut entries,
        "pyre_interpreter::executioncontext::may_ignore_finalizer",
        "pyre_interpreter::may_ignore_finalizer",
        crate::executioncontext::may_ignore_finalizer,
    );
    pa0(
        &mut entries,
        "pyre_object::object_array::itemsblock_gc_enabled",
        "pyre_object::itemsblock_gc_enabled",
        pyre_object::object_array::itemsblock_gc_enabled,
    );
    p0(
        &mut entries,
        "pyre_interpreter::call::bump_frame_entry_count",
        crate::call::bump_frame_entry_count,
    );
    p1(
        &mut entries,
        "pyre_interpreter::call::eval_current_frame_raw",
        crate::call::eval_current_frame_raw,
    );
    p1(
        &mut entries,
        "pyre_interpreter::display::jit_format_float_repr_rstr",
        crate::display::jit_format_float_repr_rstr,
    );
    p1(
        &mut entries,
        "pyre_interpreter::typedef::jit_format_complex_component_repr_rstr",
        crate::typedef::jit_format_complex_component_repr_rstr,
    );
    p0(
        &mut entries,
        "pyre_interpreter::call::py_recursion_depth",
        crate::call::py_recursion_depth,
    );
    p0(
        &mut entries,
        "pyre_interpreter::module::sys::state::recursion_limit",
        crate::module::sys::state::recursion_limit,
    );
    // The dispatch-loop safepoint entry itself paces the poll inline and
    // dispatches to the threshold and collection hooks.
    pa0(
        &mut entries,
        "pyre_object::gc_interp::safepoint",
        "pyre_object::safepoint",
        pyre_object::gc_interp::safepoint,
    );
    cp1(
        &mut entries,
        "pyre_interpreter::jit_compiler_bigint_to_rbigint",
        crate::jit_compiler_bigint_to_rbigint,
    );
    // PyPy's getconstant_w is a pre-wrapped list read. Pyre's compiler stores
    // ConstantData, so the first read realizes and atomically publishes that
    // wrapped object. Keep this temporary compiler-boundary machinery opaque
    // to source translation; all later reads return the same co_consts_w slot.
    up2(
        &mut entries,
        "pyre_interpreter::pycode::w_code_const",
        crate::pycode::w_code_const,
    );
    // `compare` residualizes its `compare_slot` tail: the slot body reads two
    // `&[u8]` through `core::slice::cmp`, which has no LLBC, so the source lift
    // fails and the whole callee becomes a residual. What was missing is only
    // the address — the callee keeps its graph, and with it a real EffectInfo,
    // so it must NOT be given `dont_look_inside` (a graphless callee gets an
    // empty rather than a top EffectInfo, and the heap optimizer would then
    // keep cached fields across a comparison that can run user `__eq__`).
    // The published address is the word-ABI bridge, not `compare_slot` itself;
    // see `compare_slot_jit_abi` for why the raw signature cannot be a row.
    cp3(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::compare_slot",
        crate::objspace::descroperation::compare_slot_jit_abi,
    );
    // `binop_impl`'s builtin-fast-path override gates.  Each is
    // `dont_look_inside` so the type-static and typeobject-registry loads stay
    // out of the traced arithmetic graph, which makes every one of them a
    // residual call the walk has to bind.
    up3(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::needs_numeric_binop_dispatch",
        crate::objspace::descroperation::needs_numeric_binop_dispatch,
    );
    up3(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::needs_bytes_binop_dispatch",
        crate::objspace::descroperation::needs_bytes_binop_dispatch,
    );
    up4(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::needs_seq_binop_dispatch",
        crate::objspace::descroperation::needs_seq_binop_dispatch,
    );
    up2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::needs_set_binop_dispatch",
        crate::objspace::descroperation::needs_set_binop_dispatch,
    );
    up2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::needs_numeric_unaryop_dispatch",
        crate::objspace::descroperation::needs_numeric_unaryop_dispatch,
    );
    // The two gates `binop_impl`'s sequence branches reach past the ones
    // above.  Each also carried its dunder names as text and so had no row
    // until it took a discriminant.
    up2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::sequence_numeric_slot_is_null",
        crate::objspace::descroperation::sequence_numeric_slot_is_null,
    );
    up2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::seq_repeat_override",
        crate::objspace::descroperation::seq_repeat_override,
    );
    // Truncated `_divrem` projections used by Rust operator shims.
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_div",
        crate::objspace::descroperation::jit_bigint_div,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_rem",
        crate::objspace::descroperation::jit_bigint_rem,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_divrem_returns_lhs_remainder",
        crate::objspace::descroperation::jit_bigint_divrem_returns_lhs_remainder,
    );
    // Floored `divmod` projections used by the zero-checked interpreter seams.
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_div_floor",
        crate::objspace::descroperation::jit_bigint_div_floor,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_mod_floor",
        crate::objspace::descroperation::jit_bigint_mod_floor,
    );
    // Machine-int-divisor legs of the same seams (`_int_floordiv` / `_int_mod`).
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_int_div_floor",
        crate::objspace::descroperation::jit_bigint_int_div_floor,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_int_mod_int_result",
        crate::objspace::descroperation::jit_bigint_int_mod_int_result,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_int_divmod",
        crate::objspace::descroperation::jit_bigint_int_divmod,
    );
    // `jit_bigint_{and,or,xor,sub,mul}` residualize the Rust RBigInt binary
    // operators (`<BigInt as BitAnd>::bitand`, …) the `front::mir` retarget
    // (`front::bigint_binop`) redirects when both operands are the opaque
    // `BigInt` ADT.  Each returns a fresh `*mut BigInt` (as i64), bound by path.
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_and",
        crate::objspace::descroperation::jit_bigint_and,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_or",
        crate::objspace::descroperation::jit_bigint_or,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_xor",
        crate::objspace::descroperation::jit_bigint_xor,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_sub",
        crate::objspace::descroperation::jit_bigint_sub,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_mul",
        crate::objspace::descroperation::jit_bigint_mul,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_add",
        crate::objspace::descroperation::jit_bigint_add,
    );
    // Mixed W_LongObject/W_IntObject descriptors call the dedicated
    // rbigint.int_* operations, preserving PyPy's no-temporary-bigint path.
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_int_add",
        crate::objspace::descroperation::jit_bigint_int_add,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_int_sub",
        crate::objspace::descroperation::jit_bigint_int_sub,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_int_mul",
        crate::objspace::descroperation::jit_bigint_int_mul,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_int_and",
        crate::objspace::descroperation::jit_bigint_int_and,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_int_or",
        crate::objspace::descroperation::jit_bigint_int_or,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_int_xor",
        crate::objspace::descroperation::jit_bigint_int_xor,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_int_eq",
        crate::objspace::descroperation::jit_bigint_int_eq,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_int_ne",
        crate::objspace::descroperation::jit_bigint_int_ne,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_int_lt",
        crate::objspace::descroperation::jit_bigint_int_lt,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_int_le",
        crate::objspace::descroperation::jit_bigint_int_le,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_int_gt",
        crate::objspace::descroperation::jit_bigint_int_gt,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_int_ge",
        crate::objspace::descroperation::jit_bigint_int_ge,
    );
    // `bigint_pow_nomod(...)?` is source-level `Result` syntax for
    // RPython's implicit MemoryError edge. The MIR front removes that shell
    // and binds the elidable pointer-ABI payload call here.
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_pow_nomod",
        crate::objspace::descroperation::jit_bigint_pow_nomod,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_int_pow_nomod",
        crate::objspace::descroperation::jit_bigint_int_pow_nomod,
    );
    // `bigint_lshift_count(...)?` carries the same implicit MemoryError shape
    // for RPython's lshift allocation.
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_lshift_count",
        crate::objspace::descroperation::jit_bigint_lshift_count,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_lshift_int_int_result",
        crate::objspace::descroperation::jit_bigint_lshift_int_int_result,
    );
    // Unary rbigint operations each take one payload pointer.
    cp1(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_neg",
        crate::objspace::descroperation::jit_bigint_neg,
    );
    cp1(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_invert",
        crate::objspace::descroperation::jit_bigint_invert,
    );
    // `jit_bigint_{shl,shr}` residualize the BigInt shift-by-`usize` operators
    // (`<BigInt as Shl<usize>>::shl`, …); `b` is the machine shift count.
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_shl",
        crate::objspace::descroperation::jit_bigint_shl,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_bigint_shr",
        crate::objspace::descroperation::jit_bigint_shr,
    );

    for (nargs, (module_path, root_path)) in CALLABLE_HELPER_PATHS.iter().enumerate() {
        if let Some(fnptr) = crate::runtime_ops::callable_call_helper(nargs) {
            push_word_accessor_alias_pair(&mut entries, module_path, root_path, fnptr);
        }
    }
    for (nargs, (module_path, root_path)) in KNOWN_BUILTIN_HELPER_PATHS.iter().enumerate() {
        if let Some(fnptr) = crate::runtime_ops::known_builtin_call_helper(nargs) {
            push_word_accessor_alias_pair(&mut entries, module_path, root_path, fnptr);
        }
    }
    for (nargs, (module_path, root_path)) in KNOWN_FUNCTION_HELPER_PATHS.iter().enumerate() {
        if let Some(fnptr) = crate::runtime_ops::known_function_call_helper(nargs) {
            push_word_accessor_alias_pair(&mut entries, module_path, root_path, fnptr);
        }
    }
    for (count, (module_path, root_path)) in LIST_BUILD_HELPER_PATHS.iter().enumerate() {
        if let Some(fnptr) = crate::runtime_ops::list_build_helper(count) {
            push_word_accessor_alias_pair(&mut entries, module_path, root_path, fnptr);
        }
    }
    for (count, (module_path, root_path)) in TUPLE_BUILD_HELPER_PATHS.iter().enumerate() {
        if let Some(fnptr) = crate::runtime_ops::tuple_build_helper(count) {
            push_word_accessor_alias_pair(&mut entries, module_path, root_path, fnptr);
        }
    }
    for (count, (module_path, root_path)) in MAP_BUILD_HELPER_PATHS.iter().enumerate() {
        if let Some(fnptr) = crate::runtime_ops::map_build_helper(count) {
            push_word_accessor_alias_pair(&mut entries, module_path, root_path, fnptr);
        }
    }

    cpa1(
        &mut entries,
        "pyre_object::intobject::jit_w_int_new",
        "pyre_object::jit_w_int_new",
        pyre_object::jit_w_int_new,
    );
    cpa1(
        &mut entries,
        "pyre_object::floatobject::jit_w_float_new",
        "pyre_object::jit_w_float_new",
        pyre_object::jit_w_float_new,
    );
    cpa2(
        &mut entries,
        "pyre_object::listobject::jit_list_append",
        "pyre_object::jit_list_append",
        pyre_object::jit_list_append,
    );
    cpa2(
        &mut entries,
        "pyre_object::listobject::jit_list_getitem",
        "pyre_object::jit_list_getitem",
        pyre_object::jit_list_getitem,
    );
    cpa3(
        &mut entries,
        "pyre_object::listobject::jit_list_setitem",
        "pyre_object::jit_list_setitem",
        pyre_object::jit_list_setitem,
    );
    cpa1(
        &mut entries,
        "pyre_object::listobject::jit_list_reverse",
        "pyre_object::jit_list_reverse",
        pyre_object::jit_list_reverse,
    );
    pa1(
        &mut entries,
        "pyre_object::longobject::jit_bigint_to_i64_fits",
        "pyre_object::jit_bigint_to_i64_fits",
        pyre_object::jit_bigint_to_i64_fits,
    );
    cpa1(
        &mut entries,
        "pyre_object::longobject::jit_bigint_from_i64",
        "pyre_object::jit_bigint_from_i64",
        pyre_object::jit_bigint_from_i64,
    );
    cpa1(
        &mut entries,
        "pyre_object::longobject::jit_bigint_from_u64",
        "pyre_object::jit_bigint_from_u64",
        pyre_object::jit_bigint_from_u64,
    );
    cpa1(
        &mut entries,
        "pyre_object::longobject::jit_bigint_clone",
        "pyre_object::jit_bigint_clone",
        pyre_object::jit_bigint_clone,
    );
    cpa2(
        &mut entries,
        "pyre_object::longobject::jit_bigint_eq",
        "pyre_object::jit_bigint_eq",
        pyre_object::jit_bigint_eq,
    );
    cpa2(
        &mut entries,
        "pyre_object::longobject::jit_bigint_ne",
        "pyre_object::jit_bigint_ne",
        pyre_object::jit_bigint_ne,
    );
    cpa2(
        &mut entries,
        "pyre_object::longobject::jit_bigint_lt",
        "pyre_object::jit_bigint_lt",
        pyre_object::jit_bigint_lt,
    );
    cpa2(
        &mut entries,
        "pyre_object::longobject::jit_bigint_le",
        "pyre_object::jit_bigint_le",
        pyre_object::jit_bigint_le,
    );
    cpa2(
        &mut entries,
        "pyre_object::longobject::jit_bigint_gt",
        "pyre_object::jit_bigint_gt",
        pyre_object::jit_bigint_gt,
    );
    cpa2(
        &mut entries,
        "pyre_object::longobject::jit_bigint_ge",
        "pyre_object::jit_bigint_ge",
        pyre_object::jit_bigint_ge,
    );
    cpa1(
        &mut entries,
        "pyre_object::longobject::jit_bigint_bits",
        "pyre_object::jit_bigint_bits",
        pyre_object::jit_bigint_bits,
    );
    cpa1(
        &mut entries,
        "pyre_object::longobject::jit_bigint_is_zero",
        "pyre_object::jit_bigint_is_zero",
        pyre_object::jit_bigint_is_zero,
    );
    cpa1(
        &mut entries,
        "pyre_object::longobject::jit_bigint_is_one",
        "pyre_object::jit_bigint_is_one",
        pyre_object::jit_bigint_is_one,
    );
    cpa1(
        &mut entries,
        "pyre_object::longobject::jit_bigint_tobool",
        "pyre_object::jit_bigint_tobool",
        pyre_object::jit_bigint_tobool,
    );
    cpa1(
        &mut entries,
        "pyre_object::longobject::jit_bigint_hash",
        "pyre_object::jit_bigint_hash",
        pyre_object::jit_bigint_hash,
    );
    pa1(
        &mut entries,
        "pyre_object::longobject::jit_bigint_to_i64_value",
        "pyre_object::jit_bigint_to_i64_value",
        pyre_object::jit_bigint_to_i64_value,
    );
    pa1(
        &mut entries,
        "pyre_object::longobject::jit_bigint_to_i64_value_or_zero",
        "pyre_object::jit_bigint_to_i64_value_or_zero",
        pyre_object::jit_bigint_to_i64_value_or_zero,
    );
    pa1(
        &mut entries,
        "pyre_object::longobject::jit_bigint_to_u64_fits",
        "pyre_object::jit_bigint_to_u64_fits",
        pyre_object::jit_bigint_to_u64_fits,
    );
    pa1(
        &mut entries,
        "pyre_object::longobject::jit_bigint_to_u64_value",
        "pyre_object::jit_bigint_to_u64_value",
        pyre_object::jit_bigint_to_u64_value,
    );
    pa1(
        &mut entries,
        "pyre_object::longobject::jit_bigint_sign_i64",
        "pyre_object::jit_bigint_sign_i64",
        pyre_object::jit_bigint_sign_i64,
    );
    pa1(
        &mut entries,
        "pyre_object::longobject::jit_bigint_to_f64_or_inf",
        "pyre_object::jit_bigint_to_f64_or_inf",
        pyre_object::jit_bigint_to_f64_or_inf,
    );
    pa1(
        &mut entries,
        "pyre_object::longobject::jit_bigint_to_f64_or_nan",
        "pyre_object::jit_bigint_to_f64_or_nan",
        pyre_object::jit_bigint_to_f64_or_nan,
    );
    // The #171 object-append fold descends `w_list_append` and folds the
    // store leaves to native ops, leaving `list_write_barrier(l)` as a
    // residual call. Register it so the codewriter resolves the residual to a
    // runtime-patchable address instead of a `symbolic_fnaddr_for_path` hash
    // the inline sub-walk must decline. The residual barrier remembers the
    // enclosing `W_ListObject`, whose trace reaches every item slot, and is
    // the only thing keeping an appended `old -> young` element reachable
    // across a minor collection.
    cpa1(
        &mut entries,
        "pyre_object::listobject::list_write_barrier",
        "pyre_object::list_write_barrier",
        pyre_object::list_write_barrier,
    );
    // The Object arm reaches the barrier through `prepare_list_ref_store`,
    // which brackets it in `push_roots` so the value survives the safepoint
    // inside the barrier's ownership query. That bracket's zero-arg
    // root-stack resolve has no registered address, so leaving it inside the
    // descended body made every object-strategy append decline the fold. The
    // wrapper is `dont_look_inside`; register it for the same reason as the
    // barrier itself.
    //
    // Register the macro-emitted `extern "C" fn(i64, i64) -> i64` call
    // trampoline, not the raw fn — the shape
    // `#[jit_module]::__majit_helper_trace_fnaddrs()` publishes for a
    // policy-bearing free fn (`majit-macros` `impl_addr_expr` routes it through
    // `__majit_call_policy_*`'s trace-target slot; the raw fn is only its
    // null-target fallback).  The wasm backend lowers an `Int`/`Ref`-result
    // residual to a direct `call_indirect` whose static type is `(i64 x n) ->
    // i64` derived from the descr alone, so a raw
    // `(*mut PyObject, *mut PyObject) -> *mut PyObject` — `(i32, i32) -> i32` on
    // wasm32 — traps `indirect call type mismatch`.  The registered paths are
    // unchanged, so `is_list_write_barrier` and the path-keyed build->runtime
    // re-pairing are unaffected.
    let prepare_list_ref_store: extern "C" fn(i64, i64) -> i64 =
        pyre_object::listobject::__majit_call_target_prepare_list_ref_store;
    cpa2(
        &mut entries,
        "pyre_object::listobject::prepare_list_ref_store",
        "pyre_object::prepare_list_ref_store",
        prepare_list_ref_store,
    );
    // `prepare_list_ref_store` returns the relocated value. The following
    // owner reload is the other half of RPython's post-safepoint pop_roots;
    // keep it residual while leaving the append's set_len/setitem leaves in
    // the descended body.
    let current_gc_ref: extern "C" fn(i64) -> i64 =
        pyre_object::listobject::__majit_call_target_current_gc_ref;
    cpa1(
        &mut entries,
        "pyre_object::listobject::current_gc_ref",
        "pyre_object::current_gc_ref",
        current_gc_ref,
    );
    // The #171 fold descends `w_list_append` as a sub-jitcode walk, so a guard
    // exit inside it is numbered against `w_list_append`'s own jitcode and is
    // resumed there in the blackhole (`resume.py:1339 jitcodes[jitcode_pos]`).
    // The resumed body then reaches the per-strategy store its arm selected —
    // `W_ListObject::object_push` for the Object strategy, `IntArray::push` /
    // `FloatArray::push` for the unwrapped ones — each a `residual_call`, and
    // `blackhole.py:1230 bhimpl_residual_call_*` takes the funcptr straight to
    // an indirect branch.  Upstream never has to bind these: `call.py:181-183
    // getfunctionptr(graph)` resolves every callee in the same translation.
    // pyre's codewriter runs in `build.rs`, so an unregistered callee keeps a
    // `symbolic_fnaddr_for_path` hash, the blackhole aborts the frame, and the
    // jd1 drain silently loses the in-flight `next()` item the resume was
    // supposed to append.  `fnaddr_for_target`'s `CallTarget::Method` fallback
    // looks the address up as `CallPath::for_impl_method(receiver, name)`, i.e.
    // the 2-segment `[receiver, method]` key `register_macro_helper_trace_fnaddr`
    // derives by stripping the leading crate segment — hence the
    // `pyre_object::<Type>::<method>` spelling here.
    let object_push: unsafe fn(&mut pyre_object::W_ListObject, pyre_object::PyObjectRef) =
        pyre_object::W_ListObject::object_push;
    up2(
        &mut entries,
        "pyre_object::W_ListObject::object_push",
        object_push,
    );
    let int_array_push: fn(&mut pyre_object::IntArray, i64) = pyre_object::IntArray::push;
    p2(&mut entries, "pyre_object::IntArray::push", int_array_push);
    let float_array_push: fn(&mut pyre_object::FloatArray, f64) = pyre_object::FloatArray::push;
    p2(
        &mut entries,
        "pyre_object::FloatArray::push",
        float_array_push,
    );
    // The same resume needs the jitcode *shells* it inline-calls to carry a
    // real address: `blackhole.py:1300-1317 bhimpl_inline_call_*` calls
    // `cpu.bh_call_*(adr2int(jitcode.fnaddr), ...)`, so a shell minted with
    // `symbolic_fnaddr_for_path` is uncallable the same way.  `w_list_append`
    // is the fold's descended body and `w_list_len` its length probe.
    let w_list_append: unsafe fn(pyre_object::PyObjectRef, pyre_object::PyObjectRef) =
        pyre_object::listobject::w_list_append;
    upa2(
        &mut entries,
        "pyre_object::listobject::w_list_append",
        "pyre_object::w_list_append",
        w_list_append,
    );
    let w_list_pop_end_inner: unsafe fn(
        pyre_object::PyObjectRef,
    ) -> Option<pyre_object::PyObjectRef> = pyre_object::listobject::w_list_pop_end_inner;
    // ABI-UNSOUND: same two-word Option as `w_list_pop_end` below. The empty
    // check now lives in this descended body (`W_ListObject.descr_pop`).
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_object::listobject::w_list_pop_end_inner",
        "pyre_object::w_list_pop_end_inner",
        w_list_pop_end_inner as *const (),
    );
    let w_list_pop_end: unsafe fn(pyre_object::PyObjectRef) -> Option<pyre_object::PyObjectRef> =
        pyre_object::listobject::w_list_pop_end;
    // ABI-UNSOUND: `Option<PyObjectRef>` is two words: a raw pointer has no niche.
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_object::listobject::w_list_pop_end",
        "pyre_object::w_list_pop_end",
        w_list_pop_end as *const (),
    );
    let w_list_len: unsafe fn(pyre_object::PyObjectRef) -> usize =
        pyre_object::listobject::w_list_len;
    upa1(
        &mut entries,
        "pyre_object::listobject::w_list_len",
        "pyre_object::w_list_len",
        w_list_len,
    );
    let w_set_len: unsafe fn(pyre_object::PyObjectRef) -> usize = pyre_object::setobject::w_set_len;
    upa1(
        &mut entries,
        "pyre_object::setobject::w_set_len",
        "pyre_object::w_set_len",
        w_set_len,
    );
    // The cold list strategy dehomogenization `switch_to_object_strategy` bulk
    // re-boxes typed int/float storage into an Object items block via
    // Vec/collect allocation the tracer cannot model. Register it so the hot
    // append/setitem paths that call it resolve the residual to a
    // runtime-patchable address instead of tracing into the transition.
    upa1(
        &mut entries,
        "pyre_object::listobject::switch_to_object_strategy",
        "pyre_object::switch_to_object_strategy",
        pyre_object::switch_to_object_strategy,
    );
    cpa2(
        &mut entries,
        "pyre_object::tupleobject::jit_tuple_getitem",
        "pyre_object::jit_tuple_getitem",
        pyre_object::jit_tuple_getitem,
    );
    cpa2(
        &mut entries,
        "pyre_object::unicodeobject::jit_str_concat",
        "pyre_object::jit_str_concat",
        pyre_object::jit_str_concat,
    );
    cpa2(
        &mut entries,
        "pyre_object::unicodeobject::jit_str_repeat",
        "pyre_object::jit_str_repeat",
        pyre_object::jit_str_repeat,
    );
    cpa2(
        &mut entries,
        "pyre_object::unicodeobject::jit_str_compare",
        "pyre_object::jit_str_compare",
        pyre_object::jit_str_compare,
    );
    cpa1(
        &mut entries,
        "pyre_object::unicodeobject::jit_str_is_true",
        "pyre_object::jit_str_is_true",
        pyre_object::jit_str_is_true,
    );
    cpa1(
        &mut entries,
        "pyre_object::unicodeobject::jit_int_str",
        "pyre_object::jit_int_str",
        pyre_object::jit_int_str,
    );
    cpa2(
        &mut entries,
        "pyre_object::unicodeobject::jit_str_getitem",
        "pyre_object::jit_str_getitem",
        pyre_object::jit_str_getitem,
    );
    // `rgc.ll_shrink_array` residual target for the StringBuilder `build` tree
    // (`_handle_rgc_call` rewrites the oopspec residual to `["jit_ll_shrink_array"]`).
    // The non-virtual shrink reallocs a raw low-level string down to its final
    // length; the virtual path is folded by `opt_call_shrink_array` and never
    // calls this.
    cpa2(
        &mut entries,
        "pyre_object::lowlevel_string::jit_ll_shrink_array",
        "pyre_object::jit_ll_shrink_array",
        pyre_object::lowlevel_string::jit_ll_shrink_array,
    );
    // `rgc.ll_arraymove` / `list.ll_arraymove` keeps PyPy's four-argument
    // residual ABI. The target recovers the registered array token from the
    // GC TYPE_INFO row, runs the before-move barrier for reference items, and
    // performs overlap-safe raw memmove.
    cpa4(
        &mut entries,
        "pyre_object::object_array::jit_ll_arraymove",
        "pyre_object::jit_ll_arraymove",
        pyre_object::object_array::jit_ll_arraymove,
    );
    // `dont_look_inside` residual append targets for the StringBuilder value:
    // `guess_call_kind` residualizes a call whose leaf is `ll_append_res0` /
    // `ll_append_res_slice` once its native fnaddr is bound. Unlike shrink, these
    // are not retargeted by `jtransform`, so the graph target path is the bare
    // leaf — the crate-root alias leaf must stay un-prefixed (strips to
    // `["ll_append_res0"]`) to satisfy both the leaf-name gate and the
    // `function_fnaddrs.contains_key` lookup; the real symbols carry `jit_`.
    cpa2(
        &mut entries,
        "pyre_object::rbuilder::ll_append_res0",
        "pyre_object::ll_append_res0",
        pyre_object::rbuilder::rbuilder_runtime::jit_ll_append_res0,
    );
    cpa4(
        &mut entries,
        "pyre_object::rbuilder::ll_append_res_slice",
        "pyre_object::ll_append_res_slice",
        pyre_object::rbuilder::rbuilder_runtime::jit_ll_append_res_slice,
    );
    cpa3(
        &mut entries,
        "pyre_object::functional::jit_range_iter_new",
        "pyre_object::jit_range_iter_new",
        pyre_object::jit_range_iter_new,
    );
    // The lowered raise path's exception materialisation, opaque so that its
    // body stays out of every JitCode that can raise.
    let pyerror_to_exc_object: extern "C" fn(i64) -> i64 =
        crate::error::__majit_call_target_pyerror_to_exc_object;
    cpa1(
        &mut entries,
        "pyre_interpreter::error::pyerror_to_exc_object",
        "pyre_interpreter::pyerror_to_exc_object",
        pyerror_to_exc_object,
    );
    // The same materialisation with the `type_error` constructor folded in, so
    // the raise site carries neither body. The typed local spells the
    // trampoline's signature at the call site; `cpa1` checks the same thing
    // through [`ResidualSlot`] / [`ResidualRet`].
    let pyerror_type_error_to_exc_object: extern "C" fn(i64) -> i64 =
        crate::error::__majit_call_target_pyerror_type_error_to_exc_object;
    cpa1(
        &mut entries,
        "pyre_interpreter::error::pyerror_type_error_to_exc_object",
        "pyre_interpreter::pyerror_type_error_to_exc_object",
        pyerror_type_error_to_exc_object,
    );
    let pyerror_zero_division_to_exc_object: extern "C" fn(i64) -> i64 =
        crate::error::__majit_call_target_pyerror_zero_division_to_exc_object;
    cpa1(
        &mut entries,
        "pyre_interpreter::error::pyerror_zero_division_to_exc_object",
        "pyre_interpreter::pyerror_zero_division_to_exc_object",
        pyerror_zero_division_to_exc_object,
    );
    // `elidable_cannot_raise` subclass-range check; the trampoline widens its
    // one-word bool return by zero-extension.
    let ll_issubclass: extern "C" fn(i64, i64) -> i64 =
        pyre_object::pyobject::__majit_call_target_ll_issubclass;
    cpa2(
        &mut entries,
        "pyre_object::pyobject::ll_issubclass",
        "pyre_object::ll_issubclass",
        ll_issubclass,
    );
    // `elidable_cannot_raise` bool singleton lookup; the trampoline widens the
    // returned pointer to one word.
    let w_bool_from: extern "C" fn(i64) -> i64 =
        pyre_object::boolobject::__majit_call_target_w_bool_from;
    cpa1(
        &mut entries,
        "pyre_object::boolobject::w_bool_from",
        "pyre_object::w_bool_from",
        w_bool_from,
    );
    cpa0(
        &mut entries,
        "pyre_object::pyobject::ensure_object_subclass_ranges_initialized",
        "pyre_object::ensure_object_subclass_ranges_initialized",
        pyre_object::pyobject::ensure_object_subclass_ranges_initialized,
    );
    cpa1(
        &mut entries,
        "pyre_object::gc_hook::try_gc_write_barrier",
        "pyre_object::try_gc_write_barrier",
        pyre_object::gc_hook::try_gc_write_barrier,
    );
    cpa1(
        &mut entries,
        "pyre_object::gc_hook::try_gc_owns_object",
        "pyre_object::try_gc_owns_object",
        pyre_object::gc_hook::try_gc_owns_object,
    );
    cpa1(
        &mut entries,
        "pyre_object::gc_hook::maybe_register_finalizer",
        "pyre_object::maybe_register_finalizer",
        pyre_object::gc_hook::maybe_register_finalizer,
    );
    cpa0(
        &mut entries,
        "pyre_object::dict_eq_hook::has_hash_w_hook",
        "pyre_object::has_hash_w_hook",
        pyre_object::dict_eq_hook::has_hash_w_hook,
    );
    cpa1(
        &mut entries,
        "pyre_object::dict_eq_hook::hash_w_hooked",
        "pyre_object::hash_w_hooked",
        pyre_object::dict_eq_hook::hash_w_hooked,
    );
    cpa0(
        &mut entries,
        "pyre_object::dict_eq_hook::has_eq_w_hook",
        "pyre_object::has_eq_w_hook",
        pyre_object::dict_eq_hook::has_eq_w_hook,
    );
    cpa2(
        &mut entries,
        "pyre_object::dict_eq_hook::eq_w_hooked",
        "pyre_object::eq_w_hooked",
        pyre_object::dict_eq_hook::eq_w_hooked,
    );
    cpa0(
        &mut entries,
        "pyre_object::dict_eq_hook::has_hash_str_hook",
        "pyre_object::has_hash_str_hook",
        pyre_object::dict_eq_hook::has_hash_str_hook,
    );
    cpa2(
        &mut entries,
        "pyre_object::dict_eq_hook::hash_str_hooked",
        "pyre_object::hash_str_hooked",
        pyre_object::dict_eq_hook::hash_str_hooked,
    );
    /*
     * Fat-pointer arguments (`&str`, `&[u8]`, `&Wtf8`) are two words, but the
     * residual-call ABI passes one register per argument slot; publishing these
     * addresses would pass one word where the callee reads two.
     */
    // ABI-UNSOUND: `Option<*mut PyObject>` does not fit one residual slot.
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_object::dictmultiobject::dict_entries_probe_object",
        "pyre_object::dict_entries_probe_object",
        pyre_object::dictmultiobject::dict_entries_probe_object as *const (),
    );
    upa2(
        &mut entries,
        "pyre_object::dictmultiobject::dict_entries_remove_object",
        "pyre_object::dict_entries_remove_object",
        pyre_object::dictmultiobject::dict_entries_remove_object,
    );
    // The checked probe / store pair, keyed on an already-hashed key.
    // ABI-UNSOUND: `Option<*mut PyObject>` does not fit one residual slot.
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_object::dictmultiobject::dict_entries_probe_hashed",
        "pyre_object::dict_entries_probe_hashed",
        pyre_object::dictmultiobject::dict_entries_probe_hashed as *const (),
    );
    // ABI-UNSOUND: `Option<*mut PyObject>` does not fit one residual slot.
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_object::dictmultiobject::dict_entries_insert_hashed",
        "pyre_object::dict_entries_insert_hashed",
        pyre_object::dictmultiobject::dict_entries_insert_hashed as *const (),
    );
    upa1(
        &mut entries,
        "pyre_object::dictmultiobject::dict_entries_pop_last",
        "pyre_object::dict_entries_pop_last",
        pyre_object::dictmultiobject::dict_entries_pop_last,
    );
    // The positional slot reads the post-scan lookup arms and the reentrant
    // key scan perform: an index the caller already settled on, so no
    // comparison runs behind these boundaries.
    upa2(
        &mut entries,
        "pyre_object::dictmultiobject::dict_entries_value_at",
        "pyre_object::dict_entries_value_at",
        pyre_object::dictmultiobject::dict_entries_value_at,
    );
    // ABI-UNSOUND: `Option<*mut PyObject>` does not fit one residual slot.
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_object::dictmultiobject::dict_entries_key_obj_at",
        "pyre_object::dict_entries_key_obj_at",
        pyre_object::dictmultiobject::dict_entries_key_obj_at as *const (),
    );
    upa2(
        &mut entries,
        "pyre_object::dictmultiobject::dict_entries_key_hash_at",
        "pyre_object::dict_entries_key_hash_at",
        pyre_object::dictmultiobject::dict_entries_key_hash_at,
    );
    upa4(
        &mut entries,
        "pyre_object::dictmultiobject::dict_entries_key_is_at",
        "pyre_object::dict_entries_key_is_at",
        pyre_object::dictmultiobject::dict_entries_key_is_at,
    );
    upa1(
        &mut entries,
        "pyre_object::dictmultiobject::dict_entries_generation",
        "pyre_object::dict_entries_generation",
        pyre_object::dictmultiobject::dict_entries_generation,
    );
    upa1(
        &mut entries,
        "pyre_object::dictmultiobject::dict_entries_slot_count",
        "pyre_object::dict_entries_slot_count",
        pyre_object::dictmultiobject::dict_entries_slot_count,
    );
    upa3(
        &mut entries,
        "pyre_object::dictmultiobject::dict_entries_value_set_at",
        "pyre_object::dict_entries_value_set_at",
        pyre_object::dictmultiobject::dict_entries_value_set_at,
    );
    upa3(
        &mut entries,
        "pyre_object::dictmultiobject::dict_entries_insert_object",
        "pyre_object::dict_entries_insert_object",
        pyre_object::dictmultiobject::dict_entries_insert_object,
    );
    // The index-returning twin of the object-key probe.
    // ABI-UNSOUND: `Option<usize>` does not fit one residual slot.
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_object::dictmultiobject::dict_entries_index_of_object",
        "pyre_object::dict_entries_index_of_object",
        pyre_object::dictmultiobject::dict_entries_index_of_object as *const (),
    );
    // The `dict.lookup` producer's two residuals bind the macro-emitted
    // `__majit_call_target_*` trampoline, not the raw fn. The wasm backend
    // lowers a Ref/Int-result residual to a `call_indirect` whose static type
    // comes from the descr alone — `(i64 x n) -> i64` — so a raw
    // `(*mut PyObject, *mut PyObject, i64, i64) -> i64`, which is
    // `(i32, i32, i64, i64) -> i64` on wasm32, traps
    // `indirect call type mismatch`. The trampoline takes and returns the
    // uniform machine word on every target. The raw fn stays reachable as
    // `__majit_call_policy_*`'s null-target fallback.
    let w_dict_unicode_lookup_index: extern "C" fn(i64, i64, i64, i64) -> i64 =
        pyre_object::dictmultiobject::__majit_call_target_w_dict_unicode_lookup_index;
    cpa4(
        &mut entries,
        "pyre_object::dictmultiobject::w_dict_unicode_lookup_index",
        "pyre_object::w_dict_unicode_lookup_index",
        w_dict_unicode_lookup_index,
    );
    let w_dict_unicode_key_hash: extern "C" fn(i64) -> i64 =
        pyre_object::dictmultiobject::__majit_call_target_w_dict_unicode_key_hash;
    cpa1(
        &mut entries,
        "pyre_object::dictmultiobject::w_dict_unicode_key_hash",
        "pyre_object::w_dict_unicode_key_hash",
        w_dict_unicode_key_hash,
    );
    // A runtime-mutable global counter, not a build-time constant: bind the
    // read seam by address so the JIT calls it instead of folding whatever
    // serial the build process saw.
    let next_version_tag_serial: fn() -> u64 = pyre_object::celldict::next_version_tag_serial;
    pa0(
        &mut entries,
        "pyre_object::celldict::next_version_tag_serial",
        "pyre_object::next_version_tag_serial",
        next_version_tag_serial,
    );
    // `quasiimmut.py _invalidate_now`, shared by both `?` fields.
    upa1(
        &mut entries,
        "pyre_object::quasiimmut::sweep_quasi_immut_field",
        "pyre_object::sweep_quasi_immut_field",
        pyre_object::quasiimmut::sweep_quasi_immut_field,
    );
    // The three typed-storage promotions: `IndexMap` construction and refill
    // end to end, so the residual boundary is the whole migration.
    upa1(
        &mut entries,
        "pyre_object::dictmultiobject::w_dict_switch_int_to_object_strategy",
        "pyre_object::w_dict_switch_int_to_object_strategy",
        pyre_object::dictmultiobject::w_dict_switch_int_to_object_strategy,
    );
    upa1(
        &mut entries,
        "pyre_object::dictmultiobject::w_dict_switch_bytes_to_object_strategy",
        "pyre_object::w_dict_switch_bytes_to_object_strategy",
        pyre_object::dictmultiobject::w_dict_switch_bytes_to_object_strategy,
    );
    upa1(
        &mut entries,
        "pyre_object::dictmultiobject::w_module_dict_switch_to_object_strategy",
        "pyre_object::w_module_dict_switch_to_object_strategy",
        pyre_object::dictmultiobject::w_module_dict_switch_to_object_strategy,
    );
    upa1(
        &mut entries,
        "pyre_object::kwargsdict::w_dict_switch_kwargs_to_object_strategy",
        "pyre_object::w_dict_switch_kwargs_to_object_strategy",
        pyre_object::kwargsdict::w_dict_switch_kwargs_to_object_strategy,
    );
    upa1(
        &mut entries,
        "pyre_object::identitydict::w_dict_switch_identity_to_object_strategy",
        "pyre_object::w_dict_switch_identity_to_object_strategy",
        pyre_object::identitydict::w_dict_switch_identity_to_object_strategy,
    );
    upa2(
        &mut entries,
        "pyre_object::identitydict::w_dict_delete_identity_strategy",
        "pyre_object::w_dict_delete_identity_strategy",
        pyre_object::identitydict::w_dict_delete_identity_strategy,
    );
    upa3(
        &mut entries,
        "pyre_object::identitydict::w_dict_store_identity_strategy",
        "pyre_object::w_dict_store_identity_strategy",
        pyre_object::identitydict::w_dict_store_identity_strategy,
    );
    pa1(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::jit_float_abs",
        "pyre_interpreter::jit_float_abs",
        crate::objspace::descroperation::jit_float_abs,
    );
    pa0(
        &mut entries,
        "pyre_interpreter::call::pyre_debug_call_enabled",
        "pyre_interpreter::pyre_debug_call_enabled",
        crate::call::pyre_debug_call_enabled,
    );
    pa0(
        &mut entries,
        "pyre_interpreter::executioncontext::arm_async_eval_breaker",
        "pyre_interpreter::arm_async_eval_breaker",
        crate::executioncontext::arm_async_eval_breaker,
    );
    cpa7(
        &mut entries,
        "pyre_interpreter::module::_warnings::show_warning",
        "pyre_interpreter::show_warning",
        crate::module::_warnings::show_warning_jit_abi,
    );
    pa0(
        &mut entries,
        "pyre_interpreter::executioncontext::disarm_async_eval_breaker",
        "pyre_interpreter::disarm_async_eval_breaker",
        crate::executioncontext::disarm_async_eval_breaker,
    );

    // Eval-breaker poll residuals: the dispatch-loop poll reads the breaker
    // word, drains a pending memory-error bit, and services stop-the-world /
    // finalization requests through these cross-crate helpers.
    //
    // The value-returning ones ride a word-ABI bridge.  A residual whose
    // result is Int/Ref lowers to a direct `call_indirect` typed
    // `(i64 x n) -> i64` (`ResidualCallAbi::Word`, the default), and a Rust
    // `-> usize` / `-> bool` / `&T` argument is narrower than a word on
    // wasm32, where the call type-checks its callee.  The two `-> ()` polls
    // keep their plain rows because they take no arguments: `() -> ()` is the
    // type the void residual family declares on every target.  A void residual
    // that DOES take arguments needs a bridge like any other, which is why
    // `frame_anchor_release` has one.
    cp0(
        &mut entries,
        "majit_ir::eval_breaker_word::load",
        majit_ir::eval_breaker_word::load_jit_abi,
    );
    cp0(
        &mut entries,
        "majit_ir::eval_breaker_word::take_memory_error",
        majit_ir::eval_breaker_word::take_memory_error_jit_abi,
    );
    // The portal's prologue arms this bit before the dispatch loop, so it is
    // the first residual an `ENTRY=start` walk of `eval_loop_jit` meets.
    let eval_breaker_set_gc_interp: fn() = majit_ir::eval_breaker_word::set_gc_interp;
    p0(
        &mut entries,
        "majit_ir::eval_breaker_word::set_gc_interp",
        eval_breaker_set_gc_interp,
    );
    let gc_safepoint_poll: fn() = majit_gc::gc_sync::safepoint_poll;
    p0(
        &mut entries,
        "majit_gc::gc_sync::safepoint_poll",
        gc_safepoint_poll,
    );
    let thread_park_if_finalizing: fn() = crate::module::thread::park_if_finalizing;
    p0(
        &mut entries,
        "pyre_interpreter::module::thread::park_if_finalizing",
        thread_park_if_finalizing,
    );
    cp1(
        &mut entries,
        "pyre_interpreter::module::thread::all_thread_hooks_current",
        crate::module::thread::all_thread_hooks_current_jit_abi,
    );
    cp2(
        &mut entries,
        "pyre_interpreter::executioncontext::space_decrement_ticker",
        crate::executioncontext::space_decrement_ticker_jit_abi,
    );
    // The `anchor` handler graph residualizes `FrameAnchor::new` itself
    // (its aggregate return keeps it out of inlining), and `push_anchored`
    // reads back through `FrameAnchor::live`; bind both under the exact
    // path spellings the codewriter hashes for method targets.  They carry
    // their own bridges rather than sharing the free slot ops': one address
    // must name one function here.  `new` is `from_raw` handed back without a
    // `Drop`, and `front::mir` aliases an `Rvalue::Ref` over a bare local to
    // that local's own Variable without emitting an address-of, so `live`'s
    // `&self` arrives as the one-word anchor's value — the depth — rather
    // than a pointer to it.
    cpa1(
        &mut entries,
        "eval::FrameAnchor::new",
        "pyre_interpreter::eval::FrameAnchor::new",
        crate::eval::frame_anchor_new_jit_abi,
    );
    cpa1(
        &mut entries,
        "eval::FrameAnchor::live",
        "pyre_interpreter::eval::FrameAnchor::live",
        crate::eval::frame_anchor_live_method_jit_abi,
    );
    cp1(
        &mut entries,
        "pyre_interpreter::eval::frame_anchor_push",
        crate::eval::frame_anchor_push_jit_abi,
    );
    cp1(
        &mut entries,
        "pyre_interpreter::eval::frame_anchor_live",
        crate::eval::frame_anchor_live_jit_abi,
    );
    cp1(
        &mut entries,
        "pyre_interpreter::eval::frame_anchor_release",
        crate::eval::frame_anchor_release_jit_abi,
    );
    pa1(
        &mut entries,
        "pyre_interpreter::executioncontext::execution_context_builtin_cache_get",
        "pyre_interpreter::execution_context_builtin_cache_get",
        crate::executioncontext::execution_context_builtin_cache_get,
    );
    cpa0(
        &mut entries,
        "pyre_object::dict_eq_hook::has_compares_by_identity_hook",
        "pyre_object::has_compares_by_identity_hook",
        pyre_object::dict_eq_hook::has_compares_by_identity_hook,
    );
    cpa1(
        &mut entries,
        "pyre_object::dict_eq_hook::compares_by_identity_hooked",
        "pyre_object::compares_by_identity_hooked",
        pyre_object::dict_eq_hook::compares_by_identity_hooked,
    );
    cpa1(
        &mut entries,
        "pyre_object::dict_eq_hook::signal_hash_error",
        "pyre_object::signal_hash_error",
        pyre_object::dict_eq_hook::signal_hash_error,
    );
    cpa0(
        &mut entries,
        "pyre_object::dict_eq_hook::take_hash_error",
        "pyre_object::take_hash_error",
        pyre_object::dict_eq_hook::take_hash_error,
    );
    cpa1(
        &mut entries,
        "pyre_object::dict_eq_hook::signal_eq_error",
        "pyre_object::signal_eq_error",
        pyre_object::dict_eq_hook::signal_eq_error,
    );
    cpa0(
        &mut entries,
        "pyre_object::dict_eq_hook::take_eq_error",
        "pyre_object::take_eq_error",
        pyre_object::dict_eq_hook::take_eq_error,
    );
    cpa0(
        &mut entries,
        "pyre_object::dict_eq_hook::eq_error_pending",
        "pyre_object::eq_error_pending",
        pyre_object::dict_eq_hook::eq_error_pending,
    );
    cpa0(
        &mut entries,
        "pyre_object::dict_eq_hook::begin_callback_free_probe",
        "pyre_object::begin_callback_free_probe",
        pyre_object::dict_eq_hook::begin_callback_free_probe,
    );
    cpa0(
        &mut entries,
        "pyre_object::dict_eq_hook::end_callback_free_probe",
        "pyre_object::end_callback_free_probe",
        pyre_object::dict_eq_hook::end_callback_free_probe,
    );
    cpa0(
        &mut entries,
        "pyre_object::dict_eq_hook::callback_free_probe_active",
        "pyre_object::callback_free_probe_active",
        pyre_object::dict_eq_hook::callback_free_probe_active,
    );
    cpa0(
        &mut entries,
        "pyre_object::dict_eq_hook::callback_free_probe_broken",
        "pyre_object::callback_free_probe_broken",
        pyre_object::dict_eq_hook::callback_free_probe_broken,
    );
    cpa0(
        &mut entries,
        "pyre_object::dict_eq_hook::break_callback_free_probe",
        "pyre_object::break_callback_free_probe",
        pyre_object::dict_eq_hook::break_callback_free_probe,
    );
    cpa0(
        &mut entries,
        "pyre_interpreter::stack_check::stack_almost_full",
        "pyre_interpreter::stack_almost_full",
        crate::stack_check::stack_almost_full,
    );

    // `@jit.elidable`-decorated inherent methods that show up as
    // `residual_call_*` in the codewriter (`call.py:181-187
    // getfunctionptr(graph)` parity).  Without an entry here
    // `direct_funcptr_value` (`jtransform.rs`) falls back to
    // `symbolic_fnaddr_for_path`, which is a deterministic hash but NOT
    // a valid function address — invoking it at the walker's
    // `execute_residual_call` (`executor.rs`) is an
    // immediate SEGV.  Path shape matches
    // `target_to_path` for inherent method calls
    // (`parse.rs`'s `CallPath::for_impl_method(impl_type_joined,
    // method)`): the `register_macro_helper_trace_fnaddr` string-strip
    // drops the leading crate segment, leaving `[module, Type, method]`
    // which is the exact 3-segment shape `for_impl_method` produces.
    //
    // PyFrame::nlocals — invoked by `eval.rs`'s `pop_value` and is the
    // funcptr the walker reaches when dispatching `PopTop`'s nested
    // `pop_value` sub-jitcode.  Same dual-shape binding as
    // `PyFrame::pop` below: the bare `self.nlocals()` spelling inside
    // the MIR-lowered `pop_value` graph resolves through
    // `impl_method_owner` to the 2-segment `["PyFrame", "nlocals"]`,
    // while the module-qualified form is the 3-segment
    // `["pyframe", "PyFrame", "nlocals"]` — register both.
    let pyframe_nlocals: fn(&crate::pyframe::PyFrame) -> usize = crate::pyframe::PyFrame::nlocals;
    pa1(
        &mut entries,
        "pyre_interpreter::pyframe::PyFrame::nlocals",
        "pyre_interpreter::PyFrame::nlocals",
        pyframe_nlocals,
    );

    // `PyFrame::pop` — invoked by `<PyFrame as SharedOpcodeHandler>::pop_value`
    // at its `Ok(self.pop())` tail (`eval.rs`).  Two CallPath shapes need binding:
    //
    // 1. The qualified `PyFrame::pop(self)` spelling resolves to the
    //    2-segment CallPath `["PyFrame", "pop"]` via `for_impl_method`.
    // 2. The bare `self.pop()` spelling goes through `target_to_path`'s
    //    suffix-match fallback (call.rs), which returns the
    //    3-segment module-qualified key `["pyframe", "PyFrame", "pop"]`
    //    that `function_graphs` actually stores inherent impl methods
    //    under (per `parse::extract_inherent_impl_methods`).
    //
    // `register_macro_helper_trace_fnaddr` strips the leading segment,
    // so we register both spellings as an alias pair: the 3-segment
    // input `pyre_interpreter::PyFrame::pop` produces the 2-segment
    // canonical, and the 4-segment input `pyre_interpreter::pyframe::PyFrame::pop`
    // produces the 3-segment module-qualified form.  Without the second
    // binding, `fnaddr_for_target` for `self.pop()` falls back to the
    // symbolic hash from [`symbolic_fnaddr_for_path`], which SEGVs at
    // trace-time call.
    let pyframe_pop: fn(&mut crate::pyframe::PyFrame) -> pyre_object::PyObjectRef =
        crate::pyframe::PyFrame::pop;
    pa1(
        &mut entries,
        "pyre_interpreter::pyframe::PyFrame::pop",
        "pyre_interpreter::PyFrame::pop",
        pyframe_pop,
    );

    // `stack_underflow_error` deliberately remains unpublished: its `&str`
    // argument is a two-word aggregate with no one-word residual-call ABI.

    // `get_current_exception` / `set_current_exception` — the named TLS
    // accessors `PyFrame::push_exc_info` / `pop_except` (`eval.rs`) call
    // for the per-thread `CURRENT_EXCEPTION` slot.  Both carry
    // `#[dont_look_inside]` (the `LocalKey::with` closure inside has no
    // extractable graph), so the codewriter classifies the calls
    // `Residual` and needs these bindings to bake real funcptrs instead
    // of `symbolic_fnaddr_for_path` hashes.  These are the
    // interpreter-side twins of the trace-side
    // `get_current_exception_fn` / `set_current_exception_fn` cpu
    // helpers — same TLS slot, same flat read/write semantics.
    let get_current_exc: fn() -> pyre_object::PyObjectRef = crate::eval::get_current_exception;
    pa0(
        &mut entries,
        "pyre_interpreter::eval::get_current_exception",
        "pyre_interpreter::get_current_exception",
        get_current_exc,
    );
    // `get_sys_exception` is the PyPy `ExecutionContext.sys_exc_info` leaf:
    // it may walk the running-generator chain, but that execution-context
    // state is runtime data and must not be folded into a trace.
    let get_sys_exc: fn() -> pyre_object::PyObjectRef = crate::eval::get_sys_exception;
    pa0(
        &mut entries,
        "pyre_interpreter::eval::get_sys_exception",
        "pyre_interpreter::get_sys_exception",
        get_sys_exc,
    );
    // `ExecutionContext._get_topmost_exception` is the loop-bearing cold arm
    // of `ExecutionContext.sys_exc_info` (`pypy/interpreter/executioncontext.py`).
    // PyPy's JitPolicy leaves that graph out of an inline trace because it
    // contains a loop, but `getfunctionptr(graph)` still gives the residual
    // call a real address. Publish the same method target here: without it the
    // codewriter leaves a symbolic hash in `get_sys_exception`'s JitCode, and
    // the builtin descent gate must reject even the ordinary handled-exception
    // arm which never calls this helper.
    let get_topmost_exception: fn(
        &crate::executioncontext::ExecutionContext,
    ) -> pyre_object::PyObjectRef =
        crate::executioncontext::ExecutionContext::_get_topmost_exception;
    pa1(
        &mut entries,
        "pyre_interpreter::executioncontext::ExecutionContext::_get_topmost_exception",
        "pyre_interpreter::ExecutionContext::_get_topmost_exception",
        get_topmost_exception,
    );
    let set_current_exc: fn(pyre_object::PyObjectRef) = crate::eval::set_current_exception;
    pa1(
        &mut entries,
        "pyre_interpreter::eval::set_current_exception",
        "pyre_interpreter::set_current_exception",
        set_current_exc,
    );

    // `w_type` / `w_object` — the `type` / `object` typeobject accessors
    // read the `W_TYPE_TYPEOBJECT` / `W_OBJECT_TYPEOBJECT` `OnceLock<usize>`
    // slots set once at startup.  Both carry `#[dont_look_inside]` (the
    // `OnceLock::get` read has no registry-resolvable accessor graph), so
    // the codewriter classifies the calls `Residual` and needs these
    // bindings to bake real funcptrs instead of `symbolic_fnaddr_for_path`
    // hashes.  Callers spell them `crate::typedef::w_type()`, the sole
    // path form, with no crate-root re-export.
    let w_type: fn() -> pyre_object::PyObjectRef = crate::typedef::w_type;
    p0(&mut entries, "pyre_interpreter::typedef::w_type", w_type);
    let w_object: fn() -> pyre_object::PyObjectRef = crate::typedef::w_object;
    p0(
        &mut entries,
        "pyre_interpreter::typedef::w_object",
        w_object,
    );
    // `_ast` keeps CPython 3.14's process-wide `Load_singleton` in a rooted
    // `OnceLock` slot.  Its accessor is opaque for the same reason as the
    // builtin type accessors above and remains a residual runtime read.
    let ast_load_singleton: fn() -> pyre_object::PyObjectRef =
        crate::module::_ast::moduledef::load_singleton;
    p0(
        &mut entries,
        "pyre_interpreter::module::_ast::moduledef::load_singleton",
        ast_load_singleton,
    );

    // Thread-local / `OnceLock` accessors that carry `#[dont_look_inside]`
    // (the `.with` closure read has no extractable graph): the codewriter
    // classifies the calls `Residual` and needs real funcptrs instead of
    // `symbolic_fnaddr_for_path` hashes.  Error-slot twins of
    // `get_current_exception` / `set_current_exception`, plus the weakref
    // proxy type singletons (twins of `w_type` / `w_object`).
    let set_call_error: fn(crate::PyError) = crate::call::set_call_error;
    // ABI-UNSOUND: `PyError` is passed by value and is wider than a word.
    push_abi_unsound_argument_fnaddr(
        &mut entries,
        &mut abi_unsound_arguments,
        "pyre_interpreter::call::set_call_error",
        set_call_error as *const (),
    );
    // `take_call_error` deliberately remains unpublished: it returns an error
    // as a value, so routing it through `BH_LAST_EXC_VALUE` would convert the
    // returned value into a raise.
    let clear_call_error: fn() = crate::call::clear_call_error;
    p0(
        &mut entries,
        "pyre_interpreter::call::clear_call_error",
        clear_call_error,
    );
    // `#[dont_look_inside]` execution-context thread-local read, a twin of
    // the call-error slot accessors above. front::mir const-folds the
    // `ThreadLocal` global to None, so its body has no extractable graph and
    // the call stays a residual read via the registered fnaddr.
    let take_last_exec_ctx: fn() -> *const crate::PyExecutionContext =
        crate::call::take_last_exec_ctx;
    p0(
        &mut entries,
        "pyre_interpreter::call::take_last_exec_ctx",
        take_last_exec_ctx,
    );
    // `take_pending_hash_error` deliberately remains unpublished: it returns
    // an error as a value, so routing it through `BH_LAST_EXC_VALUE` would
    // convert the returned value into a raise.
    let proxy_type: fn() -> pyre_object::PyObjectRef =
        crate::module::_weakref::interp__weakref::proxy_type;
    p0(
        &mut entries,
        "pyre_interpreter::module::_weakref::interp__weakref::proxy_type",
        proxy_type,
    );
    let callable_proxy_type: fn() -> pyre_object::PyObjectRef =
        crate::module::_weakref::interp__weakref::callable_proxy_type;
    p0(
        &mut entries,
        "pyre_interpreter::module::_weakref::interp__weakref::callable_proxy_type",
        callable_proxy_type,
    );

    // Stack-overflow / JIT-pending-exception bookkeeping accessors, all
    // `#[dont_look_inside]` (PYRE_STACKTOOBIG static / TL_JIT_PENDING_EXCEPTION
    // thread-local reads with no extractable graph).  The slowpath is
    // already a C-ABI residual the backend calls directly; the wrappers
    // become residual Calls.
    let stack_slowpath: extern "C" fn(usize) -> u8 =
        crate::stack_check::pyre_stack_too_big_slowpath;
    cp1(
        &mut entries,
        "pyre_interpreter::stack_check::pyre_stack_too_big_slowpath",
        stack_slowpath,
    );
    cp0(
        &mut entries,
        "pyre_interpreter::stack_check::stack_check",
        crate::stack_check::stack_check_jit_abi,
    );
    cp0(
        &mut entries,
        "pyre_interpreter::stack_check::drain_jit_pending_exception",
        crate::stack_check::drain_jit_pending_exception_jit_abi,
    );

    // `pyframe_get_pycode` / `ncells` / `npure_cellvars` / `PyFrame::ncells`
    // carry `#[elidable_cannot_raise]`.  `call.rs:has_cannot_raise_assertion`
    // only honours the assertion when `function_fnaddrs.contains_key(p)`,
    // so without a registration the descr falls back to
    // `EF_ELIDABLE_CAN_RAISE`.
    //
    // These free functions are also called unqualified inside `pyframe.rs`
    // itself (`pyframe_get_pycode(self)` / `ncells(code)` / `npure_cellvars(code)`).
    // `target_to_path` for a `FunctionPath` returns the segments verbatim,
    // so an in-module bare call resolves to a 1-segment CallPath
    // `["<name>"]` while a cross-module qualified call resolves to
    // `["pyframe", "<name>"]`.  Register both shapes as an alias pair, via
    // the strip-one-segment rule in
    // `register_macro_helper_trace_fnaddr`.
    let pyframe_get_pycode_fn: unsafe fn(&crate::pyframe::PyFrame) -> *const crate::CodeObject =
        crate::pyframe::pyframe_get_pycode;
    upa1(
        &mut entries,
        "pyre_interpreter::pyframe::pyframe_get_pycode",
        "pyre_interpreter::pyframe_get_pycode",
        pyframe_get_pycode_fn,
    );

    let report_stack_underflow: fn(&crate::pyframe::PyFrame) =
        crate::pyframe::report_stack_underflow;
    pa1(
        &mut entries,
        "pyre_interpreter::pyframe::report_stack_underflow",
        "pyre_interpreter::report_stack_underflow",
        report_stack_underflow,
    );

    let pyframe_ncells_free: fn(&crate::CodeObject) -> usize = crate::pyframe::ncells;
    pa1(
        &mut entries,
        "pyre_interpreter::pyframe::ncells",
        "pyre_interpreter::ncells",
        pyframe_ncells_free,
    );

    let pyframe_npure_cellvars: fn(&crate::CodeObject) -> usize = crate::pyframe::npure_cellvars;
    pa1(
        &mut entries,
        "pyre_interpreter::pyframe::npure_cellvars",
        "pyre_interpreter::npure_cellvars",
        pyframe_npure_cellvars,
    );

    let pyframe_ncells_method: fn(&crate::pyframe::PyFrame) -> usize =
        crate::pyframe::PyFrame::ncells;
    p1(
        &mut entries,
        "pyre_interpreter::pyframe::PyFrame::ncells",
        pyframe_ncells_method,
    );

    // LoadFast/LoadFastBorrow/LoadFastCheck arm folding helpers.  Both
    // carry `#[elidable_cannot_raise]` so `has_cannot_raise_assertion`
    // requires the fnaddr registration to fire (`call.rs`
    // gates the assertion on `function_fnaddrs.contains_key(p)`).
    // Without these the chained `Arg::get` / `VarNum::as_usize` /
    // `Vec::len` third-party helpers reach the walker as unfolded
    // `residual_call` ops and the walker's `goto_if_not` bounds-check
    // aborts with `GotoIfNotValueNotConcrete`.
    //
    // The alias pair (vs a single module-qualified path) is required because
    // the in-module call site `load_fast_var_num_to_index(var_num, op_arg)`
    // inside `pyopcode.rs` resolves to a bare-segment `CallPath`
    // (`["load_fast_var_num_to_index"]`) that the assertion-aware hint
    // walker DOES populate but the module-qualified-only fnaddr
    // registration would miss.  Register the bare alias alongside the
    // canonical `pyopcode::name` form so the assertion gate fires.
    let load_fast_var_num_to_index: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::VarNum>,
        crate::bytecode::OpArg,
    ) -> usize = crate::pyopcode::load_fast_var_num_to_index;
    pa2(
        &mut entries,
        "pyre_interpreter::pyopcode::load_fast_var_num_to_index",
        "pyre_interpreter::load_fast_var_num_to_index",
        load_fast_var_num_to_index,
    );

    let code_varnames_len: fn(&crate::CodeObject) -> usize = crate::pyopcode::code_varnames_len;
    pa1(
        &mut entries,
        "pyre_interpreter::pyopcode::code_varnames_len",
        "pyre_interpreter::code_varnames_len",
        code_varnames_len,
    );

    let code_instructions_len: fn(&crate::CodeObject) -> usize =
        crate::pyopcode::code_instructions_len;
    pa1(
        &mut entries,
        "pyre_interpreter::pyopcode::code_instructions_len",
        "pyre_interpreter::code_instructions_len",
        code_instructions_len,
    );

    cpa2(
        &mut entries,
        "pyre_interpreter::pyopcode::code_unit_at",
        "pyre_interpreter::code_unit_at",
        bh_code_unit_at,
    );

    // Paired-local index decode helpers for the LoadFastLoadFast /
    // StoreFastLoadFast / StoreFastStoreFast /
    // LoadFastBorrowLoadFastBorrow arms — same alias-pair rationale as
    // `load_fast_var_num_to_index` above.
    let var_nums_to_first_index: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::VarNums>,
        crate::bytecode::OpArg,
    ) -> usize = crate::pyopcode::var_nums_to_first_index;
    pa2(
        &mut entries,
        "pyre_interpreter::pyopcode::var_nums_to_first_index",
        "pyre_interpreter::var_nums_to_first_index",
        var_nums_to_first_index,
    );

    let var_nums_to_second_index: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::VarNums>,
        crate::bytecode::OpArg,
    ) -> usize = crate::pyopcode::var_nums_to_second_index;
    pa2(
        &mut entries,
        "pyre_interpreter::pyopcode::var_nums_to_second_index",
        "pyre_interpreter::var_nums_to_second_index",
        var_nums_to_second_index,
    );

    // Opcode oparg decode helpers for two-phase lifting. These wrap
    // RustPython's generic `Arg::get` and `CodeUnits::deref` surfaces
    // behind first-party residual calls whose return values are the
    // scalar/enum values consumed by the opcode handlers.
    let label_arg_to_usize: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::Label>,
        crate::bytecode::OpArg,
    ) -> usize = crate::pyopcode::label_arg_to_usize;
    pa2(
        &mut entries,
        "pyre_interpreter::pyopcode::label_arg_to_usize",
        "pyre_interpreter::label_arg_to_usize",
        label_arg_to_usize,
    );

    let jump_target_forward_decoded: fn(
        &crate::CodeObject,
        usize,
        crate::bytecode::Arg<crate::bytecode::oparg::Label>,
        crate::bytecode::OpArg,
    ) -> usize = crate::pyopcode::jump_target_forward_decoded;
    pa4(
        &mut entries,
        "pyre_interpreter::pyopcode::jump_target_forward_decoded",
        "pyre_interpreter::jump_target_forward_decoded",
        jump_target_forward_decoded,
    );

    let jump_target_forward_from_oparg: fn(
        &crate::CodeObject,
        usize,
        crate::bytecode::OpArg,
    ) -> usize = crate::pyopcode::jump_target_forward_from_oparg;
    pa3(
        &mut entries,
        "pyre_interpreter::pyopcode::jump_target_forward_from_oparg",
        "pyre_interpreter::jump_target_forward_from_oparg",
        jump_target_forward_from_oparg,
    );

    let jump_target_backward_decoded: fn(
        &crate::CodeObject,
        usize,
        crate::bytecode::Arg<crate::bytecode::oparg::Label>,
        crate::bytecode::OpArg,
    ) -> usize = crate::pyopcode::jump_target_backward_decoded;
    pa4(
        &mut entries,
        "pyre_interpreter::pyopcode::jump_target_backward_decoded",
        "pyre_interpreter::jump_target_backward_decoded",
        jump_target_backward_decoded,
    );

    let binary_op_arg: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::BinaryOperator>,
        crate::bytecode::OpArg,
    ) -> crate::bytecode::BinaryOperator = crate::pyopcode::binary_op_arg;
    // ABI-UNSOUND: the result is an ADT. `return_type_string_to_value_type`
    // classifies every type name it does not know as `Type::Ref`, so this
    // one-byte enum is read back as a reference.
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::binary_op_arg",
        "pyre_interpreter::binary_op_arg",
        binary_op_arg as *const (),
    );

    let comparison_op_arg: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::ComparisonOperator>,
        crate::bytecode::OpArg,
    ) -> crate::bytecode::ComparisonOperator = crate::pyopcode::comparison_op_arg;
    // ABI-UNSOUND: the result is an ADT. `return_type_string_to_value_type`
    // classifies every type name it does not know as `Type::Ref`, so this
    // one-byte enum is read back as a reference.
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::comparison_op_arg",
        "pyre_interpreter::comparison_op_arg",
        comparison_op_arg as *const (),
    );

    let invert_arg: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::Invert>,
        crate::bytecode::OpArg,
    ) -> crate::bytecode::Invert = crate::pyopcode::invert_arg;
    // ABI-UNSOUND: the result is an ADT. `return_type_string_to_value_type`
    // classifies every type name it does not know as `Type::Ref`, so this
    // one-byte enum is read back as a reference.
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::invert_arg",
        "pyre_interpreter::invert_arg",
        invert_arg as *const (),
    );

    let build_slice_arg: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::BuildSliceArgCount>,
        crate::bytecode::OpArg,
    ) -> crate::bytecode::BuildSliceArgCount = crate::pyopcode::build_slice_arg;
    // ABI-UNSOUND: the result is an ADT. `return_type_string_to_value_type`
    // classifies every type name it does not know as `Type::Ref`, so this
    // one-byte enum is read back as a reference.
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::build_slice_arg",
        "pyre_interpreter::build_slice_arg",
        build_slice_arg as *const (),
    );

    let common_constant_arg: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::CommonConstant>,
        crate::bytecode::OpArg,
    ) -> crate::bytecode::CommonConstant = crate::pyopcode::common_constant_arg;
    // ABI-UNSOUND: the result is an ADT. `return_type_string_to_value_type`
    // classifies every type name it does not know as `Type::Ref`, so this
    // one-byte enum is read back as a reference.
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::common_constant_arg",
        "pyre_interpreter::common_constant_arg",
        common_constant_arg as *const (),
    );

    let convert_value_arg: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::ConvertValueOparg>,
        crate::bytecode::OpArg,
    ) -> crate::bytecode::ConvertValueOparg = crate::pyopcode::convert_value_arg;
    // ABI-UNSOUND: the result is an ADT. `return_type_string_to_value_type`
    // classifies every type name it does not know as `Type::Ref`, so this
    // one-byte enum is read back as a reference.
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::convert_value_arg",
        "pyre_interpreter::convert_value_arg",
        convert_value_arg as *const (),
    );

    let special_method_arg: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::SpecialMethod>,
        crate::bytecode::OpArg,
    ) -> crate::bytecode::SpecialMethod = crate::pyopcode::special_method_arg;
    // ABI-UNSOUND: the result is an ADT. `return_type_string_to_value_type`
    // classifies every type name it does not know as `Type::Ref`, so this
    // one-byte enum is read back as a reference.
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::special_method_arg",
        "pyre_interpreter::special_method_arg",
        special_method_arg as *const (),
    );

    let make_function_flag_arg: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::MakeFunctionFlag>,
        crate::bytecode::OpArg,
    ) -> crate::bytecode::MakeFunctionFlag = crate::pyopcode::make_function_flag_arg;
    // ABI-UNSOUND: the result is an ADT. `return_type_string_to_value_type`
    // classifies every type name it does not know as `Type::Ref`, so this
    // one-byte enum is read back as a reference.
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::make_function_flag_arg",
        "pyre_interpreter::make_function_flag_arg",
        make_function_flag_arg as *const (),
    );

    let intrinsic_function_1_arg: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::IntrinsicFunction1>,
        crate::bytecode::OpArg,
    ) -> crate::bytecode::IntrinsicFunction1 = crate::pyopcode::intrinsic_function_1_arg;
    // ABI-UNSOUND: the result is an ADT. `return_type_string_to_value_type`
    // classifies every type name it does not know as `Type::Ref`, so this
    // one-byte enum is read back as a reference.
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::intrinsic_function_1_arg",
        "pyre_interpreter::intrinsic_function_1_arg",
        intrinsic_function_1_arg as *const (),
    );

    let intrinsic_function_2_arg: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::IntrinsicFunction2>,
        crate::bytecode::OpArg,
    ) -> crate::bytecode::IntrinsicFunction2 = crate::pyopcode::intrinsic_function_2_arg;
    // ABI-UNSOUND: the result is an ADT. `return_type_string_to_value_type`
    // classifies every type name it does not know as `Type::Ref`, so this
    // one-byte enum is read back as a reference.
    push_abi_unsound_alias_pair(
        &mut entries,
        "pyre_interpreter::pyopcode::intrinsic_function_2_arg",
        "pyre_interpreter::intrinsic_function_2_arg",
        intrinsic_function_2_arg as *const (),
    );

    let raise_kind_arg_as_usize: fn(
        crate::bytecode::Arg<crate::bytecode::oparg::RaiseKind>,
        crate::bytecode::OpArg,
    ) -> usize = crate::pyopcode::raise_kind_arg_as_usize;
    pa2(
        &mut entries,
        "pyre_interpreter::pyopcode::raise_kind_arg_as_usize",
        "pyre_interpreter::raise_kind_arg_as_usize",
        raise_kind_arg_as_usize,
    );

    // `PyError::type_error` deliberately remains unpublished: its by-value
    // `String` argument is a three-word aggregate with no one-word
    // residual-call ABI.

    // `PyError::to_exc_object` — residual exception materialization emitted by
    // the two-phase rtyper for `PyError.to_exc_object()` call sites.  This uses
    // the same impl-method CallPath shape as `type_error`, resolving to
    // `["PyError", "to_exc_object"]` after the crate segment is stripped.
    let pyerror_to_exc_object: fn(&mut crate::PyError) -> pyre_object::PyObjectRef =
        crate::PyError::to_exc_object;
    p1(
        &mut entries,
        "pyre_interpreter::PyError::to_exc_object",
        pyerror_to_exc_object,
    );

    // RPython convention (cross-reference `support.py:255-271` for
    // the C-trunc helpers, `rint.py:398/495` for the Python-floor
    // ones) is to keep the two semantic flavours under DISTINCT
    // canonical names:
    //
    //   - bare `int_mod` / `int_floordiv` — the lltype-level
    //     truncating primitive (canonical names of the
    //     `_ll_2_int_mod` / `_ll_2_int_floordiv` no-branch reverse).
    //     C-truncating output.
    //   - `int.py_mod` / `int.py_div` — the Python-semantic
    //     `@jit.oopspec("int.py_mod")` / `@jit.oopspec("int.py_div")`
    //     names that decorate `ll_int_py_mod` / `ll_int_py_div`.
    //     Python-floor output.
    //
    // Pyre's `jtransform.rs` BinOp{mod,floordiv,Int} arm emits a
    // `CallTarget::function_path(["_ll_2_int_mod"])` /
    // `CallTarget::function_path(["_ll_2_int_floordiv"])` per
    // `jtransform.py:576-577 rewrite_op_int_floordiv =
    // _do_builtin_call` (which resolves the helper through
    // `support.py` `_ll_2_int_mod` / `:255` `_ll_2_int_floordiv`).
    // The C-trunc residual call below is what a Rust `/` / `%` in a
    // descended body sees.  The Python-floor `ll_int_py_*` pair
    // registered after it is route (b): `int_floordiv` / `int_mod`
    // call the interpreter's `#[oopspec("int.py_div")]` twins, so the
    // generated `//` / `%` descent records the same elidable
    // `int.py_div` / `int.py_mod` call the hand fold did.
    //
    // `register_macro_helper_trace_fnaddr` strips the leading segment
    // from `full_path`; for a single-segment path (no `::`) the entire
    // string survives as the canonical CallPath, matching the segment
    // shape jtransform produces.
    //
    // The Rust-source graphs for the integer helpers are NOT
    // registered in `CallControl::function_graphs` (pyre has no
    // `MixLevelHelperAnnotator` to materialise a graph from a `pub
    // extern "C"` function pointer), so `call.rs`'s
    // `find_all_graphs_bfs` finds the function pointer via
    // `function_fnaddrs` lookup but cannot seed the BFS through the
    // helper's body — the helpers stay opaque to the inliner,
    // matching upstream behaviour for any `@dont_look_inside`
    // oopspec helper.  Two `support.py:inline_calls_to` entries
    // are intentionally NOT bound:
    //   * `_ll_1_int_abs` — RPython `inline_calls_to` seeds the
    //     `int_abs` helper *graph* into the BFS for actual inlining
    //     at `call.py todo.append(c_func.value._obj.graph)`.
    //     Pyre can register the fnaddr but cannot fabricate the
    //     helper body graph from an `extern "C"` function pointer
    //     (no `MixLevelHelperAnnotator.constfunc` analogue), so a
    //     fnaddr-only binding would make `int_abs` an opaque extern
    //     helper — the opposite of the upstream inlining intent.
    //     No production pyre rewrite emits `direct_call(_ll_1_int_abs)`
    //     so the binding is omitted until the rtyper-equivalent
    //     can synthesise the body graph.
    //   * `_ll_1_ll_math_ll_math_sqrt` — `rpython/rtyper/lltypesystem/
    //     module/ll_math.py ll_math_sqrt` raises
    //     `ValueError("math domain error")` on negative input, and
    //     Rust's `f64::sqrt()` returns NaN; making the fnaddr
    //     reachable would be a silent semantic regression.
    // See the TODO block at
    // `call.rs::find_all_graphs_bfs` for the convergence path.
    cp2(
        &mut entries,
        "_ll_2_int_floordiv",
        majit_metainterp::blackhole::_ll_2_int_floordiv,
    );
    cp2(
        &mut entries,
        "_ll_2_int_mod",
        majit_metainterp::blackhole::_ll_2_int_mod,
    );
    p2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::ll_int_py_div",
        crate::objspace::descroperation::ll_int_py_div,
    );
    p2(
        &mut entries,
        "pyre_interpreter::objspace::descroperation::ll_int_py_mod",
        crate::objspace::descroperation::ll_int_py_mod,
    );

    // `support.py _ll_1_cast_uint_to_float` / `_ll_1_cast_float_to_uint`
    // residual-call targets emitted by
    // `codewriter/jtransform.rs:cast_*_to_*` (mirroring
    // `jtransform.py _do_builtin_call`).  Without these the
    // codewriter falls back to `symbolic_fnaddr_for_path`, which
    // produces a deterministic but unbound hash — fine for source
    // analysis but unreachable at runtime.  The 1-segment root_path
    // alias is what `CallTarget::function_path(["cast_uint_to_float"])`
    // resolves against after `register_macro_helper_trace_fnaddr`
    // strips the crate segment.
    pa1(
        &mut entries,
        "majit_metainterp::blackhole::cast_uint_to_float",
        "majit_metainterp::cast_uint_to_float",
        majit_metainterp::blackhole::cast_uint_to_float,
    );
    pa1(
        &mut entries,
        "majit_metainterp::blackhole::cast_float_to_uint",
        "majit_metainterp::cast_float_to_uint",
        majit_metainterp::blackhole::cast_float_to_uint,
    );

    // `_ll_2_str_eq_nonnull` (`rpython/jit/codewriter/support.py-
    // 538`) is the helper canonically registered by `jtransform.py:
    // 620-624 _register_extra_helper(OS_STREQ_NONNULL, "str.eq_nonnull",
    // ...)` and `:637-641 _register_extra_helper(OS_UNIEQ_NONNULL,
    // "str.eq_nonnull", ...)`.  Pyre intentionally does NOT register
    // a host fnaddr for it: there is no `rstr.STR`-equivalent GC
    // layout in pyre-object today, so a registration would have to
    // point at a panic-stub that fails at runtime — a parity
    // violation against `support.py:526-538`'s real `s.chars[i]`
    // comparison body.
    //
    // Pyre's type state has no `Ptr(rstr.STR)` / `Ptr(rstr.UNICODE)`
    // channel yet: the elidable-promote dual hint (`PromoteOrString`)
    // falls through to the plain `<kind>_guard_value` arm, and direct
    // `hint_promote_string` / `hint_promote_unicode` calls fail loud
    // in `codewriter/jtransform.rs`.  Re-introduce the
    // registration here together with a line-by-line port of
    // `_ll_2_str_eq_nonnull`'s body in `majit-metainterp::blackhole`
    // once pyre grows the backing GC struct.

    (entries, abi_unsound_arguments)
}

/// Build-time addresses of the prebuilt static `PyType` singletons that
/// pyre source carries through the flowgraph as opaque `LOAD_GLOBAL`
/// constants (`flowcontext.py:856` pushes the per-module-globals entry
/// as `Constant(value)`).  The codewriter bakes each into
/// `JitCode.constants_i` as a build-time `ConstValue::Int(addr)`.
///
/// The translator (`majit-translate`) sits in `rpython/` layer terms
/// below the object space and must not import `pyre-object`; the driver
/// supplies these prebuilt-instance addresses across the translation
/// boundary the same way `rpython/jit` receives `Constant(GCREF)` from
/// the host rather than importing `pypy/objspace`.  Resolved here in the
/// same build-script process that runs the translator, so the captured
/// addresses are identical to a direct `&pyre_object::X` read at the
/// codewriter call site.
///
/// Keys name the static's path.  The front-end reaches them through
/// `HostStaticAddrs::pytypes` and matches with `front::mir`'s
/// `static_key_matches`, which accepts the full path, the crate-stripped
/// path, or either with the key as a `::`-boundary suffix — so both the
/// `module::NAME` spelling the rows below use and the fully-qualified
/// spelling [`pyre_class_pytype_addrs`] carries resolve the same static.
pub fn jit_static_pytype_addrs() -> Vec<(&'static str, i64)> {
    macro_rules! pytype_addr {
        ($key:literal, $($path:tt)::+) => {
            ($key, &pyre_object::$($path)::+ as *const _ as i64)
        };
    }
    let mut rows = vec![
        pytype_addr!(
            "bytearrayobject::BYTEARRAY_TYPE",
            bytearrayobject::BYTEARRAY_TYPE
        ),
        pytype_addr!("bytesobject::BYTES_TYPE", bytesobject::BYTES_TYPE),
        pytype_addr!("interp_array::ARRAY_TYPE", interp_array::ARRAY_TYPE),
        pytype_addr!(
            "celldict::OBJECT_MUTABLE_CELL_TYPE",
            celldict::OBJECT_MUTABLE_CELL_TYPE
        ),
        pytype_addr!(
            "celldict::INT_MUTABLE_CELL_TYPE",
            celldict::INT_MUTABLE_CELL_TYPE
        ),
        pytype_addr!(
            "dictmultiobject::MODULE_DICT_TYPE",
            dictmultiobject::MODULE_DICT_TYPE
        ),
        pytype_addr!(
            "dictmultiobject::DICT_KEYS_TYPE",
            dictmultiobject::DICT_KEYS_TYPE
        ),
        pytype_addr!(
            "dictmultiobject::DICT_VALUES_TYPE",
            dictmultiobject::DICT_VALUES_TYPE
        ),
        pytype_addr!(
            "dictmultiobject::DICT_ITEMS_TYPE",
            dictmultiobject::DICT_ITEMS_TYPE
        ),
        pytype_addr!(
            "dictmultiobject::DICT_KEYITERATOR_TYPE",
            dictmultiobject::DICT_KEYITERATOR_TYPE
        ),
        pytype_addr!(
            "dictmultiobject::DICT_VALUEITERATOR_TYPE",
            dictmultiobject::DICT_VALUEITERATOR_TYPE
        ),
        pytype_addr!(
            "dictmultiobject::DICT_ITEMITERATOR_TYPE",
            dictmultiobject::DICT_ITEMITERATOR_TYPE
        ),
        pytype_addr!(
            "dictmultiobject::DICT_REVERSEKEYITERATOR_TYPE",
            dictmultiobject::DICT_REVERSEKEYITERATOR_TYPE
        ),
        pytype_addr!(
            "dictmultiobject::DICT_REVERSEVALUEITERATOR_TYPE",
            dictmultiobject::DICT_REVERSEVALUEITERATOR_TYPE
        ),
        pytype_addr!(
            "dictmultiobject::DICT_REVERSEITEMITERATOR_TYPE",
            dictmultiobject::DICT_REVERSEITEMITERATOR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXCEPTION_TYPE",
            interp_exceptions::EXCEPTION_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_EXCEPTION_TYPE",
            interp_exceptions::EXC_EXCEPTION_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_ARITHMETIC_ERROR_TYPE",
            interp_exceptions::EXC_ARITHMETIC_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_OVERFLOW_ERROR_TYPE",
            interp_exceptions::EXC_OVERFLOW_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_ZERO_DIVISION_ERROR_TYPE",
            interp_exceptions::EXC_ZERO_DIVISION_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_TYPE_ERROR_TYPE",
            interp_exceptions::EXC_TYPE_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_VALUE_ERROR_TYPE",
            interp_exceptions::EXC_VALUE_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_NAME_ERROR_TYPE",
            interp_exceptions::EXC_NAME_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_UNBOUND_LOCAL_ERROR_TYPE",
            interp_exceptions::EXC_UNBOUND_LOCAL_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_INDEX_ERROR_TYPE",
            interp_exceptions::EXC_INDEX_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_KEY_ERROR_TYPE",
            interp_exceptions::EXC_KEY_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_ATTRIBUTE_ERROR_TYPE",
            interp_exceptions::EXC_ATTRIBUTE_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_RUNTIME_ERROR_TYPE",
            interp_exceptions::EXC_RUNTIME_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_STOP_ITERATION_TYPE",
            interp_exceptions::EXC_STOP_ITERATION_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_IMPORT_ERROR_TYPE",
            interp_exceptions::EXC_IMPORT_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_NOT_IMPLEMENTED_ERROR_TYPE",
            interp_exceptions::EXC_NOT_IMPLEMENTED_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_ASSERTION_ERROR_TYPE",
            interp_exceptions::EXC_ASSERTION_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_REFERENCE_ERROR_TYPE",
            interp_exceptions::EXC_REFERENCE_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_GENERATOR_EXIT_TYPE",
            interp_exceptions::EXC_GENERATOR_EXIT_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_RECURSION_ERROR_TYPE",
            interp_exceptions::EXC_RECURSION_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_OS_ERROR_TYPE",
            interp_exceptions::EXC_OS_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_FILE_NOT_FOUND_ERROR_TYPE",
            interp_exceptions::EXC_FILE_NOT_FOUND_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_UNICODE_DECODE_ERROR_TYPE",
            interp_exceptions::EXC_UNICODE_DECODE_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_UNICODE_ENCODE_ERROR_TYPE",
            interp_exceptions::EXC_UNICODE_ENCODE_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_UNICODE_TRANSLATE_ERROR_TYPE",
            interp_exceptions::EXC_UNICODE_TRANSLATE_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_SYSTEM_EXIT_TYPE",
            interp_exceptions::EXC_SYSTEM_EXIT_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_MEMORY_ERROR_TYPE",
            interp_exceptions::EXC_MEMORY_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_SYSTEM_ERROR_TYPE",
            interp_exceptions::EXC_SYSTEM_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_BUFFER_ERROR_TYPE",
            interp_exceptions::EXC_BUFFER_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_LOOKUP_ERROR_TYPE",
            interp_exceptions::EXC_LOOKUP_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_UNICODE_ERROR_TYPE",
            interp_exceptions::EXC_UNICODE_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_MODULE_NOT_FOUND_ERROR_TYPE",
            interp_exceptions::EXC_MODULE_NOT_FOUND_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_SYNTAX_ERROR_TYPE",
            interp_exceptions::EXC_SYNTAX_ERROR_TYPE
        ),
        pytype_addr!(
            "interp_exceptions::EXC_STOP_ASYNC_ITERATION_TYPE",
            interp_exceptions::EXC_STOP_ASYNC_ITERATION_TYPE
        ),
        pytype_addr!("generator::GENERATOR_TYPE", generator::GENERATOR_TYPE),
        pytype_addr!("generator::COROUTINE_TYPE", generator::COROUTINE_TYPE),
        pytype_addr!(
            "generator::ASYNC_GENERATOR_TYPE",
            generator::ASYNC_GENERATOR_TYPE
        ),
        pytype_addr!(
            "generator::COROUTINE_WRAPPER_TYPE",
            generator::COROUTINE_WRAPPER_TYPE
        ),
        pytype_addr!(
            "generator::ASYNC_GEN_VALUE_WRAPPER_TYPE",
            generator::ASYNC_GEN_VALUE_WRAPPER_TYPE
        ),
        pytype_addr!(
            "generator::ASYNC_GEN_ASEND_TYPE",
            generator::ASYNC_GEN_ASEND_TYPE
        ),
        pytype_addr!(
            "generator::ASYNC_GEN_ATHROW_TYPE",
            generator::ASYNC_GEN_ATHROW_TYPE
        ),
        pytype_addr!("pyobject::INT_TYPE", pyobject::INT_TYPE),
        pytype_addr!("pyobject::BOOL_TYPE", pyobject::BOOL_TYPE),
        pytype_addr!("pyobject::FLOAT_TYPE", pyobject::FLOAT_TYPE),
        pytype_addr!("pyobject::COMPLEX_TYPE", pyobject::COMPLEX_TYPE),
        pytype_addr!("pyobject::STR_TYPE", pyobject::STR_TYPE),
        pytype_addr!("pyobject::LIST_TYPE", pyobject::LIST_TYPE),
        pytype_addr!("pyobject::TUPLE_TYPE", pyobject::TUPLE_TYPE),
        pytype_addr!("pyobject::DICT_TYPE", pyobject::DICT_TYPE),
        pytype_addr!("pyobject::LONG_TYPE", pyobject::LONG_TYPE),
        pytype_addr!("pyobject::NONE_TYPE", pyobject::NONE_TYPE),
        pytype_addr!(
            "pyobject::NOTIMPLEMENTED_TYPE",
            pyobject::NOTIMPLEMENTED_TYPE
        ),
        pytype_addr!("pyobject::ELLIPSIS_TYPE", pyobject::ELLIPSIS_TYPE),
        pytype_addr!("pyobject::MODULE_TYPE", pyobject::MODULE_TYPE),
        pytype_addr!("pyobject::MAPPING_PROXY_TYPE", pyobject::MAPPING_PROXY_TYPE),
        pytype_addr!("pyobject::TYPE_TYPE", pyobject::TYPE_TYPE),
        pytype_addr!("pyobject::INSTANCE_TYPE", pyobject::INSTANCE_TYPE),
        pytype_addr!("setobject::SET_TYPE", setobject::SET_TYPE),
        pytype_addr!("setobject::FROZENSET_TYPE", setobject::FROZENSET_TYPE),
        pytype_addr!(
            "specialisedtupleobject::SPECIALISED_TUPLE_II_TYPE",
            specialisedtupleobject::SPECIALISED_TUPLE_II_TYPE
        ),
        pytype_addr!(
            "specialisedtupleobject::SPECIALISED_TUPLE_FF_TYPE",
            specialisedtupleobject::SPECIALISED_TUPLE_FF_TYPE
        ),
        pytype_addr!(
            "specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE",
            specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE
        ),
        pytype_addr!("weakref::GC_WEAKREF_BOX_TYPE", weakref::GC_WEAKREF_BOX_TYPE),
        pytype_addr!("nestedscope::CELL_TYPE", nestedscope::CELL_TYPE),
        pytype_addr!("sliceobject::SLICE_TYPE", sliceobject::SLICE_TYPE),
        pytype_addr!("functional::RANGE_TYPE", functional::RANGE_TYPE),
        pytype_addr!("functional::RANGE_ITER_TYPE", functional::RANGE_ITER_TYPE),
        pytype_addr!("memoryview::MEMORYVIEW_TYPE", memoryview::MEMORYVIEW_TYPE),
        pytype_addr!("iterobject::SEQ_ITER_TYPE", iterobject::SEQ_ITER_TYPE),
        pytype_addr!(
            "iterobject::STR_ASCII_ITER_TYPE",
            iterobject::STR_ASCII_ITER_TYPE
        ),
        pytype_addr!("iterobject::STR_ITER_TYPE", iterobject::STR_ITER_TYPE),
        pytype_addr!("iterobject::BYTES_ITER_TYPE", iterobject::BYTES_ITER_TYPE),
        pytype_addr!(
            "iterobject::BYTEARRAY_ITER_TYPE",
            iterobject::BYTEARRAY_ITER_TYPE
        ),
        pytype_addr!("iterobject::MEMORY_ITER_TYPE", iterobject::MEMORY_ITER_TYPE),
        pytype_addr!("iterobject::ARRAY_ITER_TYPE", iterobject::ARRAY_ITER_TYPE),
        pytype_addr!("iterobject::LIST_ITER_TYPE", iterobject::LIST_ITER_TYPE),
        pytype_addr!(
            "iterobject::LIST_REVERSE_ITER_TYPE",
            iterobject::LIST_REVERSE_ITER_TYPE
        ),
        pytype_addr!("iterobject::TUPLE_ITER_TYPE", iterobject::TUPLE_ITER_TYPE),
        pytype_addr!("setobject::SET_ITERATOR_TYPE", setobject::SET_ITERATOR_TYPE),
        pytype_addr!("function::METHOD_TYPE", function::METHOD_TYPE),
        pytype_addr!("typedef::MEMBER_TYPE", MEMBER_TYPE),
        pytype_addr!("descriptor::PROPERTY_TYPE", descriptor::PROPERTY_TYPE),
        pytype_addr!("function::STATICMETHOD_TYPE", function::STATICMETHOD_TYPE),
        pytype_addr!("function::CLASSMETHOD_TYPE", function::CLASSMETHOD_TYPE),
        pytype_addr!("typedef::GETSET_DESCRIPTOR_TYPE", GETSET_DESCRIPTOR_TYPE),
        pytype_addr!("functional::ENUMERATE_TYPE", functional::ENUMERATE_TYPE),
        pytype_addr!("functional::REVERSED_TYPE", functional::REVERSED_TYPE),
        pytype_addr!("functional::FILTER_TYPE", functional::FILTER_TYPE),
        pytype_addr!("functional::MAP_TYPE", functional::MAP_TYPE),
        pytype_addr!("functional::ZIP_TYPE", functional::ZIP_TYPE),
        pytype_addr!(
            "operation::CALLABLE_ITERATOR_TYPE",
            operation::CALLABLE_ITERATOR_TYPE
        ),
        pytype_addr!("interp_itertools::COUNT_TYPE", interp_itertools::COUNT_TYPE),
        pytype_addr!(
            "interp_itertools::REPEAT_TYPE",
            interp_itertools::REPEAT_TYPE
        ),
        pytype_addr!(
            "interp_itertools::TAKEWHILE_TYPE",
            interp_itertools::TAKEWHILE_TYPE
        ),
        pytype_addr!(
            "interp_itertools::DROPWHILE_TYPE",
            interp_itertools::DROPWHILE_TYPE
        ),
        pytype_addr!(
            "interp_itertools::FILTERFALSE_TYPE",
            interp_itertools::FILTERFALSE_TYPE
        ),
        pytype_addr!(
            "interp_itertools::COMPRESS_TYPE",
            interp_itertools::COMPRESS_TYPE
        ),
        pytype_addr!(
            "interp_itertools::STARMAP_TYPE",
            interp_itertools::STARMAP_TYPE
        ),
        pytype_addr!(
            "interp_itertools::ACCUMULATE_TYPE",
            interp_itertools::ACCUMULATE_TYPE
        ),
        pytype_addr!(
            "interp_itertools::ZIP_LONGEST_TYPE",
            interp_itertools::ZIP_LONGEST_TYPE
        ),
        pytype_addr!(
            "interp_itertools::PAIRWISE_TYPE",
            interp_itertools::PAIRWISE_TYPE
        ),
        pytype_addr!("interp_itertools::CYCLE_TYPE", interp_itertools::CYCLE_TYPE),
        pytype_addr!("interp_itertools::CHAIN_TYPE", interp_itertools::CHAIN_TYPE),
        pytype_addr!("interp_sre::SRE_SCANNER_TYPE", interp_sre::SRE_SCANNER_TYPE),
        pytype_addr!(
            "functional::LONG_RANGE_ITER_TYPE",
            functional::LONG_RANGE_ITER_TYPE
        ),
        pytype_addr!("interp_sre::SRE_MATCH_TYPE", interp_sre::SRE_MATCH_TYPE),
        pytype_addr!("interp_sre::SRE_PATTERN_TYPE", interp_sre::SRE_PATTERN_TYPE),
        pytype_addr!(
            "_pypy_generic_alias::GENERIC_ALIAS_TYPE",
            _pypy_generic_alias::GENERIC_ALIAS_TYPE
        ),
        pytype_addr!("descriptor::SUPER_TYPE", descriptor::SUPER_TYPE),
        pytype_addr!(
            "_pypy_generic_alias::UNION_TYPE",
            _pypy_generic_alias::UNION_TYPE
        ),
        // `pyre_interpreter`-local `PyType` singletons.  The `pytype_addr!`
        // macro emits `&pyre_object::$path` and cannot reach these
        // crate-local statics, so capture their addresses directly.  The
        // keys match the front-end `["pyre_interpreter", module, NAME]`
        // global-read segments via the `static_key_matches` `::`-suffix
        // rule, so the `module::NAME` form suffices.  All five are
        // compile-time `static … : PyType = new_pytype(…)` so the captured
        // address is the stable runtime identity.
        (
            "function::FUNCTION_TYPE",
            &crate::function::FUNCTION_TYPE as *const _ as i64,
        ),
        (
            "function::BUILTIN_FUNCTION_TYPE",
            &crate::function::BUILTIN_FUNCTION_TYPE as *const _ as i64,
        ),
        (
            "function::METHOD_DESCRIPTOR_TYPE",
            &crate::function::METHOD_DESCRIPTOR_TYPE as *const _ as i64,
        ),
        (
            "function::SLOT_WRAPPER_TYPE",
            &crate::function::SLOT_WRAPPER_TYPE as *const _ as i64,
        ),
        (
            "function::METHOD_WRAPPER_TYPE",
            &crate::function::METHOD_WRAPPER_TYPE as *const _ as i64,
        ),
        (
            "function::METHOD_DESCRIPTOR_TYPE",
            &crate::function::METHOD_DESCRIPTOR_TYPE as *const _ as i64,
        ),
        (
            "gateway::BUILTIN_CODE_TYPE",
            &crate::gateway::BUILTIN_CODE_TYPE as *const _ as i64,
        ),
        (
            "pycode::CODE_TYPE",
            &crate::pycode::CODE_TYPE as *const _ as i64,
        ),
        (
            "pytraceback::PYTRACEBACK_TYPE",
            &crate::pytraceback::PYTRACEBACK_TYPE as *const _ as i64,
        ),
        (
            "interp_buffer::PICKLEBUFFER_TYPE",
            &crate::module::__pypy__::interp_buffer::PICKLEBUFFER_TYPE as *const _ as i64,
        ),
        (
            "function::METHOD_DESCRIPTOR_TYPE",
            &crate::function::METHOD_DESCRIPTOR_TYPE as *const _ as i64,
        ),
    ];
    // Fold in the `#[pyre_class]` registry.  A row above names one static
    // by hand, so a `#[pyre_class]` that lands without someone adding its
    // row leaves that type's address unbound and every graph reaching it
    // walled off at translation — a drift the rows cannot prevent because
    // the macro derives the static's identifier from the struct name and
    // no source search finds it.  Deduplicated on the address, not the
    // key: the two spellings differ (`module::NAME` above, fully-qualified
    // below) and both satisfy the front-end's `static_key_matches`, so the
    // hand-written row wins for the types that already have one.
    let hand_written: std::collections::HashSet<i64> = rows.iter().map(|&(_, addr)| addr).collect();
    rows.extend(
        pyre_class_pytype_addrs()
            .into_iter()
            .filter(|(_, addr)| !hand_written.contains(addr)),
    );
    rows
}

/// Every `#[pyre_class]` type's `PyType` address, keyed by the RUST PATH
/// OF THE TYPE rather than of the static holding it.
///
/// `#[pyre_class]` emits one address under two names: the static
/// [`pyre_class_pytype_addrs`] keys on, and the associated const
/// `impl PyreClassPyTypeOf for T { const PYTYPE = &<static> }`. A flow
/// graph reading the second carries no static path — Charon renders that
/// read `<module>::<Impl>::PYTYPE`, one spelling for every trait impl in
/// the module — so the translator resolves it through the impl's `Self`
/// type and joins on the type's own path, which is what this table
/// supplies.
///
/// A type path is injective where the rendered one is not, and that is
/// the property this key has to have: it is also the linkage symbol
/// `runtime_fnaddr_patch::patch_static_addr_constants` re-pairs across
/// the build/run boundary, where two entries sharing a name would pair
/// the wrong address rather than merely fail to lower.
pub fn pyre_class_pytype_by_struct_addrs() -> Vec<(&'static str, i64)> {
    let mut rows = Vec::new();
    pyre_object::lltype::for_each_class_descriptor(|d| {
        rows.push((d.struct_path, d.pytype_ptr as usize as i64));
    });
    rows
}

/// The `PyType` static of every `#[pyre_class]` type, keyed by the
/// fully-qualified Rust path the flowgraph names the global read with
/// (`PyreClassDescriptor::pytype_path`).
///
/// Populated on every target.  Its membership still follows the module set,
/// which a cross-target build cannot see: the list is read once in the build
/// script, compiled for the host, and once at run time, compiled for the
/// target, so a module declared behind `target_arch`, `unix` or `windows`
/// appears in the first and not the second.  A name bound at build time and
/// missing at run time keeps the build-process address baked in the constant
/// pool, because `runtime_fnaddr_patch::patch_static_addr_constants` re-pairs
/// only names present in both; `disarm_unpaired_build_addrs` writes zero over
/// such an address, which is the "no address" value every call site already
/// declines.
pub fn pyre_class_pytype_addrs() -> Vec<(&'static str, i64)> {
    let mut rows = Vec::new();
    pyre_object::lltype::for_each_class_descriptor(|d| {
        rows.push((d.pytype_path, d.pytype_ptr as usize as i64));
    });
    rows
}

/// Build-time addresses of the prebuilt dict-strategy singletons pyre
/// source references as opaque ref constants.  Same translation-boundary
/// contract as [`jit_static_pytype_addrs`]; the front-end records these
/// under `ValueType::Ref(None)`.
pub fn jit_static_ref_addrs() -> Vec<(&'static str, i64)> {
    macro_rules! ref_addr {
        ($key:literal, $($path:tt)::+) => {
            ($key, &pyre_object::$($path)::+ as *const _ as i64)
        };
    }
    vec![
        ref_addr!(
            "dictmultiobject::OBJECT_DICT_STRATEGY",
            dictmultiobject::OBJECT_DICT_STRATEGY
        ),
        ref_addr!(
            "dictmultiobject::EMPTY_DICT_STRATEGY",
            dictmultiobject::EMPTY_DICT_STRATEGY
        ),
        ref_addr!(
            "dictmultiobject::EMPTY_KWARGS_DICT_STRATEGY",
            dictmultiobject::EMPTY_KWARGS_DICT_STRATEGY
        ),
        ref_addr!(
            "dictmultiobject::BYTES_DICT_STRATEGY",
            dictmultiobject::BYTES_DICT_STRATEGY
        ),
        ref_addr!(
            "dictmultiobject::UNICODE_DICT_STRATEGY",
            dictmultiobject::UNICODE_DICT_STRATEGY
        ),
        ref_addr!(
            "dictmultiobject::INT_DICT_STRATEGY",
            dictmultiobject::INT_DICT_STRATEGY
        ),
        // A translated `W_DictObject.dstrategy` holds the address of the
        // non-zero-sized holder, not the zero-sized strategy implementation.
        // These are the concrete prebuilt instances corresponding to PyPy's
        // `space.fromcache(StrategyClass)` results and are therefore GCREF
        // constants in the translated graph, exactly like the implementation
        // rows above but with the identity the live dict slot actually reads.
        ref_addr!(
            "dictmultiobject::OBJECT_DICT_STRATEGY_REF",
            dictmultiobject::OBJECT_DICT_STRATEGY_REF
        ),
        ref_addr!(
            "dictmultiobject::EMPTY_DICT_STRATEGY_REF",
            dictmultiobject::EMPTY_DICT_STRATEGY_REF
        ),
        ref_addr!(
            "dictmultiobject::EMPTY_KWARGS_DICT_STRATEGY_REF",
            dictmultiobject::EMPTY_KWARGS_DICT_STRATEGY_REF
        ),
        ref_addr!(
            "dictmultiobject::BYTES_DICT_STRATEGY_REF",
            dictmultiobject::BYTES_DICT_STRATEGY_REF
        ),
        ref_addr!(
            "dictmultiobject::UNICODE_DICT_STRATEGY_REF",
            dictmultiobject::UNICODE_DICT_STRATEGY_REF
        ),
        ref_addr!(
            "dictmultiobject::INT_DICT_STRATEGY_REF",
            dictmultiobject::INT_DICT_STRATEGY_REF
        ),
        ref_addr!(
            "identitydict::IDENTITY_DICT_STRATEGY",
            identitydict::IDENTITY_DICT_STRATEGY
        ),
        ref_addr!(
            "identitydict::IDENTITY_DICT_STRATEGY_REF",
            identitydict::IDENTITY_DICT_STRATEGY_REF
        ),
        ref_addr!(
            "kwargsdict::KWARGS_DICT_STRATEGY",
            kwargsdict::KWARGS_DICT_STRATEGY
        ),
        ref_addr!(
            "kwargsdict::KWARGS_DICT_STRATEGY_REF",
            kwargsdict::KWARGS_DICT_STRATEGY_REF
        ),
        (
            "objspace::std::mapdict::MAP_DICT_STRATEGY_REF",
            &crate::objspace::std::mapdict::MAP_DICT_STRATEGY_REF as *const _ as i64,
        ),
        // Prebuilt object singletons (`None` / `NotImplemented` /
        // `Ellipsis` / `True` / `False`).  The accessors `w_none`,
        // `w_ellipsis`, `w_not_implemented`, `w_bool_from` read these
        // statics as a bare same-file `LOAD_GLOBAL` and return their
        // address; supplying the captured address lets the front-end
        // `Expr::Path` same-file fold emit `ConstRefAddr` with the real
        // runtime identity instead of a cross-block body-`Input`.  The
        // statics are private (callers route through the accessors), so
        // the address is captured through the accessor rather than the
        // `ref_addr!` `&pyre_object::X` path form.
        (
            "noneobject::NONE_SINGLETON",
            pyre_object::w_none() as usize as i64,
        ),
        (
            "special::NOT_IMPLEMENTED_SINGLETON",
            pyre_object::w_not_implemented() as usize as i64,
        ),
        (
            "special::ELLIPSIS_SINGLETON",
            pyre_object::w_ellipsis() as usize as i64,
        ),
        (
            "boolobject::TRUE_SINGLETON",
            pyre_object::w_bool_from(true) as usize as i64,
        ),
        (
            "boolobject::FALSE_SINGLETON",
            pyre_object::w_bool_from(false) as usize as i64,
        ),
    ]
}

/// Build-time *values* of the immutable size constants pyre source reads
/// through the flowgraph as opaque `LOAD_GLOBAL` constants.  Unlike the
/// `refs`/`pytypes` siblings (which carry a static's *address*), these are
/// compile-time `const`s whose initializer is a `size_of::<T>()` the
/// front-end cannot evaluate (Charon leaves the target-dependent layout
/// symbolic).  The value is identical at the codewriter call site, so the
/// front-end bakes it directly as a `ConstInt` instead of minting an
/// accessor call no registry can resolve.
///
/// Resolved in the same build-script process the translator runs in, so
/// the captured size matches a direct `size_of::<T>()` at the call site
/// (the JIT is native — host target == runtime target).  Keys are the
/// crate-stripped `module::NAME` spelling `front::mir::static_int_value_op`
/// matches against the `FunctionPath` segments.
pub fn jit_static_int_values() -> Vec<(&'static str, i64)> {
    vec![
        (
            "function::FUNCTION_OBJECT_SIZE",
            crate::function::FUNCTION_OBJECT_SIZE as i64,
        ),
        (
            "dictmultiobject::W_DICT_OBJECT_SIZE",
            pyre_object::dictmultiobject::W_DICT_OBJECT_SIZE as i64,
        ),
        (
            "specialisedtupleobject::SPECIALISED_TUPLE_II_OBJECT_SIZE",
            pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_II_OBJECT_SIZE as i64,
        ),
        (
            "specialisedtupleobject::SPECIALISED_TUPLE_FF_OBJECT_SIZE",
            pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_FF_OBJECT_SIZE as i64,
        ),
        (
            "specialisedtupleobject::SPECIALISED_TUPLE_OO_OBJECT_SIZE",
            pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_OBJECT_SIZE as i64,
        ),
        (
            "objectobject::W_OBJECT_OBJECT_SIZE",
            pyre_object::objectobject::W_OBJECT_OBJECT_SIZE as i64,
        ),
        // `pub const CAN_BE_TAGGED: bool` (tagged-int enablement, currently
        // `true`); Charon emits the read as an opaque global rather than
        // folding it, so bake the build-time value (`true as i64` == 1).
        (
            "tagged_int::CAN_BE_TAGGED",
            pyre_object::tagged_int::CAN_BE_TAGGED as i64,
        ),
        // `i64::MAX` reached as `core::num::<Impl>::MAX` in `getindex_w`'s
        // overflow clamp. Charon leaves the associated const as a global
        // accessor path, so bake the native signed max value.
        ("core::num::<Impl>::MAX", i64::MAX),
        // `compares_by_identity_status` tri-state markers, read as opaque
        // global accessor paths in `mutated` / the `__eq__`/`__hash__`
        // fast paths. Bake the build-time `u8` values.
        (
            "typeobject::COMPARES_BY_IDENTITY_UNKNOWN",
            pyre_object::typeobject::COMPARES_BY_IDENTITY_UNKNOWN as i64,
        ),
        (
            "typeobject::COMPARES_BY_IDENTITY_YES",
            pyre_object::typeobject::COMPARES_BY_IDENTITY_YES as i64,
        ),
        (
            "typeobject::COMPARES_BY_IDENTITY_NO",
            pyre_object::typeobject::COMPARES_BY_IDENTITY_NO as i64,
        ),
        // Eval-breaker word bit masks read by the dispatch-loop poll in
        // `executioncontext.rs`. Cross-crate `pub const usize` reads reach
        // the front-end as opaque `Foreign` globals, so bake the build-time
        // bit values.
        (
            "eval_breaker_word::EB_ASYNC",
            majit_ir::eval_breaker_word::EB_ASYNC as i64,
        ),
        (
            "eval_breaker_word::EB_STW",
            majit_ir::eval_breaker_word::EB_STW as i64,
        ),
        (
            "eval_breaker_word::EB_FINALIZING",
            majit_ir::eval_breaker_word::EB_FINALIZING as i64,
        ),
        (
            "eval_breaker_word::EB_GC_INTERP",
            majit_ir::eval_breaker_word::EB_GC_INTERP as i64,
        ),
        (
            "eval_breaker_word::EB_GC",
            majit_ir::eval_breaker_word::EB_GC as i64,
        ),
        (
            "eval_breaker_word::EB_MEMORY_ERROR",
            majit_ir::eval_breaker_word::EB_MEMORY_ERROR as i64,
        ),
        (
            "eval_breaker_word::JIT_BREAKER_MASK",
            majit_ir::eval_breaker_word::JIT_BREAKER_MASK as i64,
        ),
        // Tick-counter decrement step read on the same poll path.
        (
            "executioncontext::TICK_COUNTER_STEP",
            crate::executioncontext::TICK_COUNTER_STEP as i64,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        is_abi_unsound_argument_residual, is_list_write_barrier, is_pyframe_operand_stack_accessor,
        is_rerunnable_bookkeeping_residual, jit_static_pytype_addrs, jit_static_ref_addrs,
        jit_trace_fnaddrs, pyre_class_pytype_addrs, pyre_class_pytype_by_struct_addrs,
        shadow_stack_get_word, shadow_stack_push_word, shadow_stack_try_pop_to_word,
    };
    use std::collections::HashMap;

    /// The exemption is keyed on the registered path, so a rename or a typo in
    /// one of the three patterns silently drops a helper out of the set and the
    /// walk that met only it stops taking the no-replay roads. Pin each
    /// pattern, and pin a sibling in the same module that must NOT be exempt:
    /// `pyre_stack_too_big_slowpath` shares `::stack_check::` with the one
    /// match, so a pattern loosened to the module would take it too.
    #[test]
    fn is_rerunnable_bookkeeping_residual_matches_the_registered_helpers() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();
        for path in [
            "pyre_interpreter::stack_check::stack_check",
            "pyre_object::pyobject::ensure_object_subclass_ranges_initialized",
            "pyre_object::ensure_object_subclass_ranges_initialized",
        ] {
            assert!(
                is_rerunnable_bookkeeping_residual(bindings[path] as usize),
                "{path} is registered but not exempt"
            );
        }
        assert!(!is_rerunnable_bookkeeping_residual(
            bindings["pyre_interpreter::stack_check::pyre_stack_too_big_slowpath"] as usize
        ));
        assert!(!is_rerunnable_bookkeeping_residual(0));
    }

    /// The set is filled by the publication sites, so the way to lose a helper
    /// out of it is not a typo in a name but a site moved back to the
    /// result-half publisher: the address then reads as sound and a sub-walk
    /// executes a helper whose second argument word nothing wrote. Pin one
    /// entry from each publisher form, and pin a checked publisher's address
    /// in the same module that must NOT be in the set.
    #[test]
    fn is_abi_unsound_argument_residual_matches_the_published_helpers() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();
        for path in [
            "pyre_object::dictmultiobject::wtf8_key_is_utf8",
            "pyre_object::wtf8_surrogate_key_str_object",
            "pyre_interpreter::host_seam::emit_stdout",
            "emit_stdout",
            "pyre_interpreter::call::set_call_error",
        ] {
            assert!(
                is_abi_unsound_argument_residual(bindings[path] as usize),
                "{path} is published with an argument wider than a residual slot"
            );
        }
        assert!(!is_abi_unsound_argument_residual(
            bindings["pyre_object::dictmultiobject::w_dict_len"] as usize
        ));
        assert!(!is_abi_unsound_argument_residual(0));
    }

    #[test]
    fn jit_trace_fnaddrs_contains_root_and_module_aliases() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();

        let make_fn =
            crate::runtime_ops::jit_make_function_from_globals as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::runtime_ops::jit_make_function_from_globals"],
            make_fn
        );
        assert_eq!(
            bindings["pyre_interpreter::jit_make_function_from_globals"],
            make_fn
        );

        let list_append = pyre_object::jit_list_append as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_object::listobject::jit_list_append"],
            list_append
        );
        assert_eq!(bindings["pyre_object::jit_list_append"], list_append);
    }

    #[test]
    fn jit_trace_fnaddrs_covers_frame_anchor_shadow_stack_externals() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();
        for (path, expected) in [
            (
                "majit_gc::shadow_stack::push",
                shadow_stack_push_word as *const () as usize as i64,
            ),
            (
                "majit_gc::shadow_stack::get",
                shadow_stack_get_word as *const () as usize as i64,
            ),
            (
                "majit_gc::shadow_stack::try_pop_to",
                shadow_stack_try_pop_to_word as *const () as usize as i64,
            ),
        ] {
            assert_eq!(bindings.get(path), Some(&expected), "missing {path}");
        }
    }

    /// The three published addresses answer through the word ABI a residual
    /// call reaches them with, and the round trip preserves the reference.
    ///
    /// Calling each through its registry address, transmuted to the signature
    /// the lowering emits, is what an in-module `call_indirect` does; a raw
    /// `(usize) -> usize` published here would be a different wasm32 table
    /// type and trap there while still passing an address comparison.
    #[test]
    fn the_shadow_stack_externals_answer_through_the_word_abi() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();
        let addr = |path: &str| *bindings.get(path).expect("registered") as usize;
        let push: extern "C" fn(i64) -> i64 =
            unsafe { std::mem::transmute(addr("majit_gc::shadow_stack::push")) };
        let get: extern "C" fn(i64) -> i64 =
            unsafe { std::mem::transmute(addr("majit_gc::shadow_stack::get")) };
        let try_pop_to: extern "C" fn(i64) =
            unsafe { std::mem::transmute(addr("majit_gc::shadow_stack::try_pop_to")) };

        // A word that is not a live object: these three only move it between
        // the stack and the caller.
        let marker = 0x2468_i64;
        let depth = push(marker);
        assert_eq!(get(depth), marker);
        try_pop_to(depth);
        assert_eq!(push(marker), depth, "try_pop_to left the depth unrestored");
        try_pop_to(depth);
    }

    #[test]
    fn jit_trace_fnaddrs_covers_codewriter_unary_graphs_with_word_abi_bridges() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();
        for (path, expected) in [
            (
                "pyre_interpreter::objspace::descroperation::neg",
                crate::opcode_ops::jit_descroperation_neg as *const () as usize as i64,
            ),
            (
                "pyre_interpreter::objspace::descroperation::invert",
                crate::opcode_ops::jit_descroperation_invert as *const () as usize as i64,
            ),
            (
                "pyre_interpreter::objspace::descroperation::pos",
                crate::opcode_ops::jit_descroperation_pos as *const () as usize as i64,
            ),
        ] {
            assert_eq!(bindings.get(path), Some(&expected), "missing {path}");
        }
    }

    /// Two registered functions must never share an address.
    ///
    /// `pyre-jit-trace`'s `patch_constants_i_fnaddrs` rewrites residual-call
    /// constants through a build-address → runtime-address map, so an address
    /// standing for two functions sends one callee's call to the other.  Its
    /// runtime assertion only fires once the patch path executes; this covers
    /// the registry itself.
    ///
    /// Several path spellings for one function are deliberate — the module
    /// path and the crate-root re-export both appear — and those agree on the
    /// leaf name.  Two distinct leaf names on one address means the toolchain
    /// folded unrelated functions together, which is the collision that
    /// matters.  That is not hypothetical: MSVC links with `/OPT:ICF` by
    /// default, and once `drain_list_append` lost `#[inline(never)]` its body
    /// became byte-identical to `w_list_append` and the two folded.
    ///
    /// This covers only the address space it runs in.  The Windows fold
    /// happened in the build-script binary while the test binary kept the two
    /// apart, so a registry that passes here can still feed
    /// `runtime_fnaddr_patch` an ambiguous build address — that direction is
    /// what its own assertion catches.
    /// Whether two registered paths are two spellings of one item, which is
    /// the only legitimate reason for them to share an address.
    fn are_alias_spellings(a: &str, b: &str) -> bool {
        /// One path is the other with more leading segments, on a `::`
        /// boundary — the shape a registry entry recorded with its crate
        /// segment has against the same entry recorded without one.
        fn extends(a: &str, b: &str) -> bool {
            let (short, long) = if a.len() > b.len() { (b, a) } else { (a, b) };
            long == short
                || long
                    .strip_suffix(short)
                    .is_some_and(|prefix| prefix.ends_with("::"))
        }
        /// One path is the other with a single interior segment removed —
        /// the shape an inner module's own path has against the re-export
        /// from the module that publishes it.  A one-segment deletion cannot
        /// relate two paths of equal length, so a substitution like
        /// `module::a::f` against `module::b::f` stays two functions.
        fn drops_one_segment(a: &str, b: &str) -> bool {
            // A path with a segment removed is strictly shorter in bytes, so
            // byte length orders the pair the same way segment count does.
            let (short, long) = if a.len() > b.len() { (b, a) } else { (a, b) };
            let short: Vec<&str> = short.split("::").collect();
            let long: Vec<&str> = long.split("::").collect();
            if long.len() != short.len() + 1 {
                return false;
            }
            (0..long.len()).any(|i| {
                let mut without = long.clone();
                without.remove(i);
                without == short
            })
        }
        fn split_head(path: &str) -> Option<(&str, &str)> {
            path.split_once("::")
        }
        // A crate-root re-export (`pyre_interpreter::acquire_buffered_lock`)
        // beside its defining path (`pyre_interpreter::module::_io::
        // acquire_buffered_lock`) is related by neither suffix while the crate
        // segment leads both, so drop that segment — but only when the two
        // paths lead with the same one. No crate re-exports another crate's
        // item, so `pyre_object::module::x::f` and
        // `pyre_interpreter::module::x::f` are two functions whose modules are
        // spelled alike, and comparing their tails would call them one.
        //
        // Comparing only the last segment would accept far more than either
        // rule: `module::a::type_object` and `module::b::type_object` would
        // read as aliases while address-keyed patching between them stays
        // ambiguous. Those are related by no suffix here and are reported.
        if extends(a, b) || drops_one_segment(a, b) {
            return true;
        }
        match (split_head(a), split_head(b)) {
            (Some((head_a, rest_a)), Some((head_b, rest_b))) => {
                head_a == head_b && extends(rest_a, rest_b)
            }
            _ => false,
        }
    }

    #[test]
    fn two_distinct_items_sharing_an_address_are_not_alias_spellings() {
        // The pair the leaf-name grouping used to accept.
        assert!(!are_alias_spellings(
            "pyre_interpreter::module::a::type_object",
            "pyre_interpreter::module::b::type_object",
        ));
        // Both shapes the registry actually produces.
        assert!(are_alias_spellings(
            "pyre_interpreter::acquire_buffered_lock",
            "pyre_interpreter::module::_io::acquire_buffered_lock",
        ));
        assert!(are_alias_spellings(
            "module::_io::stringio::type_object",
            "pyre_interpreter::module::_io::stringio::type_object",
        ));
        // An inner module's own path beside the re-export the enclosing
        // module publishes: `jit_libffi` defines its cfg-selected bodies in
        // `imp` and re-exports them, and the LLBC carries both spellings.
        assert!(are_alias_spellings(
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::exchange_size",
            "pyre_interpreter::module::_cffi_backend::jit_libffi::exchange_size",
        ));
        // Two crates cannot re-export one another's item, so identical module
        // paths under different crates are two functions, not two spellings.
        assert!(!are_alias_spellings(
            "pyre_object::module::x::type_object",
            "pyre_interpreter::module::x::type_object",
        ));
    }

    #[test]
    fn registered_paths_sharing_an_address_are_alias_spellings() {
        let mut by_addr: HashMap<i64, Vec<&'static str>> = HashMap::new();
        for (path, addr) in jit_trace_fnaddrs() {
            by_addr.entry(addr).or_default().push(path);
        }
        // Collect every colliding address before failing. Asserting inside
        // the loop reports whichever collision the hash order reached first
        // and hides the rest, so each repair looks complete and the next run
        // names a different pair.
        let mut collisions: Vec<String> = Vec::new();
        for (addr, paths) in &by_addr {
            let mut unrelated: Vec<(&str, &str)> = Vec::new();
            for (i, a) in paths.iter().enumerate() {
                for b in &paths[i + 1..] {
                    if !are_alias_spellings(a, b) {
                        unrelated.push((a, b));
                    }
                }
            }
            if !unrelated.is_empty() {
                unrelated.sort_unstable();
                collisions.push(format!("{addr:#x} {unrelated:?}"));
            }
        }
        collisions.sort();
        assert!(
            collisions.is_empty(),
            "{} fnaddr(s) claimed by unrelated functions:\n  {}",
            collisions.len(),
            collisions.join("\n  "),
        );
    }

    #[test]
    fn jit_static_pytype_addrs_covers_interpreter_function_types() {
        let bindings: HashMap<&'static str, i64> = jit_static_pytype_addrs().into_iter().collect();

        assert_eq!(
            bindings["function::METHOD_DESCRIPTOR_TYPE"],
            &crate::function::METHOD_DESCRIPTOR_TYPE as *const _ as i64
        );
        assert_eq!(
            bindings["function::METHOD_WRAPPER_TYPE"],
            &crate::function::METHOD_WRAPPER_TYPE as *const _ as i64
        );
    }

    /// The struct-keyed table names each class exactly once, and names
    /// the same address its static-keyed sibling does.
    ///
    /// Injectivity is the whole point of this key rather than a nicety.
    /// It is what the rendered `<module>::<Impl>::PYTYPE` spelling lacks —
    /// every trait impl in a module flattens onto it — and it is also what
    /// `patch_static_addr_constants` needs to re-pair the right address
    /// across the build/run boundary, where a shared key pairs the wrong
    /// one instead of merely failing to lower. `rpython`'s own object ->
    /// name layer holds itself to this: `translator/gensupp.py`'s
    /// `NameManager.uniquename` numbers a colliding basename rather than
    /// letting two objects share it.
    #[test]
    fn the_struct_keyed_pytype_table_names_each_class_exactly_once() {
        let rows = pyre_class_pytype_by_struct_addrs();
        let by_path: HashMap<&'static str, i64> = pyre_class_pytype_addrs().into_iter().collect();

        assert!(
            !rows.is_empty(),
            "no `#[pyre_class]` descriptor was registered at all, so this \
             table cannot be read as empty-because-correct"
        );

        let mut seen: HashMap<&'static str, i64> = HashMap::new();
        for (struct_path, addr) in &rows {
            if let Some(first) = seen.insert(struct_path, *addr) {
                panic!(
                    "{struct_path} appears twice (addresses {first:#x} and \
                     {addr:#x}); the key must name one type, or the \
                     build/run re-pairing binds whichever row it meets last"
                );
            }
            assert_ne!(*addr, 0, "{struct_path} has no address");
        }
        assert_eq!(
            seen.len(),
            by_path.len(),
            "the struct-keyed and static-keyed tables describe the same \
             classes, so they must have the same length"
        );

        // Every address here is one the static-keyed table also carries:
        // the two are two keys on one set of singletons, not two sets.
        let addrs_by_static: std::collections::HashSet<i64> = by_path.values().copied().collect();
        for (struct_path, addr) in &rows {
            assert!(
                addrs_by_static.contains(addr),
                "{struct_path} binds {addr:#x}, which no `pytype_path` row \
                 names; the two tables have drifted apart"
            );
        }
    }

    /// A struct path is not its `PyType` static's path, and the pair is
    /// what lets a reader join on either.
    ///
    /// Pinned because the macro derives both from `module_path!()` and a
    /// `stringify!`, so a refactor that made them coincide would leave the
    /// join silently reading the wrong column.
    #[test]
    fn the_struct_path_and_the_pytype_path_name_different_items() {
        let mut checked = 0usize;
        pyre_object::lltype::for_each_class_descriptor(|d| {
            assert_ne!(
                d.struct_path, d.pytype_path,
                "{} names the type and the static identically",
                d.pyname
            );
            let (struct_mod, _) = d.struct_path.rsplit_once("::").expect("a qualified path");
            let (pytype_mod, _) = d.pytype_path.rsplit_once("::").expect("a qualified path");
            assert_eq!(
                struct_mod, pytype_mod,
                "{}'s type and static disagree about their module; the \
                 translator resolves the static through the type, so they \
                 have to be co-located",
                d.pyname
            );
            checked += 1;
        });
        assert!(checked > 0, "no descriptor was visited");
    }

    #[test]
    fn jit_static_ref_addrs_covers_live_dict_strategy_holders() {
        let bindings: HashMap<&'static str, i64> = jit_static_ref_addrs().into_iter().collect();

        assert_eq!(
            bindings["dictmultiobject::OBJECT_DICT_STRATEGY_REF"],
            &pyre_object::dictmultiobject::OBJECT_DICT_STRATEGY_REF as *const _ as i64
        );
        assert_eq!(
            bindings["identitydict::IDENTITY_DICT_STRATEGY_REF"],
            &pyre_object::identitydict::IDENTITY_DICT_STRATEGY_REF as *const _ as i64
        );
        assert_eq!(
            bindings["kwargsdict::KWARGS_DICT_STRATEGY_REF"],
            &pyre_object::kwargsdict::KWARGS_DICT_STRATEGY_REF as *const _ as i64
        );
        assert_eq!(
            bindings["objspace::std::mapdict::MAP_DICT_STRATEGY_REF"],
            &crate::objspace::std::mapdict::MAP_DICT_STRATEGY_REF as *const _ as i64
        );
    }

    #[test]
    fn jit_trace_fnaddrs_covers_random_genrand32_residual() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();
        let genrand32: fn(&mut crate::module::_random::Random) -> u32 =
            crate::module::_random::Random::genrand32;
        let expected = genrand32 as *const () as usize as i64;

        assert_eq!(
            bindings["pyre_interpreter::module::_random::Random::genrand32"],
            expected
        );
        assert_eq!(bindings["module::_random::Random::genrand32"], expected);
    }

    /// Every `#[pyre_methods]` `type_object()` accessor publishes its residual
    /// address.  Both the crate-qualified path (the residual `FunctionPath`)
    /// and the crate-stripped alias resolve to the accessor.
    #[test]
    fn jit_trace_fnaddrs_covers_deque_iter_type_object_residual() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();
        let expected =
            crate::module::_collections::deque_iter::type_object as *const () as usize as i64;

        assert_eq!(
            bindings["pyre_interpreter::module::_collections::deque_iter::type_object"],
            expected
        );
        assert_eq!(
            bindings["module::_collections::deque_iter::type_object"],
            expected
        );
    }

    /// The `_csv::dialect_class::type_object` accessor is hand-written (not
    /// `#[pyre_methods]` / `py_class!`), yet the front recognizer stamps every
    /// `type_object` accessor `dont_look_inside`.  It must still publish a
    /// residual address, or a traced `_csv.Dialect` type lookup residualizes to
    /// a symbolic fnaddr and inline JIT descent aborts.  Guards the invariant
    /// that every `type_object` generator (macro or hand-written) registers.
    #[test]
    fn jit_trace_fnaddrs_covers_hand_written_csv_dialect_type_object() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();
        assert!(
            bindings.contains_key("pyre_interpreter::module::_csv::dialect_class::type_object"),
            "hand-written _csv::dialect_class::type_object must publish a residual fnaddr",
        );
        assert!(
            bindings.contains_key("module::_csv::dialect_class::type_object"),
            "the crate-stripped alias must resolve too",
        );
    }

    /// Every `BUILTIN_WRAPPER_DESCRIPTORS` member must carry the
    /// `__majit_wrap_` leaf.  That prefix is how
    /// `CallControl::compute_builtin_wrapper_indirect_graphs` populates the
    /// `BuiltinCode.func` PBC family, and the family is what seeds the
    /// codewriter BFS and materialises a jitcode per member.  A descriptor
    /// spelled any other way still publishes an address and still binds at
    /// runtime, so nothing fails — it simply joins no family, gets no
    /// jitcode, and leaves every indirect site that would dispatch to it
    /// residual.
    ///
    /// `descr_typecheck_fget_getdictscope` was spelled that way, and the
    /// `f_locals` getset was what it cost: the walker reached the gateway as
    /// a residual `CallMayForce` and escaped, so the
    /// `jit_force_virtualizable` the gateway carries was never deleted from a
    /// looked-inside copy.
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn every_builtin_wrapper_descriptor_carries_the_family_prefix() {
        let stray: Vec<&str> = crate::gateway::BUILTIN_WRAPPER_DESCRIPTORS
            .iter()
            .map(|wrapper| wrapper.path)
            .filter(|path| {
                !path
                    .rsplit("::")
                    .next()
                    .is_some_and(|leaf| leaf.starts_with("__majit_wrap_"))
            })
            .collect();
        assert!(
            stray.is_empty(),
            "these wrapper descriptors cannot join the BuiltinCode.func PBC \
             family, so they get no jitcode: {stray:?}",
        );
    }

    /// `BUILTIN_WRAPPER_DESCRIPTORS` is only pushed into the binding table off
    /// wasm32, so the lookup below has nothing to find there.
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn jit_trace_fnaddrs_covers_int_bit_length_gateway_wrapper() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();
        let expected =
            crate::typedef::__majit_wrap_int_descr_bit_length as *const () as usize as i64;

        assert_eq!(
            bindings["pyre_interpreter::typedef::__majit_wrap_int_descr_bit_length"],
            expected,
        );
    }

    #[test]
    fn jit_trace_fnaddrs_covers_generated_runtime_helper_families() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();

        let callable3 =
            crate::runtime_ops::callable_call_helper(3).expect("callable helper") as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::runtime_ops::jit_call_callable_3"],
            callable3
        );
        assert_eq!(bindings["pyre_interpreter::jit_call_callable_3"], callable3);

        let tuple2 =
            crate::runtime_ops::tuple_build_helper(2).expect("tuple build helper") as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::runtime_ops::jit_build_tuple_2"],
            tuple2
        );
        assert_eq!(bindings["pyre_interpreter::jit_build_tuple_2"], tuple2);
    }

    /// `front::rbigint_call::lshift_count_residual_path` retargets both the
    /// long-count and Signed×Signed-overflow forms.  Keep both exact paths
    /// resolvable: otherwise the latter silently falls back to a symbolic
    /// fnaddr when an overflowing machine-int left shift promotes to rbigint.
    #[test]
    fn jit_trace_fnaddrs_covers_both_rbigint_lshift_residuals() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();

        let count =
            crate::objspace::descroperation::jit_bigint_lshift_count as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::objspace::descroperation::jit_bigint_lshift_count"],
            count
        );

        let int_int = crate::objspace::descroperation::jit_bigint_lshift_int_int_result as *const ()
            as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::objspace::descroperation::jit_bigint_lshift_int_int_result"],
            int_int
        );
    }

    #[test]
    fn jit_trace_fnaddrs_covers_store_subscr_helpers() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();

        let execute_store_subscr =
            crate::opcode_ops::bh_execute_store_subscr as *const () as usize as i64;
        assert_eq!(bindings["execute_store_subscr"], execute_store_subscr);

        let store_subscr_fn = crate::opcode_ops::bh_store_subscr_fn as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::opcode_ops::bh_store_subscr_fn"],
            store_subscr_fn
        );
        assert_eq!(
            bindings["pyre_interpreter::bh_store_subscr_fn"],
            store_subscr_fn
        );
    }

    /// These path spellings are what keep the `pop_value` / paired-local
    /// / exception-TLS residual calls off the `symbolic_fnaddr_for_path`
    /// fallback (which SEGVs at trace time); a typo in either the
    /// module-qualified or root alias would silently regress to a
    /// symbolic hash, so pin both spellings against the live fnaddr.
    /// The lowered raise path spells both of these as string literals in
    /// another crate (`front::result_exc`, which takes the fused leaf from its
    /// `FUSED_KIND_CTORS` table), and nothing links the two spellings at build
    /// time: a typo on either side degrades the residual call to a
    /// `symbolic_fnaddr_for_path` hash instead of failing to compile.  Pinning
    /// the registration against the live trampoline catches a drift on this
    /// side; a drift in the consumer's literal still shows up only as a
    /// declined descent.
    #[test]
    fn jit_trace_fnaddrs_covers_raise_path_exception_materialisation() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();

        let materialise: extern "C" fn(i64) -> i64 =
            crate::error::__majit_call_target_pyerror_to_exc_object;
        let materialise = materialise as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::error::pyerror_to_exc_object"],
            materialise
        );
        assert_eq!(
            bindings["pyre_interpreter::pyerror_to_exc_object"],
            materialise
        );

        let fused: extern "C" fn(i64) -> i64 =
            crate::error::__majit_call_target_pyerror_type_error_to_exc_object;
        let fused = fused as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::error::pyerror_type_error_to_exc_object"],
            fused
        );
        assert_eq!(
            bindings["pyre_interpreter::pyerror_type_error_to_exc_object"],
            fused
        );

        let zero_division: extern "C" fn(i64) -> i64 =
            crate::error::__majit_call_target_pyerror_zero_division_to_exc_object;
        let zero_division = zero_division as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::error::pyerror_zero_division_to_exc_object"],
            zero_division
        );
        assert_eq!(
            bindings["pyre_interpreter::pyerror_zero_division_to_exc_object"],
            zero_division
        );
    }

    #[test]
    fn jit_trace_fnaddrs_covers_pop_value_and_exception_tls_helpers() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();

        let nlocals: fn(&crate::pyframe::PyFrame) -> usize = crate::pyframe::PyFrame::nlocals;
        let nlocals = nlocals as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::pyframe::PyFrame::nlocals"],
            nlocals
        );
        assert_eq!(bindings["pyre_interpreter::PyFrame::nlocals"], nlocals);

        let get_exc: fn() -> pyre_object::PyObjectRef = crate::eval::get_current_exception;
        let get_exc = get_exc as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::eval::get_current_exception"],
            get_exc
        );
        assert_eq!(bindings["pyre_interpreter::get_current_exception"], get_exc);

        let get_sys_exc: fn() -> pyre_object::PyObjectRef = crate::eval::get_sys_exception;
        let get_sys_exc = get_sys_exc as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::eval::get_sys_exception"],
            get_sys_exc
        );
        assert_eq!(bindings["pyre_interpreter::get_sys_exception"], get_sys_exc);

        let get_topmost_exception: fn(
            &crate::executioncontext::ExecutionContext,
        ) -> pyre_object::PyObjectRef =
            crate::executioncontext::ExecutionContext::_get_topmost_exception;
        let get_topmost_exception = get_topmost_exception as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::executioncontext::ExecutionContext::_get_topmost_exception"],
            get_topmost_exception,
        );
        assert_eq!(
            bindings["pyre_interpreter::ExecutionContext::_get_topmost_exception"],
            get_topmost_exception,
        );

        let set_exc: fn(pyre_object::PyObjectRef) = crate::eval::set_current_exception;
        let set_exc = set_exc as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::eval::set_current_exception"],
            set_exc
        );
        assert_eq!(bindings["pyre_interpreter::set_current_exception"], set_exc);

        let first: fn(
            crate::bytecode::Arg<crate::bytecode::oparg::VarNums>,
            crate::bytecode::OpArg,
        ) -> usize = crate::pyopcode::var_nums_to_first_index;
        let first = first as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::pyopcode::var_nums_to_first_index"],
            first
        );
        assert_eq!(bindings["pyre_interpreter::var_nums_to_first_index"], first);

        let second: fn(
            crate::bytecode::Arg<crate::bytecode::oparg::VarNums>,
            crate::bytecode::OpArg,
        ) -> usize = crate::pyopcode::var_nums_to_second_index;
        let second = second as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::pyopcode::var_nums_to_second_index"],
            second
        );
        assert_eq!(
            bindings["pyre_interpreter::var_nums_to_second_index"],
            second
        );
    }

    /// The dispatch-loop safepoint's global-state readers residualize by
    /// qualified path; a typo in either the module-qualified or root alias
    /// silently regresses the `#[dont_look_inside]` call to a symbolic hash,
    /// so pin both spellings against the live fnaddr (siblings of the
    /// `gc_interp::enabled` registration).
    #[test]
    fn jit_trace_fnaddrs_covers_interp_gc_safepoint_readers() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();

        let collect_enabled = pyre_object::gc_interp::collect_enabled as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_object::gc_interp::collect_enabled"],
            collect_enabled
        );
        assert_eq!(bindings["pyre_object::collect_enabled"], collect_enabled);

        let at_outermost =
            pyre_object::gc_interp::at_outermost_activation as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_object::gc_interp::at_outermost_activation"],
            at_outermost
        );
        assert_eq!(
            bindings["pyre_object::at_outermost_activation"],
            at_outermost
        );

        let collect_oldgen =
            pyre_object::gc_hook::try_gc_collect_oldgen as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_object::gc_hook::try_gc_collect_oldgen"],
            collect_oldgen
        );
        assert_eq!(
            bindings["pyre_object::try_gc_collect_oldgen"],
            collect_oldgen
        );

        let itemsblock =
            pyre_object::object_array::itemsblock_gc_enabled as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_object::object_array::itemsblock_gc_enabled"],
            itemsblock
        );
        assert_eq!(bindings["pyre_object::itemsblock_gc_enabled"], itemsblock);

        let bump = crate::call::bump_frame_entry_count as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::call::bump_frame_entry_count"],
            bump
        );

        let py_recursion_depth = crate::call::py_recursion_depth as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::call::py_recursion_depth"],
            py_recursion_depth
        );

        let recursion_limit =
            crate::module::sys::state::recursion_limit as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::module::sys::state::recursion_limit"],
            recursion_limit
        );

        let safepoint = pyre_object::gc_interp::safepoint as *const () as usize as i64;
        assert_eq!(bindings["pyre_object::gc_interp::safepoint"], safepoint);
        assert_eq!(bindings["pyre_object::safepoint"], safepoint);
    }

    /// `rgc.py` keeps `may_ignore_finalizer` opaque with `@jit.dont_look_inside`.  Both names the
    /// LLBC call-path resolver can produce must therefore bind the live helper
    /// address or the residual call would carry an unpatchable symbolic hash.
    #[test]
    fn jit_trace_fnaddrs_covers_may_ignore_finalizer() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();
        let expected = crate::executioncontext::may_ignore_finalizer as *const () as usize as i64;

        assert_eq!(
            bindings["pyre_interpreter::executioncontext::may_ignore_finalizer"],
            expected
        );
        assert_eq!(bindings["pyre_interpreter::may_ignore_finalizer"], expected);
    }

    /// `is_pyframe_operand_stack_accessor` must recognise the funcptr the
    /// codewriter bakes for `PyFrame::pop` — the `pop_value` sub-jitcode
    /// residual the full-body walk must not concretely execute against the
    /// paused outer frame — and must NOT flag `PyFrame::nlocals`, a registered
    /// `PyFrame` method that is a constant read, safe to fold during a walk.
    #[test]
    fn is_pyframe_operand_stack_accessor_matches_registered_pop() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();
        let pop = bindings["pyre_interpreter::pyframe::PyFrame::pop"];
        assert!(is_pyframe_operand_stack_accessor(pop as usize));
        let nlocals = bindings["pyre_interpreter::pyframe::PyFrame::nlocals"];
        assert!(!is_pyframe_operand_stack_accessor(nlocals as usize));
        assert!(!is_pyframe_operand_stack_accessor(0));
    }

    #[test]
    fn is_list_write_barrier_matches_registered_barrier() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();
        let barrier = bindings["pyre_object::listobject::list_write_barrier"];
        assert!(is_list_write_barrier(barrier as usize));
        let nlocals = bindings["pyre_interpreter::pyframe::PyFrame::nlocals"];
        assert!(!is_list_write_barrier(nlocals as usize));
        assert!(!is_list_write_barrier(0));
    }

    /// Negative parity guard: pyre intentionally does NOT publish a
    /// host fnaddr for `_ll_2_str_eq_nonnull` (see the comment block
    /// at `jit_trace_fnaddrs` next to the `cast_float_to_uint`
    /// registration).  A stub registration would fail at runtime
    /// inside any guard-failure recovery; better to surface the
    /// missing helper at codewriter time via the fail-loud
    /// `PromoteString` / `PromoteUnicode` rewrite arms.
    #[test]
    fn jit_trace_fnaddrs_omits_str_eq_nonnull_helper_until_rstr_str_layout_lands() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();
        assert_eq!(
            bindings.get("_ll_2_str_eq_nonnull").copied(),
            None,
            "no `_ll_2_str_eq_nonnull` fnaddr should be published while pyre \
             lacks an `rstr.STR`-equivalent GC layout — registering one would \
             point at a panic-stub that fails at runtime, contradicting \
             `rpython/jit/codewriter/support.py:526-538`'s real comparison body"
        );
    }
}
