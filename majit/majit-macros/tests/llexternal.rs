//! `llexternal!` forwards `link_name` onto the extern funcptr and still marks
//! `ccall_<name>` with `call_aroundstate_target` when `macro` is absent.
#![deny(unused_attributes)]

use majit_macros::llexternal;

llexternal! {
    #[link_name = "strlen"]
    c_my_strlen = "my_strlen",
    [*const core::ffi::c_char],
    usize,
    sandboxsafe = true,
    releasegil = true,
}

llexternal! {
    #[cfg_attr(any(), link_name = "does_not_exist")]
    c_plain_strlen = "strlen",
    [*const core::ffi::c_char],
    usize,
    sandboxsafe = true,
    releasegil = true,
}

llexternal! {
    #[cfg_attr(all(), link_name = "strlen")]
    c_cfg_strlen = "does_not_exist",
    [*const core::ffi::c_char],
    usize,
    sandboxsafe = true,
    releasegil = true,
}

fn strlen_of_hi(len: usize) {
    assert_eq!(len, 2);
}

#[test]
fn unconditional_link_name_calls_strlen() {
    let text = c"hi";
    strlen_of_hi(unsafe { c_my_strlen(text.as_ptr()) });
}

#[test]
fn inactive_cfg_attr_link_name_keeps_the_plain_symbol() {
    let text = c"hi";
    strlen_of_hi(unsafe { c_plain_strlen(text.as_ptr()) });
}

#[test]
fn active_cfg_attr_link_name_calls_strlen() {
    let text = c"hi";
    strlen_of_hi(unsafe { c_cfg_strlen(text.as_ptr()) });
}

fn same_funcptr(
    marker: unsafe extern "C" fn(*const core::ffi::c_char) -> usize,
    func: unsafe extern "C" fn(*const core::ffi::c_char) -> usize,
) {
    assert_eq!(marker as *const (), func as *const ());
}

#[test]
fn link_name_keeps_call_aroundstate_target() {
    let (funcptr, save_err) = _call_aroundstate_target_ccall_c_my_strlen;
    assert_eq!(save_err, majit_rlib::rffi::RFFI_ERR_NONE);
    same_funcptr(funcptr, __rffi_fp_c_my_strlen);

    let (cfg_funcptr, cfg_save_err) = _call_aroundstate_target_ccall_c_cfg_strlen;
    assert_eq!(cfg_save_err, majit_rlib::rffi::RFFI_ERR_NONE);
    same_funcptr(cfg_funcptr, __rffi_fp_c_cfg_strlen);

    let (plain_funcptr, plain_save_err) = _call_aroundstate_target_ccall_c_plain_strlen;
    assert_eq!(plain_save_err, majit_rlib::rffi::RFFI_ERR_NONE);
    same_funcptr(plain_funcptr, __rffi_fp_c_plain_strlen);
}
