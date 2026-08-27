//! The `datetime` C API -- PyPy `cpyext/cdatetime.py`.
//!
//! `datetime` is a Python module here as it is upstream, so the accessors an
//! extension spells as macros over a struct cannot read a struct: each one
//! reads the attribute it names (`cdatetime.py:379-508`), and the declarations
//! generated for them are what the header leaves the macro names to.
//!
//! The two fields an extension does read out of a block are `hastzinfo` and
//! `tzinfo`, which is what [`attach`] fills.

use super::pyerrors::trap;
use super::pyobject::{self, CPyObject};
use super::typeobject::CPyTypeObject;
use pyre_object::{PY_NULL, PyObjectRef};
use std::ffi::{c_int, c_uchar, c_void};

/// `PyDateTime_CAPI` — the table `PyDateTime_IMPORT` binds `PyDateTimeAPI` to.
#[repr(C)]
pub struct CPyDateTimeCAPI {
    pub date_type: *mut CPyTypeObject,
    pub datetime_type: *mut CPyTypeObject,
    pub time_type: *mut CPyTypeObject,
    pub delta_type: *mut CPyTypeObject,
    pub tzinfo_type: *mut CPyTypeObject,
    pub timezone_utc: *mut CPyObject,
    pub date_from_date: *const c_void,
    pub datetime_from_date_and_time: *const c_void,
    pub time_from_time: *const c_void,
    pub delta_from_delta: *const c_void,
    pub timezone_from_timezone: *const c_void,
    pub datetime_from_timestamp: *const c_void,
    pub date_from_timestamp: *const c_void,
    pub datetime_from_date_and_time_and_fold: *const c_void,
    pub time_from_time_and_fold: *const c_void,
}

/// `PyDateTime_Delta` — the broken-down fields a `timedelta` block carries.
#[repr(C)]
pub struct CPyDateTimeDelta {
    pub ob_base: CPyObject,
    pub days: c_int,
    pub seconds: c_int,
    pub microseconds: c_int,
}

/// `PyDateTime_Time` and `PyDateTime_DateTime`, which upstream declares as two
/// structs of one shape (`cpyext_datetime.h`) and reads through either name.
#[repr(C)]
pub struct CPyDateTimeWithTZInfo {
    pub ob_base: CPyObject,
    pub hastzinfo: c_uchar,
    pub tzinfo: *mut CPyObject,
}

const _: () = {
    assert!(std::mem::offset_of!(CPyDateTimeDelta, days) == 3 * size_of::<usize>());
    assert!(std::mem::offset_of!(CPyDateTimeWithTZInfo, tzinfo) == 4 * size_of::<usize>());
    assert!(size_of::<CPyDateTimeWithTZInfo>() == 5 * size_of::<usize>());
};

// ── the module the whole of this reaches through ────────────────────────

/// The `datetime` module, or null when nothing has imported it.
///
/// Read out of `sys.modules` rather than imported: everything below answers
/// for an object of one of the module's classes, and there can be none of
/// those before something has imported it.  [`_PyDateTime_Import`] is the one
/// entry point that imports, because that is what an extension calls it for.
fn datetime_module() -> PyObjectRef {
    crate::importing::check_sys_modules("datetime").unwrap_or(PY_NULL)
}

/// A class of the `datetime` module, or null.
///
/// A module that is only part-way through its own body has none of them yet,
/// and this is reached from a mirror attach, where nothing is going to look
/// at an error: the lookup that fails answers null and clears.
fn datetime_class(name: &str) -> PyObjectRef {
    let module = datetime_module();
    if module.is_null() {
        return PY_NULL;
    }
    let Some(class) = trap(crate::baseobjspace::getattr_str(module, name)) else {
        unsafe { super::pyerrors::PyErr_Clear() };
        return PY_NULL;
    };
    class
}

/// `cdatetime.py:31 PyImport_Import(space.newtext("datetime"))` — the import
/// the API table is built out of.
fn import_datetime() -> Result<PyObjectRef, crate::PyError> {
    super::import_::import_module("datetime")
}

fn class_of(module: PyObjectRef, name: &str) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::getattr_str(module, name)
}

// ── the API table ───────────────────────────────────────────────────────

/// `PyDateTimeAPI`, which `PyDateTime_IMPORT` assigns from
/// [`_PyDateTime_Import`].
///
/// A data symbol rather than the per-translation-unit `static` CPython's own
/// header declares, which is `cpyext/include/datetime.h`'s
/// `PyAPI_DATA(PyDateTime_CAPI*) PyDateTimeAPI`.
#[unsafe(no_mangle)]
pub static mut PyDateTimeAPI: *mut CPyDateTimeCAPI = std::ptr::null_mut();

/// The table, built once — `state.datetimeAPI` (`cdatetime.py:25-27`).
///
/// The references it holds are never released: the table is a process-wide
/// singleton an extension keeps a pointer to for as long as it is loaded.
static API_TABLE: super::ForkMutex<usize> = super::ForkMutex::new(0);

/// `cdatetime.py:22-105 _PyDateTime_Import`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyDateTime_Import() -> *mut CPyDateTimeCAPI {
    let existing = *API_TABLE.lock();
    if existing != 0 {
        return existing as *mut CPyDateTimeCAPI;
    }
    let table = match build_api() {
        Ok(table) => table,
        Err(error) => {
            super::pyerrors::set_pending_error(error);
            return std::ptr::null_mut();
        }
    };
    let table = Box::into_raw(Box::new(table));
    *API_TABLE.lock() = table as usize;
    table
}

fn build_api() -> Result<CPyDateTimeCAPI, crate::PyError> {
    let module = import_datetime()?;
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(module);
    let reload = || pyre_object::gc_roots::shadow_stack_get(module_slot);

    let mut mirror = |name: &str| -> Result<*mut CPyTypeObject, crate::PyError> {
        let class = class_of(reload(), name)?;
        Ok(pyobject::make_ref(class) as *mut CPyTypeObject)
    };
    let date_type = mirror("date")?;
    let datetime_type = mirror("datetime")?;
    let time_type = mirror("time")?;
    let delta_type = mirror("timedelta")?;
    let tzinfo_type = mirror("tzinfo")?;

    // `cdatetime.py:60-63`: the singleton is `datetime.timezone.utc`.
    let timezone = class_of(reload(), "timezone")?;
    let timezone_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(timezone);
    let utc = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(timezone_slot),
        "utc",
    )?;
    let timezone_utc = pyobject::make_ref(utc);

    Ok(CPyDateTimeCAPI {
        date_type,
        datetime_type,
        time_type,
        delta_type,
        tzinfo_type,
        timezone_utc,
        date_from_date: _PyDate_FromDate as *const c_void,
        datetime_from_date_and_time: _PyDateTime_FromDateAndTime as *const c_void,
        time_from_time: _PyTime_FromTime as *const c_void,
        delta_from_delta: _PyDelta_FromDelta as *const c_void,
        timezone_from_timezone: _PyTimeZone_FromTimeZone as *const c_void,
        datetime_from_timestamp: _PyDateTime_FromTimestamp as *const c_void,
        date_from_timestamp: _PyDate_FromTimestamp as *const c_void,
        datetime_from_date_and_time_and_fold: _PyDateTime_FromDateAndTimeAndFold as *const c_void,
        time_from_time_and_fold: _PyTime_FromTimeAndFold as *const c_void,
    })
}

pub(super) unsafe fn after_fork_child() {
    unsafe {
        API_TABLE.reinit_after_fork();
        ATTACHED.reinit_after_fork();
    }
}

// ── the constructors ────────────────────────────────────────────────────

/// Call `type` with `values`, and with `fold` as the one keyword when it is
/// named — `Arguments(..., keyword_names_w=['fold'])` (`cdatetime.py:275-283`).
fn construct(
    type_: *mut CPyTypeObject,
    values: &[i64],
    tzinfo: *mut CPyObject,
    fold: Option<i64>,
) -> *mut CPyObject {
    let callable = super::typeobject::interpreter_type(type_);
    if callable.is_null() {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "the datetime C API was called with a type that is not ready",
        ));
        return std::ptr::null_mut();
    }
    let roots = pyre_object::gc_roots::push_roots();
    let callable_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(callable);
    let arguments_base = pyre_object::gc_roots::shadow_stack_len();
    let mut count = 0usize;
    for &value in values {
        let _ = roots.pin_root(pyre_object::intobject::w_int_new(value));
        count += 1;
    }
    if !tzinfo.is_null() {
        let _ = roots.pin_root(unsafe { pyobject::from_ref(tzinfo) });
        count += 1;
    }
    let fold_slot = pyre_object::gc_roots::shadow_stack_len();
    if let Some(fold) = fold {
        let _ = roots.pin_root(pyre_object::intobject::w_int_new(fold));
    }
    // Each root was taken before the allocation that follows it, so what the
    // call is handed is read back rather than the pre-move words above.
    let arguments: Vec<PyObjectRef> = (0..count)
        .map(|index| pyre_object::gc_roots::shadow_stack_get(arguments_base + index))
        .collect();
    let callable = pyre_object::gc_roots::shadow_stack_get(callable_slot);
    let called = match fold {
        None => crate::call::call_function_impl_result(callable, &arguments),
        // `Arguments(..., keyword_names_w=['fold'])` (`cdatetime.py:275-283`).
        Some(_) => super::object::call_keyword(
            callable,
            &arguments,
            "fold",
            pyre_object::gc_roots::shadow_stack_get(fold_slot),
        ),
    };
    super::object::result(called)
}

/// `cdatetime.py:218-229 _PyDate_FromDate`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyDate_FromDate(
    year: c_int,
    month: c_int,
    day: c_int,
    type_: *mut CPyTypeObject,
) -> *mut CPyObject {
    construct(
        type_,
        &[year as i64, month as i64, day as i64],
        std::ptr::null_mut(),
        None,
    )
}

/// `cdatetime.py:230-243 _PyTime_FromTime`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyTime_FromTime(
    hour: c_int,
    minute: c_int,
    second: c_int,
    usecond: c_int,
    tzinfo: *mut CPyObject,
    type_: *mut CPyTypeObject,
) -> *mut CPyObject {
    construct(
        type_,
        &[hour as i64, minute as i64, second as i64, usecond as i64],
        tzinfo,
        None,
    )
}

/// `cdatetime.py:293-312 _PyTime_FromTimeAndFold`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyTime_FromTimeAndFold(
    hour: c_int,
    minute: c_int,
    second: c_int,
    usecond: c_int,
    tzinfo: *mut CPyObject,
    fold: c_int,
    type_: *mut CPyTypeObject,
) -> *mut CPyObject {
    construct(
        type_,
        &[hour as i64, minute as i64, second as i64, usecond as i64],
        tzinfo,
        Some(fold as i64),
    )
}

/// `cdatetime.py:244-264 _PyDateTime_FromDateAndTime`.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyDateTime_FromDateAndTime(
    year: c_int,
    month: c_int,
    day: c_int,
    hour: c_int,
    minute: c_int,
    second: c_int,
    usecond: c_int,
    tzinfo: *mut CPyObject,
    type_: *mut CPyTypeObject,
) -> *mut CPyObject {
    construct(
        type_,
        &[
            year as i64,
            month as i64,
            day as i64,
            hour as i64,
            minute as i64,
            second as i64,
            usecond as i64,
        ],
        tzinfo,
        None,
    )
}

/// `cdatetime.py:266-291 _PyDateTime_FromDateAndTimeAndFold`.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyDateTime_FromDateAndTimeAndFold(
    year: c_int,
    month: c_int,
    day: c_int,
    hour: c_int,
    minute: c_int,
    second: c_int,
    usecond: c_int,
    tzinfo: *mut CPyObject,
    fold: c_int,
    type_: *mut CPyTypeObject,
) -> *mut CPyObject {
    construct(
        type_,
        &[
            year as i64,
            month as i64,
            day as i64,
            hour as i64,
            minute as i64,
            second as i64,
            usecond as i64,
        ],
        tzinfo,
        Some(fold as i64),
    )
}

/// `cdatetime.py:348-362 _PyDelta_FromDelta`.
///
/// `normalize` is not passed on: `timedelta` normalizes what it is given, so
/// there is nothing the argument could select between.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyDelta_FromDelta(
    days: c_int,
    seconds: c_int,
    useconds: c_int,
    _normalize: c_int,
    type_: *mut CPyTypeObject,
) -> *mut CPyObject {
    construct(
        type_,
        &[days as i64, seconds as i64, useconds as i64],
        std::ptr::null_mut(),
        None,
    )
}

/// `cdatetime.py:365-377 _PyTimeZone_FromTimeZone`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyTimeZone_FromTimeZone(
    offset: *mut CPyObject,
    name: *mut CPyObject,
) -> *mut CPyObject {
    let called = (|| -> Result<PyObjectRef, crate::PyError> {
        let module = import_datetime()?;
        let roots = pyre_object::gc_roots::push_roots();
        let module_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(module);
        let timezone = class_of(
            pyre_object::gc_roots::shadow_stack_get(module_slot),
            "timezone",
        )?;
        let timezone_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(timezone);
        // Both are realized before either is read: realizing one allocates,
        // and the interpreter object the other names would move.
        super::object::realize_all([offset, name]);
        let mut arguments = vec![unsafe { pyobject::from_ref(offset) }];
        if !name.is_null() {
            arguments.push(unsafe { pyobject::from_ref(name) });
        }
        crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(timezone_slot),
            &arguments,
        )
    })();
    super::object::result(called)
}

/// Call `type.<name>(*args, **kwds)` — the shape both `FromTimestamp` entry
/// points share (`cdatetime.py:323-346`).
fn from_timestamp(
    type_: *mut CPyObject,
    name: &str,
    args: *mut CPyObject,
    kwds: *mut CPyObject,
) -> *mut CPyObject {
    let w_type = unsafe { pyobject::from_ref(type_) };
    if w_type.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let roots = pyre_object::gc_roots::push_roots();
    let type_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_type);
    let found =
        crate::baseobjspace::getattr_str(pyre_object::gc_roots::shadow_stack_get(type_slot), name);
    let Some(method) = trap(found) else {
        return std::ptr::null_mut();
    };
    let method = pyobject::make_ref(method);
    let made = unsafe { super::object::PyObject_Call(method, args, kwds) };
    unsafe { pyobject::decref(method) };
    made
}

/// `cdatetime.py:323-330 _PyDateTime_FromTimestamp`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyDateTime_FromTimestamp(
    type_: *mut CPyObject,
    args: *mut CPyObject,
    kwds: *mut CPyObject,
) -> *mut CPyObject {
    from_timestamp(type_, "fromtimestamp", args, kwds)
}

/// `cdatetime.py:341-346 _PyDate_FromTimestamp`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyDate_FromTimestamp(
    type_: *mut CPyObject,
    args: *mut CPyObject,
) -> *mut CPyObject {
    from_timestamp(type_, "fromtimestamp", args, std::ptr::null_mut())
}

/// `cdatetime.py:314-321 PyDateTime_FromTimestamp`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_FromTimestamp(args: *mut CPyObject) -> *mut CPyObject {
    let class = pyobject::make_ref(datetime_class_or_import("datetime"));
    let made = unsafe { _PyDateTime_FromTimestamp(class, args, std::ptr::null_mut()) };
    unsafe { pyobject::decref(class) };
    made
}

/// `cdatetime.py:332-339 PyDate_FromTimestamp`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDate_FromTimestamp(args: *mut CPyObject) -> *mut CPyObject {
    let class = pyobject::make_ref(datetime_class_or_import("date"));
    let made = unsafe { _PyDate_FromTimestamp(class, args) };
    unsafe { pyobject::decref(class) };
    made
}

fn datetime_class_or_import(name: &str) -> PyObjectRef {
    if datetime_module().is_null() {
        let _ = import_datetime();
    }
    datetime_class(name)
}

// ── the check functions ─────────────────────────────────────────────────

/// `cdatetime.py:107-121 make_check_function`'s `check`.
fn check(object: *mut CPyObject, name: &str) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    let class = datetime_class(name);
    if object.is_null() || class.is_null() {
        return 0;
    }
    unsafe { crate::baseobjspace::isinstance_w(object, class) as c_int }
}

/// `make_check_function`'s `check_exact`.
fn check_exact(object: *mut CPyObject, name: &str) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    let class = datetime_class(name);
    if object.is_null() || class.is_null() {
        return 0;
    }
    match crate::typedef::r#type(object) {
        Some(w_type) => std::ptr::eq(w_type.as_ptr(), class) as c_int,
        None => 0,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDate_Check(object: *mut CPyObject) -> c_int {
    check(object, "date")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDate_CheckExact(object: *mut CPyObject) -> c_int {
    check_exact(object, "date")
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_Check(object: *mut CPyObject) -> c_int {
    check(object, "datetime")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_CheckExact(object: *mut CPyObject) -> c_int {
    check_exact(object, "datetime")
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyTime_Check(object: *mut CPyObject) -> c_int {
    check(object, "time")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyTime_CheckExact(object: *mut CPyObject) -> c_int {
    check_exact(object, "time")
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDelta_Check(object: *mut CPyObject) -> c_int {
    check(object, "timedelta")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDelta_CheckExact(object: *mut CPyObject) -> c_int {
    check_exact(object, "timedelta")
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyTZInfo_Check(object: *mut CPyObject) -> c_int {
    check(object, "tzinfo")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyTZInfo_CheckExact(object: *mut CPyObject) -> c_int {
    check_exact(object, "tzinfo")
}
fn ensure_checks_linked() {
    std::hint::black_box(PyDate_Check as *const ());
    std::hint::black_box(PyDate_CheckExact as *const ());
    std::hint::black_box(PyDateTime_Check as *const ());
    std::hint::black_box(PyDateTime_CheckExact as *const ());
    std::hint::black_box(PyTime_Check as *const ());
    std::hint::black_box(PyTime_CheckExact as *const ());
    std::hint::black_box(PyDelta_Check as *const ());
    std::hint::black_box(PyDelta_CheckExact as *const ());
    std::hint::black_box(PyTZInfo_Check as *const ());
    std::hint::black_box(PyTZInfo_CheckExact as *const ());
}

// ── the accessors ───────────────────────────────────────────────────────

/// The attribute `name` of `object` as a C `int`, and 0 where there is none.
///
/// `cdatetime.py:410-416` states why a missing attribute is not an error: a
/// library that reads an hour off a `date` gets nonsense from the macro
/// upstream too, and it does not crash.
fn int_attr(w_obj: PyObjectRef, name: &str) -> c_int {
    if w_obj.is_null() {
        return 0;
    }
    let Some(value) = trap(crate::baseobjspace::getattr_str(w_obj, name)) else {
        unsafe { super::pyerrors::PyErr_Clear() };
        return 0;
    };
    let read = trap(crate::baseobjspace::gateway_int_w(value));
    if read.is_none() {
        unsafe { super::pyerrors::PyErr_Clear() };
    }
    read.unwrap_or(0) as c_int
}

fn field(object: *mut c_void, name: &str) -> c_int {
    int_attr(
        unsafe { pyobject::from_ref(object as *mut CPyObject) },
        name,
    )
}

/// The `tzinfo` of `object`, borrowed, and `Py_None` where there is none
/// (`cdatetime.py:443-454`).
fn tzinfo_of(object: *mut c_void) -> *mut CPyObject {
    let none = pyobject::borrow_mirror(pyre_object::w_none());
    let raw = object as *mut CPyObject;
    let w_obj = unsafe { pyobject::from_ref(raw) };
    if w_obj.is_null() {
        return none;
    }
    let Some(w_tzinfo) = trap(crate::baseobjspace::getattr_str(w_obj, "tzinfo")) else {
        unsafe { super::pyerrors::PyErr_Clear() };
        return none;
    };
    if w_tzinfo.is_null() {
        return none;
    }
    // Borrowed, as the macro it stands in for is: what keeps the reference
    // alive is the block it was read through.
    pyobject::borrow_from(raw, w_tzinfo)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_GET_YEAR(object: *mut c_void) -> c_int {
    field(object, "year")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_GET_MONTH(object: *mut c_void) -> c_int {
    field(object, "month")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_GET_DAY(object: *mut c_void) -> c_int {
    field(object, "day")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_GET_FOLD(object: *mut c_void) -> c_int {
    field(object, "fold")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_DATE_GET_HOUR(object: *mut c_void) -> c_int {
    field(object, "hour")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_DATE_GET_MINUTE(object: *mut c_void) -> c_int {
    field(object, "minute")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_DATE_GET_SECOND(object: *mut c_void) -> c_int {
    field(object, "second")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_DATE_GET_MICROSECOND(object: *mut c_void) -> c_int {
    field(object, "microsecond")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_TIME_GET_HOUR(object: *mut c_void) -> c_int {
    field(object, "hour")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_TIME_GET_MINUTE(object: *mut c_void) -> c_int {
    field(object, "minute")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_TIME_GET_SECOND(object: *mut c_void) -> c_int {
    field(object, "second")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_TIME_GET_MICROSECOND(object: *mut c_void) -> c_int {
    field(object, "microsecond")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_TIME_GET_FOLD(object: *mut c_void) -> c_int {
    field(object, "fold")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_DELTA_GET_DAYS(object: *mut c_void) -> c_int {
    field(object, "days")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_DELTA_GET_SECONDS(object: *mut c_void) -> c_int {
    field(object, "seconds")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_DELTA_GET_MICROSECONDS(object: *mut c_void) -> c_int {
    field(object, "microseconds")
}

fn ensure_accessors_linked() {
    std::hint::black_box(PyDateTime_GET_YEAR as *const ());
    std::hint::black_box(PyDateTime_GET_MONTH as *const ());
    std::hint::black_box(PyDateTime_GET_DAY as *const ());
    std::hint::black_box(PyDateTime_GET_FOLD as *const ());
    std::hint::black_box(PyDateTime_DATE_GET_HOUR as *const ());
    std::hint::black_box(PyDateTime_DATE_GET_MINUTE as *const ());
    std::hint::black_box(PyDateTime_DATE_GET_SECOND as *const ());
    std::hint::black_box(PyDateTime_DATE_GET_MICROSECOND as *const ());
    std::hint::black_box(PyDateTime_TIME_GET_HOUR as *const ());
    std::hint::black_box(PyDateTime_TIME_GET_MINUTE as *const ());
    std::hint::black_box(PyDateTime_TIME_GET_SECOND as *const ());
    std::hint::black_box(PyDateTime_TIME_GET_MICROSECOND as *const ());
    std::hint::black_box(PyDateTime_TIME_GET_FOLD as *const ());
    std::hint::black_box(PyDateTime_DELTA_GET_DAYS as *const ());
    std::hint::black_box(PyDateTime_DELTA_GET_SECONDS as *const ());
    std::hint::black_box(PyDateTime_DELTA_GET_MICROSECONDS as *const ());
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_DATE_GET_TZINFO(object: *mut c_void) -> *mut CPyObject {
    tzinfo_of(object)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyDateTime_TIME_GET_TZINFO(object: *mut c_void) -> *mut CPyObject {
    tzinfo_of(object)
}

// ── the fields a block carries ──────────────────────────────────────────

type BlockSet = super::address_table::HeldSet;

/// The blocks [`attach`] filled a `tzinfo` reference into.
///
/// A set of addresses rather than a size test, for the reason
/// `pyerrors::ATTACHED` states: what decides whether the word at that offset
/// is a reference is whether this module wrote it.
use super::address_table::{AddressTable, hold};

static ATTACHED: AddressTable<BlockSet> =
    AddressTable::new(BlockSet::with_hasher(std::hash::BuildHasherDefault::new()));

/// Which of the two blocks with fields a class of the `datetime` module takes
/// — `cdatetime.py:143-160 init_datetime`'s three `basestruct`s, of which
/// `date` gets one it does not use and `_PyDateTime_Import` sizes back down.
#[derive(Clone, Copy)]
enum Shape {
    /// `timedelta`, which carries its three broken-down fields.
    Delta,
    /// `datetime` and `time`, which carry `hastzinfo` and the word beside it.
    WithTZInfo,
}

impl Shape {
    fn size(self) -> isize {
        match self {
            Shape::Delta => size_of::<CPyDateTimeDelta>() as isize,
            Shape::WithTZInfo => size_of::<CPyDateTimeWithTZInfo>() as isize,
        }
    }
}

/// The shape a class asks for, by name — the question `get_typedescr` answers
/// off a typedef, asked once per type when its mirror is sized.
///
/// `None` until something has imported `datetime`, which is also when the
/// first object of one of these classes can exist.
fn declared_shape(w_type: PyObjectRef) -> Option<Shape> {
    if w_type.is_null() {
        return None;
    }
    let derived = |name: &str| {
        let class = datetime_class(name);
        !class.is_null() && unsafe { crate::baseobjspace::issubtype_w(w_type, class) }
    };
    if derived("timedelta") {
        return Some(Shape::Delta);
    }
    // `datetime` before `date`, because it is one: only the more derived of
    // the two carries the `tzinfo` word.
    if derived("datetime") || derived("time") {
        return Some(Shape::WithTZInfo);
    }
    None
}

/// What `tp_basicsize` a synthesized mirror of `w_type` carries.
pub(super) fn basicsize(w_type: PyObjectRef) -> isize {
    declared_shape(w_type).map_or(0, Shape::size)
}

/// The shape a block carries, read off its type mirror — `type_attach`
/// comparing `py_obj.c_ob_type` against `state.datetimeAPI[0]`'s type slots.
///
/// This is asked of every block this runtime mirrors, so it asks nothing of
/// the interpreter in turn: resolving the classes by name would be two dict
/// probes per object, and a thread foreign code owns may be running before
/// there is anything to probe them with.
fn block_shape(tp: *mut CPyTypeObject) -> Option<Shape> {
    if tp.is_null() {
        return None;
    }
    // Almost every block is smaller than either shape, and that is one load.
    let room = unsafe { (*tp).tp_basicsize };
    if room < Shape::Delta.size().min(Shape::WithTZInfo.size()) {
        return None;
    }
    let table = *API_TABLE.lock() as *const CPyDateTimeCAPI;
    if table.is_null() {
        return None;
    }
    let mut tp = tp;
    while !tp.is_null() {
        let table = unsafe { &*table };
        if std::ptr::eq(tp, table.delta_type) {
            return Some(Shape::Delta);
        }
        if std::ptr::eq(tp, table.datetime_type) || std::ptr::eq(tp, table.time_type) {
            return Some(Shape::WithTZInfo);
        }
        // `date` is below `datetime` and carries no field of its own, so a
        // chain that has reached it passed nothing that does.
        if std::ptr::eq(tp, table.date_type) {
            return None;
        }
        tp = unsafe { (*tp).tp_base };
    }
    None
}

/// Fill a freshly allocated mirror — `cdatetime.py:162-189 type_attach` and
/// `206-216 timedeltatype_attach`.
pub(super) fn attach(raw: *mut CPyObject, w_obj: PyObjectRef) {
    let tp = unsafe { (*raw).ob_type };
    let Some(shape) = block_shape(tp) else {
        return;
    };
    // The block was sized from this same mirror, so the two agree unless the
    // module was replaced between them: a class kept from an import that has
    // since been undone names a `datetime` this no longer knows, and the words
    // are not there to write.
    if unsafe { (*tp).tp_basicsize } < shape.size() {
        return;
    }
    if let Shape::Delta = shape {
        let delta = raw as *mut CPyDateTimeDelta;
        unsafe {
            (*delta).days = int_attr(w_obj, "_days");
            (*delta).seconds = int_attr(w_obj, "_seconds");
            (*delta).microseconds = int_attr(w_obj, "_microseconds");
        }
        return;
    }
    // The word is filled on every block that has room for it, subclasses
    // included; `type_attach` compares `ob_type` for equality instead, which
    // leaves an instance of a subclass reading as naive.
    let w_tzinfo = trap(crate::baseobjspace::getattr_str(w_obj, "tzinfo")).unwrap_or(PY_NULL);
    let has = !w_tzinfo.is_null() && !unsafe { pyre_object::is_none(w_tzinfo) };
    let block = raw as *mut CPyDateTimeWithTZInfo;
    unsafe {
        (*block).hastzinfo = has as c_uchar;
        (*block).tzinfo = match has {
            true => pyobject::make_ref(w_tzinfo),
            false => std::ptr::null_mut(),
        };
    }
    if has {
        ATTACHED.lock().insert(hold(raw as usize));
    }
}

/// Release the reference a `time` or `datetime` mirror owns —
/// `cdatetime.py:191-204 type_dealloc`.
pub(super) fn forget_block(raw: *mut CPyObject) {
    if !ATTACHED.discard(raw as usize) {
        return;
    }
    let block = raw as *mut CPyDateTimeWithTZInfo;
    unsafe {
        pyobject::decref((*block).tzinfo);
        (*block).tzinfo = std::ptr::null_mut();
        (*block).hastzinfo = 0;
    }
}

pub(super) fn ensure_linked() {
    std::hint::black_box(&raw const PyDateTimeAPI);
    std::hint::black_box(_PyDateTime_Import as *const ());
    std::hint::black_box(_PyDate_FromDate as *const ());
    std::hint::black_box(_PyTime_FromTime as *const ());
    std::hint::black_box(_PyTime_FromTimeAndFold as *const ());
    std::hint::black_box(_PyDateTime_FromDateAndTime as *const ());
    std::hint::black_box(_PyDateTime_FromDateAndTimeAndFold as *const ());
    std::hint::black_box(_PyDelta_FromDelta as *const ());
    std::hint::black_box(_PyTimeZone_FromTimeZone as *const ());
    std::hint::black_box(_PyDateTime_FromTimestamp as *const ());
    std::hint::black_box(_PyDate_FromTimestamp as *const ());
    std::hint::black_box(PyDateTime_FromTimestamp as *const ());
    std::hint::black_box(PyDate_FromTimestamp as *const ());
    std::hint::black_box(PyDateTime_DATE_GET_TZINFO as *const ());
    std::hint::black_box(PyDateTime_TIME_GET_TZINFO as *const ());
    ensure_checks_linked();
    ensure_accessors_linked();
}
