//! winreg module — PyPy: pypy/module/_winreg/ (`applevel_name = 'winreg'`).
//!
//! `importlib._bootstrap_external` does an eager `import winreg` on every
//! `sys.platform == "win32"` build, so the module must exist for the import
//! machinery — and therefore `import site` — to come up.  The integer
//! constants are always present; the `PyHKEY` handle object and the
//! Open/Query/Enum/Set/Delete key & value functions are registered when the
//! `host_env` feature is on (they delegate every Win32 `Reg*` call to
//! `rustpython_host_env::winreg`).
crate::py_module! {
    "winreg",
    int_constants: {
        // Predefined root handles (winreg.h). Python surfaces these as the
        // unsigned handle value, so cast through u32.
        "HKEY_CLASSES_ROOT" => 0x8000_0000u32,
        "HKEY_CURRENT_USER" => 0x8000_0001u32,
        "HKEY_LOCAL_MACHINE" => 0x8000_0002u32,
        "HKEY_USERS" => 0x8000_0003u32,
        "HKEY_PERFORMANCE_DATA" => 0x8000_0004u32,
        "HKEY_CURRENT_CONFIG" => 0x8000_0005u32,
        "HKEY_DYN_DATA" => 0x8000_0006u32,

        // Access-right masks (winnt.h).
        "KEY_QUERY_VALUE" => 0x0001,
        "KEY_SET_VALUE" => 0x0002,
        "KEY_CREATE_SUB_KEY" => 0x0004,
        "KEY_ENUMERATE_SUB_KEYS" => 0x0008,
        "KEY_NOTIFY" => 0x0010,
        "KEY_CREATE_LINK" => 0x0020,
        "KEY_WOW64_64KEY" => 0x0100,
        "KEY_WOW64_32KEY" => 0x0200,
        "KEY_WRITE" => 0x0002_0006,
        "KEY_READ" => 0x0002_0019,
        "KEY_EXECUTE" => 0x0002_0019,
        "KEY_ALL_ACCESS" => 0x000F_003F,

        // Value types (winnt.h REG_*).
        "REG_NONE" => 0,
        "REG_SZ" => 1,
        "REG_EXPAND_SZ" => 2,
        "REG_BINARY" => 3,
        "REG_DWORD" => 4,
        "REG_DWORD_LITTLE_ENDIAN" => 4,
        "REG_DWORD_BIG_ENDIAN" => 5,
        "REG_LINK" => 6,
        "REG_MULTI_SZ" => 7,
        "REG_RESOURCE_LIST" => 8,
        "REG_FULL_RESOURCE_DESCRIPTOR" => 9,
        "REG_RESOURCE_REQUIREMENTS_LIST" => 10,
        "REG_QWORD" => 11,
        "REG_QWORD_LITTLE_ENDIAN" => 11,

        // Key-creation options and disposition (winnt.h).
        "REG_OPTION_RESERVED" => 0,
        "REG_OPTION_NON_VOLATILE" => 0,
        "REG_OPTION_VOLATILE" => 1,
        "REG_OPTION_CREATE_LINK" => 2,
        "REG_OPTION_BACKUP_RESTORE" => 4,
        "REG_OPTION_OPEN_LINK" => 8,
        "REG_CREATED_NEW_KEY" => 1,
        "REG_OPENED_EXISTING_KEY" => 2,
        "REG_WHOLE_HIVE_VOLATILE" => 1,
        "REG_REFRESH_HIVE" => 2,
        "REG_NO_LAZY_FLUSH" => 4,

        // RegNotifyChangeKeyValue filters (winnt.h).
        "REG_NOTIFY_CHANGE_NAME" => 1,
        "REG_NOTIFY_CHANGE_ATTRIBUTES" => 2,
        "REG_NOTIFY_CHANGE_LAST_SET" => 4,
        "REG_NOTIFY_CHANGE_SECURITY" => 8,
        "REG_LEGAL_CHANGE_FILTER" => 0x000F,
        "REG_LEGAL_OPTION" => 0x000F,
    },
    extra_init: |ns| {
        #[cfg(feature = "host_env")]
        imp::install(ns);
    }
}

#[cfg(feature = "host_env")]
mod imp {
    use pyre_object::rbigint::RBigInt as BigInt;
    use pyre_object::*;
    use rustpython_host_env::winreg::{self as host_reg, HKEY};
    use std::ffi::OsStr;
    use widestring::WideCString;

    /// A raw handle/pointer value → a Python int (`PyLong_FromVoidPtr`).  The
    /// predefined roots sign-extend past `i64::MAX` on 64-bit, so route those
    /// through a long.
    fn int_from_ptr(raw: HKEY) -> PyObjectRef {
        let value = raw as usize as u64;
        if value <= i64::MAX as u64 {
            w_int_new(value as i64)
        } else {
            pyre_object::longobject::w_long_new(BigInt::from(value))
        }
    }

    // ── PyHKEY handle object ──
    // The raw `HKEY` is stored on the instance dict under `_handle` as its
    // pointer value; `int(key)` and passing a key where a handle is expected
    // read it back.  A closed handle is left as 0.
    crate::py_class! {
        "PyHKEY",
        methods: {
            fn Close(self_obj: PyObjectRef) -> Result<(), crate::PyError> {
                let raw = take_handle(self_obj);
                if !raw.is_null() {
                    host_reg::close_key(raw);
                }
                Ok(())
            }
            fn Detach(self_obj: PyObjectRef) -> PyObjectRef {
                int_from_ptr(take_handle(self_obj))
            }
            fn __int__(self_obj: PyObjectRef) -> PyObjectRef {
                int_from_ptr(get_handle(self_obj))
            }
            fn __bool__(self_obj: PyObjectRef) -> bool {
                !get_handle(self_obj).is_null()
            }
            fn __enter__(self_obj: PyObjectRef) -> PyObjectRef {
                self_obj
            }
            fn __exit__(
                self_obj: PyObjectRef,
                _exc_type: PyObjectRef,
                _exc_value: PyObjectRef,
                _traceback: PyObjectRef,
            ) -> Result<bool, crate::PyError> {
                let raw = take_handle(self_obj);
                if !raw.is_null() {
                    host_reg::close_key(raw);
                }
                Ok(false)
            }
        }
    }

    fn store_handle(obj: PyObjectRef, raw: HKEY) {
        let d = crate::baseobjspace::getdict(obj);
        if !d.is_null() {
            unsafe { w_dict_setitem_str(d, "_handle", int_from_ptr(raw)) };
        }
    }

    fn get_handle(obj: PyObjectRef) -> HKEY {
        stored_handle(obj).unwrap_or(core::ptr::null_mut())
    }

    /// Read the stored handle and reset it to 0 (used by `Close`/`Detach`).
    fn take_handle(obj: PyObjectRef) -> HKEY {
        let raw = get_handle(obj);
        store_handle(obj, core::ptr::null_mut());
        raw
    }

    fn make_pyhkey(raw: HKEY) -> PyObjectRef {
        let obj = w_instance_new(type_object());
        store_handle(obj, raw);
        obj
    }

    /// The stored `_handle` value, if `obj` is a `PyHKEY`.  `uint_w` reads both
    /// the small-int and long forms a handle pointer may take.
    fn stored_handle(obj: PyObjectRef) -> Option<HKEY> {
        let d = crate::baseobjspace::getdict(obj);
        if d.is_null() {
            return None;
        }
        let value = unsafe { w_dict_getitem_str(d, "_handle") }?;
        crate::baseobjspace::uint_w(value)
            .ok()
            .map(|u| u as usize as HKEY)
    }

    /// Resolve a key argument — a `PyHKEY` or an integer handle (the predefined
    /// `HKEY_*` roots) — to a raw `HKEY`.
    fn key_arg(args: &[PyObjectRef], idx: usize) -> Result<HKEY, crate::PyError> {
        let obj = arg(args, idx)?;
        if let Some(raw) = stored_handle(obj) {
            return Ok(raw);
        }
        if unsafe { is_int_or_long(obj) } {
            return Ok(crate::baseobjspace::uint_w(obj)? as usize as HKEY);
        }
        Err(crate::PyError::type_error(
            "The object is not a PyHKEY object",
        ))
    }

    fn arg(args: &[PyObjectRef], idx: usize) -> Result<PyObjectRef, crate::PyError> {
        args.get(idx)
            .copied()
            .ok_or_else(|| crate::PyError::type_error("required argument missing"))
    }

    /// A `str` or `None` argument → owned `String` / `None`.
    fn opt_str(args: &[PyObjectRef], idx: usize) -> Result<Option<String>, crate::PyError> {
        match args.get(idx).copied() {
            None => Ok(None),
            Some(obj) if unsafe { is_none(obj) } => Ok(None),
            Some(obj) if unsafe { is_str(obj) } => {
                Ok(Some(unsafe { w_str_get_value(obj) }.to_string()))
            }
            Some(_) => Err(crate::PyError::type_error(
                "argument must be a string or None",
            )),
        }
    }

    fn req_str(args: &[PyObjectRef], idx: usize) -> Result<String, crate::PyError> {
        let obj = arg(args, idx)?;
        if unsafe { is_str(obj) } {
            Ok(unsafe { w_str_get_value(obj) }.to_string())
        } else {
            Err(crate::PyError::type_error("argument must be a string"))
        }
    }

    fn req_int(args: &[PyObjectRef], idx: usize) -> Result<i64, crate::PyError> {
        crate::baseobjspace::int_w(arg(args, idx)?)
    }

    /// Map a Win32 error code to an `OSError`.
    fn win_err(code: u32) -> crate::PyError {
        let error = std::io::Error::from_raw_os_error(code as i32);
        crate::PyError::os_error_syscall(
            crate::builtins::io_error_posix_errno(&error, 0),
            pyre_object::PY_NULL,
        )
    }

    fn check(code: u32) -> Result<PyObjectRef, crate::PyError> {
        if code != 0 {
            Err(win_err(code))
        } else {
            Ok(w_none())
        }
    }

    fn utf16_units(data: &[u8]) -> Vec<u16> {
        data.chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect()
    }

    /// `Reg2Py` — registry bytes + type → a Python value.
    fn reg2py(data: &[u8], typ: u32) -> PyObjectRef {
        match typ {
            host_reg::REG_DWORD => {
                let value = if data.len() >= 4 {
                    u32::from_le_bytes([data[0], data[1], data[2], data[3]])
                } else {
                    0
                };
                w_int_new(value as i64)
            }
            host_reg::REG_QWORD => {
                let value = if data.len() >= 8 {
                    u64::from_le_bytes(data[..8].try_into().unwrap())
                } else {
                    0
                };
                // Registry QWORDs above i64::MAX are vanishingly rare; the raw
                // 64-bit pattern is preserved.
                w_int_new(value as i64)
            }
            host_reg::REG_SZ | host_reg::REG_EXPAND_SZ | host_reg::REG_LINK => {
                let units = utf16_units(data);
                let end = units.iter().position(|&c| c == 0).unwrap_or(units.len());
                w_str_new(&String::from_utf16_lossy(&units[..end]))
            }
            host_reg::REG_MULTI_SZ => {
                let units = utf16_units(data);
                let mut items = Vec::new();
                for chunk in units.split(|&c| c == 0) {
                    if chunk.is_empty() {
                        break;
                    }
                    items.push(w_str_new(&String::from_utf16_lossy(chunk)));
                }
                w_list_new(items)
            }
            _ => {
                if data.is_empty() {
                    w_none()
                } else {
                    pyre_object::bytesobject::w_bytes_from_bytes(data)
                }
            }
        }
    }

    fn encode_utf16z(s: &str) -> Vec<u8> {
        let mut units: Vec<u16> = s.encode_utf16().collect();
        units.push(0);
        units.iter().flat_map(|u| u.to_le_bytes()).collect()
    }

    /// `Py2Reg` — a Python value + type → registry bytes.
    fn py2reg(value: PyObjectRef, typ: u32) -> Result<Vec<u8>, crate::PyError> {
        match typ {
            host_reg::REG_DWORD => Ok((crate::baseobjspace::int_w(value)? as u32)
                .to_le_bytes()
                .to_vec()),
            host_reg::REG_QWORD => Ok((crate::baseobjspace::int_w(value)? as u64)
                .to_le_bytes()
                .to_vec()),
            host_reg::REG_SZ | host_reg::REG_EXPAND_SZ => {
                if unsafe { !is_str(value) } {
                    return Err(crate::PyError::type_error("value must be a string"));
                }
                Ok(encode_utf16z(unsafe { w_str_get_value(value) }))
            }
            host_reg::REG_MULTI_SZ => {
                let len = unsafe { w_list_len(value) };
                let mut out = Vec::new();
                for i in 0..len {
                    let item = unsafe { w_list_getitem(value, i as i64) };
                    let Some(item) = item else { continue };
                    if unsafe { !is_str(item) } {
                        return Err(crate::PyError::type_error(
                            "REG_MULTI_SZ value must be a list of strings",
                        ));
                    }
                    let units: Vec<u16> = unsafe { w_str_get_value(item) }.encode_utf16().collect();
                    out.extend(units.iter().flat_map(|u| u.to_le_bytes()));
                    out.extend_from_slice(&[0, 0]);
                }
                out.extend_from_slice(&[0, 0]);
                Ok(out)
            }
            _ => {
                // REG_BINARY / REG_NONE and the rest — a bytes-like value, or
                // `None` for an empty write.
                if unsafe { is_none(value) } {
                    return Ok(Vec::new());
                }
                if unsafe { pyre_object::bytesobject::is_bytes_like(value) } {
                    Ok(unsafe { pyre_object::bytesobject::bytes_like_data(value) }.to_vec())
                } else {
                    Err(crate::PyError::type_error("value must be bytes"))
                }
            }
        }
    }

    // ── module functions ──
    fn open_key(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        // (key, sub_key, reserved=0, access=KEY_READ)
        let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
        let key = key_arg(pos, 0)?;
        let sub_key = opt_str(pos, 1)?;
        let access = match crate::builtins::kwarg_get(kwargs, "access") {
            Some(v) => crate::baseobjspace::int_w(v)? as u32,
            None => match pos.get(3) {
                Some(&v) => crate::baseobjspace::int_w(v)? as u32,
                None => host_reg::KEY_READ,
            },
        };
        let wide = WideCString::from_str_truncate(sub_key.as_deref().unwrap_or(""));
        let mut out: HKEY = core::ptr::null_mut();
        let rc = unsafe { host_reg::open_key_ex(key, &wide, 0, access, &mut out) };
        if rc != 0 {
            return Err(win_err(rc));
        }
        Ok(make_pyhkey(out))
    }

    fn close_key(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let obj = arg(args, 0)?;
        let raw = if unsafe { is_int(obj) } {
            (unsafe { w_int_get_value(obj) } as usize) as HKEY
        } else {
            take_handle(obj)
        };
        if !raw.is_null() {
            host_reg::close_key(raw);
        }
        Ok(w_none())
    }

    fn query_value(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        use rustpython_host_env::winreg::QueryStringError;
        let key = key_arg(args, 0)?;
        let sub_key = opt_str(args, 1)?;
        match host_reg::query_default_value(key, sub_key.as_deref().map(OsStr::new)) {
            Ok(value) => Ok(w_str_new(&value)),
            Err(QueryStringError::Code(code)) => Err(win_err(code)),
            Err(QueryStringError::Utf16(_)) => Err(crate::PyError::value_error(
                "registry value is not valid UTF-16",
            )),
        }
    }

    fn query_value_ex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let key = key_arg(args, 0)?;
        let name = opt_str(args, 1)?.unwrap_or_default();
        match host_reg::query_value_bytes(key, OsStr::new(&name)) {
            Ok((data, typ)) => Ok(w_tuple_new(vec![reg2py(&data, typ), w_int_new(typ as i64)])),
            Err(code) => Err(win_err(code)),
        }
    }

    fn enum_key(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let key = key_arg(args, 0)?;
        let index = req_int(args, 1)? as u32;
        // Registry key names are capped at 255 chars (winnt.h MAX_KEY_LENGTH).
        let mut buffer = [0u16; 257];
        let mut len = buffer.len() as u32;
        let rc = unsafe { host_reg::enum_key_ex(key, index, buffer.as_mut_ptr(), &mut len) };
        if rc != 0 {
            return Err(win_err(rc));
        }
        Ok(w_str_new(&String::from_utf16_lossy(
            &buffer[..len as usize],
        )))
    }

    fn enum_value(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let key = key_arg(args, 0)?;
        let index = req_int(args, 1)? as u32;
        let mut max_name = 0u32;
        let mut max_data = 0u32;
        let rc = unsafe {
            host_reg::query_info_key(
                key,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                &mut max_name,
                &mut max_data,
            )
        };
        if rc != 0 {
            return Err(win_err(rc));
        }
        let mut name = vec![0u16; (max_name + 1) as usize];
        let mut data = vec![0u8; max_data as usize];
        let mut name_len = name.len() as u32;
        let mut data_len = data.len() as u32;
        let mut typ = 0u32;
        let data_ptr = if data.is_empty() {
            core::ptr::null_mut()
        } else {
            data.as_mut_ptr()
        };
        let rc = unsafe {
            host_reg::enum_value(
                key,
                index,
                name.as_mut_ptr(),
                &mut name_len,
                &mut typ,
                data_ptr,
                &mut data_len,
            )
        };
        if rc != 0 {
            return Err(win_err(rc));
        }
        let name_s = String::from_utf16_lossy(&name[..name_len as usize]);
        Ok(w_tuple_new(vec![
            w_str_new(&name_s),
            reg2py(&data[..data_len as usize], typ),
            w_int_new(typ as i64),
        ]))
    }

    fn query_info_key(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let key = key_arg(args, 0)?;
        match host_reg::query_info_key_full(key) {
            Ok(info) => Ok(w_tuple_new(vec![
                w_int_new(info.sub_keys as i64),
                w_int_new(info.values as i64),
                w_int_new(info.last_write_time as i64),
            ])),
            Err(code) => Err(win_err(code)),
        }
    }

    fn flush_key(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        check(host_reg::flush_key(key_arg(args, 0)?))
    }

    fn create_key(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let key = key_arg(args, 0)?;
        let sub_key = opt_str(args, 1)?;
        let wide = WideCString::from_str_truncate(sub_key.as_deref().unwrap_or(""));
        let mut out: HKEY = core::ptr::null_mut();
        let rc = unsafe { host_reg::create_key(key, &wide, &mut out) };
        if rc != 0 {
            return Err(win_err(rc));
        }
        Ok(make_pyhkey(out))
    }

    fn set_value(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        // (key, sub_key, type, value) — SetValue only writes REG_SZ.
        let key = key_arg(args, 0)?;
        let sub_key = opt_str(args, 1)?.unwrap_or_default();
        let typ = req_int(args, 2)? as u32;
        if typ != host_reg::REG_SZ {
            return Err(crate::PyError::type_error("Type must be winreg.REG_SZ"));
        }
        let value = req_str(args, 3)?;
        check(host_reg::set_default_value(
            key,
            OsStr::new(&sub_key),
            typ,
            OsStr::new(&value),
        ))
    }

    fn set_value_ex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        // (key, value_name, reserved, type, value)
        let key = key_arg(args, 0)?;
        let name = opt_str(args, 1)?;
        let typ = req_int(args, 3)? as u32;
        let data = py2reg(arg(args, 4)?, typ)?;
        let wide_name = name.as_deref().map(WideCString::from_str_truncate);
        let rc = unsafe {
            host_reg::set_value_ex(
                key,
                wide_name.as_deref(),
                typ,
                data.as_ptr(),
                data.len() as u32,
            )
        };
        if rc != 0 {
            return Err(win_err(rc));
        }
        Ok(w_none())
    }

    fn delete_key(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let key = key_arg(args, 0)?;
        let sub_key = req_str(args, 1)?;
        let wide = WideCString::from_str_truncate(&sub_key);
        check(unsafe { host_reg::delete_key(key, &wide) })
    }

    fn delete_value(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let key = key_arg(args, 0)?;
        let name = opt_str(args, 1)?;
        let wide_name = name.as_deref().map(WideCString::from_str_truncate);
        check(unsafe { host_reg::delete_value(key, wide_name.as_deref()) })
    }

    fn connect_registry(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        // (computer_name, key)
        let computer = opt_str(args, 0)?;
        let key = key_arg(args, 1)?;
        let wide = computer.as_deref().map(WideCString::from_str_truncate);
        let mut out: HKEY = core::ptr::null_mut();
        let rc = unsafe { host_reg::connect_registry(wide.as_deref(), key, &mut out) };
        if rc != 0 {
            return Err(win_err(rc));
        }
        Ok(make_pyhkey(out))
    }

    fn expand_environment_strings(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        use rustpython_host_env::winreg::ExpandEnvironmentStringsError;
        let input = req_str(args, 0)?;
        match host_reg::expand_environment_strings(OsStr::new(&input)) {
            Ok(value) => Ok(w_str_new(&value)),
            Err(ExpandEnvironmentStringsError::Os) => Err(win_err(
                std::io::Error::last_os_error().raw_os_error().unwrap_or(0) as u32,
            )),
            Err(ExpandEnvironmentStringsError::Utf16(_)) => Err(crate::PyError::value_error(
                "expanded value is not valid UTF-16",
            )),
        }
    }

    fn disable_reflection_key(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        check(host_reg::disable_reflection_key(key_arg(args, 0)?))
    }

    fn enable_reflection_key(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        check(host_reg::enable_reflection_key(key_arg(args, 0)?))
    }

    fn query_reflection_key(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let key = key_arg(args, 0)?;
        let mut disabled: i32 = 0;
        let rc = unsafe { host_reg::query_reflection_key(key, &mut disabled) };
        if rc != 0 {
            return Err(win_err(rc));
        }
        Ok(w_bool_from(disabled != 0))
    }

    pub fn install(ns: PyObjectRef) {
        crate::module_ns_store(ns, "PyHKEY", type_object());
        // The predefined roots are exposed as their full (sign-extended)
        // pointer value, matching `PyLong_FromVoidPtr` — this overrides the
        // 32-bit fallbacks the always-present `int_constants` set.
        for (name, root) in [
            ("HKEY_CLASSES_ROOT", host_reg::HKEY_CLASSES_ROOT),
            ("HKEY_CURRENT_USER", host_reg::HKEY_CURRENT_USER),
            ("HKEY_LOCAL_MACHINE", host_reg::HKEY_LOCAL_MACHINE),
            ("HKEY_USERS", host_reg::HKEY_USERS),
            ("HKEY_PERFORMANCE_DATA", host_reg::HKEY_PERFORMANCE_DATA),
            ("HKEY_CURRENT_CONFIG", host_reg::HKEY_CURRENT_CONFIG),
            ("HKEY_DYN_DATA", host_reg::HKEY_DYN_DATA),
        ] {
            crate::module_ns_store(ns, name, int_from_ptr(root));
        }
        for (name, func) in [
            (
                "OpenKey",
                open_key as fn(&[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>,
            ),
            ("OpenKeyEx", open_key),
            ("CloseKey", close_key),
            ("QueryValue", query_value),
            ("QueryValueEx", query_value_ex),
            ("EnumKey", enum_key),
            ("EnumValue", enum_value),
            ("QueryInfoKey", query_info_key),
            ("FlushKey", flush_key),
            ("CreateKey", create_key),
            ("SetValue", set_value),
            ("SetValueEx", set_value_ex),
            ("DeleteKey", delete_key),
            ("DeleteValue", delete_value),
            ("ConnectRegistry", connect_registry),
            ("ExpandEnvironmentStrings", expand_environment_strings),
            ("DisableReflectionKey", disable_reflection_key),
            ("EnableReflectionKey", enable_reflection_key),
            ("QueryReflectionKey", query_reflection_key),
        ] {
            crate::module_ns_store(ns, name, crate::make_builtin_function(name, func));
        }
    }
}
