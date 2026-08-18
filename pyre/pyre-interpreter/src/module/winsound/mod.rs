//! winsound module — the Windows sound calls.
//!
//! `PlaySound`, `Beep` and `MessageBeep` are the whole surface, over
//! `PlaySoundW` (winmm), `Beep` (kernel32) and `MessageBeep` (user32).  PyPy
//! has no counterpart, so the shapes below follow `PC/winsound.c`.
//!
//! `sound` takes its own converter rather than the shared path one: this call
//! wants the name as wide characters, and it tells a `bytes` argument apart
//! from a `__fspath__` that answered with `bytes` — the filesystem-encoded
//! bytes the shared converter produces have forgotten which it was.

use pyre_object::PyObjectRef;

/// Play from a buffer rather than from a name.
const SND_MEMORY: i32 = 0x0004;
/// Return before the sound finishes.
const SND_ASYNC: i32 = 0x0001;

/// The wide, NUL-terminated name `PlaySoundW` takes.
///
/// `bytes` is turned away even though it names a file elsewhere: the argument
/// is a buffer under `SND_MEMORY`, and accepting it here as a name too would
/// make the same value mean two things.
fn sound_name(sound: PyObjectRef) -> Result<Vec<u16>, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let sound_slot = roots.base();
    roots.pin_root(sound);

    let w_name = if unsafe { pyre_object::is_str(sound) } {
        roots.get(sound_slot)
    } else if unsafe { pyre_object::bytesobject::is_bytes(sound) } {
        return Err(crate::PyError::type_error(
            "'sound' must be str, os.PathLike, or None, not bytes",
        ));
    } else {
        // `type(sound).__fspath__(sound)` — the descriptor read off the type is
        // unbound, so the object is supplied as the sole argument.
        let Some(fspath_fn) = crate::typedef::r#type(sound).and_then(|pt| unsafe {
            crate::baseobjspace::lookup_in_type(pt.as_ptr(), "__fspath__")
        }) else {
            return Err(crate::PyError::type_error(format!(
                "expected str, bytes or os.PathLike object, not {}",
                crate::gateway::short_type_name(sound)
            )));
        };
        let fspath_slot = pyre_object::gc_roots::shadow_stack_len();
        roots.pin_root(fspath_fn);
        let resolved = crate::call::call_function_impl_result(
            roots.get(fspath_slot),
            &[roots.get(sound_slot)],
        )?;
        let resolved_slot = pyre_object::gc_roots::shadow_stack_len();
        roots.pin_root(resolved);
        let resolved = roots.get(resolved_slot);
        if !unsafe { pyre_object::is_str(resolved) } {
            return Err(crate::PyError::type_error(format!(
                "'sound' must resolve to str, not {}",
                crate::gateway::short_type_name(resolved)
            )));
        }
        resolved
    };

    // The name goes to the host as spelled, unpaired surrogates included: it
    // takes UTF-16, and a lossy re-decode would address a different file.
    let mut units: Vec<u16> = unsafe { pyre_object::w_str_get_wtf8(w_name) }
        .encode_wide()
        .collect();
    if units.contains(&0) {
        return Err(crate::PyError::value_error("embedded null character"));
    }
    units.push(0);
    Ok(units)
}

crate::py_module! {
    "winsound",
    int_constants: {
        // `PlaySound` flags (mmsystem.h).  SND_SYNC is the absence of
        // SND_ASYNC rather than a bit of its own.
        "SND_SYNC" => 0x0000,
        "SND_ASYNC" => SND_ASYNC,
        "SND_NODEFAULT" => 0x0002,
        "SND_MEMORY" => SND_MEMORY,
        "SND_LOOP" => 0x0008,
        "SND_NOSTOP" => 0x0010,
        "SND_PURGE" => 0x0040,
        "SND_APPLICATION" => 0x0080,
        "SND_NOWAIT" => 0x0000_2000,
        "SND_ALIAS" => 0x0001_0000,
        "SND_FILENAME" => 0x0002_0000,
        "SND_SENTRY" => 0x0008_0000,
        "SND_SYSTEM" => 0x0020_0000,
        // `MessageBeep` sound ids (winuser.h).  Several are aliases: HAND,
        // ERROR and STOP are one sound, as are EXCLAMATION and WARNING, and
        // ASTERISK and INFORMATION.
        "MB_OK" => 0x0000,
        "MB_ICONHAND" => 0x0010,
        "MB_ICONERROR" => 0x0010,
        "MB_ICONSTOP" => 0x0010,
        "MB_ICONQUESTION" => 0x0020,
        "MB_ICONEXCLAMATION" => 0x0030,
        "MB_ICONWARNING" => 0x0030,
        "MB_ICONASTERISK" => 0x0040,
        "MB_ICONINFORMATION" => 0x0040,
    },
    inline_functions: {
        fn PlaySound(sound: PyObjectRef, flags: PyIndexCInt) -> Result<(), crate::PyError> {
            // `None` answers first, so it silences the device whatever else
            // the flags asked for.
            let name = if unsafe { pyre_object::is_none(sound) } {
                None
            } else if flags & SND_MEMORY != 0 {
                if flags & SND_ASYNC != 0 {
                    // The buffer would have to outlive the call, and nothing
                    // here can keep it alive that long.
                    return Err(crate::PyError::runtime_error(
                        "Cannot play asynchronously from memory",
                    ));
                }
                let data = crate::baseobjspace::simple_buffer_bytes(sound)?.ok_or_else(|| {
                    crate::PyError::type_error(format!(
                        "a bytes-like object is required, not '{}'",
                        crate::gateway::short_type_name(sound)
                    ))
                })?;
                // `PlaySoundW` takes the buffer through its `LPCWSTR`
                // parameter, so the bytes are repacked into wide units with
                // their order kept.  An odd length gains a trailing zero
                // byte, which the RIFF header's own length keeps out of the
                // sound.
                let bytes = data.as_bytes().to_vec();
                data.release();
                // `PlaySoundW` reads a header at the pointer it is given, and
                // an empty `Vec`'s pointer is the alignment sentinel rather
                // than storage, so an empty buffer is backed by one zeroed
                // unit: the header fails to parse and the call answers the
                // failure an unplayable sound answers.
                let mut units = vec![0u16; bytes.len().div_ceil(2).max(1)];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        units.as_mut_ptr() as *mut u8,
                        bytes.len(),
                    );
                }
                Some(units)
            } else {
                Some(sound_name(sound)?)
            };
            let played = {
                // A synchronous play runs to the end of the sound, so the
                // thread that will stop it has to be able to run.
                let _blocked = crate::module::thread::before_external_block();
                unsafe {
                    windows_sys::Win32::Media::Audio::PlaySoundW(
                        name.as_ref().map_or(std::ptr::null(), |units| units.as_ptr()),
                        std::ptr::null_mut(),
                        flags as u32,
                    )
                }
            };
            if played == 0 {
                return Err(crate::PyError::runtime_error("Failed to play sound"));
            }
            Ok(())
        }
        fn Beep(frequency: PyIndexCInt, duration: PyIndexCInt) -> Result<(), crate::PyError> {
            // The range `Beep` itself accepts; naming one outside it is a
            // ValueError rather than a failed call.
            if !(37..=32767).contains(&frequency) {
                return Err(crate::PyError::value_error(
                    "frequency must be in 37 thru 32767",
                ));
            }
            let beeped = {
                let _blocked = crate::module::thread::before_external_block();
                unsafe {
                    windows_sys::Win32::System::Diagnostics::Debug::Beep(
                        frequency as u32,
                        duration as u32,
                    )
                }
            };
            if beeped == 0 {
                return Err(crate::PyError::runtime_error("Failed to beep"));
            }
            Ok(())
        }
        // The BOOL is dropped: a sound id the system has no sound for is not
        // an error, it just plays nothing.
        fn MessageBeep(#[default(0i32)] type_: PyIndexCInt) {
            unsafe {
                windows_sys::Win32::System::Diagnostics::Debug::MessageBeep(type_ as u32);
            }
        }
    }
}
