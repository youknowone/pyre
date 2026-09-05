use std::ffi::OsString;
use std::path::Path;

/// Spell a native host path in the POSIX path grammar used by wasm32.
///
/// A Windows drive path is not absolute to `std::path::Path` on wasm32. Give
/// it a leading slash and normalize separators while it is in the guest, then
/// remove that transport-only slash in `guest_path_to_host`.
pub(crate) fn host_path_to_guest(path: &Path) -> Option<Vec<u8>> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        Some(path.as_os_str().as_bytes().to_vec())
    }
    #[cfg(windows)]
    {
        let path = path.to_str()?.replace('\\', "/");
        let drive_absolute = path.as_bytes().first().is_some_and(u8::is_ascii_alphabetic)
            && path.as_bytes().get(1) == Some(&b':')
            && path.as_bytes().get(2) == Some(&b'/');
        Some(match drive_absolute {
            true => format!("/{path}").into_bytes(),
            false => path.into_bytes(),
        })
    }
}

/// Decode the POSIX spelling used by the wasm guest into a native host path.
pub(crate) fn guest_path_to_host(bytes: Vec<u8>) -> Option<OsString> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        Some(OsString::from_vec(bytes))
    }
    #[cfg(windows)]
    {
        let mut path = String::from_utf8(bytes).ok()?;
        let transport_drive = path.as_bytes().first() == Some(&b'/')
            && path.as_bytes().get(1).is_some_and(u8::is_ascii_alphabetic)
            && path.as_bytes().get(2) == Some(&b':')
            && (path.len() == 3 || path.as_bytes().get(3) == Some(&b'/'));
        if transport_drive {
            path.remove(0);
            if path.len() == 2 {
                path.push('/');
            }
        }
        Some(OsString::from(path.replace('/', "\\")))
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::{guest_path_to_host, host_path_to_guest};
    use std::path::Path;

    #[test]
    fn windows_drive_path_round_trips_through_guest_absolute_path() {
        let host = Path::new(r"Z:\pyre\lib-python\3");
        let guest = host_path_to_guest(host).unwrap();
        assert_eq!(guest, b"/Z:/pyre/lib-python/3");
        assert_eq!(guest_path_to_host(guest).unwrap(), host.as_os_str());
    }

    #[test]
    fn windows_relative_path_round_trips() {
        let host = Path::new(r"build\fixture.py");
        let guest = host_path_to_guest(host).unwrap();
        assert_eq!(guest, b"build/fixture.py");
        assert_eq!(guest_path_to_host(guest).unwrap(), host.as_os_str());
    }

    #[test]
    fn windows_unc_path_is_guest_absolute() {
        let host = Path::new(r"\\server\share\fixture.py");
        let guest = host_path_to_guest(host).unwrap();
        assert_eq!(guest, b"//server/share/fixture.py");
        assert_eq!(guest_path_to_host(guest).unwrap(), host.as_os_str());
    }

    #[test]
    fn guest_drive_component_decodes_as_host_drive_root() {
        assert_eq!(
            guest_path_to_host(b"/Z:".to_vec()).unwrap(),
            Path::new(r"Z:\").as_os_str()
        );
    }
}
