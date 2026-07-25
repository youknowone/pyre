//! winreg module — PyPy: pypy/module/_winreg/ (`applevel_name = 'winreg'`).
//!
//! `importlib._bootstrap_external` does an eager `import winreg` on every
//! `sys.platform == "win32"` build, so the module must exist for the import
//! machinery — and therefore `import site` — to come up.  Its one bootstrap
//! consumer, `WindowsRegistryFinder`, is deprecated and never installed onto
//! `sys.meta_path`, and touches `winreg.OpenKey`/`QueryValue` only inside its
//! (unreached) methods, so the registry-access functions are not ported here;
//! this exposes the integer constants a caller reads, leaving the RegOpenKeyEx
//! family for a follow-up.
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
    }
}
