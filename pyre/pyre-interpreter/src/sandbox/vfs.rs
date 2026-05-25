use indexmap::IndexMap;
use std::cell::Cell;
use std::fs;
use std::io::{self, Cursor, Read};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

// vfs.py mirrors stat-module bitmasks that are POSIX-standard regardless of
// host OS, so the constants are hard-coded here rather than pulled from
// libc (which omits these symbols on Windows targets).
pub type Mode = u32;
pub type FsNode = Rc<dyn FSObject>;
pub type VfsResult<T> = Result<T, VfsError>;

// vfs.py:4
pub const UID: u32 = 1000;
// vfs.py:5
pub const GID: u32 = 1000;
// vfs.py:6
pub const ATIME: i64 = 0;
// vfs.py:6
pub const MTIME: i64 = 0;
// vfs.py:6
pub const CTIME: i64 = 0;
// vfs.py:7
static INO_COUNTER: AtomicU64 = AtomicU64::new(0);

const S_IFDIR: Mode = 0o040000;
const S_IFREG: Mode = 0o100000;
const S_IFMT: Mode = 0o170000;
const S_IWUSR: Mode = 0o000200;
const S_IRUSR: Mode = 0o000400;
const S_IRGRP: Mode = 0o000040;
const S_IROTH: Mode = 0o000004;
const S_IXUSR: Mode = 0o000100;
const S_IXGRP: Mode = 0o000010;
const S_IXOTH: Mode = 0o000001;
const S_IRWXO: Mode = 0o000007;
const S_IRWXU: Mode = 0o000700;
const S_IRWXG: Mode = 0o000070;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatResult {
    pub st_mode: Mode,
    pub st_ino: u64,
    pub st_dev: u64,
    pub st_nlink: u64,
    pub st_uid: u32,
    pub st_gid: u32,
    pub st_size: u64,
    pub st_atime: i64,
    pub st_mtime: i64,
    pub st_ctime: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VfsError {
    pub errno: i32,
    pub object: String,
}

#[derive(Debug, Default)]
pub struct FSObjectState {
    st_ino: Cell<Option<u64>>,
}

// vfs.py:10
pub trait FSObject {
    // Rust support for vfs.py:15
    fn state(&self) -> &FSObjectState;

    // Rust support for vfs.py:23
    fn kind(&self) -> Mode;

    // vfs.py:11
    fn read_only(&self) -> bool {
        true
    }

    // vfs.py:13
    fn stat(&self) -> VfsResult<StatResult> {
        let st_ino = match self.state().st_ino.get() {
            Some(st_ino) => st_ino,
            None => {
                let st_ino = INO_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
                self.state().st_ino.set(Some(st_ino));
                st_ino
            }
        };
        let st_dev = 1;
        let st_nlink = 1;
        let st_size = self.getsize()?;
        let mut st_mode = self.kind();
        st_mode |= S_IWUSR | S_IRUSR | S_IRGRP | S_IROTH;
        if is_dir(self.kind()) {
            st_mode |= S_IXUSR | S_IXGRP | S_IXOTH;
        }
        let (st_uid, st_gid) = if self.read_only() { (0, 0) } else { (UID, GID) };
        Ok(StatResult {
            st_mode,
            st_ino,
            st_dev,
            st_nlink,
            st_uid,
            st_gid,
            st_size,
            st_atime: ATIME,
            st_mtime: MTIME,
            st_ctime: CTIME,
        })
    }

    // vfs.py:40
    fn access(&self, mode: Mode) -> VfsResult<bool> {
        let s = self.stat()?;
        let mut e_mode = s.st_mode & S_IRWXO;
        if UID == s.st_uid {
            e_mode |= (s.st_mode & S_IRWXU) >> 6;
        }
        if GID == s.st_gid {
            e_mode |= (s.st_mode & S_IRWXG) >> 3;
        }
        Ok((e_mode & mode) == mode)
    }

    // vfs.py:49
    fn keys(&self) -> VfsResult<Vec<String>> {
        Err(VfsError {
            errno: libc::ENOTDIR,
            object: "self".to_owned(),
        })
    }

    // vfs.py:52
    fn open(&self) -> VfsResult<Box<dyn Read>> {
        Err(VfsError {
            errno: libc::EACCES,
            object: "self".to_owned(),
        })
    }

    // vfs.py:55
    fn getsize(&self) -> VfsResult<u64> {
        Ok(0)
    }
}

// vfs.py:59
#[derive(Default)]
pub struct Dir {
    state: FSObjectState,
    pub entries: IndexMap<String, FsNode>,
}

impl Dir {
    // vfs.py:61
    pub fn new(entries: IndexMap<String, FsNode>) -> Self {
        Self {
            state: FSObjectState::default(),
            entries,
        }
    }

    // vfs.py:65
    pub fn join(&self, name: &str) -> VfsResult<FsNode> {
        self.entries.get(name).cloned().ok_or_else(|| VfsError {
            errno: libc::ENOENT,
            object: name.to_owned(),
        })
    }
}

impl FSObject for Dir {
    // Rust support for vfs.py:15
    fn state(&self) -> &FSObjectState {
        &self.state
    }

    // vfs.py:60
    fn kind(&self) -> Mode {
        S_IFDIR
    }

    // vfs.py:63
    fn keys(&self) -> VfsResult<Vec<String>> {
        Ok(self.entries.keys().cloned().collect())
    }
}

// vfs.py:71
pub struct RealDir {
    state: FSObjectState,
    pub path: PathBuf,
    pub show_dotfiles: bool,
    pub follow_links: bool,
    pub exclude: Vec<String>,
}

impl RealDir {
    // vfs.py:79
    pub fn new(
        path: impl Into<PathBuf>,
        show_dotfiles: bool,
        follow_links: bool,
        exclude: Vec<String>,
    ) -> Self {
        Self {
            state: FSObjectState::default(),
            path: path.into(),
            show_dotfiles,
            follow_links,
            exclude: exclude
                .into_iter()
                .map(|excl| excl.to_lowercase())
                .collect(),
        }
    }

    // vfs.py:85
    pub fn repr(&self) -> String {
        format!("<RealDir {}>", self.path.display())
    }

    // vfs.py:94
    //
    // Sandbox-hardening deviation from RPython upstream
    // `rpython/translator/sandbox/vfs.py:94`, which does a bare
    // `os.path.join(self.path, name)`.  RPython's sandbox is a separate
    // trusted parent process that intercepts every syscall; pyre's VFS
    // runs in-process so we must reject any `name` that would escape the
    // base directory.  Forbidden inputs: absolute paths (`PathBuf::join`
    // silently swaps the base when joined with an absolute), parent
    // traversal (`..`) and embedded path separators (single-component
    // child names only).  This keeps the join inside `self.path`
    // unconditionally — the rest of the sandbox depends on that
    // invariant.
    pub fn join(&self, name: &str) -> VfsResult<FsNode> {
        if name.is_empty()
            || name == ".."
            || name.contains(std::path::MAIN_SEPARATOR)
            || std::path::Path::new(name).is_absolute()
        {
            return Err(VfsError {
                errno: libc::ENOENT,
                object: name.to_owned(),
            });
        }
        if name.starts_with('.') && !self.show_dotfiles {
            return Err(VfsError {
                errno: libc::ENOENT,
                object: name.to_owned(),
            });
        }
        let lower_name = name.to_lowercase();
        for excl in &self.exclude {
            if lower_name.ends_with(excl) {
                return Err(VfsError {
                    errno: libc::ENOENT,
                    object: name.to_owned(),
                });
            }
        }

        let path = self.path.join(name);
        let st = if self.follow_links {
            fs::metadata(&path)
        } else {
            fs::symlink_metadata(&path)
        }
        .map_err(|err| io_error(err, path.display().to_string()))?;

        if st.file_type().is_dir() {
            Ok(Rc::new(RealDir::new(
                path,
                self.show_dotfiles,
                self.follow_links,
                self.exclude.clone(),
            )))
        } else if st.file_type().is_file() {
            Ok(Rc::new(RealFile::new(path, 0)))
        } else {
            Err(VfsError {
                errno: libc::EACCES,
                object: path.display().to_string(),
            })
        }
    }
}

impl FSObject for RealDir {
    // Rust support for vfs.py:15
    fn state(&self) -> &FSObjectState {
        &self.state
    }

    // vfs.py:60
    fn kind(&self) -> Mode {
        S_IFDIR
    }

    // vfs.py:87
    fn keys(&self) -> VfsResult<Vec<String>> {
        let mut names = Vec::new();
        for entry in fs::read_dir(&self.path)
            .map_err(|err| io_error(err, self.path.display().to_string()))?
        {
            let entry = entry.map_err(|err| io_error(err, self.path.display().to_string()))?;
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
        if !self.show_dotfiles {
            names.retain(|name| !name.starts_with('.'));
        }
        for excl in &self.exclude {
            names.retain(|name| !name.to_lowercase().ends_with(excl));
        }
        Ok(names)
    }
}

// vfs.py:115
pub struct File {
    state: FSObjectState,
    pub data: Vec<u8>,
}

impl File {
    // vfs.py:117
    pub fn new(data: impl AsRef<[u8]>) -> Self {
        Self {
            state: FSObjectState::default(),
            data: data.as_ref().to_vec(),
        }
    }
}

impl FSObject for File {
    // Rust support for vfs.py:15
    fn state(&self) -> &FSObjectState {
        &self.state
    }

    // vfs.py:116
    fn kind(&self) -> Mode {
        S_IFREG
    }

    // vfs.py:119
    fn getsize(&self) -> VfsResult<u64> {
        Ok(self.data.len() as u64)
    }

    // vfs.py:121
    fn open(&self) -> VfsResult<Box<dyn Read>> {
        Ok(Box::new(Cursor::new(self.data.clone())))
    }
}

// vfs.py:125
pub struct RealFile {
    state: FSObjectState,
    pub path: PathBuf,
    pub kind: Mode,
}

impl RealFile {
    // vfs.py:126
    pub fn new(path: impl Into<PathBuf>, mode: Mode) -> Self {
        Self {
            state: FSObjectState::default(),
            path: path.into(),
            kind: S_IFREG | mode,
        }
    }

    // vfs.py:129
    pub fn repr(&self) -> String {
        format!("<RealFile {}>", self.path.display())
    }
}

impl FSObject for RealFile {
    // Rust support for vfs.py:15
    fn state(&self) -> &FSObjectState {
        &self.state
    }

    // vfs.py:128
    fn kind(&self) -> Mode {
        self.kind
    }

    // vfs.py:131
    fn getsize(&self) -> VfsResult<u64> {
        fs::metadata(&self.path)
            .map(|st| st.len())
            .map_err(|err| io_error(err, self.path.display().to_string()))
    }

    // vfs.py:133
    fn open(&self) -> VfsResult<Box<dyn Read>> {
        fs::File::open(&self.path)
            .map(|file| Box::new(file) as Box<dyn Read>)
            .map_err(|err| io_error(err, self.path.display().to_string()))
    }
}

// vfs.py:25
fn is_dir(mode: Mode) -> bool {
    (mode & S_IFMT) == S_IFDIR
}

// vfs.py:134
fn io_error(err: io::Error, object: String) -> VfsError {
    VfsError {
        errno: err.raw_os_error().unwrap_or(libc::EIO),
        object,
    }
}
