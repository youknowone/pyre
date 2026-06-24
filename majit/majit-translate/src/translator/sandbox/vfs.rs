//! RPython `rpython/translator/sandbox/vfs.py`.

use std::borrow::Cow;
use std::cell::OnceCell;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

pub const UID: u32 = 1000;
pub const GID: u32 = 1000;
pub const ATIME: u64 = 0;
pub const MTIME: u64 = 0;
pub const CTIME: u64 = 0;

static INO_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatResult {
    pub st_mode: u32,
    pub st_ino: u64,
    pub st_dev: u64,
    pub st_nlink: u64,
    pub st_uid: u32,
    pub st_gid: u32,
    pub st_size: u64,
    pub st_atime: u64,
    pub st_mtime: u64,
    pub st_ctime: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VfsError {
    NotDir,
    Access,
    NoEnt(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FSObject {
    Dir(Dir),
    RealDir(RealDir),
    File(File),
    RealFile(RealFile),
}

impl FSObject {
    pub fn stat(&self) -> StatResult {
        let st_ino = *self
            .st_ino_cell()
            .get_or_init(|| INO_COUNTER.fetch_add(1, Ordering::Relaxed) + 1);
        let read_only = self.read_only();
        let st_mode = self.kind() | 0o644 | if self.is_dir() { 0o111 } else { 0 };
        let (st_uid, st_gid) = if read_only { (0, 0) } else { (UID, GID) };
        StatResult {
            st_mode,
            st_ino,
            st_dev: 1,
            st_nlink: 1,
            st_uid,
            st_gid,
            st_size: self.getsize(),
            st_atime: ATIME,
            st_mtime: MTIME,
            st_ctime: CTIME,
        }
    }

    pub fn access(&self, mode: u32) -> bool {
        let s = self.stat();
        let mut e_mode = s.st_mode & 0o007;
        if UID == s.st_uid {
            e_mode |= (s.st_mode & 0o700) >> 6;
        }
        if GID == s.st_gid {
            e_mode |= (s.st_mode & 0o070) >> 3;
        }
        (e_mode & mode) == mode
    }

    pub fn keys(&self) -> Result<Vec<String>, VfsError> {
        match self {
            FSObject::Dir(dir) => Ok(dir.keys()),
            FSObject::RealDir(dir) => dir.keys(),
            _ => Err(VfsError::NotDir),
        }
    }

    pub fn join(&self, name: &str) -> Result<Cow<'_, FSObject>, VfsError> {
        match self {
            FSObject::Dir(dir) => dir.join(name).map(Cow::Borrowed),
            FSObject::RealDir(dir) => dir.join(name).map(Cow::Owned),
            _ => Err(VfsError::NotDir),
        }
    }

    pub fn open(&self) -> Result<Vec<u8>, VfsError> {
        match self {
            FSObject::File(file) => Ok(file.data.clone()),
            FSObject::RealFile(file) => std::fs::read(&file.path).map_err(|_| VfsError::Access),
            _ => Err(VfsError::Access),
        }
    }

    pub fn getsize(&self) -> u64 {
        match self {
            FSObject::Dir(_) | FSObject::RealDir(_) => 0,
            FSObject::File(file) => file.data.len() as u64,
            FSObject::RealFile(file) => std::fs::metadata(&file.path).map(|m| m.len()).unwrap_or(0),
        }
    }

    fn kind(&self) -> u32 {
        match self {
            FSObject::Dir(_) | FSObject::RealDir(_) => 0o040000,
            FSObject::File(_) => 0o100000,
            FSObject::RealFile(file) => 0o100000 | file.mode,
        }
    }

    fn st_ino_cell(&self) -> &OnceCell<u64> {
        match self {
            FSObject::Dir(dir) => &dir.st_ino,
            FSObject::RealDir(dir) => &dir.st_ino,
            FSObject::File(file) => &file.st_ino,
            FSObject::RealFile(file) => &file.st_ino,
        }
    }

    fn is_dir(&self) -> bool {
        matches!(self, FSObject::Dir(_) | FSObject::RealDir(_))
    }

    fn read_only(&self) -> bool {
        true
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Dir {
    pub entries: BTreeMap<String, FSObject>,
    st_ino: OnceCell<u64>,
}

impl Dir {
    pub fn new(entries: BTreeMap<String, FSObject>) -> Self {
        Self {
            entries,
            st_ino: OnceCell::new(),
        }
    }

    pub fn keys(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    pub fn join(&self, name: &str) -> Result<&FSObject, VfsError> {
        self.entries
            .get(name)
            .ok_or_else(|| VfsError::NoEnt(name.to_string()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RealDir {
    pub path: PathBuf,
    pub show_dotfiles: bool,
    pub follow_links: bool,
    pub exclude: Vec<String>,
    st_ino: OnceCell<u64>,
}

impl RealDir {
    pub fn new(
        path: PathBuf,
        show_dotfiles: bool,
        follow_links: bool,
        exclude: Vec<String>,
    ) -> Self {
        Self {
            path,
            show_dotfiles,
            follow_links,
            exclude: exclude.into_iter().map(|s| s.to_lowercase()).collect(),
            st_ino: OnceCell::new(),
        }
    }

    pub fn keys(&self) -> Result<Vec<String>, VfsError> {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&self.path).map_err(|_| VfsError::Access)? {
            let entry = entry.map_err(|_| VfsError::Access)?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if !self.show_dotfiles && name.starts_with('.') {
                continue;
            }
            if self
                .exclude
                .iter()
                .any(|suffix| name.to_lowercase().ends_with(suffix))
            {
                continue;
            }
            names.push(name);
        }
        Ok(names)
    }

    pub fn join(&self, name: &str) -> Result<FSObject, VfsError> {
        if name.starts_with('.') && !self.show_dotfiles {
            return Err(VfsError::NoEnt(name.to_string()));
        }
        for excl in &self.exclude {
            if name.to_lowercase().ends_with(excl) {
                return Err(VfsError::NoEnt(name.to_string()));
            }
        }
        let path = self.path.join(name);
        let st = if self.follow_links {
            std::fs::metadata(&path)
        } else {
            std::fs::symlink_metadata(&path)
        }
        .map_err(|_| VfsError::NoEnt(name.to_string()))?;
        if st.is_dir() {
            Ok(FSObject::RealDir(RealDir::new(
                path,
                self.show_dotfiles,
                self.follow_links,
                self.exclude.clone(),
            )))
        } else if st.is_file() {
            Ok(FSObject::RealFile(RealFile::new(path, 0)))
        } else {
            // don't allow access to symlinks and other special files
            Err(VfsError::Access)
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct File {
    pub data: Vec<u8>,
    st_ino: OnceCell<u64>,
}

impl File {
    pub fn new(data: impl Into<Vec<u8>>) -> Self {
        Self {
            data: data.into(),
            st_ino: OnceCell::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RealFile {
    pub path: PathBuf,
    pub mode: u32,
    st_ino: OnceCell<u64>,
}

impl RealFile {
    pub fn new(path: PathBuf, mode: u32) -> Self {
        Self {
            path,
            mode,
            st_ino: OnceCell::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dir_join_and_keys_match_entries() {
        let mut entries = BTreeMap::new();
        entries.insert("x".to_string(), FSObject::File(File::new(b"abc".to_vec())));
        let dir = Dir::new(entries);

        assert_eq!(dir.keys(), vec!["x".to_string()]);
        assert_eq!(dir.join("x").unwrap().getsize(), 3);
        assert_eq!(
            dir.join("missing"),
            Err(VfsError::NoEnt("missing".to_string()))
        );
    }

    #[test]
    fn file_open_returns_data() {
        let file = FSObject::File(File::new(b"hello".to_vec()));
        assert_eq!(file.open().unwrap(), b"hello");
    }

    #[test]
    fn stat_caches_inode_per_node() {
        let file = FSObject::File(File::new(b"x".to_vec()));
        assert_eq!(file.stat().st_ino, file.stat().st_ino);
    }

    #[test]
    fn realfile_mode_appears_in_stat() {
        let exe = FSObject::RealFile(RealFile::new(PathBuf::from("/no/such/file"), 0o111));
        assert_eq!(exe.stat().st_mode & 0o111, 0o111);
        assert!(exe.access(1)); // X_OK granted via the executable mode bits

        let plain = FSObject::RealFile(RealFile::new(PathBuf::from("/no/such/file"), 0));
        assert!(!plain.access(1)); // no mode bits -> X_OK denied
    }

    #[test]
    fn fsobject_join_borrows_dir_entry() {
        let mut entries = BTreeMap::new();
        entries.insert("x".to_string(), FSObject::File(File::new(b"abc".to_vec())));
        let dir = FSObject::Dir(Dir::new(entries));

        assert_eq!(dir.join("x").unwrap().getsize(), 3);
        assert!(matches!(dir.join("missing"), Err(VfsError::NoEnt(_))));
    }

    #[test]
    fn realdir_join_filters_dotfiles_and_excludes() {
        let dir = RealDir::new(
            PathBuf::from("/tmp"),
            false,
            false,
            vec![".pyc".to_string()],
        );
        assert!(matches!(dir.join(".hidden"), Err(VfsError::NoEnt(_))));
        assert!(matches!(dir.join("mod.pyc"), Err(VfsError::NoEnt(_))));
    }
}
