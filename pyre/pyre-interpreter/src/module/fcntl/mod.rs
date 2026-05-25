//! fcntl module — PyPy: pypy/module/fcntl/
//!
//! fcntl(fd, cmd, arg=0) / ioctl(fd, request, arg=0) / flock(fd, op) /
//! lockf(fd, cmd, len=0, start=0, whence=0).  Integer-argument forms
//! only.

crate::pyre_module_init!(interp_fcntl);
