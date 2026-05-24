//! termios module — PyPy: pypy/module/termios/
//!
//! `tcgetattr(fd)` returns the 7-list `[iflag, oflag, cflag, lflag,
//! ispeed, ospeed, [cc_chars]]`.  `tcsetattr(fd, when, attrs)` takes the
//! same shape and writes it back via `termios::Termios`.

pub mod interp_termios;
pub mod moduledef;
