//! termios module — PyPy: pypy/module/termios/
//!
//! `tcgetattr(fd)` returns the 7-list `[iflag, oflag, cflag, lflag,
//! ispeed, ospeed, [cc_chars]]`.  `tcsetattr(fd, when, attrs)` takes the
//! same shape and writes it back via `termios::Termios`.

crate::pyre_module_init!(interp_termios);
