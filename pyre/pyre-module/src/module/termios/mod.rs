//! termios module — PyPy: pypy/module/termios/
//!
//! `tcgetattr(fd)` returns the 7-list `[iflag, oflag, cflag, lflag,
//! ispeed, ospeed, [cc_chars]]`.  `tcsetattr(fd, when, attrs)` takes the
//! same shape and writes it back through `rtermios.tcsetattr`.

pyre_interpreter::pyre_module_init!(interp_termios);
