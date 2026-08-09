import os


# pyre's socket type and the stdlib wrapper it enables are currently a POSIX
# module (`interp_socket.rs` registers `socket` under `#[cfg(unix)]`).
if os.name != "posix":
    print("OK")
    raise SystemExit


import _socket  # noqa: E402
import socket  # noqa: E402


assert _socket.timeout is TimeoutError
assert socket.timeout is TimeoutError
assert socket.timeout.__module__ == "builtins"

print("OK")
