import _socket
import socket


assert _socket.timeout is TimeoutError
assert socket.timeout is TimeoutError
assert socket.timeout.__module__ == "builtins"

print("OK")
