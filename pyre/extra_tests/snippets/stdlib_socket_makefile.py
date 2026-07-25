import socket


with socket.socket() as listener:
    assert type(listener) is socket.socket
    assert hasattr(listener, "makefile")
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    with socket.socket() as client:
        client.connect(listener.getsockname())
        server, _ = listener.accept()
        with server:
            assert type(server) is socket.socket
            raw = server.makefile("rb", buffering=0)
            client.sendall(b"ping")
            assert raw.read(4) == b"ping"
            raw.close()

# Exercise pack_inet_addr's hostname path, which must use a re-entrant
# resolver because socketserver and logging handlers call it from workers.
with socket.socket() as sock:
    assert isinstance(sock.connect_ex(("localhost", 9)), int)

print("stdlib socket makefile ok")
