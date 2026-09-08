# pyre-check: gate=1
# `ssl3_read_n` takes a record's 5-byte header and then exactly the body its
# length names, so a TLS read never lifts more off the transport than the one
# record it is parsing.  That is what leaves whatever the peer wrote after its
# close_notify in the kernel, for the plain socket `unwrap()` hands back.  A
# reader that instead takes whatever the transport has drops those bytes into
# the TLS stream, where nothing can give them back -- the plaintext is simply
# lost, with no error to say so.
#
# The peer here is a memory-BIO server because a `wrap_socket` one cannot
# produce the input: TLS shutdown is bidirectional, so its `unwrap()` would
# block for the client's close_notify and could never put close_notify plus
# trailing plaintext on the wire ahead of the client's read.  Through
# `wrap_bio`, `unwrap()` emits close_notify into the outgoing BIO and raises
# instead of blocking, so the record, the close_notify and the trailing bytes
# all go out in one `sendall` before the client reads anything.  Nothing is
# left to timing.

import os
import socket
import ssl
import sys
import threading

CERT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "lib-python", "3", "test", "certdata", "keycert.pem",
)
RECORD = b"HELLO"
TRAILING = b"TRAILING"

failure = []


def serve(listener, sent):
    conn = None
    try:
        conn, _ = listener.accept()
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(CERT)
        incoming, outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
        tls = ctx.wrap_bio(incoming, outgoing, server_side=True)
        while True:
            try:
                tls.do_handshake()
                break
            except ssl.SSLWantReadError:
                pending = outgoing.read()
                if pending:
                    conn.sendall(pending)
                chunk = conn.recv(16384)
                if not chunk:
                    raise AssertionError("peer closed during the handshake")
                incoming.write(chunk)
        pending = outgoing.read()
        if pending:
            conn.sendall(pending)

        tls.write(RECORD)
        try:
            tls.unwrap()
        except ssl.SSLWantReadError:
            # Our close_notify is queued; the peer's cannot have arrived yet.
            pass
        # One write: the application record, the close_notify, and the bytes
        # that belong to the socket the client's `unwrap()` returns.
        conn.sendall(outgoing.read() + TRAILING)
        sent.set()
        # Stay open, so a client read that overshoots reports the loss rather
        # than an end of file.
        conn.recv(16384)
    except BaseException as error:  # noqa: BLE001 - reported through `failure`
        failure.append(error)
        sent.set()
    finally:
        if conn is not None:
            conn.close()


def run(use_msg_callback):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    sent = threading.Event()
    thread = threading.Thread(target=serve, args=(listener, sent), daemon=True)
    thread.start()
    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        if use_msg_callback:
            # A message callback takes the read off the pump that holds the
            # released region across the whole exchange and onto the path that
            # returns after every record; both stop at the same boundary.
            ctx._msg_callback = lambda *args: None
        raw = socket.create_connection(listener.getsockname())
        tls = ctx.wrap_socket(raw, server_hostname="localhost")
        try:
            assert sent.wait(30), "server never reached its write"
            assert not failure, f"server failed: {failure[0]!r}"
            assert tls.recv(len(RECORD)) == RECORD
            # `unwrap()` hands back the same descriptor, so the plain socket
            # is what gets closed from here on -- closing `tls` too would
            # close it twice.
            plain = tls.unwrap()
        except BaseException:
            tls.close()
            raise
        try:
            got = plain.recv(len(TRAILING))
        finally:
            plain.close()
        assert got == TRAILING, (
            f"the bytes after close_notify came back as {got!r}; a read that "
            f"crossed the record boundary took them into the TLS stream"
        )
    finally:
        listener.close()
        thread.join(30)


if not ssl.HAS_TLSv1_2 and not ssl.HAS_TLSv1_3:
    print("OK")
    sys.exit(0)

for use_msg_callback in (False, True):
    failure.clear()
    run(use_msg_callback)

print("OK")
