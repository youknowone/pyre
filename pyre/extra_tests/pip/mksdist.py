"""Write the `stpkg` source distribution.

Run by the interpreter under test rather than by the driver, so the archive is
one the runtime produced: it goes through `tarfile` over `gzip` over `zlib`,
and the driver reads the result back with its own `tarfile` before pip is
allowed near it.  A tarball that only the writer can open would otherwise
surface as an unrelated failure inside the build.
"""

import io
import os
import sys
import tarfile
import time

DIST = "stpkg-0.2.0"
PKG_INFO = b"Metadata-Version: 2.1\nName: stpkg\nVersion: 0.2.0\n"


def main(source, destination):
    parent = os.path.dirname(destination)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for name in ("pyproject.toml", "stpkg.py"):
            archive.add(os.path.join(source, name), arcname=f"{DIST}/{name}")
        info = tarfile.TarInfo(f"{DIST}/PKG-INFO")
        info.size = len(PKG_INFO)
        info.mtime = int(time.time())
        archive.addfile(info, io.BytesIO(PKG_INFO))
    print(destination)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
