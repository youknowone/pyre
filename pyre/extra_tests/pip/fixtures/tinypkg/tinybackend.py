"""A PEP 517 backend with no build dependencies, written on the stdlib alone.

Installing this fixture exercises the hook protocol itself -- the isolated
environment, `get_requires_for_build_wheel`, `build_wheel`, and the unpacking
of what it returns -- with nothing to resolve and nothing to download.  When it
passes and the `stpkg` fixture beside it does not, the defect is in what the
build environment installs rather than in the protocol.
"""

import base64
import hashlib
import os
import zipfile

NAME = "tinypkg"
VERSION = "0.1.0"
DIST = f"{NAME}-{VERSION}"
METADATA = f"Metadata-Version: 2.1\nName: {NAME}\nVersion: {VERSION}\n"
WHEEL = (
    "Wheel-Version: 1.0\n"
    "Generator: tinybackend\n"
    "Root-Is-Purelib: true\n"
    "Tag: py3-none-any\n"
)


def get_requires_for_build_wheel(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    info = os.path.join(metadata_directory, f"{DIST}.dist-info")
    os.makedirs(info, exist_ok=True)
    with open(os.path.join(info, "METADATA"), "w", encoding="utf-8") as out:
        out.write(METADATA)
    with open(os.path.join(info, "WHEEL"), "w", encoding="utf-8") as out:
        out.write(WHEEL)
    return f"{DIST}.dist-info"


def _record_line(name, payload):
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
    return f"{name},sha256={digest.rstrip(b'=').decode()},{len(payload)}\n"


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    filename = f"{DIST}-py3-none-any.whl"
    with open(os.path.join(os.path.dirname(__file__), "tinypkg.py"), "rb") as source:
        module = source.read()
    info = f"{DIST}.dist-info"
    entries = [
        ("tinypkg.py", module),
        (f"{info}/METADATA", METADATA.encode()),
        (f"{info}/WHEEL", WHEEL.encode()),
    ]
    # `RECORD` names itself with an empty hash, which is the one entry whose
    # digest cannot be taken before the file exists.
    record = "".join(_record_line(name, data) for name, data in entries)
    record += f"{info}/RECORD,,\n"
    with zipfile.ZipFile(os.path.join(wheel_directory, filename), "w") as wheel:
        for name, data in entries:
            wheel.writestr(name, data)
        wheel.writestr(f"{info}/RECORD", record)
    return filename


def build_sdist(sdist_directory, config_settings=None):
    raise NotImplementedError("tinypkg is installed from its directory")
