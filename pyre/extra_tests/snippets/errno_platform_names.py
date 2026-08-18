# pyre-check: gate=1
# `interp_errno.py` declares one name list for every platform and drops only
# the entries the host does not define, so these names are absent solely where
# the platform lacks them.  `from errno import ESTALE` is an unconditional
# import in widely used packages.
import errno
import sys

# Defined by the C runtime on every platform pyre builds for.
for name in ('EILSEQ', 'ECANCELED', 'ENOTSUP', 'EOWNERDEAD', 'ENOTRECOVERABLE'):
    assert hasattr(errno, name), name

if sys.platform != 'win32':
    # Present through libc on posix; on windows these reach the module through
    # the WSA-derived aliases, which this snippet does not assert because the
    # reference interpreter does not carry all of them there.
    for name in (
        'ESTALE',
        'EUSERS',
        'EREMOTE',
        'ETOOMANYREFS',
        'EPFNOSUPPORT',
        'ESOCKTNOSUPPORT',
    ):
        assert hasattr(errno, name), name

if sys.platform == 'darwin':
    for name in (
        'ENOATTR',
        'EAUTH',
        'EBADARCH',
        'EBADEXEC',
        'EBADMACHO',
        'EBADRPC',
        'EDEVERR',
        'EFTYPE',
        'ENEEDAUTH',
        'ENOPOLICY',
        'EPROCLIM',
        'EPROCUNAVAIL',
        'EPROGMISMATCH',
        'EPROGUNAVAIL',
        'EPWROFF',
        'EQFULL',
        'ERPCMISMATCH',
        'ESHLIBVERS',
    ):
        assert hasattr(errno, name), name

# Every exported code is reachable through `errorcode`, which is what makes
# `os.strerror(errno.errorcode[...])` and the traceback formatting work.
for attribute, value in vars(errno).items():
    if attribute.isupper():
        assert value in errno.errorcode, attribute
