import _imp
import os
import platform
import sys
import struct
from shutil import which

try:
    so_ext = _imp.extension_suffixes()[0]
except IndexError:
    # EXT_SUFFIX is build ABI metadata, and remains useful to wheel tooling
    # when this interpreter build has no native-extension loader.  CPython's
    # sysconfig tests likewise distinguish an absent loader from an absent
    # build suffix.  Keep PyPy's `<impl>-<abi>-<multiarch>` shape.
    multiarch = sys.implementation._multiarch
    shared_ext = '.pyd' if sys.platform == 'win32' else '.so'
    so_ext = '.pyre314-pyre0%s%s' % (
        ('-' + multiarch) if multiarch else '', shared_ext)

pydot = '%d.%d' % sys.version_info[:2]

build_time_vars = {
    'ABIFLAGS': '',
    # SOABI is PEP 3149 compliant, but CPython3 has so_ext.split('.')[1]
    # ("ABI tag"-"platform tag") where this is ABI tag only. Wheel 0.34.2
    # depends on this value, so don't make it CPython compliant without
    # checking wheel: it uses pep425tags.get_abi_tag with special handling
    # for CPython
    "SOABI": '-'.join(so_ext.split('.')[1].split('-')[:2]),
    "SO": so_ext,  # deprecated in Python 3, for backward compatibility
    'MULTIARCH': sys.implementation._multiarch,
    'CC': "cc -pthread",
    'CXX': "c++ -pthread",
    'OPT': "-DNDEBUG -O2",
    'CFLAGS': "-DNDEBUG -O2",
    'CCSHARED': "-fPIC",
    'LDFLAGS': "-Wl,-Bsymbolic-functions",
    'LDSHARED': "cc -pthread -shared -Wl,-Bsymbolic-functions",
    'LDCXXSHARED': "c++ -shared -Wl,-O1 -Wl,-Bsymbolic-functions",
    'EXT_SUFFIX': so_ext,
    'SHLIB_SUFFIX': ".so",
    'AR': "ar",
    'ARFLAGS': "rc",
    'EXE': "",
    'VERSION': pydot,
    'LDVERSION': pydot,
    'Py_DEBUG': 0,  # cpyext never uses this
    'Py_GIL_DISABLED': 0,
    'Py_ENABLE_SHARED': 0,  # if 1, will add python so to link like -lpython3.7
    # Pyre currently has neither a CPython-compatible C API nor a separately
    # linkable runtime library.  Keep build ABI metadata above for wheel tags,
    # but never invent files that are absent from the installation.
    'LIBRARY': '',
    'LDLIBRARY': '',
    'LIBPYTHON': '',
    'INCLUDEPY': '',
    'CONFINCLUDEPY': '',
    'LIBDIR': '',
    'SIZEOF_VOID_P': struct.calcsize("P"),
    # CPython 3.14's relocation check reads these from the generated data.
    'prefix': sys.base_prefix,
    'exec_prefix': sys.base_exec_prefix,
    'srcdir': sys.base_prefix,
}

# Keep PyPy's relocatable zoneinfo search rooted at base_prefix.  The C-runtime
# path block above intentionally differs: PyPy ships libpypy beside its binary,
# whereas Pyre has no separate library to name.
mybase = sys.base_prefix
if sys.platform != 'win32':
    # try paths relative to sys.base_prefix first
    tzpaths = [
        os.path.join(mybase, 'share', 'zoneinfo'),
        os.path.join(mybase, 'lib', 'zoneinfo'),
        os.path.join(mybase, 'share', 'lib', 'zoneinfo'),
        os.path.join(mybase, '..', 'etc', 'zoneinfo'),
    ]
    # add absolute system paths if sys.base_prefix != "/usr"
    # (then we'd be adding duplicates)
    if mybase != '/usr':
        tzpaths.extend([
            '/usr/share/zoneinfo',
            '/usr/lib/zoneinfo',
            '/usr/share/lib/zoneinfo',
            '/etc/zoneinfo',
        ])
    build_time_vars['TZPATH'] = ':'.join(tzpaths)

if which("gcc"):
    build_time_vars.update({
        "CC": "gcc -pthread",
        "GNULD": "yes",
        "LDSHARED": "gcc -pthread -shared" + " " + build_time_vars["LDFLAGS"] ,
    })
    if which("g++"):
        build_time_vars["CXX"] = "g++ -pthread"

if sys.platform[:6] == "darwin":
    arch = platform.machine()
    build_time_vars['CC'] += ' -arch %s' % (arch,)
    build_time_vars["LDFLAGS"] = "-undefined dynamic_lookup"
    build_time_vars["LDSHARED"] = "clang -bundle -undefined dynamic_lookup "
    build_time_vars["LDCXXSHARED"] = "clang++ -bundle -undefined dynamic_lookup "
    # scikit-build checks this, it is left over from the NextStep rld linker
    build_time_vars['WITH_DYLD'] = 1
    if "CXX" in build_time_vars:
        build_time_vars['CXX'] += ' -arch %s' % (arch,)
    # This was added to solve problems that may have been
    # solved elsewhere. Can we remove it? See cibuildwheel PR 185 and
    # pypa/wheel. Need to check: interaction with build_cffi_imports.py
    #
    # In any case, keep this in sync with DARWIN_VERSION_MIN in
    # rpython/translator/platform/darwin.py and Lib/_osx_support.py
    if arch == "arm64":
        build_time_vars['MACOSX_DEPLOYMENT_TARGET'] = '11.0'
    else:
        build_time_vars['MACOSX_DEPLOYMENT_TARGET'] = '10.15'
