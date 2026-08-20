/* The header a C extension includes.
 *
 * Nothing else here is meant to be included directly. The entry points
 * are declared in `pyre_decl.h`, written by `scripts/cpyext-abi.py`
 * from the exports themselves; the rest is the hand-written half --
 * the structs an extension lays out, and the macros it expands.
 */
/* The guard is the name CPython and PyPy both give it: an extension that
 * wants to know whether it has Python's headers at all tests `Py_PYTHON_H`,
 * and Cython's generated C refuses to compile without it. */
#ifndef Py_PYTHON_H
#define Py_PYTHON_H

#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>

/* `pyport.h` first: `patchlevel.h` declares one object, and `PyAPI_DATA` is
 * what declares it. */
#include "pyport.h"
#include "patchlevel.h"
#include "pymacro.h"
#include "pytypedefs.h"
#include "object.h"
#include "typeslots.h"
/* Name the types entry points below take and answer with by value:
   `PyThreadState` and `PyGILState_STATE`, and `Py_complex`. */
#include "pystate.h"
#include "lock.h"
#include "pythread.h"
#include "complexobject.h"

/* Every exported entry point. It sits here because the headers above
   name the types it uses, and the ones below define `static inline`
   functions that call it. */
#include "pyre_decl.h"
#include "refcount.h"
#include "pymem.h"
#include "objimpl.h"
#include "methodobject.h"
#include "structmember.h"
#include "moduleobject.h"
#include "pyerrors.h"
/* The `%`-format engine and the three variadic entry points over it. It
   follows `pyerrors.h` for the `PyExc_*` names it reports failures through. */
#include "pyre_format.h"
/* The variadic warning entry points, written over that engine. */
#include "warnings.h"
#include "bytesobject.h"
#include "unicodeobject.h"
#include "longobject.h"
#include "floatobject.h"
#include "tupleobject.h"
#include "listobject.h"
#include "dictobject.h"
#include "setobject.h"
#include "sliceobject.h"
#include "memoryobject.h"
#include "pycapsule.h"
#include "code.h"
#include "funcobject.h"
#include "frameobject.h"
#include "traceback.h"
#include "import.h"
#include "modsupport.h"
#include "audit.h"
#include "abstract.h"

#endif /* !Py_PYTHON_H */
