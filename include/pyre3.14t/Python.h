/* The header a C extension includes.
 *
 * Nothing else here is meant to be included directly. The entry points
 * are declared in `pyre_decl.h`, written by `scripts/cpyext-abi.py`
 * from the exports themselves; the rest is the hand-written half --
 * the structs an extension lays out, and the macros it expands.
 */
#ifndef PYRE_PYTHON_H
#define PYRE_PYTHON_H

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

#include "patchlevel.h"
#include "pyport.h"
#include "pymacro.h"
#include "pytypedefs.h"
#include "object.h"
#include "typeslots.h"

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
#include "import.h"
#include "modsupport.h"
#include "abstract.h"

#endif /* !PYRE_PYTHON_H */
