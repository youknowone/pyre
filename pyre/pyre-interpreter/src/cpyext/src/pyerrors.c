/* `PyErr_Format` and `PyErr_FormatUnraisable`: the format engine with the
 * message handed to `PyErr_SetObject` and to the unraisable hook.
 *
 * Compiled into the interpreter and exported from it, which is where
 * `pypy/module/cpyext/src/pyerrors.c` puts the same bodies.  Everything here is
 * built out of the non-variadic entry points `pyre_decl.h` declares, and the
 * headers carry the declarations alone -- an extension resolves these at load
 * time rather than compiling a copy of each.
 */
#include "Python.h"

/* `PyErr_Format` is `PyUnicode_FromFormat` with the message handed to
   `PyErr_SetObject`, so the two describe the same conversions.

   The pending error is dropped first: building the message runs whatever
   `__repr__` an argument has, which must not start with one already set. */
PyObject *PyErr_Format(PyObject *type, const char *format, ...)
{
    va_list va;
    PyObject *message;
    PyErr_Clear();
    va_start(va, format);
    message = PyUnicode_FromFormatV(format, va);
    va_end(va);
    if (message == NULL) {
        return NULL;
    }
    PyErr_SetObject(type, message);
    Py_DECREF(message);
    return NULL;
}

/* `PyErr_FormatUnraisable` states what was going on where
   `PyErr_WriteUnraisable` names the object, so the message is built with this
   engine and handed to the core the two share.

   The exception to be reported is already pending when this is called, and
   building the message can raise: a `%S`/`%R`/`%A` conversion runs Python, a
   non-ASCII format byte and an out-of-range `%c` each set an error of their
   own.  So the indicator is held aside across the format and a failure there
   is dropped -- what gets reported must be the caller's exception, not
   whatever the message-building hit.  A NULL format is the spelling for "no
   message": the report then carries the exception alone, with `err_msg` None,
   and the formatter is never entered. */
void PyErr_FormatUnraisable(const char *format, ...)
{
    va_list va;
    PyObject *message = NULL;
    PyObject *pending = PyErr_GetRaisedException();
    if (format != NULL) {
        va_start(va, format);
        message = PyUnicode_FromFormatV(format, va);
        va_end(va);
        if (message == NULL) {
            PyErr_Clear();
        }
    }
    PyErr_SetRaisedException(pending);
    _PyPyre_WriteUnraisable(message, NULL);
    Py_XDECREF(message);
}
