/* The exception types and the pending-error interface.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_PYERRORS_H
#define PYRE_PYERRORS_H

#ifdef __cplusplus
extern "C" {
#endif
/* Exceptions. */
PyAPI_DATA(PyObject *) PyExc_BaseException;
PyAPI_DATA(PyObject *) PyExc_Exception;
PyAPI_DATA(PyObject *) PyExc_ArithmeticError;
PyAPI_DATA(PyObject *) PyExc_AssertionError;
PyAPI_DATA(PyObject *) PyExc_AttributeError;
PyAPI_DATA(PyObject *) PyExc_BufferError;
PyAPI_DATA(PyObject *) PyExc_EOFError;
PyAPI_DATA(PyObject *) PyExc_FileNotFoundError;
PyAPI_DATA(PyObject *) PyExc_FloatingPointError;
PyAPI_DATA(PyObject *) PyExc_GeneratorExit;
PyAPI_DATA(PyObject *) PyExc_ImportError;
PyAPI_DATA(PyObject *) PyExc_IndexError;
PyAPI_DATA(PyObject *) PyExc_KeyError;
PyAPI_DATA(PyObject *) PyExc_KeyboardInterrupt;
PyAPI_DATA(PyObject *) PyExc_LookupError;
PyAPI_DATA(PyObject *) PyExc_MemoryError;
PyAPI_DATA(PyObject *) PyExc_ModuleNotFoundError;
PyAPI_DATA(PyObject *) PyExc_NameError;
PyAPI_DATA(PyObject *) PyExc_NotImplementedError;
PyAPI_DATA(PyObject *) PyExc_OSError;
PyAPI_DATA(PyObject *) PyExc_OverflowError;
PyAPI_DATA(PyObject *) PyExc_RecursionError;
PyAPI_DATA(PyObject *) PyExc_ReferenceError;
PyAPI_DATA(PyObject *) PyExc_RuntimeError;
PyAPI_DATA(PyObject *) PyExc_StopAsyncIteration;
PyAPI_DATA(PyObject *) PyExc_StopIteration;
PyAPI_DATA(PyObject *) PyExc_SyntaxError;
PyAPI_DATA(PyObject *) PyExc_SystemError;
PyAPI_DATA(PyObject *) PyExc_SystemExit;
PyAPI_DATA(PyObject *) PyExc_TypeError;
PyAPI_DATA(PyObject *) PyExc_UnboundLocalError;
PyAPI_DATA(PyObject *) PyExc_UnicodeDecodeError;
PyAPI_DATA(PyObject *) PyExc_UnicodeEncodeError;
PyAPI_DATA(PyObject *) PyExc_UnicodeError;
PyAPI_DATA(PyObject *) PyExc_UnicodeTranslateError;
PyAPI_DATA(PyObject *) PyExc_ValueError;
PyAPI_DATA(PyObject *) PyExc_ZeroDivisionError;

/* `%S`, `%R`, `%U` and `%A` take a `PyObject *`; everything else is handed to
   `snprintf` one conversion at a time. */
static inline PyObject *PyErr_Format(PyObject *type, const char *format, ...)
{
    char message[1024];
    size_t filled = 0;
    va_list va;
    va_start(va, format);
    for (const char *cursor = format; *cursor && filled + 1 < sizeof(message);) {
        if (*cursor != '%') {
            message[filled++] = *cursor++;
            continue;
        }
        const char *start = cursor++;
        while (*cursor && strchr("0123456789.-+ #lzhj", *cursor) != NULL) {
            cursor++;
        }
        char code = *cursor;
        if (code == '\0') {
            break;
        }
        cursor++;
        char spec[32];
        size_t spec_length = (size_t)(cursor - start);
        if (spec_length >= sizeof(spec)) {
            spec_length = sizeof(spec) - 1;
        }
        memcpy(spec, start, spec_length);
        spec[spec_length] = '\0';
        size_t room = sizeof(message) - filled;
        int written = 0;
        switch (code) {
        case '%':
            message[filled++] = '%';
            continue;
        case 'S': case 'R': case 'A': case 'U': case 'V': {
            PyObject *object = va_arg(va, PyObject *);
            PyObject *text = (code == 'R' || code == 'A') ? PyObject_Repr(object)
                                                          : PyObject_Str(object);
            const char *utf8 = text == NULL ? "<unprintable>" : PyUnicode_AsUTF8(text);
            written = snprintf(message + filled, room, "%s", utf8 ? utf8 : "<unprintable>");
            Py_XDECREF(text);
            break;
        }
        case 's':
            written = snprintf(message + filled, room, spec, va_arg(va, const char *));
            break;
        case 'p':
            written = snprintf(message + filled, room, spec, va_arg(va, void *));
            break;
        case 'f': case 'g': case 'e':
            written = snprintf(message + filled, room, spec, va_arg(va, double));
            break;
        case 'c':
            written = snprintf(message + filled, room, spec, va_arg(va, int));
            break;
        default:
            if (strstr(spec, "ll") != NULL) {
                written = snprintf(message + filled, room, spec, va_arg(va, long long));
            } else if (strchr(spec, 'l') != NULL || strchr(spec, 'z') != NULL) {
                written = snprintf(message + filled, room, spec, va_arg(va, long));
            } else {
                written = snprintf(message + filled, room, spec, va_arg(va, int));
            }
            break;
        }
        if (written < 0) {
            break;
        }
        filled += (size_t)written < room ? (size_t)written : room - 1;
    }
    va_end(va);
    message[filled < sizeof(message) ? filled : sizeof(message) - 1] = '\0';
    return _PyPyre_ErrFormatted(type, message);
}

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_PYERRORS_H */
