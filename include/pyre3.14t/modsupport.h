/* The variadic entry points: argument parsing and value building.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_MODSUPPORT_H
#define PYRE_MODSUPPORT_H

#ifdef __cplusplus
extern "C" {
#endif
/* ── The variadic entry points ─────────────────────────────────────────
 *
 * `PyArg_ParseTuple`, `Py_BuildValue` and `PyErr_Format` are C functions with
 * C bodies, as they are upstream (`pypy/module/cpyext/src/getargs.c`).  Pyre
 * ships no companion library, so they are `static inline` here and every
 * extension compiles its own copy; each one is built entirely out of the
 * non-variadic entry points declared above.
 */

static inline void _PyPyre_ArgError(const char *fname, const char *message)
{
    char buffer[256];
    snprintf(buffer, sizeof(buffer), "%s() %s", fname ? fname : "function", message);
    PyErr_SetString(PyExc_TypeError, buffer);
}

/* Declared here because the nested unit below counts the units it contains. */
static inline void _PyPyre_ArgCount(const char *format, Py_ssize_t *total,
                                    Py_ssize_t *required, const char **fname);

/* The buffers an `es`/`et` unit allocated -- `getargs.c addcleanup`, whose
   freelist exists so that a unit failing further along the format hands back
   what the units before it took.  Sixteen is past any format an extension
   writes; a format naming more gets the error rather than a silent leak. */
#define _PyPyre_ARG_OWNED_MAX 16
typedef struct {
    Py_ssize_t count;
    char **buffers[_PyPyre_ARG_OWNED_MAX];
} _PyPyre_ArgOwned;

static inline int _PyPyre_ArgOwn(_PyPyre_ArgOwned *owned, char **buffer)
{
    if (owned->count >= _PyPyre_ARG_OWNED_MAX) {
        return 0;
    }
    owned->buffers[owned->count++] = buffer;
    return 1;
}

/* `cleanup_ptr`: hand every buffer back and leave its destination naming
   nothing, so a caller that frees on failure anyway frees nothing twice. */
static inline void _PyPyre_ArgDisown(_PyPyre_ArgOwned *owned)
{
    for (Py_ssize_t index = 0; index < owned->count; index++) {
        PyMem_Free(*owned->buffers[index]);
        *owned->buffers[index] = NULL;
    }
    owned->count = 0;
}

/* Convert one argument according to *format, advancing it past the unit. */
static inline int _PyPyre_ArgConvert(PyObject *arg, const char **format,
                                     va_list *va, const char *fname,
                                     _PyPyre_ArgOwned *owned)
{
    char code = **format;
    (*format)++;
    switch (code) {
    case 'b': case 'B': case 'h': case 'H': case 'i': case 'I':
    case 'l': case 'k': case 'L': case 'K': case 'n': case 'p': {
        long long value;
        if (code == 'p') {
            int truth = PyObject_IsTrue(arg);
            if (truth < 0) {
                return 0;
            }
            value = truth;
        } else {
            value = PyLong_AsLongLong(arg);
            if (value == -1 && PyErr_Occurred()) {
                return 0;
            }
        }
        switch (code) {
        case 'b': *va_arg(*va, char *) = (char)value; break;
        case 'B': *va_arg(*va, unsigned char *) = (unsigned char)value; break;
        case 'h': *va_arg(*va, short *) = (short)value; break;
        case 'H': *va_arg(*va, unsigned short *) = (unsigned short)value; break;
        case 'i': case 'p': *va_arg(*va, int *) = (int)value; break;
        case 'I': *va_arg(*va, unsigned int *) = (unsigned int)value; break;
        case 'l': *va_arg(*va, long *) = (long)value; break;
        case 'k': *va_arg(*va, unsigned long *) = (unsigned long)value; break;
        case 'L': *va_arg(*va, long long *) = value; break;
        case 'K': *va_arg(*va, unsigned long long *) = (unsigned long long)value; break;
        case 'n': *va_arg(*va, Py_ssize_t *) = (Py_ssize_t)value; break;
        default: break;
        }
        return 1;
    }
    case 'f': case 'd': {
        double value = PyFloat_AsDouble(arg);
        if (value == -1.0 && PyErr_Occurred()) {
            return 0;
        }
        if (code == 'f') {
            *va_arg(*va, float *) = (float)value;
        } else {
            *va_arg(*va, double *) = value;
        }
        return 1;
    }
    case 's': case 'z': case 'y': case 'w': {
        if (**format == '*') {
            /* Fill a `Py_buffer` the caller releases.  A `str` is its own
               UTF-8 storage, so the view borrows that rather than encoding
               into a temporary; everything else goes through the buffer
               protocol. */
            (*format)++;
            Py_buffer *view = va_arg(*va, Py_buffer *);
            if (code == 'z' && Py_IsNone(arg)) {
                return PyBuffer_FillInfo(view, NULL, NULL, 0, 1, 0) == 0;
            }
            if ((code == 's' || code == 'z') && PyUnicode_Check(arg)) {
                Py_ssize_t length = 0;
                const char *text = PyUnicode_AsUTF8AndSize(arg, &length);
                if (text == NULL) {
                    return 0;
                }
                return PyBuffer_FillInfo(view, arg, (void *)text, length, 1, 0) == 0;
            }
            if (PyObject_GetBuffer(arg, view,
                                   code == 'w' ? PyBUF_WRITABLE : PyBUF_SIMPLE) < 0) {
                return 0;
            }
            if (!PyBuffer_IsContiguous(view, 'C')) {
                PyBuffer_Release(view);
                _PyPyre_ArgError(fname, "argument must be a contiguous buffer");
                return 0;
            }
            return 1;
        }
        if (code == 'w') {
            _PyPyre_ArgError(fname, "'w' argument format requires a '*'");
            return 0;
        }
        int with_size = (**format == '#');
        if (with_size) {
            (*format)++;
        }
        const char *text = NULL;
        Py_ssize_t length = 0;
        if (code == 'z' && Py_IsNone(arg)) {
            text = NULL;
        } else if (code == 'y' || (with_size && PyBytes_Check(arg))) {
            /* `s#` and `z#` take a read-only bytes-like object as well as a
               str; `s` and `z` take a str alone. A `bytearray` is neither,
               having a release the pointer this hands over cannot run. */
            char *buffer = NULL;
            if (PyBytes_AsStringAndSize(arg, &buffer, &length) < 0) {
                return 0;
            }
            text = buffer;
        } else {
            text = PyUnicode_AsUTF8AndSize(arg, &length);
            if (text == NULL) {
                return 0;
            }
        }
        /* Without a size the pointer is all the caller is given, so what it
           names has to end where the object does. */
        if (!with_size && text != NULL && strlen(text) != (size_t)length) {
            PyErr_SetString(PyExc_ValueError, code == 'y'
                                                  ? "embedded null byte"
                                                  : "embedded null character");
            return 0;
        }
        *va_arg(*va, const char **) = text;
        if (with_size) {
            *va_arg(*va, Py_ssize_t *) = length;
        }
        return 1;
    }
    case 'c': {
        char *buffer = NULL;
        Py_ssize_t length = 0;
        if (PyBytes_AsStringAndSize(arg, &buffer, &length) < 0) {
            return 0;
        }
        if (length != 1) {
            _PyPyre_ArgError(fname, "argument must be a byte string of length 1");
            return 0;
        }
        *va_arg(*va, char *) = buffer[0];
        return 1;
    }
    case 'C': {
        Py_ssize_t length = PyUnicode_GetLength(arg);
        const char *text = PyUnicode_AsUTF8(arg);
        if (text == NULL) {
            return 0;
        }
        if (length != 1) {
            _PyPyre_ArgError(fname, "argument must be a string of length 1");
            return 0;
        }
        *va_arg(*va, int *) = (unsigned char)text[0];
        return 1;
    }
    case 'e': {
        /* `es`/`et`, with the encoding as the unit's own first argument. The
           buffer is freshly allocated and the caller frees it with
           `PyMem_Free`; `et` takes an object already in that encoding as it
           stands, `es` re-encodes everything through `str`. */
        const char *encoding = va_arg(*va, const char *);
        char recode = **format;
        if (recode != 's' && recode != 't') {
            _PyPyre_ArgError(fname, "(unknown parser marker combination)");
            return 0;
        }
        (*format)++;
        char **buffer = va_arg(*va, char **);
        int with_size = (**format == '#');
        Py_ssize_t *psize = NULL;
        if (with_size) {
            (*format)++;
            psize = va_arg(*va, Py_ssize_t *);
        }
        if (buffer == NULL || (with_size && psize == NULL)) {
            _PyPyre_ArgError(fname, "(buffer is NULL)");
            return 0;
        }
        PyObject *encoded = NULL;
        const char *text = NULL;
        Py_ssize_t size = 0;
        if (recode == 't' && PyByteArray_Check(arg)) {
            encoded = Py_NewRef(arg);
            text = PyByteArray_AS_STRING(arg);
            size = PyByteArray_GET_SIZE(arg);
        } else if (recode == 't' && PyBytes_Check(arg)) {
            encoded = Py_NewRef(arg);
            text = PyBytes_AS_STRING(arg);
            size = PyBytes_GET_SIZE(arg);
        } else if (PyUnicode_Check(arg)) {
            encoded = PyUnicode_AsEncodedString(arg, encoding, NULL);
            if (encoded == NULL) {
                return 0;
            }
            text = PyBytes_AS_STRING(encoded);
            size = PyBytes_GET_SIZE(encoded);
        } else {
            _PyPyre_ArgError(fname, recode == 's'
                                        ? "argument must be str"
                                        : "argument must be str, bytes or bytearray");
            return 0;
        }
        if (!with_size && (Py_ssize_t)strlen(text) != size) {
            Py_DECREF(encoded);
            _PyPyre_ArgError(fname, "encoded string without null bytes");
            return 0;
        }
        if (with_size && *buffer != NULL) {
            /* The caller's own buffer, whose room `*psize` states.  It has to
               hold the trailing zero as well as the bytes. */
            if (size + 1 > *psize) {
                Py_DECREF(encoded);
                PyErr_Format(PyExc_ValueError,
                             "encoded string too long (%zd, maximum length %zd)",
                             size, *psize - 1);
                return 0;
            }
        } else {
            *buffer = (char *)PyMem_Malloc((size_t)size + 1);
            if (*buffer == NULL) {
                Py_DECREF(encoded);
                PyErr_NoMemory();
                return 0;
            }
            if (!_PyPyre_ArgOwn(owned, buffer)) {
                PyMem_Free(*buffer);
                *buffer = NULL;
                Py_DECREF(encoded);
                _PyPyre_ArgError(fname, "(cleanup problem)");
                return 0;
            }
        }
        memcpy(*buffer, text, (size_t)size);
        (*buffer)[size] = '\0';
        if (with_size) {
            *psize = size;
        }
        Py_DECREF(encoded);
        return 1;
    }
    case 'S': {
        if (!PyBytes_Check(arg)) {
            _PyPyre_ArgError(fname, "argument must be bytes");
            return 0;
        }
        *va_arg(*va, PyObject **) = arg;
        return 1;
    }
    case 'U': {
        if (!PyUnicode_Check(arg)) {
            _PyPyre_ArgError(fname, "argument must be str");
            return 0;
        }
        *va_arg(*va, PyObject **) = arg;
        return 1;
    }
    case '(': {
        /* A sequence whose items the units up to the matching `)` convert.
           `str` is excluded the way `converttuple` excludes it: a two-item
           string is not a two-item argument list. */
        Py_ssize_t items = 0, ignored = 0;
        _PyPyre_ArgCount(*format, &items, &ignored, NULL);
        char buffer[128];
        if (!PySequence_Check(arg) || PyUnicode_Check(arg)) {
            snprintf(buffer, sizeof(buffer), "argument must be %zd-item tuple, not %s",
                     items, arg == Py_None ? "None" : Py_TYPE(arg)->tp_name);
            _PyPyre_ArgError(fname, buffer);
            return 0;
        }
        Py_ssize_t length = PySequence_Size(arg);
        if (length < 0) {
            return 0;
        }
        if (length != items) {
            snprintf(buffer, sizeof(buffer),
                     "argument must be tuple of length %zd, not %zd", items, length);
            _PyPyre_ArgError(fname, buffer);
            return 0;
        }
        for (Py_ssize_t index = 0; index < items; index++) {
            PyObject *item = PySequence_GetItem(arg, index);
            if (item == NULL) {
                return 0;
            }
            int converted = _PyPyre_ArgConvert(item, format, va, fname, owned);
            Py_DECREF(item);
            if (!converted) {
                return 0;
            }
        }
        if (**format == ')') {
            (*format)++;
        }
        return 1;
    }
    case 'O': {
        if (**format == '!') {
            (*format)++;
            PyTypeObject *expected = va_arg(*va, PyTypeObject *);
            /* The layout, not `isinstance`: the caller goes on to read the
               object through the fields that type declares, so an object
               whose `__class__` merely answers with it must be refused. */
            if (!PyType_IsSubtype(Py_TYPE(arg), expected)) {
                char buffer[128];
                snprintf(buffer, sizeof(buffer), "argument must be %.50s, not %.50s",
                         expected->tp_name,
                         arg == Py_None ? "None" : Py_TYPE(arg)->tp_name);
                _PyPyre_ArgError(fname, buffer);
                return 0;
            }
            *va_arg(*va, PyObject **) = arg;
            return 1;
        }
        if (**format == '&') {
            (*format)++;
            converter convert = (converter)va_arg(*va, converter);
            void *target = va_arg(*va, void *);
            return convert(arg, target) != 0;
        }
        *va_arg(*va, PyObject **) = arg;
        return 1;
    }
    default:
        _PyPyre_ArgError(fname, "uses an argument format this build does not support");
        return 0;
    }
}

/* Step over one format unit, consuming its destination pointers without
   writing them -- `skipitem()` in getargs.c.  The pointers still have to leave
   the `va_list`, or the unit after an absent optional reads the slot the
   absent one would have written. */
static inline void _PyPyre_ArgSkip(const char **format, va_list *va)
{
    char code = **format;
    (*format)++;
    switch (code) {
    case 's': case 'z': case 'y': case 'w':
        (void)va_arg(*va, void *);
        if (**format == '*') {
            (*format)++;
        } else if (**format == '#') {
            (*format)++;
            (void)va_arg(*va, void *);
        }
        break;
    case 'e':
        /* The encoding, then the buffer, and the size if the unit names one. */
        (void)va_arg(*va, void *);
        if (**format == 's' || **format == 't') {
            (*format)++;
        }
        (void)va_arg(*va, void *);
        if (**format == '#') {
            (*format)++;
            (void)va_arg(*va, void *);
        }
        break;
    case 'O':
        if (**format == '!' || **format == '&') {
            (*format)++;
            (void)va_arg(*va, void *);
        }
        (void)va_arg(*va, void *);
        break;
    case '(':
        /* Every unit inside the nested one, then its closing paren. */
        while (**format != ')' && **format != '\0') {
            _PyPyre_ArgSkip(format, va);
        }
        if (**format == ')') {
            (*format)++;
        }
        break;
    default:
        (void)va_arg(*va, void *);
        break;
    }
}

/* How many argument units the format string describes, and where the optional
   tail begins.  `|`, `$`, `:` and `;` are markers rather than units. */
static inline void _PyPyre_ArgCount(const char *format, Py_ssize_t *total,
                                    Py_ssize_t *required, const char **fname)
{
    Py_ssize_t count = 0;
    Py_ssize_t minimum = -1;
    for (const char *cursor = format; *cursor; cursor++) {
        switch (*cursor) {
        case '|': case '$':
            if (minimum < 0) {
                minimum = count;
            }
            break;
        case ':': case ';':
            if (*cursor == ':' && fname != NULL) {
                *fname = cursor + 1;
            }
            goto done;
        case '#': case '!': case '&': case '*':
            break;
        case 'e':
            /* One argument, spelled `es`, `et`, `es#` or `et#`: the letter
               that follows says which and is not a unit of its own. */
            if (cursor[1] == 's' || cursor[1] == 't') {
                cursor++;
            }
            count++;
            break;
        case ')':
            /* The end of the nested unit this format is the inside of. */
            goto done;
        case '(': {
            /* One argument, however many units name its items. */
            int depth = 0;
            do {
                depth += *cursor == '(';
                depth -= *cursor == ')';
                if (depth == 0) {
                    break;
                }
                cursor++;
            } while (*cursor);
            count++;
            break;
        }
        default:
            count++;
            break;
        }
    }
done:
    *total = count;
    *required = minimum < 0 ? count : minimum;
}

static inline int _PyPyre_VaParse(PyObject *args, PyObject *kwargs,
                                  const char *format,
                                  PY_CXX_CONST char *const *keywords,
                                  va_list *va, const char *fname)
{
    Py_ssize_t total = 0;
    Py_ssize_t required = 0;
    /* Whatever an `es`/`et` unit allocated, so that a unit failing after it
       hands it back: one exit does that for every path below. */
    _PyPyre_ArgOwned owned = {0, {NULL}};
    const char *cursor = format;
    _PyPyre_ArgCount(format, &total, &required, &fname);
    Py_ssize_t given = args == NULL ? 0 : PyTuple_Size(args);
    if (given < 0) {
        goto failed;
    }
    if (given > total) {
        char buffer[128];
        snprintf(buffer, sizeof(buffer),
                 "takes at most %zd arguments (%zd given)", total, given);
        _PyPyre_ArgError(fname, buffer);
        goto failed;
    }
    for (Py_ssize_t index = 0; index < total; index++) {
        while (*cursor == '|' || *cursor == '$') {
            cursor++;
        }
        if (*cursor == '\0' || *cursor == ':' || *cursor == ';') {
            break;
        }
        PyObject *arg = NULL;
        if (index < given) {
            arg = PyTuple_GetItem(args, index);
            if (arg == NULL) {
                goto failed;
            }
        } else if (kwargs != NULL && keywords != NULL && keywords[index] != NULL) {
            arg = PyDict_GetItemString(kwargs, keywords[index]);
        }
        if (arg == NULL) {
            if (index < required) {
                char buffer[128];
                snprintf(buffer, sizeof(buffer),
                         "requires at least %zd arguments (%zd given)", required, given);
                _PyPyre_ArgError(fname, buffer);
                goto failed;
            }
            /* Absent optional: the destination keeps whatever the caller left
               in it, but its pointer still leaves the `va_list`. */
            _PyPyre_ArgSkip(&cursor, va);
            continue;
        }
        if (!_PyPyre_ArgConvert(arg, &cursor, va, fname, &owned)) {
            goto failed;
        }
    }
    if (kwargs != NULL && keywords != NULL) {
        /* An unexpected keyword is an error, which is the only reason
           `PyArg_ParseTupleAndKeywords` needs the name list at all once the
           positional mapping above is done. */
        Py_ssize_t size = PyDict_Size(kwargs);
        Py_ssize_t matched = 0;
        for (Py_ssize_t index = 0; index < total && keywords[index] != NULL; index++) {
            if (index >= given && PyDict_GetItemString(kwargs, keywords[index]) != NULL) {
                matched++;
            }
        }
        if (matched != size) {
            _PyPyre_ArgError(fname, "got an unexpected keyword argument");
            goto failed;
        }
    }
    return 1;

failed:
    _PyPyre_ArgDisown(&owned);
    return 0;
}

static inline int PyArg_ParseTuple(PyObject *args, const char *format, ...)
{
    va_list va;
    va_start(va, format);
    int parsed = _PyPyre_VaParse(args, NULL, format, NULL, &va, NULL);
    va_end(va);
    return parsed;
}

/* The pre-2.0 parser: `args` is the single argument rather than a tuple of
   them, so a format naming one unit converts `args` itself and a format naming
   more has nowhere to take them from. */
static inline int PyArg_Parse(PyObject *args, const char *format, ...)
{
    Py_ssize_t total = 0, required = 0;
    const char *fname = NULL;
    _PyPyre_ArgCount(format, &total, &required, &fname);
    if (total > 1) {
        PyErr_SetString(PyExc_SystemError, "old style getargs format uses new features");
        return 0;
    }
    if (total == 0) {
        if (args == NULL) {
            return 1;
        }
        _PyPyre_ArgError(fname, "takes no arguments");
        return 0;
    }
    if (args == NULL) {
        _PyPyre_ArgError(fname, "takes at least one argument");
        return 0;
    }
    va_list va;
    va_start(va, format);
    const char *cursor = format;
    _PyPyre_ArgOwned owned = {0, {NULL}};
    int parsed = _PyPyre_ArgConvert(args, &cursor, &va, fname, &owned);
    va_end(va);
    if (!parsed) {
        _PyPyre_ArgDisown(&owned);
    }
    return parsed;
}

static inline int PyArg_ParseTupleAndKeywords(PyObject *args, PyObject *kwargs,
                                              const char *format,
                                              PY_CXX_CONST char *const *keywords,
                                              ...)
{
    va_list va;
    va_start(va, keywords);
    int parsed = _PyPyre_VaParse(args, kwargs, format, keywords, &va, NULL);
    va_end(va);
    return parsed;
}

static inline int PyArg_UnpackTuple(PyObject *args, const char *name,
                                    Py_ssize_t least, Py_ssize_t most, ...)
{
    Py_ssize_t given = PyTuple_Size(args);
    if (given < 0) {
        return 0;
    }
    if (given < least || given > most) {
        char buffer[128];
        snprintf(buffer, sizeof(buffer),
                 "expected %zd-%zd arguments (%zd given)", least, most, given);
        _PyPyre_ArgError(name, buffer);
        return 0;
    }
    va_list va;
    va_start(va, most);
    for (Py_ssize_t index = 0; index < most; index++) {
        PyObject **target = va_arg(va, PyObject **);
        *target = index < given ? PyTuple_GetItem(args, index) : NULL;
    }
    va_end(va);
    return 1;
}

static inline PyObject *_PyPyre_BuildValue(const char **format, va_list *va);

/* One `Py_BuildValue` unit.  Containers recurse until their closing bracket. */
static inline PyObject *_PyPyre_BuildOne(const char **format, va_list *va)
{
    char code = **format;
    (*format)++;
    switch (code) {
    case 'i': case 'b': case 'h':
        return PyLong_FromLong((long)va_arg(*va, int));
    case 'B': case 'H': case 'I':
        return PyLong_FromUnsignedLong((unsigned long)va_arg(*va, unsigned int));
    case 'l':
        return PyLong_FromLong(va_arg(*va, long));
    case 'k':
        return PyLong_FromUnsignedLong(va_arg(*va, unsigned long));
    case 'L':
        return PyLong_FromLongLong(va_arg(*va, long long));
    case 'K':
        return PyLong_FromUnsignedLongLong(va_arg(*va, unsigned long long));
    case 'n':
        return PyLong_FromSsize_t(va_arg(*va, Py_ssize_t));
    case 'f': case 'd':
        return PyFloat_FromDouble(va_arg(*va, double));
    case 'c': {
        char value = (char)va_arg(*va, int);
        return PyBytes_FromStringAndSize(&value, 1);
    }
    case 'C': {
        char value = (char)va_arg(*va, int);
        return PyUnicode_FromStringAndSize(&value, 1);
    }
    case 's': case 'z': {
        const char *text = va_arg(*va, const char *);
        Py_ssize_t length = -1;
        if (**format == '#') {
            (*format)++;
            length = va_arg(*va, Py_ssize_t);
        }
        if (text == NULL) {
            Py_INCREF(Py_None);
            return Py_None;
        }
        return length < 0 ? PyUnicode_FromString(text)
                          : PyUnicode_FromStringAndSize(text, length);
    }
    case 'y': {
        const char *text = va_arg(*va, const char *);
        Py_ssize_t length = -1;
        if (**format == '#') {
            (*format)++;
            length = va_arg(*va, Py_ssize_t);
        }
        if (text == NULL) {
            Py_INCREF(Py_None);
            return Py_None;
        }
        return length < 0 ? PyBytes_FromString(text)
                          : PyBytes_FromStringAndSize(text, length);
    }
    case 'O': case 'S': case 'N': {
        PyObject *value = va_arg(*va, PyObject *);
        if (value == NULL) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_SystemError,
                                "Py_BuildValue: NULL object passed to O format");
            }
            return NULL;
        }
        if (code != 'N') {
            Py_INCREF(value);
        }
        return value;
    }
    case '(': case '[': {
        char closing = code == '(' ? ')' : ']';
        PyObject *items = PyList_New(0);
        if (items == NULL) {
            return NULL;
        }
        while (**format && **format != closing) {
            if (**format == ',' || **format == ' ') {
                (*format)++;
                continue;
            }
            PyObject *item = _PyPyre_BuildOne(format, va);
            if (item == NULL || PyList_Append(items, item) < 0) {
                Py_XDECREF(item);
                Py_DECREF(items);
                return NULL;
            }
            Py_DECREF(item);
        }
        if (**format == closing) {
            (*format)++;
        }
        if (closing == ']') {
            return items;
        }
        Py_ssize_t size = PyList_Size(items);
        PyObject *tuple = PyTuple_New(size);
        if (tuple == NULL) {
            Py_DECREF(items);
            return NULL;
        }
        for (Py_ssize_t index = 0; index < size; index++) {
            PyObject *item = PyList_GetItem(items, index);
            Py_INCREF(item);
            PyTuple_SetItem(tuple, index, item);
        }
        Py_DECREF(items);
        return tuple;
    }
    case '{': {
        PyObject *mapping = PyDict_New();
        if (mapping == NULL) {
            return NULL;
        }
        while (**format && **format != '}') {
            if (**format == ',' || **format == ' ' || **format == ':') {
                (*format)++;
                continue;
            }
            PyObject *key = _PyPyre_BuildOne(format, va);
            if (key == NULL) {
                Py_DECREF(mapping);
                return NULL;
            }
            while (**format == ',' || **format == ' ' || **format == ':') {
                (*format)++;
            }
            PyObject *value = _PyPyre_BuildOne(format, va);
            if (value == NULL || PyDict_SetItem(mapping, key, value) < 0) {
                Py_DECREF(key);
                Py_XDECREF(value);
                Py_DECREF(mapping);
                return NULL;
            }
            Py_DECREF(key);
            Py_DECREF(value);
        }
        if (**format == '}') {
            (*format)++;
        }
        return mapping;
    }
    default:
        PyErr_SetString(PyExc_SystemError, "Py_BuildValue: unsupported format");
        return NULL;
    }
}

static inline PyObject *_PyPyre_BuildValue(const char **format, va_list *va)
{
    while (**format == ' ' || **format == ',') {
        (*format)++;
    }
    if (**format == '\0') {
        Py_INCREF(Py_None);
        return Py_None;
    }
    const char *lookahead = *format;
    PyObject *first = _PyPyre_BuildOne(format, va);
    if (first == NULL) {
        return NULL;
    }
    while (**format == ' ' || **format == ',') {
        (*format)++;
    }
    if (**format == '\0') {
        return first;
    }
    (void)lookahead;
    /* More than one unit builds a tuple, exactly as `Py_BuildValue`
       documents. */
    PyObject *items = PyList_New(0);
    if (items == NULL || PyList_Append(items, first) < 0) {
        Py_DECREF(first);
        Py_XDECREF(items);
        return NULL;
    }
    Py_DECREF(first);
    while (**format) {
        if (**format == ' ' || **format == ',') {
            (*format)++;
            continue;
        }
        PyObject *item = _PyPyre_BuildOne(format, va);
        if (item == NULL || PyList_Append(items, item) < 0) {
            Py_XDECREF(item);
            Py_DECREF(items);
            return NULL;
        }
        Py_DECREF(item);
    }
    Py_ssize_t size = PyList_Size(items);
    PyObject *tuple = PyTuple_New(size);
    if (tuple == NULL) {
        Py_DECREF(items);
        return NULL;
    }
    for (Py_ssize_t index = 0; index < size; index++) {
        PyObject *item = PyList_GetItem(items, index);
        Py_INCREF(item);
        PyTuple_SetItem(tuple, index, item);
    }
    Py_DECREF(items);
    return tuple;
}

static inline PyObject *Py_BuildValue(const char *format, ...)
{
    va_list va;
    va_start(va, format);
    const char *cursor = format;
    PyObject *value = _PyPyre_BuildValue(&cursor, &va);
    va_end(va);
    return value;
}

/* The same build over a list the caller already opened, which is how a
   variadic entry point of its own passes its arguments on.

   The walk goes over a `va_copy`, not over `&va`: an argument of type
   `va_list` is an array on some ABIs, where it decays to a pointer and `&va`
   names the parameter slot rather than the list.  Copying gives an object of
   the type the walker takes the address of, on every ABI, and leaves the
   caller's own list where it was. */
static inline PyObject *Py_VaBuildValue(const char *format, va_list va)
{
    va_list copy;
    va_copy(copy, va);
    const char *cursor = format;
    PyObject *value = _PyPyre_BuildValue(&cursor, &copy);
    va_end(copy);
    return value;
}

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_MODSUPPORT_H */
