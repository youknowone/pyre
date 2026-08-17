/* The `%`-format engine, and `PyUnicode_FromFormat` over it.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. `PyErr_Format` is here rather than beside the rest of the
 * exception interface because it is this engine with the result handed to
 * `PyErr_SetObject`, which is the shape `Python/errors.c:1210` gives it too.
 *
 * Pyre ships no companion library, so a variadic entry point is `static
 * inline` here and every extension compiles its own copy, built out of the
 * non-variadic entry points `pyre_decl.h` declares. The pieces are assembled
 * as `str` objects rather than as bytes: width pads to a count of characters
 * and only the interpreter knows how many a conversion produced.
 */
#ifndef PYRE_PYRE_FORMAT_H
#define PYRE_PYRE_FORMAT_H

#ifdef __cplusplus
extern "C" {
#endif

/* `count` spaces, which is what a conversion narrower than its width is
   padded with. */
static inline PyObject *_PyPyre_Spaces(Py_ssize_t count)
{
    char chunk[64];
    PyObject *result = PyUnicode_FromStringAndSize("", 0);
    memset(chunk, ' ', sizeof(chunk));
    while (result != NULL && count > 0) {
        Py_ssize_t take = count < (Py_ssize_t)sizeof(chunk) ? count : (Py_ssize_t)sizeof(chunk);
        PyUnicode_AppendAndDel(&result, PyUnicode_FromStringAndSize(chunk, take));
        count -= take;
    }
    return result;
}

/* `text` cut to `precision` characters and padded to `width`, taking the
   caller's reference to `text` either way. */
static inline PyObject *_PyPyre_Pad(PyObject *text, Py_ssize_t width, Py_ssize_t precision,
                                    int left_align)
{
    if (text == NULL) {
        return NULL;
    }
    if (precision >= 0 && PyUnicode_GetLength(text) > precision) {
        PyObject *cut = PyUnicode_Substring(text, 0, precision);
        Py_DECREF(text);
        text = cut;
        if (text == NULL) {
            return NULL;
        }
    }
    Py_ssize_t length = PyUnicode_GetLength(text);
    if (length < 0) {
        Py_DECREF(text);
        return NULL;
    }
    if (width < 0 || length >= width) {
        return text;
    }
    PyObject *spaces = _PyPyre_Spaces(width - length);
    PyObject *padded = spaces == NULL ? NULL
                       : left_align   ? PyUnicode_Concat(text, spaces)
                                      : PyUnicode_Concat(spaces, text);
    Py_DECREF(text);
    Py_XDECREF(spaces);
    return padded;
}

/* `snprintf` of one integer conversion, whose length modifier is rewritten to
   `ll` so a single call renders every width the argument was read at. */
static inline PyObject *_PyPyre_FormatInteger(const char *start, const char *modifiers,
                                              size_t modifier_length, char code, va_list *va)
{
    int is_signed = (code == 'd' || code == 'i');
    long long value;
    if (modifier_length == 0) {
        value = is_signed ? (long long)va_arg(*va, int) : (long long)va_arg(*va, unsigned int);
    } else if (modifiers[0] == 'z') {
        value = is_signed ? (long long)va_arg(*va, Py_ssize_t) : (long long)va_arg(*va, size_t);
    } else if (modifiers[0] == 'j') {
        value = is_signed ? (long long)va_arg(*va, intmax_t) : (long long)va_arg(*va, uintmax_t);
    } else if (modifiers[0] == 't') {
        value = (long long)va_arg(*va, ptrdiff_t);
    } else if (modifier_length >= 2) {
        value = is_signed ? va_arg(*va, long long) : (long long)va_arg(*va, unsigned long long);
    } else {
        value = is_signed ? (long long)va_arg(*va, long) : (long long)va_arg(*va, unsigned long);
    }

    char spec[48];
    size_t head = (size_t)(modifiers - start);
    if (head + 4 > sizeof(spec)) {
        head = sizeof(spec) - 4;
    }
    memcpy(spec, start, head);
    spec[head] = 'l';
    spec[head + 1] = 'l';
    spec[head + 2] = code;
    spec[head + 3] = '\0';

    char rendered[160];
    int written = is_signed ? snprintf(rendered, sizeof(rendered), spec, value)
                            : snprintf(rendered, sizeof(rendered), spec, (unsigned long long)value);
    if (written < 0) {
        PyErr_SetString(PyExc_SystemError, "invalid format string");
        return NULL;
    }
    if (written >= (int)sizeof(rendered)) {
        written = (int)sizeof(rendered) - 1;
    }
    return PyUnicode_FromStringAndSize(rendered, written);
}

/* A pointer, with the leading `0x` that `%p` is free not to produce. */
static inline PyObject *_PyPyre_FormatPointer(void *pointer)
{
    char rendered[64];
    int written = snprintf(rendered, sizeof(rendered), "%p", pointer);
    if (written < 0) {
        PyErr_SetString(PyExc_SystemError, "invalid format string");
        return NULL;
    }
    if (rendered[0] == '0' && (rendered[1] == 'x' || rendered[1] == 'X')) {
        rendered[1] = 'x';
        return PyUnicode_FromString(rendered);
    }
    char prefixed[66];
    snprintf(prefixed, sizeof(prefixed), "0x%s", rendered);
    return PyUnicode_FromString(prefixed);
}

/* A type's fully qualified name, `module:qualname` when `alternate` asks for
   the colon spelling. Either way the module is left off a builtin and off
   `__main__`, as `_PyType_GetFullyQualifiedName` (`Objects/typeobject.c:1589`)
   does. */
static inline PyObject *_PyPyre_TypeName(PyTypeObject *type, int alternate)
{
    if (!alternate) {
        return PyType_GetFullyQualifiedName(type);
    }
    PyObject *qualified = PyType_GetQualName(type);
    PyObject *module = PyType_GetModuleName(type);
    if (qualified == NULL || module == NULL) {
        PyErr_Clear();
        Py_XDECREF(module);
        return qualified;
    }
    if (PyUnicode_CompareWithASCIIString(module, "builtins") == 0
        || PyUnicode_CompareWithASCIIString(module, "__main__") == 0) {
        Py_DECREF(module);
        return qualified;
    }
    PyUnicode_AppendAndDel(&module, PyUnicode_FromString(":"));
    PyUnicode_AppendAndDel(&module, qualified);
    return module;
}

/* One `%` conversion as a `str`, `*cursor` left just past it. NULL with the
   error recorded for a conversion this does not describe. */
static inline PyObject *_PyPyre_FormatOne(const char **cursor, va_list *va)
{
    const char *start = *cursor; /* the '%' */
    const char *f = start + 1;
    if (*f == '%') {
        *cursor = f + 1;
        return PyUnicode_FromStringAndSize("%", 1);
    }
    int left_align = 0;
    int alternate = 0;
    while (*f == '-' || *f == '0' || *f == '+' || *f == ' ' || *f == '#') {
        left_align = left_align || *f == '-';
        alternate = alternate || *f == '#';
        f++;
    }
    Py_ssize_t width = -1;
    while (*f >= '0' && *f <= '9') {
        width = (width < 0 ? 0 : width) * 10 + (*f++ - '0');
    }
    Py_ssize_t precision = -1;
    if (*f == '.') {
        f++;
        precision = 0;
        while (*f >= '0' && *f <= '9') {
            precision = precision * 10 + (*f++ - '0');
        }
    }
    const char *modifiers = f;
    while (*f == 'l' || *f == 'z' || *f == 'j' || *f == 't') {
        f++;
    }
    size_t modifier_length = (size_t)(f - modifiers);
    char code = *f;

    PyObject *piece = NULL;
    switch (code) {
    case 'c': {
        int point = va_arg(*va, int);
        if (point < 0 || point > 0x10ffff) {
            PyErr_SetString(PyExc_OverflowError, "character argument not in range(0x110000)");
            return NULL;
        }
        piece = PyUnicode_FromOrdinal(point);
        precision = -1;
        break;
    }
    case 's': {
        /* The precision bounds the bytes read, not the characters they
           decode to, and text C hands over need not be valid UTF-8. */
        const char *text = va_arg(*va, const char *);
        Py_ssize_t length = 0;
        if (precision < 0) {
            length = (Py_ssize_t)strlen(text);
        } else {
            while (length < precision && text[length] != '\0') {
                length++;
            }
        }
        piece = PyUnicode_DecodeUTF8(text, length, "replace");
        precision = -1;
        break;
    }
    case 'S': case 'R': case 'A': case 'U': case 'V': {
        PyObject *object = va_arg(*va, PyObject *);
        /* `%V` names a str and the text to use instead when it is NULL, which
           is the one conversion taking two arguments. */
        const char *fallback = code == 'V' ? va_arg(*va, const char *) : NULL;
        if (object == NULL && fallback != NULL) {
            piece = PyUnicode_DecodeUTF8(fallback, (Py_ssize_t)strlen(fallback), "replace");
        } else if (code == 'R') {
            piece = PyObject_Repr(object);
        } else if (code == 'A') {
            piece = PyObject_ASCII(object);
        } else if (code == 'S') {
            piece = PyObject_Str(object);
        } else {
            piece = Py_XNewRef(object);
        }
        break;
    }
    case 'T': case 'N': {
        /* `%T` names an object's type and `%N` a type itself; `#` asks for
           the module and the qualified name separated by a colon. */
        PyObject *object = va_arg(*va, PyObject *);
        PyObject *type = code == 'T' ? (PyObject *)Py_TYPE(object) : object;
        if (code == 'N' && !PyType_Check(type)) {
            PyErr_SetString(PyExc_TypeError, "%N argument must be a type");
            return NULL;
        }
        piece = _PyPyre_TypeName((PyTypeObject *)type, alternate);
        break;
    }
    case 'd': case 'i': case 'u': case 'o': case 'x': case 'X':
        piece = _PyPyre_FormatInteger(start, modifiers, modifier_length, code, va);
        width = -1; /* rendered by the conversion itself, digits being ASCII */
        precision = -1;
        break;
    case 'p':
        piece = _PyPyre_FormatPointer(va_arg(*va, void *));
        width = -1;
        precision = -1;
        break;
    default: {
        char message[256];
        snprintf(message, sizeof(message), "invalid format string: %s", start);
        PyErr_SetString(PyExc_SystemError, message);
        return NULL;
    }
    }
    *cursor = f + 1;
    return _PyPyre_Pad(piece, width, precision, left_align);
}

static inline PyObject *PyUnicode_FromFormatV(const char *format, va_list vargs)
{
    va_list va;
    PyObject *pieces = PyList_New(0);
    PyObject *result = NULL;
    const char *cursor = format;
    if (pieces == NULL) {
        return NULL;
    }
    va_copy(va, vargs);
    while (*cursor != '\0') {
        PyObject *piece;
        if (*cursor == '%') {
            piece = _PyPyre_FormatOne(&cursor, &va);
        } else {
            const char *run = cursor;
            while (*cursor != '\0' && *cursor != '%') {
                if ((unsigned char)*cursor >= 128) {
                    char message[128];
                    snprintf(message, sizeof(message),
                             "PyUnicode_FromFormatV() expects an ASCII-encoded format "
                             "string, got a non-ASCII byte: 0x%02x",
                             (unsigned char)*cursor);
                    PyErr_SetString(PyExc_ValueError, message);
                    goto done;
                }
                cursor++;
            }
            piece = PyUnicode_FromStringAndSize(run, cursor - run);
        }
        if (piece == NULL) {
            goto done;
        }
        if (PyList_Append(pieces, piece) < 0) {
            Py_DECREF(piece);
            goto done;
        }
        Py_DECREF(piece);
    }
    {
        PyObject *empty = PyUnicode_FromStringAndSize("", 0);
        if (empty != NULL) {
            result = PyUnicode_Join(empty, pieces);
            Py_DECREF(empty);
        }
    }
done:
    va_end(va);
    Py_DECREF(pieces);
    return result;
}

static inline PyObject *PyUnicode_FromFormat(const char *format, ...)
{
    va_list va;
    va_start(va, format);
    PyObject *result = PyUnicode_FromFormatV(format, va);
    va_end(va);
    return result;
}

/* `PyErr_Format` is `PyUnicode_FromFormat` with the message handed to
   `PyErr_SetObject`, so the two describe the same conversions.

   The pending error is dropped first: building the message runs whatever
   `__repr__` an argument has, which must not start with one already set. */
static inline PyObject *PyErr_Format(PyObject *type, const char *format, ...)
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

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_PYRE_FORMAT_H */
