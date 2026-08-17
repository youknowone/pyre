/* The `str` entry points, and the `%`-format engine three of them share.

   Each function answers the observable outcome, so a Python-side comparison
   against CPython running the same code says whether the two agree. */

#include <Python.h>
#include <string.h>

/* The exception's class name and message, which is what a refusal is compared
   by -- the two agree on both or the row differs. */
static PyObject *refused(void)
{
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    PyObject *text = value == NULL ? PyUnicode_FromString("") : PyObject_Str(value);
    PyObject *name =
        PyUnicode_FromString(type == NULL ? "?" : ((PyTypeObject *)type)->tp_name);
    PyObject *pair = Py_BuildValue("(OO)", name, text);
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
    Py_XDECREF(text);
    Py_XDECREF(name);
    return pair;
}

/* ── the format engine ────────────────────────────────────────────────── */

/* One row per conversion, named so a comparison says which one differs. */
static PyObject *format_rows(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *object;
    if (!PyArg_ParseTuple(args, "O", &object)) {
        return NULL;
    }
    PyObject *rows = PyList_New(0);
    if (rows == NULL) {
        return NULL;
    }
    PyObject *text = PyUnicode_FromString("abcdef");
    if (text == NULL) {
        Py_DECREF(rows);
        return NULL;
    }

#define ROW(label, expr)                                          \
    do {                                                          \
        PyObject *value = (expr);                                 \
        PyObject *pair;                                           \
        if (value == NULL) {                                      \
            pair = refused();                                     \
            if (pair != NULL) {                                   \
                PyObject *named = Py_BuildValue("(sO)", label, pair); \
                Py_DECREF(pair);                                  \
                pair = named;                                     \
            }                                                     \
        } else {                                                  \
            pair = Py_BuildValue("(sO)", label, value);           \
            Py_DECREF(value);                                     \
        }                                                         \
        if (pair == NULL || PyList_Append(rows, pair) < 0) {      \
            Py_XDECREF(pair);                                     \
            Py_DECREF(text);                                      \
            Py_DECREF(rows);                                      \
            return NULL;                                          \
        }                                                         \
        Py_DECREF(pair);                                          \
    } while (0)

    ROW("literal", PyUnicode_FromFormat("plain"));
    ROW("%%", PyUnicode_FromFormat("100%%"));
    ROW("%c ascii", PyUnicode_FromFormat("[%c]", 'A'));
    ROW("%c latin", PyUnicode_FromFormat("[%c]", 0xe9));
    ROW("%c astral", PyUnicode_FromFormat("[%c]", 0x1f600));
    ROW("%c twice", PyUnicode_FromFormat("[%c%c]", 0x41, 0x42));
    ROW("%c negative", PyUnicode_FromFormat("[%c]", -1));
    ROW("%c too big", PyUnicode_FromFormat("[%c]", 0x110000));

    ROW("%d", PyUnicode_FromFormat("[%d]", -42));
    ROW("%i", PyUnicode_FromFormat("[%i]", 42));
    ROW("%u", PyUnicode_FromFormat("[%u]", 42u));
    ROW("%ld", PyUnicode_FromFormat("[%ld]", -42L));
    ROW("%lu", PyUnicode_FromFormat("[%lu]", 42UL));
    ROW("%lld", PyUnicode_FromFormat("[%lld]", -42LL));
    ROW("%llu", PyUnicode_FromFormat("[%llu]", 42ULL));
    ROW("%zd", PyUnicode_FromFormat("[%zd]", (Py_ssize_t)-42));
    ROW("%zu", PyUnicode_FromFormat("[%zu]", (size_t)42));
    ROW("%x", PyUnicode_FromFormat("[%x]", 255));
    ROW("%X", PyUnicode_FromFormat("[%X]", 255));
    ROW("%o", PyUnicode_FromFormat("[%o]", 8));
    ROW("%5d", PyUnicode_FromFormat("[%5d]", 42));
    ROW("%-5d", PyUnicode_FromFormat("[%-5d]", 42));
    ROW("%05d", PyUnicode_FromFormat("[%05d]", 42));

    ROW("%s", PyUnicode_FromFormat("[%s]", "text"));
    ROW("%s utf8", PyUnicode_FromFormat("[%s]", "\xc3\xa9t\xc3\xa9"));
    ROW("%s invalid", PyUnicode_FromFormat("[%s]", "bad\xff\xfeutf8"));
    ROW("%.2s", PyUnicode_FromFormat("[%.2s]", "abcdef"));
    /* The precision counts the bytes read, so it can stop inside a character
       -- two bytes of "été" is its first one. */
    ROW("%.2s utf8", PyUnicode_FromFormat("[%.2s]", "\xc3\xa9t\xc3\xa9"));
    ROW("%.0s", PyUnicode_FromFormat("[%.0s]", "abc"));
    ROW("%10s", PyUnicode_FromFormat("[%10s]", "ab"));
    /* The width counts characters, so the padding is three and not one. */
    ROW("%6s utf8", PyUnicode_FromFormat("[%6s]", "\xc3\xa9t\xc3\xa9"));

    ROW("%S", PyUnicode_FromFormat("[%S]", object));
    ROW("%R", PyUnicode_FromFormat("[%R]", object));
    ROW("%A", PyUnicode_FromFormat("[%A]", object));
    ROW("%.2S", PyUnicode_FromFormat("[%.2S]", object));
    ROW("%6S", PyUnicode_FromFormat("[%6S]", object));
    ROW("%-6S", PyUnicode_FromFormat("[%-6S]", object));
    ROW("%U", PyUnicode_FromFormat("[%U]", text));
    ROW("%.3U", PyUnicode_FromFormat("[%.3U]", text));
    ROW("%V", PyUnicode_FromFormat("[%V]", text, "fallback"));
    ROW("%V null", PyUnicode_FromFormat("[%V]", NULL, "fallback"));

    ROW("two", PyUnicode_FromFormat("%s=%d", "n", 7));
    /* Longer than any fixed buffer the engine might start with. */
    ROW("long", PyUnicode_FromFormat(
                    "%s%s%s%s%s%s%s%s",
                    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"));

    ROW("unknown code", PyUnicode_FromFormat("[%q]", 1));
    ROW("float code", PyUnicode_FromFormat("[%.2f]", 1.5));
    ROW("trailing", PyUnicode_FromFormat("ends with %"));
    ROW("non-ascii format", PyUnicode_FromFormat("caf\xc3\xa9"));
#undef ROW
    Py_DECREF(text);
    return rows;
}

/* `%T` names an object's type and `%N` a type; `#` asks for the module and
   the qualified name with a colon between them. */
static PyObject *format_type(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *object;
    if (!PyArg_ParseTuple(args, "O", &object)) {
        return NULL;
    }
    PyObject *plain = PyUnicode_FromFormat("[%T]", object);
    if (plain == NULL) {
        return refused();
    }
    PyObject *qualified = PyUnicode_FromFormat("[%#T]", object);
    if (qualified == NULL) {
        Py_DECREF(plain);
        return refused();
    }
    PyObject *pair = Py_BuildValue("(OO)", plain, qualified);
    Py_DECREF(plain);
    Py_DECREF(qualified);
    return pair;
}

static PyObject *format_type_name(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *object;
    if (!PyArg_ParseTuple(args, "O", &object)) {
        return NULL;
    }
    PyObject *plain = PyUnicode_FromFormat("[%N]", object);
    return plain ? plain : refused();
}

/* `%p` is the one conversion whose answer is the platform's, so only its
   shape is compared. */
static PyObject *format_pointer(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    return PyUnicode_FromFormat("%p", (void *)0);
}

/* The same engine reached through the error path, including the pending error
   it has to drop before running an argument's `__repr__`. */
static PyObject *format_error(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *object;
    if (!PyArg_ParseTuple(args, "O", &object)) {
        return NULL;
    }
    PyErr_SetString(PyExc_KeyError, "left over");
    PyErr_Format(PyExc_ValueError, "%s=%d %S %c", "n", 7, object, 'Z');
    return refused();
}

/* ── the entry points ─────────────────────────────────────────────────── */

static PyObject *str_concat(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *left, *right;
    if (!PyArg_ParseTuple(args, "OO", &left, &right)) {
        return NULL;
    }
    PyObject *joined = PyUnicode_Concat(left, right);
    return joined ? joined : refused();
}

static PyObject *str_append(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *left, *right;
    if (!PyArg_ParseTuple(args, "OO", &left, &right)) {
        return NULL;
    }
    PyObject *held = Py_NewRef(left);
    PyUnicode_Append(&held, right);
    if (held == NULL) {
        return refused();
    }
    return held;
}

/* `AppendAndDel` gives up the caller's reference to the right operand, so it
   is handed one of its own. */
static PyObject *str_append_and_del(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *left, *right;
    if (!PyArg_ParseTuple(args, "OO", &left, &right)) {
        return NULL;
    }
    PyObject *held = Py_NewRef(left);
    PyUnicode_AppendAndDel(&held, Py_NewRef(right));
    if (held == NULL) {
        return refused();
    }
    return held;
}

static PyObject *str_substring(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *text;
    Py_ssize_t start, end;
    if (!PyArg_ParseTuple(args, "Onn", &text, &start, &end)) {
        return NULL;
    }
    PyObject *cut = PyUnicode_Substring(text, start, end);
    return cut ? cut : refused();
}

static PyObject *str_join(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *separator, *sequence;
    if (!PyArg_ParseTuple(args, "OO", &separator, &sequence)) {
        return NULL;
    }
    PyObject *joined = PyUnicode_Join(separator, sequence);
    return joined ? joined : refused();
}

static PyObject *str_find_char(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *text;
    int point, direction;
    Py_ssize_t start, end;
    if (!PyArg_ParseTuple(args, "Oinni", &text, &point, &start, &end, &direction)) {
        return NULL;
    }
    Py_ssize_t at = PyUnicode_FindChar(text, (Py_UCS4)point, start, end, direction);
    if (at == -2) {
        return refused();
    }
    return PyLong_FromSsize_t(at);
}

static PyObject *str_contains(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *container, *element;
    if (!PyArg_ParseTuple(args, "OO", &container, &element)) {
        return NULL;
    }
    int found = PyUnicode_Contains(container, element);
    if (found < 0) {
        return refused();
    }
    return PyBool_FromLong(found);
}

static PyObject *str_compare(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *left, *right;
    if (!PyArg_ParseTuple(args, "OO", &left, &right)) {
        return NULL;
    }
    int order = PyUnicode_Compare(left, right);
    if (order == -1 && PyErr_Occurred()) {
        return refused();
    }
    return PyLong_FromLong(order);
}

static PyObject *str_compare_ascii(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *left;
    const char *right;
    if (!PyArg_ParseTuple(args, "Os", &left, &right)) {
        return NULL;
    }
    return PyLong_FromLong(PyUnicode_CompareWithASCIIString(left, right));
}

static PyObject *str_rich_compare(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *left, *right;
    int op;
    if (!PyArg_ParseTuple(args, "OOi", &left, &right, &op)) {
        return NULL;
    }
    PyObject *answer = PyUnicode_RichCompare(left, right, op);
    return answer ? answer : refused();
}

static PyObject *str_equal(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *left, *right;
    if (!PyArg_ParseTuple(args, "OO", &left, &right)) {
        return NULL;
    }
    int equal = PyUnicode_Equal(left, right);
    if (equal < 0) {
        return refused();
    }
    return PyBool_FromLong(equal);
}

static PyObject *str_equal_utf8(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *left;
    const char *right;
    Py_ssize_t length;
    if (!PyArg_ParseTuple(args, "Os#", &left, &right, &length)) {
        return NULL;
    }
    return Py_BuildValue("(OO)",
                         PyUnicode_EqualToUTF8(left, right) ? Py_True : Py_False,
                         PyUnicode_EqualToUTF8AndSize(left, right, length) ? Py_True : Py_False);
}

static PyObject *str_from_ordinal(PyObject *self, PyObject *arg)
{
    (void)self;
    long point = PyLong_AsLong(arg);
    if (point == -1 && PyErr_Occurred()) {
        return NULL;
    }
    PyObject *made = PyUnicode_FromOrdinal((int)point);
    return made ? made : refused();
}

/* The copy a subclass instance becomes, named by its exact type. */
static PyObject *str_from_object(PyObject *self, PyObject *arg)
{
    (void)self;
    PyObject *exact = PyUnicode_FromObject(arg);
    if (exact == NULL) {
        return refused();
    }
    PyObject *answer = Py_BuildValue("(OsO)", exact, Py_TYPE(exact)->tp_name,
                                     exact == arg ? Py_True : Py_False);
    Py_DECREF(exact);
    return answer;
}

/* Two strings with the same characters intern to one object. */
static PyObject *str_intern(PyObject *self, PyObject *arg)
{
    (void)self;
    const char *text = PyUnicode_AsUTF8(arg);
    if (text == NULL) {
        return NULL;
    }
    PyObject *first = PyUnicode_InternFromString(text);
    PyObject *second = PyUnicode_InternFromString(text);
    if (first == NULL || second == NULL) {
        Py_XDECREF(first);
        Py_XDECREF(second);
        return refused();
    }
    PyObject *answer = Py_BuildValue("(OO)", first, first == second ? Py_True : Py_False);
    Py_DECREF(first);
    Py_DECREF(second);
    return answer;
}

/* Interning in place replaces the pointer with the shared object. */
static PyObject *str_intern_in_place(PyObject *self, PyObject *arg)
{
    (void)self;
    const char *text = PyUnicode_AsUTF8(arg);
    if (text == NULL) {
        return NULL;
    }
    PyObject *shared = PyUnicode_InternFromString(text);
    PyObject *held = PyUnicode_FromString(text);
    if (shared == NULL || held == NULL) {
        Py_XDECREF(shared);
        Py_XDECREF(held);
        return refused();
    }
    PyUnicode_InternInPlace(&held);
    PyObject *answer = Py_BuildValue("(OO)", held, held == shared ? Py_True : Py_False);
    Py_DECREF(shared);
    Py_DECREF(held);
    return answer;
}

static PyObject *str_decode_utf8(PyObject *self, PyObject *args)
{
    (void)self;
    const char *bytes;
    Py_ssize_t length;
    const char *errors;
    if (!PyArg_ParseTuple(args, "y#z", &bytes, &length, &errors)) {
        return NULL;
    }
    PyObject *decoded = PyUnicode_DecodeUTF8(bytes, length, errors);
    return decoded ? decoded : refused();
}

static PyMethodDef methods[] = {
    {"format_rows", format_rows, METH_VARARGS, "every format conversion"},
    {"format_type", format_type, METH_VARARGS, "the '%T' conversion"},
    {"format_type_name", format_type_name, METH_VARARGS, "the '%N' conversion"},
    {"format_pointer", format_pointer, METH_NOARGS, "the '%p' conversion"},
    {"format_error", format_error, METH_VARARGS, "the engine through PyErr_Format"},
    {"str_concat", str_concat, METH_VARARGS, NULL},
    {"str_append", str_append, METH_VARARGS, NULL},
    {"str_append_and_del", str_append_and_del, METH_VARARGS, NULL},
    {"str_substring", str_substring, METH_VARARGS, NULL},
    {"str_join", str_join, METH_VARARGS, NULL},
    {"str_find_char", str_find_char, METH_VARARGS, NULL},
    {"str_contains", str_contains, METH_VARARGS, NULL},
    {"str_compare", str_compare, METH_VARARGS, NULL},
    {"str_compare_ascii", str_compare_ascii, METH_VARARGS, NULL},
    {"str_rich_compare", str_rich_compare, METH_VARARGS, NULL},
    {"str_equal", str_equal, METH_VARARGS, NULL},
    {"str_equal_utf8", str_equal_utf8, METH_VARARGS, NULL},
    {"str_decode_utf8", str_decode_utf8, METH_VARARGS, NULL},
    {"str_from_ordinal", str_from_ordinal, METH_O, NULL},
    {"str_from_object", str_from_object, METH_O, NULL},
    {"str_intern", str_intern, METH_O, NULL},
    {"str_intern_in_place", str_intern_in_place, METH_O, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef def = {PyModuleDef_HEAD_INIT, "cpyext_str", NULL, -1, methods};

PyMODINIT_FUNC PyInit_cpyext_str(void)
{
    return PyModule_Create(&def);
}
