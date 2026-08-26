/* `bytearray`, `complex`, `memoryview`, `weakref` and `struct.Struct` through
   their concrete C API.

   Each function answers the observable outcome, so a Python-side comparison
   against CPython running the same code says whether the two agree. */

#include <Python.h>
#include <string.h>

static PyObject *failed(const char *what)
{
    PyErr_Clear();
    return PyUnicode_FromString(what);
}

/* ── bytearray ────────────────────────────────────────────────────────── */
static PyObject *ba_check(PyObject *s, PyObject *o)
{ (void)s; return PyBool_FromLong(PyByteArray_Check(o)); }
static PyObject *ba_check_exact(PyObject *s, PyObject *o)
{ (void)s; return PyBool_FromLong(PyByteArray_CheckExact(o)); }

static PyObject *ba_size(PyObject *s, PyObject *o)
{ (void)s; Py_ssize_t n = PyByteArray_Size(o); PyErr_Clear(); return PyLong_FromSsize_t(n); }

/* The length is given separately, so a copy shorter than the source can be
   asked for -- `y#` fills one of its own from the buffer. */
static PyObject *ba_from_string_and_size(PyObject *s, PyObject *args)
{
    (void)s;
    const char *bytes;
    Py_ssize_t available, wanted;
    if (!PyArg_ParseTuple(args, "y#n", &bytes, &available, &wanted)) return NULL;
    if (wanted > available) wanted = available;
    return PyByteArray_FromStringAndSize(bytes, wanted);
}

/* A NULL source asks for that many zero bytes. */
static PyObject *ba_from_null(PyObject *s, PyObject *arg)
{
    (void)s;
    Py_ssize_t length = PyLong_AsSsize_t(arg);
    if (length == -1 && PyErr_Occurred()) return NULL;
    return PyByteArray_FromStringAndSize(NULL, length);
}

static PyObject *ba_from_object(PyObject *s, PyObject *o)
{ (void)s; PyObject *r = PyByteArray_FromObject(o); return r ? r : failed("fromobject-failed"); }

static PyObject *ba_concat(PyObject *s, PyObject *args)
{
    (void)s;
    PyObject *left, *right;
    if (!PyArg_ParseTuple(args, "OO", &left, &right)) return NULL;
    PyObject *r = PyByteArray_Concat(left, right);
    return r ? r : failed("concat-failed");
}

/* The payload as C sees it, plus the terminator one past the length. */
static PyObject *ba_as_string(PyObject *s, PyObject *o)
{
    (void)s;
    char *data = PyByteArray_AsString(o);
    if (data == NULL) return failed("asstring-failed");
    Py_ssize_t length = PyByteArray_Size(o);
    return Py_BuildValue("(y#O)", data, length,
                         data[length] == '\0' ? Py_True : Py_False);
}

/* Writing through the pointer has to reach the object Python holds. */
static PyObject *ba_write_through(PyObject *s, PyObject *o)
{
    (void)s;
    char *data = PyByteArray_AsString(o);
    if (data == NULL) return failed("asstring-failed");
    if (PyByteArray_Size(o) < 1) return failed("too-short");
    data[0] = 'Z';
    Py_RETURN_NONE;
}

static PyObject *ba_resize(PyObject *s, PyObject *args)
{
    (void)s;
    PyObject *target;
    Py_ssize_t length;
    if (!PyArg_ParseTuple(args, "On", &target, &length)) return NULL;
    if (PyByteArray_Resize(target, length) < 0) return failed("resize-failed");
    Py_RETURN_NONE;
}

/* ── complex ──────────────────────────────────────────────────────────── */
static PyObject *cx_check(PyObject *s, PyObject *o)
{ (void)s; return PyBool_FromLong(PyComplex_Check(o)); }
static PyObject *cx_check_exact(PyObject *s, PyObject *o)
{ (void)s; return PyBool_FromLong(PyComplex_CheckExact(o)); }

static PyObject *cx_from_doubles(PyObject *s, PyObject *args)
{
    (void)s;
    double real, imag;
    if (!PyArg_ParseTuple(args, "dd", &real, &imag)) return NULL;
    return PyComplex_FromDoubles(real, imag);
}

/* Round-trip through the by-value struct, which is its own calling convention. */
static PyObject *cx_round_trip(PyObject *s, PyObject *o)
{
    (void)s;
    Py_complex value = PyComplex_AsCComplex(o);
    if (value.real == -1.0 && PyErr_Occurred()) return failed("ascomplex-failed");
    return PyComplex_FromCComplex(value);
}

static PyObject *cx_parts(PyObject *s, PyObject *o)
{
    (void)s;
    double real = PyComplex_RealAsDouble(o);
    if (real == -1.0 && PyErr_Occurred()) return failed("real-failed");
    double imag = PyComplex_ImagAsDouble(o);
    if (imag == -1.0 && PyErr_Occurred()) return failed("imag-failed");
    return Py_BuildValue("(dd)", real, imag);
}

static PyObject *cx_as_ccomplex(PyObject *s, PyObject *o)
{
    (void)s;
    Py_complex value = PyComplex_AsCComplex(o);
    if (value.real == -1.0 && PyErr_Occurred()) return failed("ascomplex-failed");
    return Py_BuildValue("(dd)", value.real, value.imag);
}

/* The pair a `complex` block carries, read straight out of the block rather
   than through an accessor: the block only has room for it if `tp_basicsize`
   says so. */
static PyObject *cx_block(PyObject *s, PyObject *o)
{
    (void)s;
    if (!PyComplex_Check(o)) return failed("not-a-complex");
    if (Py_TYPE(o)->tp_basicsize < (Py_ssize_t)sizeof(PyComplexObject)) {
        return failed("block-too-small");
    }
    Py_complex value = ((PyComplexObject *)o)->cval;
    return Py_BuildValue("(dd)", value.real, value.imag);
}

/* What `complex` reports an instance is sized as, beside what this extension
   was compiled believing. */
static PyObject *cx_basicsize(PyObject *s, PyObject *unused)
{
    (void)s;
    (void)unused;
    return Py_BuildValue("(nn)", PyComplex_Type.tp_basicsize,
                         (Py_ssize_t)sizeof(PyComplexObject));
}

/* ── memoryview ───────────────────────────────────────────────────────── */

/* `mode` is 'r' or 'w' for the two buffer types, and anything else is passed
   through so that the refusal can be observed; `order` goes through
   untouched for the same reason. */
static PyObject *mv_contiguous(PyObject *s, PyObject *args)
{
    (void)s;
    PyObject *object;
    const char *mode;
    const char *order;
    if (!PyArg_ParseTuple(args, "Oss", &object, &mode, &order)) return NULL;
    int buffertype = mode[0] == 'r' ? PyBUF_READ : mode[0] == 'w' ? PyBUF_WRITE : 0;
    PyObject *view = PyMemoryView_GetContiguous(object, buffertype, order[0]);
    if (view == NULL) {
        PyObject *raised = PyErr_GetRaisedException();
        PyObject *text = raised == NULL ? NULL : PyObject_Str(raised);
        PyObject *pair = Py_BuildValue("(sO)", raised == NULL ? "?" : Py_TYPE(raised)->tp_name,
                                       text == NULL ? Py_None : text);
        Py_XDECREF(text);
        Py_XDECREF(raised);
        return pair;
    }
    return view;
}

/* ── weakref ──────────────────────────────────────────────────────────── */
static PyObject *wr_check(PyObject *s, PyObject *o)
{ (void)s; return PyBool_FromLong(PyWeakref_Check(o)); }
static PyObject *wr_check_ref(PyObject *s, PyObject *o)
{ (void)s; return PyBool_FromLong(PyWeakref_CheckRef(o)); }
static PyObject *wr_check_proxy(PyObject *s, PyObject *o)
{ (void)s; return PyBool_FromLong(PyWeakref_CheckProxy(o)); }

static PyObject *wr_new_ref(PyObject *s, PyObject *o)
{ (void)s; PyObject *r = PyWeakref_NewRef(o, NULL); return r ? r : failed("newref-failed"); }

static PyObject *wr_new_ref_with_callback(PyObject *s, PyObject *args)
{
    (void)s;
    PyObject *object, *callback;
    if (!PyArg_ParseTuple(args, "OO", &object, &callback)) return NULL;
    PyObject *r = PyWeakref_NewRef(object, callback);
    return r ? r : failed("newref-failed");
}

static PyObject *wr_new_proxy(PyObject *s, PyObject *o)
{ (void)s; PyObject *r = PyWeakref_NewProxy(o, NULL); return r ? r : failed("newproxy-failed"); }

/* Borrowed, so a new reference is what goes back to Python. */
static PyObject *wr_get_object(PyObject *s, PyObject *o)
{
    (void)s;
    PyObject *referent = PyWeakref_GetObject(o);
    if (referent == NULL) return failed("getobject-failed");
    return Py_NewRef(referent);
}

static PyObject *wr_get_ref(PyObject *s, PyObject *o)
{
    (void)s;
    PyObject *referent = NULL;
    int live = PyWeakref_GetRef(o, &referent);
    if (live < 0) return failed("getref-failed");
    if (referent == NULL) return Py_BuildValue("(iO)", live, Py_None);
    PyObject *pair = Py_BuildValue("(iO)", live, referent);
    Py_DECREF(referent);
    return pair;
}

static PyObject *wr_is_dead(PyObject *s, PyObject *o)
{
    (void)s;
    int dead = PyWeakref_IsDead(o);
    if (dead < 0) return failed("isdead-failed");
    return PyBool_FromLong(dead);
}

/* `_cffi_backend.c ctypedescr_dealloc` breaks the weak references to a dying
   object from the object's own deallocator, which `weakrefobject.c
   PyObject_ClearWeakRefs` states is the only place it may be called from: it
   rejects a receiver whose count has not fallen to zero.  What matters here is
   what the call leaves behind, because the next entry point inherits it. */
typedef struct {
    PyObject_HEAD
    PyObject *weaklist;
} ClearedObject;

static long wr_cleared;

static void cleared_dealloc(PyObject *self)
{
    PyObject_ClearWeakRefs(self);
    wr_cleared += 1;
    Py_TYPE(self)->tp_free(self);
}

static PyTypeObject ClearedType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "cpyext_object_families.Cleared",
    .tp_basicsize = sizeof(ClearedObject),
    .tp_weaklistoffset = offsetof(ClearedObject, weaklist),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyType_GenericNew,
    .tp_dealloc = cleared_dealloc,
};

static PyObject *wr_cleared_count(PyObject *s, PyObject *unused)
{ (void)s; (void)unused; return PyLong_FromLong(wr_cleared); }

#define M(name, fn) {name, fn, METH_O, NULL}
/* A one-dimensional view whose format names two members per item.  Such a
   view usually comes from `_testbuffer`'s `ndarray`; this is the same shape
   built by hand, so the fixture needs no second extension. */
static unsigned int compound_storage[4] = {1, 2, 3, 4};
static Py_ssize_t compound_shape[1] = {2};
static Py_ssize_t compound_strides[1] = {2 * sizeof(unsigned int)};

static PyObject *mv_compound_format(PyObject *self, PyObject *unused)
{
    Py_buffer view;
    (void)self;
    (void)unused;
    memset(&view, 0, sizeof(view));
    view.buf = compound_storage;
    view.len = (Py_ssize_t)sizeof(compound_storage);
    view.itemsize = 2 * (Py_ssize_t)sizeof(unsigned int);
    view.format = "II";
    view.ndim = 1;
    view.shape = compound_shape;
    view.strides = compound_strides;
    view.readonly = 1;
    return PyMemoryView_FromBuffer(&view);
}

/* Three native formats one word wide, differing only in signedness: `n` is a
   `Py_ssize_t`, `N` a `size_t` and `P` a `void *`.  A reader that takes the
   item's width and ignores its code answers a word with the top bit set the
   same way for all three. */
static unsigned long long word_storage[2] = {0xffffffffffffffffULL, 1};
static Py_ssize_t word_shape[1] = {2};
static Py_ssize_t word_strides[1] = {sizeof(unsigned long long)};

static PyObject *mv_word_format(PyObject *self, PyObject *args)
{
    Py_buffer view;
    const char *format = NULL;
    (void)self;
    if (!PyArg_ParseTuple(args, "s", &format)) {
        return NULL;
    }
    memset(&view, 0, sizeof(view));
    view.buf = word_storage;
    view.len = (Py_ssize_t)sizeof(word_storage);
    view.itemsize = (Py_ssize_t)sizeof(word_storage[0]);
    view.format = (char *)format;
    view.ndim = 1;
    view.shape = word_shape;
    view.strides = word_strides;
    view.readonly = 1;
    return PyMemoryView_FromBuffer(&view);
}

/* ── struct.Struct ── */

/* `_struct.c` keeps the byte size and the value count as fields of
   `PyStructObject` and declares that struct in its own source, so the only way
   to read them is to copy the prefix and cast -- which is what
   `Modules/_testbuffer.c` does to size the tuple it packs one item from. */
typedef struct {
    PyObject_HEAD
    Py_ssize_t s_size;
    Py_ssize_t s_len;
} PyPartialStructObject;

static PyObject *struct_counts(PyObject *self, PyObject *value)
{
    PyPartialStructObject *s = (PyPartialStructObject *)value;
    (void)self;
    return Py_BuildValue("(nn)", s->s_size, s->s_len);
}

static PyMethodDef methods[] = {
    M("ba_check", ba_check), M("ba_check_exact", ba_check_exact),
    M("ba_size", ba_size), M("ba_from_null", ba_from_null),
    M("ba_from_object", ba_from_object), M("ba_as_string", ba_as_string),
    M("ba_write_through", ba_write_through),
    {"ba_from_string_and_size", ba_from_string_and_size, METH_VARARGS, NULL},
    {"ba_concat", ba_concat, METH_VARARGS, NULL},
    {"ba_resize", ba_resize, METH_VARARGS, NULL},

    M("cx_check", cx_check), M("cx_check_exact", cx_check_exact),
    M("cx_round_trip", cx_round_trip), M("cx_parts", cx_parts),
    M("cx_as_ccomplex", cx_as_ccomplex), M("cx_block", cx_block),
    {"cx_from_doubles", cx_from_doubles, METH_VARARGS, NULL},
    {"cx_basicsize", cx_basicsize, METH_NOARGS, NULL},

    {"mv_contiguous", mv_contiguous, METH_VARARGS, NULL},
    {"mv_compound_format", mv_compound_format, METH_NOARGS, NULL},
    {"mv_word_format", mv_word_format, METH_VARARGS, NULL},

    M("wr_check", wr_check), M("wr_check_ref", wr_check_ref),
    M("wr_check_proxy", wr_check_proxy), M("wr_new_ref", wr_new_ref),
    M("wr_new_proxy", wr_new_proxy), M("wr_get_object", wr_get_object),
    M("wr_get_ref", wr_get_ref), M("wr_is_dead", wr_is_dead),
    {"wr_new_ref_with_callback", wr_new_ref_with_callback, METH_VARARGS, NULL},
    {"wr_cleared_count", wr_cleared_count, METH_NOARGS, NULL},
    {"struct_counts", struct_counts, METH_O, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef def = {
    PyModuleDef_HEAD_INIT, "cpyext_object_families", NULL, -1, methods};

PyMODINIT_FUNC PyInit_cpyext_object_families(void)
{
    PyObject *module = PyModule_Create(&def);
    if (module == NULL) {
        return NULL;
    }
    if (PyType_Ready(&ClearedType) < 0
        || PyModule_AddObjectRef(module, "Cleared", (PyObject *)&ClearedType) < 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
