/* `bytearray`, `complex` and `weakref` through their concrete C API.

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

#define M(name, fn) {name, fn, METH_O, NULL}
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
    M("cx_as_ccomplex", cx_as_ccomplex),
    {"cx_from_doubles", cx_from_doubles, METH_VARARGS, NULL},

    M("wr_check", wr_check), M("wr_check_ref", wr_check_ref),
    M("wr_check_proxy", wr_check_proxy), M("wr_new_ref", wr_new_ref),
    M("wr_new_proxy", wr_new_proxy), M("wr_get_object", wr_get_object),
    M("wr_get_ref", wr_get_ref), M("wr_is_dead", wr_is_dead),
    {"wr_new_ref_with_callback", wr_new_ref_with_callback, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef def = {
    PyModuleDef_HEAD_INIT, "cpyext_object_families", NULL, -1, methods};

PyMODINIT_FUNC PyInit_cpyext_object_families(void)
{
    return PyModule_Create(&def);
}
