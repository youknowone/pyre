/* The shapes a Cython module compiles to, each of which reached a field or an
   entry point that was not there.

   Every one of these was found by building SQLAlchemy's `cyextension` package
   against these headers and importing it, so the C here is the generated code's
   shape rather than an invention. */

#include <Python.h>

static PyObject *failed(const char *what)
{
    PyErr_Clear();
    return PyUnicode_FromString(what);
}

/* ── a callable in the vectorcall shape ──────────────────────────────────
   A module-level `def` becomes an object whose `tp_call` is
   `PyVectorcall_Call` and whose function is named by a
   `__vectorcalloffset__` member -- a spec has no slot id for the offset.
   Answering such a call through `tp_call` reaches this very entry point
   again, so the two recurse until the stack runs out. */
typedef struct {
    PyObject_HEAD
    vectorcallfunc vectorcall;
} Caller;

static PyObject *caller_vectorcall(PyObject *self, PyObject *const *args,
                                   size_t nargsf, PyObject *kwnames)
{
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    PyObject *names = kwnames ? Py_NewRef(kwnames) : Py_NewRef(Py_None);
    PyObject *first = nargs > 0 ? Py_NewRef(args[0]) : Py_NewRef(Py_None);
    PyObject *last = kwnames && PyTuple_GET_SIZE(kwnames) > 0
                         ? Py_NewRef(args[nargs + PyTuple_GET_SIZE(kwnames) - 1])
                         : Py_NewRef(Py_None);
    (void)self;
    return Py_BuildValue("(nNNN)", nargs, names, first, last);
}

static PyObject *caller_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Caller *self;
    (void)args; (void)kwds;
    self = (Caller *)type->tp_alloc(type, 0);
    if (self != NULL) self->vectorcall = caller_vectorcall;
    return (PyObject *)self;
}

static PyMemberDef caller_members[] = {
    {"__vectorcalloffset__", T_PYSSIZET, offsetof(Caller, vectorcall), READONLY, NULL},
    {NULL, 0, 0, 0, NULL}};

static PyType_Slot caller_slots[] = {
    {Py_tp_new, (void *)caller_new},
    {Py_tp_call, (void *)PyVectorcall_Call},
    {Py_tp_members, (void *)caller_members},
    {0, NULL}};

static PyType_Spec caller_spec = {
    "cpyext_cython_shapes.Caller", sizeof(Caller), 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, caller_slots};

/* ── the ancestry a `cdef class` is laid out against ─────────────────────
   `__Pyx_MergeVtables` takes `PyTuple_GET_SIZE(type->tp_bases)`, so a null
   there is a size read off nothing. */
static PyObject *c_ancestry(PyObject *self, PyObject *object)
{
    PyTypeObject *type;
    (void)self;
    if (!PyType_Check(object)) return failed("not-a-type");
    type = (PyTypeObject *)object;
    if (type->tp_bases == NULL) return failed("tp_bases-null");
    if (type->tp_mro == NULL) return failed("tp_mro-null");
    return Py_BuildValue("(nn)", PyTuple_Size(type->tp_bases),
                         PyTuple_Size(type->tp_mro));
}

static PyObject *c_first_base(PyObject *self, PyObject *object)
{
    PyTypeObject *type;
    (void)self;
    if (!PyType_Check(object)) return failed("not-a-type");
    type = (PyTypeObject *)object;
    if (type->tp_bases == NULL || PyTuple_Size(type->tp_bases) < 1)
        return failed("tp_bases-empty");
    return Py_NewRef(PyTuple_GetItem(type->tp_bases, 0));
}

/* ── the set protocol on a subclass that overrides the method ────────────
   `cdef class OrderedSet(set)` writes `def add` and calls `PySet_Add` from
   it; an entry point that dispatched on the receiver would come straight
   back. */
static PyObject *c_set_add(PyObject *self, PyObject *args)
{
    PyObject *set, *key;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &set, &key)) return NULL;
    if (PySet_Add(set, key) < 0) return failed("add-failed");
    return PyLong_FromSsize_t(PySet_Size(set));
}

static PyObject *c_set_contains(PyObject *self, PyObject *args)
{
    PyObject *set, *key;
    int found;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &set, &key)) return NULL;
    found = PySet_Contains(set, key);
    if (found < 0) return failed("contains-failed");
    return PyBool_FromLong(found);
}

static PyObject *c_set_discard(PyObject *self, PyObject *args)
{
    PyObject *set, *key;
    int removed;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &set, &key)) return NULL;
    removed = PySet_Discard(set, key);
    if (removed < 0) return failed("discard-failed");
    return PyLong_FromLong(removed);
}

static PyObject *c_set_pop(PyObject *self, PyObject *set)
{
    (void)self;
    PyObject *popped = PySet_Pop(set);
    if (popped == NULL) return failed("pop-failed");
    return popped;
}

static PyObject *c_set_clear(PyObject *self, PyObject *set)
{
    (void)self;
    if (PySet_Clear(set) < 0) return failed("clear-failed");
    return PyLong_FromSsize_t(PySet_Size(set));
}

/* ── the spellings the generated C reaches for ───────────────────────── */
static PyObject *c_dict_get_size(PyObject *self, PyObject *mapping)
{ (void)self; return PyLong_FromSsize_t(PyDict_GET_SIZE(mapping)); }

static PyObject *c_exactness(PyObject *self, PyObject *object)
{
    (void)self;
    return Py_BuildValue("(NNNN)",
                         PyBool_FromLong(PySet_CheckExact(object)),
                         PyBool_FromLong(PyFrozenSet_CheckExact(object)),
                         PyBool_FromLong(PyAnySet_CheckExact(object)),
                         PyBool_FromLong(PyAnySet_Check(object)));
}

static PyObject *c_recursive_call(PyObject *self, PyObject *object)
{
    (void)self; (void)object;
    if (Py_EnterRecursiveCall(" while testing") != 0) return failed("entered-failed");
    Py_LeaveRecursiveCall();
    return PyUnicode_FromString("entered-and-left");
}

static PyMethodDef methods[] = {
    {"ancestry", c_ancestry, METH_O, NULL},
    {"first_base", c_first_base, METH_O, NULL},
    {"set_add", c_set_add, METH_VARARGS, NULL},
    {"set_contains", c_set_contains, METH_VARARGS, NULL},
    {"set_discard", c_set_discard, METH_VARARGS, NULL},
    {"set_pop", c_set_pop, METH_O, NULL},
    {"set_clear", c_set_clear, METH_O, NULL},
    {"dict_get_size", c_dict_get_size, METH_O, NULL},
    {"exactness", c_exactness, METH_O, NULL},
    {"recursive_call", c_recursive_call, METH_O, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef def = {
    PyModuleDef_HEAD_INIT, "cpyext_cython_shapes", NULL, -1, methods};

PyMODINIT_FUNC PyInit_cpyext_cython_shapes(void)
{
    PyObject *module = PyModule_Create(&def);
    PyObject *caller;
    if (module == NULL) return NULL;
    caller = PyType_FromSpec(&caller_spec);
    if (caller == NULL || PyModule_AddObject(module, "Caller", caller) < 0) {
        Py_XDECREF(caller);
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
