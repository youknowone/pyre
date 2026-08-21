/* What a class derived in Python from a C type is handed to.

   A constructor written for C ends in `t->tp_alloc(t, 0)`, where `t` is the
   type being built rather than the one that declared the constructor -- the
   shape every Cython `cdef class` compiles to.  Derive from one in Python and
   the type handed over is the interpreter's own, so the slots it carries and
   the size it reports are what decide whether the C fields the constructor
   fills are inside the block it was given. */
#include "Python.h"

typedef struct {
    PyObject_HEAD
    long value;
    PyObject *tag;
} CellObject;

/* Reads the slots off `t` the way a C constructor does, and refuses to guess
   when one is missing: a NULL `tp_alloc` here is the call this fixture exists
   to make, so it is reported rather than dispatched. */
static PyObject *cell_new(PyTypeObject *t, PyObject *args, PyObject *kwds)
{
    allocfunc alloc;
    PyObject *made;
    (void)args;
    (void)kwds;
    alloc = t->tp_alloc;
    if (alloc == NULL) {
        PyErr_SetString(PyExc_AssertionError, "tp_alloc is NULL");
        return NULL;
    }
    if (t->tp_basicsize < (Py_ssize_t)sizeof(CellObject)) {
        PyErr_SetString(PyExc_AssertionError, "tp_basicsize is too small");
        return NULL;
    }
    made = alloc(t, 0);
    if (made == NULL) {
        return NULL;
    }
    ((CellObject *)made)->value = -1;
    ((CellObject *)made)->tag = NULL;
    return made;
}

static int cell_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    long value = 0;
    PyObject *tag = NULL;
    PyObject *previous;
    static char *names[] = {"value", "tag", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|lO", names, &value, &tag)) {
        return -1;
    }
    ((CellObject *)self)->value = value;
    previous = ((CellObject *)self)->tag;
    Py_XINCREF(tag);
    ((CellObject *)self)->tag = tag;
    Py_XDECREF(previous);
    return 0;
}

static int cell_traverse(PyObject *self, visitproc visit, void *arg)
{
    Py_VISIT(((CellObject *)self)->tag);
    return 0;
}

/* The `tp_clear` the collector reaches for a block whose interpreter object is
   already gone.  `PyObject_ClearManagedDict` is what Cython's own clear calls
   there, and it returns void: an error recorded inside it is one nothing takes,
   so the next call into this module is what would find it. */
static int cell_clear(PyObject *self)
{
    PyObject_ClearManagedDict(self);
    Py_CLEAR(((CellObject *)self)->tag);
    return 0;
}

static void cell_dealloc(PyObject *self)
{
    PyTypeObject *tp = Py_TYPE(self);
    PyObject_GC_UnTrack(self);
    Py_CLEAR(((CellObject *)self)->tag);
    tp->tp_free(self);
}

static PyObject *cell_get_value(PyObject *self, void *closure)
{
    (void)closure;
    return PyLong_FromLong(((CellObject *)self)->value);
}

static PyObject *cell_get_tag(PyObject *self, void *closure)
{
    PyObject *tag = ((CellObject *)self)->tag;
    (void)closure;
    if (tag == NULL) {
        Py_RETURN_NONE;
    }
    return Py_NewRef(tag);
}

/* The setter is what lets a cycle be built through the C field, which is what
   the collector has to reach `tp_clear` on. */
static int cell_set_tag(PyObject *self, PyObject *value, void *closure)
{
    PyObject *previous = ((CellObject *)self)->tag;
    (void)closure;
    Py_XINCREF(value);
    ((CellObject *)self)->tag = value;
    Py_XDECREF(previous);
    return 0;
}

static PyGetSetDef cell_getset[] = {
    {"value", cell_get_value, NULL, NULL, NULL},
    {"tag", cell_get_tag, cell_set_tag, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL}};

static PyType_Slot cell_slots[] = {
    {Py_tp_new, (void *)cell_new},
    {Py_tp_init, (void *)cell_init},
    {Py_tp_dealloc, (void *)cell_dealloc},
    {Py_tp_traverse, (void *)cell_traverse},
    {Py_tp_clear, (void *)cell_clear},
    {Py_tp_getset, (void *)cell_getset},
    {0, NULL},
};

static PyType_Spec cell_spec = {
    "cpyext_derive.Cell",
    sizeof(CellObject),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    cell_slots,
};

/* What the type an extension is handed carries, read off the struct rather
   than asked for through the object protocol. */
static PyObject *slots_of(PyObject *self, PyObject *type)
{
    PyTypeObject *t;
    (void)self;
    if (!PyType_Check(type)) {
        PyErr_SetString(PyExc_TypeError, "not a type");
        return NULL;
    }
    t = (PyTypeObject *)type;
    return Py_BuildValue(
        "{s:n,s:n,s:O,s:O,s:O,s:O,s:O}",
        "basicsize", (Py_ssize_t)t->tp_basicsize,
        "itemsize", (Py_ssize_t)t->tp_itemsize,
        "alloc", t->tp_alloc == NULL ? Py_False : Py_True,
        "free", t->tp_free == NULL ? Py_False : Py_True,
        "new", t->tp_new == NULL ? Py_False : Py_True,
        "getattro", t->tp_getattro == NULL ? Py_False : Py_True,
        "base", t->tp_base == NULL ? Py_None : (PyObject *)t->tp_base);
}

/* Whether the constructor the subtype carries is the one its base declared. */
static PyObject *shares_new(PyObject *self, PyObject *args)
{
    PyObject *first;
    PyObject *second;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &first, &second)) {
        return NULL;
    }
    if (!PyType_Check(first) || !PyType_Check(second)) {
        PyErr_SetString(PyExc_TypeError, "not a type");
        return NULL;
    }
    return PyBool_FromLong(
        ((PyTypeObject *)first)->tp_new == ((PyTypeObject *)second)->tp_new);
}

/* A call that answers only when nothing has left an error behind it. */
static PyObject *undisturbed(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_TRUE;
}

static PyMethodDef methods[] = {
    {"slots_of", slots_of, METH_O, NULL},
    {"shares_new", shares_new, METH_VARARGS, NULL},
    {"undisturbed", undisturbed, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}};

static int derive_exec(PyObject *module)
{
    PyObject *made = PyType_FromModuleAndSpec(module, &cell_spec, NULL);
    if (made == NULL) {
        return -1;
    }
    if (PyModule_AddObjectRef(module, "Cell", made) < 0) {
        Py_DECREF(made);
        return -1;
    }
    Py_DECREF(made);
    return 0;
}

static PyModuleDef_Slot slots[] = {
    {Py_mod_exec, (void *)derive_exec},
    {0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cpyext_derive",
    "pyre cpyext derived-type module",
    0,
    methods,
    slots,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC
PyInit_cpyext_derive(void)
{
    return PyModuleDef_Init(&moduledef);
}
