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

/* A `tp_repr` written in C, so that a class derived from this type in Python
   without a `__repr__` of its own has one to inherit. */
static PyObject *cell_repr(PyObject *self)
{
    return PyUnicode_FromFormat("<Cell %ld>", ((CellObject *)self)->value);
}

static PyType_Slot cell_slots[] = {
    {Py_tp_repr, (void *)cell_repr},
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

/* Walk `o` the way a compiled loop does: take `tp_iter` off the type, call
   it, then call the iterator's own `tp_iternext` until it answers NULL.  A
   NULL slot is reported rather than called, because that is what a caller
   would otherwise dereference. */
static PyObject *walk_slots(PyObject *self, PyObject *o)
{
    (void)self;
    getiterfunc get = Py_TYPE(o)->tp_iter;
    if (get == NULL) {
        return PyUnicode_FromString("no-tp_iter");
    }
    PyObject *it = get(o);
    if (it == NULL) {
        return NULL;
    }
    iternextfunc next = Py_TYPE(it)->tp_iternext;
    if (next == NULL) {
        Py_DECREF(it);
        return PyUnicode_FromString("no-tp_iternext");
    }
    PyObject *seen = PyList_New(0);
    if (seen == NULL) {
        Py_DECREF(it);
        return NULL;
    }
    for (;;) {
        PyObject *item = next(it);
        if (item == NULL) {
            break;
        }
        if (PyList_Append(seen, item) < 0) {
            Py_DECREF(item);
            Py_DECREF(seen);
            Py_DECREF(it);
            return NULL;
        }
        Py_DECREF(item);
    }
    Py_DECREF(it);
    /* Exhaustion leaves nothing behind; anything else is the walk failing. */
    if (PyErr_Occurred()) {
        Py_DECREF(seen);
        return NULL;
    }
    return seen;
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

/* Read one slot off `o`'s type by name and call it, the way a compiled module
   reaches a method without going through the abstract entry point.  A NULL
   slot is reported as `"none"` rather than called: that is the word for what
   a caller would otherwise dereference, and it is the answer this fixture
   exists to tell apart from a result. */
static PyObject *call_slot(PyObject *self, PyObject *args)
{
    (void)self;
    const char *which;
    PyObject *o;
    PyObject *operand = NULL;
    PyObject *value = NULL;
    if (!PyArg_ParseTuple(args, "sO|OO", &which, &o, &operand, &value)) {
        return NULL;
    }
    PyTypeObject *t = Py_TYPE(o);
    PyNumberMethods *nb = t->tp_as_number;
    PySequenceMethods *sq = t->tp_as_sequence;
    PyMappingMethods *mp = t->tp_as_mapping;

    /* Which field the name reads, and the shape of the call it makes.  The
       shape is what decides how a failure is told from an answer, so it is
       settled from the name alone -- a NULL field is still reported as the
       absent slot it is. */
    enum { NONE, UNARY, LENGTH, HASH, BINARY, COUNTED, POWER, ASSIGN, ASSIGN_AT } shape =
        NONE;
    void *slot = NULL;
#define PICK(name, kind, field)                                              \
    if (strcmp(which, name) == 0) {                                          \
        shape = kind;                                                        \
        slot = (void *)(field);                                              \
    } else
#define SUITE(suite, member) ((suite) == NULL ? NULL : (suite)->member)
    PICK("tp_repr", UNARY, t->tp_repr)
    PICK("tp_str", UNARY, t->tp_str)
    PICK("nb_int", UNARY, SUITE(nb, nb_int))
    PICK("nb_float", UNARY, SUITE(nb, nb_float))
    PICK("nb_index", UNARY, SUITE(nb, nb_index))
    PICK("nb_negative", UNARY, SUITE(nb, nb_negative))
    PICK("nb_positive", UNARY, SUITE(nb, nb_positive))
    PICK("nb_absolute", UNARY, SUITE(nb, nb_absolute))
    PICK("nb_invert", UNARY, SUITE(nb, nb_invert))
    PICK("tp_hash", HASH, t->tp_hash)
    PICK("sq_length", LENGTH, SUITE(sq, sq_length))
    PICK("mp_length", LENGTH, SUITE(mp, mp_length))
    PICK("nb_add", BINARY, SUITE(nb, nb_add))
    PICK("nb_subtract", BINARY, SUITE(nb, nb_subtract))
    PICK("nb_multiply", BINARY, SUITE(nb, nb_multiply))
    PICK("nb_remainder", BINARY, SUITE(nb, nb_remainder))
    PICK("nb_and", BINARY, SUITE(nb, nb_and))
    PICK("sq_concat", BINARY, SUITE(sq, sq_concat))
    PICK("mp_subscript", BINARY, SUITE(mp, mp_subscript))
    PICK("sq_item", COUNTED, SUITE(sq, sq_item))
    PICK("sq_repeat", COUNTED, SUITE(sq, sq_repeat))
    PICK("nb_power", POWER, SUITE(nb, nb_power))
    PICK("mp_ass_subscript", ASSIGN, SUITE(mp, mp_ass_subscript))
    PICK("sq_ass_item", ASSIGN_AT, SUITE(sq, sq_ass_item))
    {
        PyErr_Format(PyExc_ValueError, "the fixture does not offer %s", which);
        return NULL;
    }
#undef SUITE
#undef PICK

    if (slot == NULL) {
        return PyUnicode_FromString("none");
    }
    if (shape == UNARY) {
        return ((unaryfunc)slot)(o);
    }
    if (shape == LENGTH) {
        Py_ssize_t n = ((lenfunc)slot)(o);
        return n < 0 && PyErr_Occurred() ? NULL : PyLong_FromSsize_t(n);
    }
    if (shape == HASH) {
        Py_hash_t h = ((hashfunc)slot)(o);
        return h == -1 && PyErr_Occurred() ? NULL : PyLong_FromSsize_t((Py_ssize_t)h);
    }
    if (operand == NULL) {
        PyErr_Format(PyExc_TypeError, "%s takes an operand", which);
        return NULL;
    }
    if (shape == BINARY) {
        return ((binaryfunc)slot)(o, operand);
    }
    if (shape == POWER) {
        /* The modulus a caller without one of its own passes. */
        return ((ternaryfunc)slot)(o, operand, value == NULL ? Py_None : value);
    }
    if (shape == COUNTED || shape == ASSIGN_AT) {
        Py_ssize_t index = PyNumber_AsSsize_t(operand, PyExc_IndexError);
        if (index == -1 && PyErr_Occurred()) {
            return NULL;
        }
        if (shape == COUNTED) {
            return ((ssizeargfunc)slot)(o, index);
        }
        if (((ssizeobjargproc)slot)(o, index, value) < 0) {
            return NULL;
        }
        Py_RETURN_NONE;
    }
    /* `ASSIGN`: a fourth argument is the value, and no fourth argument is the
       deletion the same slot answers for. */
    if (((objobjargproc)slot)(o, operand, value) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* Whether the suite a type names is the block its own `PyHeapTypeObject`
   declares.  An extension that casts a heap type to `PyHeapTypeObject *` and
   one that reads `tp_as_number` off it have to reach the same words.

   Only a heap type has that block to compare against; a static type carries
   suites allocated apart from it, and casting one is what this refuses. */
static PyObject *suites_are_embedded(PyObject *self, PyObject *o)
{
    (void)self;
    if (!PyType_Check(o)) {
        PyErr_SetString(PyExc_TypeError, "a type was expected");
        return NULL;
    }
    if (!PyType_HasFeature((PyTypeObject *)o, Py_TPFLAGS_HEAPTYPE)) {
        return PyUnicode_FromString("not-a-heap-type");
    }
    PyTypeObject *t = (PyTypeObject *)o;
    PyHeapTypeObject *ht = (PyHeapTypeObject *)o;
    return Py_BuildValue(
        "(iiiii)", t->tp_as_async == &ht->as_async ? 1 : 0,
        t->tp_as_number == &ht->as_number ? 1 : 0,
        t->tp_as_sequence == &ht->as_sequence ? 1 : 0,
        t->tp_as_mapping == &ht->as_mapping ? 1 : 0,
        t->tp_as_buffer == &ht->as_buffer ? 1 : 0);
}

static PyMethodDef methods[] = {
    {"slots_of", slots_of, METH_O, NULL},
    {"walk_slots", walk_slots, METH_O, NULL},
    {"call_slot", call_slot, METH_VARARGS, NULL},
    {"suites_are_embedded", suites_are_embedded, METH_O, NULL},
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
