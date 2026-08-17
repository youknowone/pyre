/* A `Py_TPFLAGS_HAVE_GC` type whose one field is a strong reference, plus the
   two things a test needs to see it: how many instances are alive, and a C
   global that roots one from outside anything the interpreter can walk. */

#include <Python.h>

static long alive = 0;

typedef struct {
    PyObject_HEAD
    PyObject *ref;
} Node;

static PyObject *node_new(PyTypeObject *tp, PyObject *args, PyObject *kwds)
{
    (void)args;
    (void)kwds;
    Node *self = (Node *)tp->tp_alloc(tp, 0);
    if (self == NULL) {
        return NULL;
    }
    self->ref = NULL;
    alive++;
    return (PyObject *)self;
}

static int node_traverse(PyObject *self, visitproc visit, void *arg)
{
    Py_VISIT(((Node *)self)->ref);
    return 0;
}

static int node_clear(PyObject *self)
{
    Py_CLEAR(((Node *)self)->ref);
    return 0;
}

static void node_dealloc(PyObject *self)
{
    PyObject_GC_UnTrack(self);
    node_clear(self);
    alive--;
    Py_TYPE(self)->tp_free(self);
}

static PyObject *node_get(PyObject *self, void *closure)
{
    (void)closure;
    PyObject *ref = ((Node *)self)->ref;
    return Py_NewRef(ref ? ref : Py_None);
}

static int node_set(PyObject *self, PyObject *value, void *closure)
{
    (void)closure;
    Py_XDECREF(((Node *)self)->ref);
    ((Node *)self)->ref = Py_XNewRef(value);
    return 0;
}

static PyGetSetDef node_getset[] = {
    {"ref", node_get, node_set, "the one reference the block holds", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyTypeObject NodeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "cpyext_cycles.Node",
    .tp_basicsize = sizeof(Node),
    .tp_dealloc = node_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = node_traverse,
    .tp_clear = node_clear,
    .tp_getset = node_getset,
    .tp_new = node_new,
};

/* An external root no traversal can discover, so whatever it names has to
   survive however the rest of the graph looks. */
static PyObject *pinned = NULL;

static PyObject *m_pin(PyObject *self, PyObject *object)
{
    (void)self;
    Py_XDECREF(pinned);
    pinned = object == Py_None ? NULL : Py_NewRef(object);
    Py_RETURN_NONE;
}

static PyObject *m_pinned_ref(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    return Py_NewRef(pinned ? pinned : Py_None);
}

static PyObject *m_alive(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    return PyLong_FromLong(alive);
}

static PyMethodDef methods[] = {
    {"alive", m_alive, METH_NOARGS, "how many Nodes have not been deallocated"},
    {"pin", m_pin, METH_O, "hold one reference in a C global"},
    {"pinned_ref", m_pinned_ref, METH_NOARGS, "what the C global holds"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "cpyext_cycles",
    "a collectable type whose only field is a C-held reference",
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit_cpyext_cycles(void)
{
    if (PyType_Ready(&NodeType) < 0) {
        return NULL;
    }
    PyObject *m = PyModule_Create(&module);
    if (m == NULL) {
        return NULL;
    }
    Py_INCREF(&NodeType);
    if (PyModule_AddObject(m, "Node", (PyObject *)&NodeType) < 0) {
        Py_DECREF(&NodeType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
