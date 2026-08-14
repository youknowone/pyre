/* A multi-phase extension defining C types: `tp_new`/`tp_init`, methods,
   members, getsets, `tp_repr`/`tp_str`/`tp_hash`/`tp_call`,
   `tp_iter`/`tp_iternext`, `tp_richcompare`, inheritance through `tp_base`
   and an exception class built with `PyErr_NewExceptionWithDoc`. */

#include <Python.h>
#include <string.h>

/* ── Point: the whole slot surface on one type ──────────────────────── */

typedef struct {
    PyObject_HEAD
    long x;
    long y;
    PyObject *label;
    double scale;
} PointObject;

static PyTypeObject PointType;

static PyObject *point_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PointObject *self = (PointObject *)PyType_GenericAlloc(type, 0);
    if (self == NULL) {
        return NULL;
    }
    self->x = 0;
    self->y = 0;
    self->scale = 1.0;
    self->label = PyUnicode_FromString("");
    if (self->label == NULL) {
        return NULL;
    }
    return (PyObject *)self;
}

static int point_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char *keywords[] = {"x", "y", "label", NULL};
    PointObject *point = (PointObject *)self;
    long x = 0;
    long y = 0;
    const char *label = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|lls", keywords, &x, &y, &label)) {
        return -1;
    }
    point->x = x;
    point->y = y;
    if (label != NULL) {
        PyObject *text = PyUnicode_FromString(label);
        if (text == NULL) {
            return -1;
        }
        Py_SETREF(point->label, text);
    }
    return 0;
}

static PyObject *point_repr(PyObject *self)
{
    PointObject *point = (PointObject *)self;
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "Point(%ld, %ld)", point->x, point->y);
    return PyUnicode_FromString(buffer);
}

static PyObject *point_str(PyObject *self)
{
    PointObject *point = (PointObject *)self;
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%ld/%ld", point->x, point->y);
    return PyUnicode_FromString(buffer);
}

static Py_hash_t point_hash(PyObject *self)
{
    PointObject *point = (PointObject *)self;
    return (Py_hash_t)(point->x * 1000003 + point->y);
}

/* tp_call: `point(a, b)` shifts by the pair. */
static PyObject *point_call(PyObject *self, PyObject *args, PyObject *kwds)
{
    PointObject *point = (PointObject *)self;
    long dx = 0;
    long dy = 0;
    static char *keywords[] = {"dx", "dy", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ll", keywords, &dx, &dy)) {
        return NULL;
    }
    return Py_BuildValue("(ll)", point->x + dx, point->y + dy);
}

static PyObject *point_richcompare(PyObject *self, PyObject *other, int op)
{
    if (!PyObject_TypeCheck(other, &PointType)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    PointObject *left = (PointObject *)self;
    PointObject *right = (PointObject *)other;
    long a = left->x * left->x + left->y * left->y;
    long b = right->x * right->x + right->y * right->y;
    int result;
    switch (op) {
    case Py_LT: result = a < b; break;
    case Py_LE: result = a <= b; break;
    case Py_EQ: result = a == b; break;
    case Py_NE: result = a != b; break;
    case Py_GT: result = a > b; break;
    default: result = a >= b; break;
    }
    return PyBool_FromLong(result);
}

static PyObject *point_translate(PyObject *self, PyObject *args)
{
    PointObject *point = (PointObject *)self;
    long dx = 0;
    long dy = 0;
    if (!PyArg_ParseTuple(args, "ll", &dx, &dy)) {
        return NULL;
    }
    point->x += dx;
    point->y += dy;
    Py_INCREF(self);
    return self;
}

static PyObject *point_norm(PyObject *self, PyObject *unused)
{
    PointObject *point = (PointObject *)self;
    return PyLong_FromLong(point->x * point->x + point->y * point->y);
}

static PyObject *point_named(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char *keywords[] = {"prefix", NULL};
    const char *prefix = "p";
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|s", keywords, &prefix)) {
        return NULL;
    }
    PointObject *point = (PointObject *)self;
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "%s:%ld", prefix, point->x);
    return PyUnicode_FromString(buffer);
}

static PyMethodDef point_methods[] = {
    {"translate", (PyCFunction)point_translate, METH_VARARGS, "shift in place"},
    {"norm", (PyCFunction)point_norm, METH_NOARGS, "squared length"},
    {"named", (PyCFunction)(void (*)(void))point_named, METH_VARARGS | METH_KEYWORDS,
     "prefixed name"},
    {NULL, NULL, 0, NULL},
};

static PyMemberDef point_members[] = {
    {"x", Py_T_LONG, offsetof(PointObject, x), 0, "the abscissa"},
    {"y", Py_T_LONG, offsetof(PointObject, y), Py_READONLY, "the ordinate"},
    {"label", T_OBJECT, offsetof(PointObject, label), 0, "a free-form label"},
    {"scale", Py_T_DOUBLE, offsetof(PointObject, scale), 0, "a scale factor"},
    {NULL, 0, 0, 0, NULL},
};

static PyObject *point_get_total(PyObject *self, void *closure)
{
    PointObject *point = (PointObject *)self;
    return PyLong_FromLong(point->x + point->y + (long)(intptr_t)closure);
}

static int point_set_total(PyObject *self, PyObject *value, void *closure)
{
    PointObject *point = (PointObject *)self;
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "cannot delete total");
        return -1;
    }
    long total = PyLong_AsLong(value);
    if (total == -1 && PyErr_Occurred()) {
        return -1;
    }
    point->x = total;
    point->y = 0;
    return 0;
}

static PyObject *point_get_frozen(PyObject *self, void *closure)
{
    return PyUnicode_FromString("frozen");
}

static PyGetSetDef point_getset[] = {
    {"total", point_get_total, point_set_total, "x + y + closure", (void *)(intptr_t)100},
    {"frozen", point_get_frozen, NULL, "read-only property", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

PyDoc_STRVAR(point_doc, "a two-dimensional point defined in C");

static PyTypeObject PointType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "cpyext_types.Point",                       /* tp_name */
    sizeof(PointObject),                        /* tp_basicsize */
    0,                                          /* tp_itemsize */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_vectorcall_offset */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_as_async */
    point_repr,                                 /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    point_hash,                                 /* tp_hash */
    (ternaryfunc)point_call,                    /* tp_call */
    point_str,                                  /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
    point_doc,                                  /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    point_richcompare,                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    point_methods,                              /* tp_methods */
    point_members,                              /* tp_members */
    point_getset,                               /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    point_init,                                 /* tp_init */
    0,                                          /* tp_alloc */
    point_new,                                  /* tp_new */
};

/* ── Point3: inherits every slot it leaves null ─────────────────────── */

typedef struct {
    PointObject base;
    long z;
} Point3Object;

static PyMemberDef point3_members[] = {
    {"z", Py_T_LONG, offsetof(Point3Object, z), 0, "the applicate"},
    {NULL, 0, 0, 0, NULL},
};

static PyObject *point3_depth(PyObject *self, PyObject *unused)
{
    return PyLong_FromLong(((Point3Object *)self)->z);
}

static PyMethodDef point3_methods[] = {
    {"depth", point3_depth, METH_NOARGS, "the third coordinate"},
    {NULL, NULL, 0, NULL},
};

static PyTypeObject Point3Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "cpyext_types.Point3",                      /* tp_name */
    sizeof(Point3Object),                       /* tp_basicsize */
    0,                                          /* tp_itemsize */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_vectorcall_offset */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_as_async */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    "a three-dimensional point",                /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    point3_methods,                             /* tp_methods */
    point3_members,                             /* tp_members */
    0,                                          /* tp_getset */
    &PointType,                                 /* tp_base */
};

/* ── Counter: tp_iter / tp_iternext ─────────────────────────────────── */

typedef struct {
    PyObject_HEAD
    long next;
    long stop;
} CounterObject;

static PyObject *counter_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    long stop = 0;
    if (!PyArg_ParseTuple(args, "l", &stop)) {
        return NULL;
    }
    CounterObject *self = (CounterObject *)PyType_GenericAlloc(type, 0);
    if (self == NULL) {
        return NULL;
    }
    self->next = 0;
    self->stop = stop;
    return (PyObject *)self;
}

static PyObject *counter_iter(PyObject *self)
{
    Py_INCREF(self);
    return self;
}

static PyObject *counter_iternext(PyObject *self)
{
    CounterObject *counter = (CounterObject *)self;
    if (counter->next >= counter->stop) {
        return NULL;
    }
    return PyLong_FromLong(counter->next++);
}

static PyTypeObject CounterType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "cpyext_types.Counter",                     /* tp_name */
    sizeof(CounterObject),                      /* tp_basicsize */
    0,                                          /* tp_itemsize */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_vectorcall_offset */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_as_async */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    "counts up to a bound",                     /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    counter_iter,                               /* tp_iter */
    counter_iternext,                           /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    counter_new,                                /* tp_new */
};

/* ── module ─────────────────────────────────────────────────────────── */

static PyObject *TypesError;

static PyObject *m_make(PyObject *self, PyObject *args)
{
    long x = 0;
    long y = 0;
    if (!PyArg_ParseTuple(args, "ll", &x, &y)) {
        return NULL;
    }
    PointObject *point = (PointObject *)PyType_GenericAlloc(&PointType, 0);
    if (point == NULL) {
        return NULL;
    }
    point->x = x;
    point->y = y;
    point->scale = 1.0;
    point->label = PyUnicode_FromString("made");
    if (point->label == NULL) {
        return NULL;
    }
    return (PyObject *)point;
}

static PyObject *m_is_point(PyObject *self, PyObject *arg)
{
    return PyBool_FromLong(PyObject_TypeCheck(arg, &PointType));
}

static PyObject *m_sum_x(PyObject *self, PyObject *arg)
{
    if (!PyObject_TypeCheck(arg, &PointType)) {
        PyErr_SetString(TypesError, "not a Point");
        return NULL;
    }
    return PyLong_FromLong(((PointObject *)arg)->x);
}

static PyObject *m_flags(PyObject *self, PyObject *unused)
{
    unsigned long flags = PyType_GetFlags(&PointType);
    return Py_BuildValue("(ii)", (flags & Py_TPFLAGS_READY) != 0,
                         (flags & Py_TPFLAGS_BASETYPE) != 0);
}

static PyObject *m_is_subtype(PyObject *self, PyObject *unused)
{
    return Py_BuildValue("(ii)", PyType_IsSubtype(&Point3Type, &PointType),
                         PyType_IsSubtype(&PointType, &Point3Type));
}

static PyMethodDef methods[] = {
    {"make", m_make, METH_VARARGS, "build a Point without calling the class"},
    {"is_point", m_is_point, METH_O, "PyObject_TypeCheck"},
    {"sum_x", m_sum_x, METH_O, "read x, raising the module exception otherwise"},
    {"flags", m_flags, METH_NOARGS, "PyType_GetFlags on Point"},
    {"is_subtype", m_is_subtype, METH_NOARGS, "PyType_IsSubtype both ways"},
    {NULL, NULL, 0, NULL},
};

static int types_exec(PyObject *module)
{
    if (PyType_Ready(&PointType) < 0) {
        return -1;
    }
    if (PyModule_AddType(module, &PointType) < 0) {
        return -1;
    }
    if (PyModule_AddType(module, &Point3Type) < 0) {
        return -1;
    }
    if (PyModule_AddType(module, &CounterType) < 0) {
        return -1;
    }
    TypesError = PyErr_NewExceptionWithDoc("cpyext_types.TypesError",
                                           "raised by the fixture", NULL, NULL);
    if (TypesError == NULL) {
        return -1;
    }
    return PyModule_AddObjectRef(module, "TypesError", TypesError);
}

static PyModuleDef_Slot slots[] = {
    {Py_mod_exec, (void *)types_exec},
    {0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cpyext_types",
    "pyre cpyext type module",
    0,
    methods,
    slots,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC
PyInit_cpyext_types(void)
{
    return PyModuleDef_Init(&moduledef);
}
