/* A multi-phase extension defining C types: `tp_new`/`tp_init`, methods,
   members, getsets, `tp_repr`/`tp_str`/`tp_hash`/`tp_call`,
   `tp_iter`/`tp_iternext`, `tp_richcompare`, inheritance through `tp_base`
   and an exception class built with `PyErr_NewExceptionWithDoc`. */

#include <Python.h>
#include <string.h>

static struct PyModuleDef moduledef;

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
        Py_DECREF(self);
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

/* `label` is a strong reference the instance owns, so it is released here;
   Point3 inherits this slot. */
static void point_dealloc(PyObject *self)
{
    Py_CLEAR(((PointObject *)self)->label);
    Py_TYPE(self)->tp_free(self);
}

static PyTypeObject PointType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "cpyext_types.Point",                       /* tp_name */
    sizeof(PointObject),                        /* tp_basicsize */
    0,                                          /* tp_itemsize */
    point_dealloc,                              /* tp_dealloc */
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

/* ── Vec: the number table ──────────────────────────────────────────── */

typedef struct {
    PyObject_HEAD
    long value;
} VecObject;

static PyTypeObject VecType;

static PyObject *vec_from(long value)
{
    VecObject *self = (VecObject *)PyType_GenericAlloc(&VecType, 0);
    if (self == NULL) {
        return NULL;
    }
    self->value = value;
    return (PyObject *)self;
}

/* -1 when the operand is neither a Vec nor an int. */
static int vec_operand(PyObject *object, long *out)
{
    if (PyObject_TypeCheck(object, &VecType)) {
        *out = ((VecObject *)object)->value;
        return 0;
    }
    if (PyLong_Check(object)) {
        *out = PyLong_AsLong(object);
        return (*out == -1 && PyErr_Occurred()) ? -1 : 0;
    }
    return 1;
}

#define VEC_BINARY(name, expression)                                    \
    static PyObject *name(PyObject *left, PyObject *right)              \
    {                                                                   \
        long a = 0;                                                     \
        long b = 0;                                                     \
        int first = vec_operand(left, &a);                              \
        int second = vec_operand(right, &b);                            \
        if (first < 0 || second < 0) {                                  \
            return NULL;                                                \
        }                                                               \
        if (first > 0 || second > 0) {                                  \
            Py_RETURN_NOTIMPLEMENTED;                                   \
        }                                                               \
        return vec_from(expression);                                    \
    }

VEC_BINARY(vec_add, a + b)
VEC_BINARY(vec_sub, a - b)
VEC_BINARY(vec_mul, a * b)

static PyObject *vec_negative(PyObject *self)
{
    return vec_from(-((VecObject *)self)->value);
}

static PyObject *vec_absolute(PyObject *self)
{
    long value = ((VecObject *)self)->value;
    return vec_from(value < 0 ? -value : value);
}

static int vec_bool(PyObject *self)
{
    return ((VecObject *)self)->value != 0;
}

static PyObject *vec_int(PyObject *self)
{
    return PyLong_FromLong(((VecObject *)self)->value);
}

static PyObject *vec_float(PyObject *self)
{
    return PyFloat_FromDouble((double)((VecObject *)self)->value);
}

static PyObject *vec_power(PyObject *left, PyObject *right, PyObject *modulus)
{
    long a = 0;
    long b = 0;
    if (vec_operand(left, &a) != 0 || vec_operand(right, &b) != 0) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    long result = 1;
    for (long i = 0; i < b; i++) {
        result *= a;
    }
    if (modulus != NULL && !Py_IsNone(modulus)) {
        long m = 0;
        if (vec_operand(modulus, &m) != 0 || m == 0) {
            Py_RETURN_NOTIMPLEMENTED;
        }
        result %= m;
    }
    return vec_from(result);
}

/* The in-place slot mutates and hands back the same object. */
static PyObject *vec_inplace_add(PyObject *left, PyObject *right)
{
    long b = 0;
    if (!PyObject_TypeCheck(left, &VecType) || vec_operand(right, &b) != 0) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    ((VecObject *)left)->value += b;
    Py_INCREF(left);
    return left;
}

static PyNumberMethods vec_as_number = {
    vec_add,                                    /* nb_add */
    vec_sub,                                    /* nb_subtract */
    vec_mul,                                    /* nb_multiply */
    0,                                          /* nb_remainder */
    0,                                          /* nb_divmod */
    vec_power,                                  /* nb_power */
    vec_negative,                               /* nb_negative */
    0,                                          /* nb_positive */
    vec_absolute,                               /* nb_absolute */
    vec_bool,                                   /* nb_bool */
    0,                                          /* nb_invert */
    0,                                          /* nb_lshift */
    0,                                          /* nb_rshift */
    0,                                          /* nb_and */
    0,                                          /* nb_xor */
    0,                                          /* nb_or */
    vec_int,                                    /* nb_int */
    0,                                          /* nb_reserved */
    vec_float,                                  /* nb_float */
    vec_inplace_add,                            /* nb_inplace_add */
};

static PyObject *vec_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    long value = 0;
    if (!PyArg_ParseTuple(args, "l", &value)) {
        return NULL;
    }
    return vec_from(value);
}

static PyObject *vec_repr(PyObject *self)
{
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "Vec(%ld)", ((VecObject *)self)->value);
    return PyUnicode_FromString(buffer);
}

static PyMemberDef vec_members[] = {
    {"value", Py_T_LONG, offsetof(VecObject, value), Py_READONLY, "the scalar"},
    {NULL, 0, 0, 0, NULL},
};

static PyTypeObject VecType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "cpyext_types.Vec",                         /* tp_name */
    sizeof(VecObject),                          /* tp_basicsize */
    0,                                          /* tp_itemsize */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_vectorcall_offset */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_as_async */
    vec_repr,                                   /* tp_repr */
    &vec_as_number,                             /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    "a scalar with a number table",             /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    vec_members,                                /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    vec_new,                                    /* tp_new */
};

/* ── Bag: the sequence table ────────────────────────────────────────── */

#define BAG_CAPACITY 8

typedef struct {
    PyObject_HEAD
    Py_ssize_t count;
    long items[BAG_CAPACITY];
} BagObject;

static PyTypeObject BagType;

static PyObject *bag_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    BagObject *self = (BagObject *)PyType_GenericAlloc(type, 0);
    if (self == NULL) {
        return NULL;
    }
    self->count = 0;
    Py_ssize_t given = PyTuple_Size(args);
    if (given > BAG_CAPACITY) {
        PyErr_SetString(PyExc_ValueError, "too many items");
        Py_DECREF(self);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < given; i++) {
        PyObject *item = PyTuple_GetItem(args, i);
        if (item == NULL) {
            Py_DECREF(self);
            return NULL;
        }
        long value = PyLong_AsLong(item);
        if (value == -1 && PyErr_Occurred()) {
            Py_DECREF(self);
            return NULL;
        }
        self->items[self->count++] = value;
    }
    return (PyObject *)self;
}

static Py_ssize_t bag_length(PyObject *self)
{
    return ((BagObject *)self)->count;
}

static PyObject *bag_item(PyObject *self, Py_ssize_t index)
{
    BagObject *bag = (BagObject *)self;
    if (index < 0 || index >= bag->count) {
        PyErr_SetString(PyExc_IndexError, "bag index out of range");
        return NULL;
    }
    return PyLong_FromLong(bag->items[index]);
}

static int bag_ass_item(PyObject *self, Py_ssize_t index, PyObject *value)
{
    BagObject *bag = (BagObject *)self;
    if (index < 0 || index >= bag->count) {
        PyErr_SetString(PyExc_IndexError, "bag index out of range");
        return -1;
    }
    if (value == NULL) {
        for (Py_ssize_t i = index; i + 1 < bag->count; i++) {
            bag->items[i] = bag->items[i + 1];
        }
        bag->count--;
        return 0;
    }
    long item = PyLong_AsLong(value);
    if (item == -1 && PyErr_Occurred()) {
        return -1;
    }
    bag->items[index] = item;
    return 0;
}

static int bag_contains(PyObject *self, PyObject *value)
{
    BagObject *bag = (BagObject *)self;
    long item = PyLong_AsLong(value);
    if (item == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }
    for (Py_ssize_t i = 0; i < bag->count; i++) {
        if (bag->items[i] == item) {
            return 1;
        }
    }
    return 0;
}

static PyObject *bag_repeat(PyObject *self, Py_ssize_t count)
{
    BagObject *bag = (BagObject *)self;
    PyObject *out = PyList_New(0);
    if (out == NULL) {
        return NULL;
    }
    for (Py_ssize_t round = 0; round < count; round++) {
        for (Py_ssize_t i = 0; i < bag->count; i++) {
            PyObject *item = PyLong_FromLong(bag->items[i]);
            if (item == NULL || PyList_Append(out, item) < 0) {
                Py_XDECREF(item);
                Py_DECREF(out);
                return NULL;
            }
            Py_DECREF(item);
        }
    }
    return out;
}

static PySequenceMethods bag_as_sequence = {
    bag_length,                                 /* sq_length */
    0,                                          /* sq_concat */
    bag_repeat,                                 /* sq_repeat */
    bag_item,                                   /* sq_item */
    0,                                          /* was_sq_slice */
    bag_ass_item,                               /* sq_ass_item */
    0,                                          /* was_sq_ass_slice */
    bag_contains,                               /* sq_contains */
};

static PyTypeObject BagType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "cpyext_types.Bag",                         /* tp_name */
    sizeof(BagObject),                          /* tp_basicsize */
    0,                                          /* tp_itemsize */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_vectorcall_offset */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_as_async */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    &bag_as_sequence,                           /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    "a fixed-capacity sequence",                /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
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
    bag_new,                                    /* tp_new */
};

/* ── Table: the mapping table ───────────────────────────────────────── */

typedef struct {
    PyObject_HEAD
    PyObject *store;
} TableObject;

static PyObject *table_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    TableObject *self = (TableObject *)PyType_GenericAlloc(type, 0);
    if (self == NULL) {
        return NULL;
    }
    self->store = PyDict_New();
    if (self->store == NULL) {
        Py_DECREF(self);
        return NULL;
    }
    return (PyObject *)self;
}

static Py_ssize_t table_length(PyObject *self)
{
    return PyDict_Size(((TableObject *)self)->store);
}

static PyObject *table_subscript(PyObject *self, PyObject *key)
{
    PyObject *value = PyDict_GetItem(((TableObject *)self)->store, key);
    if (value == NULL) {
        PyErr_SetObject(PyExc_KeyError, key);
        return NULL;
    }
    Py_INCREF(value);
    return value;
}

static int table_ass_subscript(PyObject *self, PyObject *key, PyObject *value)
{
    TableObject *table = (TableObject *)self;
    if (value == NULL) {
        return PyDict_DelItem(table->store, key);
    }
    return PyDict_SetItem(table->store, key, value);
}

static PyObject *table_keys(PyObject *self, PyObject *unused)
{
    return PyMapping_Keys(((TableObject *)self)->store);
}

static PyMethodDef table_methods[] = {
    {"keys", table_keys, METH_NOARGS, "the stored keys"},
    {NULL, NULL, 0, NULL},
};

static PyMappingMethods table_as_mapping = {
    table_length,                               /* mp_length */
    table_subscript,                            /* mp_subscript */
    table_ass_subscript,                        /* mp_ass_subscript */
};

static void table_dealloc(PyObject *self)
{
    Py_CLEAR(((TableObject *)self)->store);
    Py_TYPE(self)->tp_free(self);
}

static PyTypeObject TableType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "cpyext_types.Table",                       /* tp_name */
    sizeof(TableObject),                        /* tp_basicsize */
    0,                                          /* tp_itemsize */
    table_dealloc,                              /* tp_dealloc */
    0,                                          /* tp_vectorcall_offset */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_as_async */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    &table_as_mapping,                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    "a dict-backed mapping",                    /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    table_methods,                              /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    table_new,                                  /* tp_new */
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
        Py_DECREF(point);
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

/* The abstract protocols, driven from C over whatever object is handed in. */
static PyObject *m_protocol(PyObject *self, PyObject *args)
{
    const char *what = NULL;
    PyObject *first = NULL;
    PyObject *second = NULL;
    if (!PyArg_ParseTuple(args, "sO|O", &what, &first, &second)) {
        return NULL;
    }
    if (strcmp(what, "add") == 0) {
        return PyNumber_Add(first, second);
    }
    if (strcmp(what, "multiply") == 0) {
        return PyNumber_Multiply(first, second);
    }
    if (strcmp(what, "power") == 0) {
        return PyNumber_Power(first, second, Py_None);
    }
    if (strcmp(what, "negative") == 0) {
        return PyNumber_Negative(first);
    }
    if (strcmp(what, "index") == 0) {
        return PyNumber_Index(first);
    }
    if (strcmp(what, "float") == 0) {
        return PyNumber_Float(first);
    }
    if (strcmp(what, "number_check") == 0) {
        return PyBool_FromLong(PyNumber_Check(first));
    }
    if (strcmp(what, "as_ssize") == 0) {
        Py_ssize_t value = PyNumber_AsSsize_t(first, PyExc_OverflowError);
        if (value == -1 && PyErr_Occurred()) {
            return NULL;
        }
        return PyLong_FromSsize_t(value);
    }
    if (strcmp(what, "sequence_check") == 0) {
        return PyBool_FromLong(PySequence_Check(first));
    }
    if (strcmp(what, "size") == 0) {
        Py_ssize_t size = PySequence_Size(first);
        return size < 0 ? NULL : PyLong_FromSsize_t(size);
    }
    if (strcmp(what, "getitem") == 0) {
        Py_ssize_t index = PyNumber_AsSsize_t(second, NULL);
        if (index == -1 && PyErr_Occurred()) {
            return NULL;
        }
        return PySequence_GetItem(first, index);
    }
    if (strcmp(what, "contains") == 0) {
        int found = PySequence_Contains(first, second);
        return found < 0 ? NULL : PyBool_FromLong(found);
    }
    if (strcmp(what, "list") == 0) {
        return PySequence_List(first);
    }
    if (strcmp(what, "tuple") == 0) {
        return PySequence_Tuple(first);
    }
    if (strcmp(what, "seq_index") == 0) {
        Py_ssize_t index = PySequence_Index(first, second);
        return index < 0 ? NULL : PyLong_FromSsize_t(index);
    }
    if (strcmp(what, "repeat") == 0) {
        Py_ssize_t count = PyNumber_AsSsize_t(second, NULL);
        if (count == -1 && PyErr_Occurred()) {
            return NULL;
        }
        return PySequence_Repeat(first, count);
    }
    if (strcmp(what, "mapping_check") == 0) {
        return PyBool_FromLong(PyMapping_Check(first));
    }
    if (strcmp(what, "keys") == 0) {
        return PyMapping_Keys(first);
    }
    if (strcmp(what, "values") == 0) {
        return PyMapping_Values(first);
    }
    if (strcmp(what, "items") == 0) {
        return PyMapping_Items(first);
    }
    if (strcmp(what, "getstring") == 0) {
        const char *key = PyUnicode_AsUTF8(second);
        return key == NULL ? NULL : PyMapping_GetItemString(first, key);
    }
    if (strcmp(what, "haskey") == 0) {
        return PyBool_FromLong(PyMapping_HasKey(first, second));
    }
    PyErr_SetString(PyExc_ValueError, "unknown protocol operation");
    return NULL;
}

/* ── Doubler: tp_descr_get and tp_descr_set ─────────────────────────── */

typedef struct {
    PyObject_HEAD
    long held;
} DoublerObject;

static PyObject *doubler_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    DoublerObject *self = (DoublerObject *)PyType_GenericAlloc(type, 0);
    if (self == NULL) {
        return NULL;
    }
    self->held = 0;
    return (PyObject *)self;
}

static PyObject *doubler_get(PyObject *self, PyObject *instance, PyObject *owner)
{
    if (instance == NULL) {
        Py_INCREF(self);
        return self;
    }
    return PyLong_FromLong(((DoublerObject *)self)->held);
}

static int doubler_set(PyObject *self, PyObject *instance, PyObject *value)
{
    if (value == NULL) {
        ((DoublerObject *)self)->held = 0;
        return 0;
    }
    long given = PyLong_AsLong(value);
    if (given == -1 && PyErr_Occurred()) {
        return -1;
    }
    ((DoublerObject *)self)->held = given * 2;
    return 0;
}

static PyTypeObject DoublerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "cpyext_types.Doubler",                     /* tp_name */
    sizeof(DoublerObject),                      /* tp_basicsize */
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
    "a data descriptor defined in C",           /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    doubler_get,                                /* tp_descr_get */
    doubler_set,                                /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    doubler_new,                                /* tp_new */
};

/* ── Owner: an observable tp_dealloc, and a slot that can close a cycle ─ */

typedef struct {
    PyObject_HEAD
    PyObject *held;
} OwnerObject;

static long owner_deallocs;

static PyObject *owner_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject *held = NULL;
    if (!PyArg_ParseTuple(args, "|O", &held)) {
        return NULL;
    }
    OwnerObject *self = (OwnerObject *)PyType_GenericAlloc(type, 0);
    if (self == NULL) {
        return NULL;
    }
    Py_XINCREF(held);
    self->held = held;
    return (PyObject *)self;
}

static void owner_dealloc(PyObject *self)
{
    OwnerObject *owner = (OwnerObject *)self;
    owner_deallocs += 1;
    Py_CLEAR(owner->held);
    Py_TYPE(self)->tp_free(self);
}

static PyObject *owner_get_held(PyObject *self, PyObject *unused)
{
    OwnerObject *owner = (OwnerObject *)self;
    PyObject *held = owner->held == NULL ? Py_None : owner->held;
    Py_INCREF(held);
    return held;
}

static PyObject *owner_set_held(PyObject *self, PyObject *value)
{
    OwnerObject *owner = (OwnerObject *)self;
    PyObject *previous = owner->held;
    Py_INCREF(value);
    owner->held = value;
    Py_XDECREF(previous);
    Py_RETURN_NONE;
}

static PyMethodDef owner_methods[] = {
    {"held", owner_get_held, METH_NOARGS, "the object this instance holds"},
    {"hold", owner_set_held, METH_O, "hold an object"},
    {NULL, NULL, 0, NULL},
};

static PyObject *m_owner_deallocs(PyObject *self, PyObject *unused)
{
    return PyLong_FromLong(owner_deallocs);
}

static PyTypeObject OwnerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "cpyext_types.Owner",                       /* tp_name */
    sizeof(OwnerObject),                        /* tp_basicsize */
    0,                                          /* tp_itemsize */
    owner_dealloc,                              /* tp_dealloc */
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
    "counts its own deallocations",             /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    owner_methods,                              /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    owner_new,                                  /* tp_new */
};

/* ── Blob: the buffer table ─────────────────────────────────────────── */

typedef struct {
    PyObject_HEAD
    char bytes[8];
    Py_ssize_t length;
    long exports;
} BlobObject;

static PyObject *blob_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    const char *text = "";
    Py_ssize_t size = 0;
    if (!PyArg_ParseTuple(args, "|y#", &text, &size)) {
        return NULL;
    }
    if (size > (Py_ssize_t)sizeof(((BlobObject *)0)->bytes)) {
        PyErr_SetString(PyExc_ValueError, "a Blob holds at most eight bytes");
        return NULL;
    }
    BlobObject *self = (BlobObject *)PyType_GenericAlloc(type, 0);
    if (self == NULL) {
        return NULL;
    }
    memcpy(self->bytes, text, (size_t)size);
    self->length = size;
    self->exports = 0;
    return (PyObject *)self;
}

static int blob_getbuffer(PyObject *self, Py_buffer *view, int flags)
{
    BlobObject *blob = (BlobObject *)self;
    if (PyBuffer_FillInfo(view, self, blob->bytes, blob->length, 0, flags) < 0) {
        return -1;
    }
    /* `internal` is the exporter's own state: it has to survive until the
       paired release, which is what proves the structure is handed back. */
    view->internal = (void *)blob;
    blob->exports += 1;
    return 0;
}

static void blob_releasebuffer(PyObject *self, Py_buffer *view)
{
    BlobObject *blob = (BlobObject *)view->internal;
    if (blob != NULL) {
        blob->exports -= 1;
    }
}

static PyObject *blob_exports(PyObject *self, PyObject *unused)
{
    return PyLong_FromLong(((BlobObject *)self)->exports);
}

/* PyObject_GetBuffer over whatever object is handed in, from C. */
static PyObject *blob_read(PyObject *self, PyObject *source)
{
    Py_buffer view;
    if (PyObject_GetBuffer(source, &view, PyBUF_SIMPLE) < 0) {
        return NULL;
    }
    PyObject *copy = PyBytes_FromStringAndSize((const char *)view.buf, view.len);
    PyBuffer_Release(&view);
    return copy;
}

static PyMethodDef blob_methods[] = {
    {"exports", blob_exports, METH_NOARGS, "live export count"},
    {"read", blob_read, METH_O, "PyObject_GetBuffer over any exporter"},
    {NULL, NULL, 0, NULL},
};

static PyBufferProcs blob_as_buffer = {
    blob_getbuffer,                             /* bf_getbuffer */
    blob_releasebuffer,                         /* bf_releasebuffer */
};

static PyTypeObject BlobType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "cpyext_types.Blob",                        /* tp_name */
    sizeof(BlobObject),                         /* tp_basicsize */
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
    &blob_as_buffer,                            /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    "a byte exporter defined in C",             /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    blob_methods,                               /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    blob_new,                                   /* tp_new */
};

/* ── Ticker: the async table ────────────────────────────────────────── */

typedef struct {
    PyObject_HEAD
    long remaining;
} TickerObject;

static PyTypeObject TickerType;

static PyObject *ticker_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    long count = 0;
    if (!PyArg_ParseTuple(args, "|l", &count)) {
        return NULL;
    }
    TickerObject *self = (TickerObject *)PyType_GenericAlloc(type, 0);
    if (self == NULL) {
        return NULL;
    }
    self->remaining = count;
    return (PyObject *)self;
}

/* `__await__` answers with an ordinary iterator over the remaining ticks. */
static PyObject *ticker_await(PyObject *self)
{
    PyObject *countdown = PyList_New(0);
    if (countdown == NULL) {
        return NULL;
    }
    for (long i = ((TickerObject *)self)->remaining; i > 0; i--) {
        PyObject *item = PyLong_FromLong(i);
        if (item == NULL || PyList_Append(countdown, item) < 0) {
            Py_XDECREF(item);
            Py_DECREF(countdown);
            return NULL;
        }
        Py_DECREF(item);
    }
    PyObject *iterator = PyObject_GetIter(countdown);
    Py_DECREF(countdown);
    return iterator;
}

static PyObject *ticker_aiter(PyObject *self)
{
    Py_INCREF(self);
    return self;
}

/* NULL with no exception set is the end, as it is for `tp_iternext`. */
static PyObject *ticker_anext(PyObject *self)
{
    TickerObject *ticker = (TickerObject *)self;
    if (ticker->remaining <= 0) {
        return NULL;
    }
    ticker->remaining -= 1;
    return PyLong_FromLong(ticker->remaining);
}

static PySendResult ticker_send(PyObject *self, PyObject *value, PyObject **result)
{
    TickerObject *ticker = (TickerObject *)self;
    if (ticker->remaining <= 0) {
        *result = PyLong_FromLong(-1);
        return *result == NULL ? PYGEN_ERROR : PYGEN_RETURN;
    }
    ticker->remaining -= 1;
    *result = PyLong_FromLong(ticker->remaining);
    return *result == NULL ? PYGEN_ERROR : PYGEN_NEXT;
}

static PyAsyncMethods ticker_as_async = {
    ticker_await,                               /* am_await */
    ticker_aiter,                               /* am_aiter */
    ticker_anext,                               /* am_anext */
    ticker_send,                                /* am_send */
};

/* Drive `PyIter_Send` from C so `am_send` is reached. */
static PyObject *m_send(PyObject *self, PyObject *args)
{
    PyObject *iterator = NULL;
    PyObject *value = Py_None;
    if (!PyArg_ParseTuple(args, "O|O", &iterator, &value)) {
        return NULL;
    }
    PyObject *stepped = NULL;
    PySendResult status = PyIter_Send(iterator, value, &stepped);
    if (status == PYGEN_ERROR) {
        return NULL;
    }
    PyObject *label = PyUnicode_FromString(status == PYGEN_RETURN ? "return" : "next");
    if (label == NULL) {
        Py_XDECREF(stepped);
        return NULL;
    }
    PyObject *pair = Py_BuildValue("(OO)", label, stepped);
    Py_DECREF(label);
    Py_XDECREF(stepped);
    return pair;
}

static PyTypeObject TickerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "cpyext_types.Ticker",                      /* tp_name */
    sizeof(TickerObject),                       /* tp_basicsize */
    0,                                          /* tp_itemsize */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_vectorcall_offset */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    &ticker_as_async,                           /* tp_as_async */
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
    "an async iterator defined in C",           /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
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
    ticker_new,                                 /* tp_new */
};

/* ── Spec: a heap type built with PyType_FromSpec ───────────────────── */

typedef struct {
    PyObject_HEAD
    long code;
} SpecObject;

static PyObject *spec_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    long code = 0;
    if (!PyArg_ParseTuple(args, "|l", &code)) {
        return NULL;
    }
    SpecObject *self = (SpecObject *)PyType_GenericAlloc(type, 0);
    if (self == NULL) {
        return NULL;
    }
    self->code = code;
    return (PyObject *)self;
}

static PyObject *spec_repr(PyObject *self)
{
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "Spec(%ld)", ((SpecObject *)self)->code);
    return PyUnicode_FromString(buffer);
}

static PyObject *spec_double(PyObject *self, PyObject *unused)
{
    return PyLong_FromLong(((SpecObject *)self)->code * 2);
}

static Py_ssize_t spec_length(PyObject *self)
{
    return ((SpecObject *)self)->code;
}

static PyMethodDef spec_methods[] = {
    {"double", spec_double, METH_NOARGS, "twice the code"},
    {NULL, NULL, 0, NULL},
};

static PyMemberDef spec_members[] = {
    {"code", Py_T_LONG, offsetof(SpecObject, code), 0, "the code"},
    {NULL, 0, 0, 0, NULL},
};

static PyType_Slot spec_slots[] = {
    {Py_tp_new, spec_new},
    {Py_tp_repr, spec_repr},
    {Py_tp_methods, spec_methods},
    {Py_tp_members, spec_members},
    {Py_tp_doc, (void *)"a heap type built from a spec"},
    {Py_sq_length, spec_length},
    /* Py_TP_USE_SPEC: the spec's own address becomes the token. */
    {Py_tp_token, Py_TP_USE_SPEC},
    {0, NULL},
};

static PyType_Spec spec_spec = {
    "cpyext_types.Spec",
    sizeof(SpecObject),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    spec_slots,
};

/* ── Extra: a spec declaring storage relative to its base's ─────────── */

typedef struct {
    long tag;
    double weight;
} extra_data;

static PyObject *extra_set(PyObject *self, PyObject *args)
{
    long tag = 0;
    double weight = 0.0;
    if (!PyArg_ParseTuple(args, "ld", &tag, &weight)) {
        return NULL;
    }
    extra_data *data = PyObject_GetTypeData(self, Py_TYPE(self));
    if (data == NULL) {
        return NULL;
    }
    data->tag = tag;
    data->weight = weight;
    Py_RETURN_NONE;
}

static PyObject *extra_get(PyObject *self, PyObject *unused)
{
    (void)unused;
    extra_data *data = PyObject_GetTypeData(self, Py_TYPE(self));
    if (data == NULL) {
        return NULL;
    }
    /* The code the base declared is still readable through the base's own
       fields, which is what a relative basicsize is for. */
    return Py_BuildValue("(ldl)", data->tag, data->weight,
                         ((SpecObject *)self)->code);
}

static PyMethodDef extra_methods[] = {
    {"set", extra_set, METH_VARARGS, "fill the extra type data"},
    {"get", extra_get, METH_NOARGS, "read the extra type data back"},
    {NULL, NULL, 0, NULL},
};

static char extra_token;

static PyType_Slot extra_slots[] = {
    {Py_tp_methods, extra_methods},
    {Py_tp_token, &extra_token},
    {0, NULL},
};

static PyType_Spec extra_spec = {
    "cpyext_types.Extra",
    -(int)sizeof(extra_data),
    0,
    Py_TPFLAGS_DEFAULT,
    extra_slots,
};

/* ── capsules and imports ───────────────────────────────────────────── */

static long capsule_payload = 4242;
static PyObject *module_capsule;

static void capsule_destructor(PyObject *capsule)
{
    (void)capsule;
}

static PyObject *m_capsule_read(PyObject *self, PyObject *capsule)
{
    void *pointer = PyCapsule_GetPointer(capsule, "cpyext_types.PAYLOAD");
    if (pointer == NULL) {
        return NULL;
    }
    return PyLong_FromLong(*(long *)pointer);
}

static PyObject *m_capsule_facts(PyObject *self, PyObject *capsule)
{
    const char *name = PyCapsule_GetName(capsule);
    if (name == NULL && PyErr_Occurred()) {
        return NULL;
    }
    int valid = PyCapsule_IsValid(capsule, "cpyext_types.PAYLOAD");
    int mismatched = PyCapsule_IsValid(capsule, "other");
    int exact = PyCapsule_CheckExact(capsule);
    void *context = PyCapsule_GetContext(capsule);
    return Py_BuildValue("(siiii)", name, valid, mismatched, exact, context != NULL);
}

static PyObject *m_capsule_wrong_name(PyObject *self, PyObject *capsule)
{
    void *pointer = PyCapsule_GetPointer(capsule, "cpyext_types.other");
    if (pointer == NULL) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *m_capsule_import(PyObject *self, PyObject *unused)
{
    void *pointer = PyCapsule_Import("cpyext_types.PAYLOAD", 0);
    if (pointer == NULL) {
        return NULL;
    }
    return PyLong_FromLong(*(long *)pointer);
}

static PyObject *m_import(PyObject *self, PyObject *args)
{
    const char *name = NULL;
    const char *attribute = NULL;
    if (!PyArg_ParseTuple(args, "ss", &name, &attribute)) {
        return NULL;
    }
    PyObject *module = PyImport_ImportModule(name);
    if (module == NULL) {
        return NULL;
    }
    PyObject *value = PyObject_GetAttrString(module, attribute);
    Py_DECREF(module);
    return value;
}

static PyObject *m_import_ref(PyObject *self, PyObject *args)
{
    const char *name = NULL;
    if (!PyArg_ParseTuple(args, "s", &name)) {
        return NULL;
    }
    return PyImport_AddModuleRef(name);
}

/* Build a type whose base is `int` and report the fast-subclass flag it
   earned, alongside the flag `Point` (based on `object`) earned. */
static PyObject *m_subclass_flags(PyObject *self, PyObject *unused)
{
    (void)self; (void)unused;
    PyObject *zero = PyLong_FromLong(0);
    if (zero == NULL) {
        return NULL;
    }
    PyObject *int_type = (PyObject *)Py_TYPE(zero);
    Py_INCREF(int_type);
    Py_DECREF(zero);

    PyType_Slot slots[] = {
        {Py_tp_base, int_type},
        {0, NULL},
    };
    PyType_Spec spec;
    spec.name = "cpyext_types.Counted";
    spec.basicsize = 0;
    spec.itemsize = 0;
    spec.flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    spec.slots = slots;

    PyObject *made = PyType_FromSpec(&spec);
    Py_DECREF(int_type);
    if (made == NULL) {
        return NULL;
    }
    unsigned long made_flags = PyType_GetFlags((PyTypeObject *)made);
    unsigned long point_flags = PyType_GetFlags(&PointType);
    PyObject *result = Py_BuildValue(
        "(Oiii)", made,
        (made_flags & Py_TPFLAGS_LONG_SUBCLASS) != 0,
        (made_flags & Py_TPFLAGS_UNICODE_SUBCLASS) != 0,
        (point_flags & Py_TPFLAGS_LONG_SUBCLASS) != 0);
    Py_DECREF(made);
    return result;
}

/* What a spec type reports about the module and the token it was built with. */
static PyObject *m_type_owner(PyObject *self, PyObject *type)
{
    if (!PyType_Check(type)) {
        PyErr_SetString(PyExc_TypeError, "type_owner wants a type");
        return NULL;
    }
    PyTypeObject *cls = (PyTypeObject *)type;
    PyObject *module = PyType_GetModule(cls);  /* borrowed */
    if (module == NULL) {
        return NULL;
    }
    PyObject *by_def = PyType_GetModuleByDef(cls, &moduledef);  /* borrowed */
    if (by_def == NULL) {
        return NULL;
    }
    PyObject *module_name = PyType_GetModuleName(cls);
    if (module_name == NULL) {
        return NULL;
    }
    PyObject *qualified = PyType_GetFullyQualifiedName(cls);
    if (qualified == NULL) {
        Py_DECREF(module_name);
        return NULL;
    }
    PyObject *result = Py_BuildValue(
        "(OiNN)", module, by_def == module, module_name, qualified);
    return result;
}

/* `PyType_GetBaseByToken` against the two tokens the fixture declares, and
   against one no type carries. */
static PyObject *m_type_token(PyObject *self, PyObject *type)
{
    static char stranger;
    PyTypeObject *cls = (PyTypeObject *)type;
    PyTypeObject *from_spec = NULL;
    int spec_found = PyType_GetBaseByToken(cls, &spec_spec, &from_spec);
    if (spec_found < 0) {
        return NULL;
    }
    PyTypeObject *from_extra = NULL;
    int extra_found = PyType_GetBaseByToken(cls, &extra_token, &from_extra);
    if (extra_found < 0) {
        Py_XDECREF(from_spec);
        return NULL;
    }
    int absent = PyType_GetBaseByToken(cls, &stranger, NULL);
    if (absent < 0) {
        Py_XDECREF(from_spec);
        Py_XDECREF(from_extra);
        return NULL;
    }
    PyObject *result = Py_BuildValue(
        "(iOiOi)",
        spec_found, from_spec ? (PyObject *)from_spec : Py_None,
        extra_found, from_extra ? (PyObject *)from_extra : Py_None,
        absent);
    Py_XDECREF(from_spec);
    Py_XDECREF(from_extra);
    return result;
}

/* A NULL token is the caller's error, not a miss. */
static PyObject *m_type_token_null(PyObject *self, PyObject *type)
{
    if (PyType_GetBaseByToken((PyTypeObject *)type, NULL, NULL) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *m_type_data_size(PyObject *self, PyObject *type)
{
    return PyLong_FromSsize_t(PyType_GetTypeDataSize((PyTypeObject *)type));
}

/* `PyType_Freeze` on a class handed in from Python, plus the cache calls the
   C side is expected to make after rewriting a namespace. */
static PyObject *m_freeze(PyObject *self, PyObject *type)
{
    if (!PyType_Check(type)) {
        PyErr_SetString(PyExc_TypeError, "freeze wants a type");
        return NULL;
    }
    if (PyType_Freeze((PyTypeObject *)type) < 0) {
        return NULL;
    }
    PyType_Modified((PyTypeObject *)type);
    PyType_ClearCache();
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"type_owner", m_type_owner, METH_O, "the module a spec type belongs to"},
    {"type_token", m_type_token, METH_O, "PyType_GetBaseByToken three ways"},
    {"type_token_null", m_type_token_null, METH_O, "a NULL token is an error"},
    {"type_data_size", m_type_data_size, METH_O, "PyType_GetTypeDataSize"},
    {"freeze", m_freeze, METH_O, "PyType_Freeze a class"},
    {"capsule_read", m_capsule_read, METH_O, "read the capsule payload"},
    {"capsule_facts", m_capsule_facts, METH_O, "name, validity and context"},
    {"capsule_wrong_name", m_capsule_wrong_name, METH_O, "fetch under a wrong name"},
    {"capsule_import", m_capsule_import, METH_NOARGS, "PyCapsule_Import round trip"},
    {"import_attr", m_import, METH_VARARGS, "PyImport_ImportModule then getattr"},
    {"add_module_ref", m_import_ref, METH_VARARGS, "PyImport_AddModuleRef"},
    {"protocol", m_protocol, METH_VARARGS, "drive one abstract protocol call"},
    {"send", m_send, METH_VARARGS, "PyIter_Send one step"},
    {"make", m_make, METH_VARARGS, "build a Point without calling the class"},
    {"is_point", m_is_point, METH_O, "PyObject_TypeCheck"},
    {"sum_x", m_sum_x, METH_O, "read x, raising the module exception otherwise"},
    {"flags", m_flags, METH_NOARGS, "PyType_GetFlags on Point"},
    {"subclass_flags", m_subclass_flags, METH_NOARGS, "fast-subclass flags"},
    {"is_subtype", m_is_subtype, METH_NOARGS, "PyType_IsSubtype both ways"},
    {"owner_deallocs", m_owner_deallocs, METH_NOARGS, "how often Owner.tp_dealloc ran"},
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
    if (PyModule_AddType(module, &VecType) < 0) {
        return -1;
    }
    if (PyModule_AddType(module, &BagType) < 0) {
        return -1;
    }
    if (PyModule_AddType(module, &TableType) < 0) {
        return -1;
    }
    if (PyModule_AddType(module, &DoublerType) < 0) {
        return -1;
    }
    if (PyModule_AddType(module, &BlobType) < 0) {
        return -1;
    }
    if (PyModule_AddType(module, &TickerType) < 0) {
        return -1;
    }
    if (PyModule_AddType(module, &OwnerType) < 0) {
        return -1;
    }
    if (PyModule_AddStringConstant(module, "ANSWER_TYPES", "types") < 0) {
        return -1;
    }
    PyObject *spec_type = PyType_FromModuleAndSpec(module, &spec_spec, NULL);
    if (spec_type == NULL) {
        return -1;
    }
    if (PyType_GetSlot((PyTypeObject *)spec_type, Py_tp_repr) != (void *)spec_repr) {
        PyErr_SetString(PyExc_SystemError, "PyType_GetSlot did not answer with tp_repr");
        Py_DECREF(spec_type);
        return -1;
    }
    /* `Extra` extends `Spec`: its spec asks for storage relative to its base
       rather than for a whole block, which is what `PyType_FromMetaclass`
       takes an explicit metaclass and bases for. */
    PyObject *extra_type = PyType_FromMetaclass(NULL, module, &extra_spec, spec_type);
    if (extra_type == NULL) {
        Py_DECREF(spec_type);
        return -1;
    }
    int added = PyModule_AddObjectRef(module, "Extra", extra_type);
    Py_DECREF(extra_type);
    if (added < 0) {
        Py_DECREF(spec_type);
        return -1;
    }
    added = PyModule_AddObjectRef(module, "Spec", spec_type);
    Py_DECREF(spec_type);
    if (added < 0) {
        return -1;
    }
    TypesError = PyErr_NewExceptionWithDoc("cpyext_types.TypesError",
                                           "raised by the fixture", NULL, NULL);
    if (TypesError == NULL) {
        return -1;
    }
    if (PyModule_AddObjectRef(module, "TypesError", TypesError) < 0) {
        return -1;
    }
    module_capsule = PyCapsule_New(&capsule_payload, "cpyext_types.PAYLOAD",
                                   capsule_destructor);
    if (module_capsule == NULL) {
        return -1;
    }
    if (PyCapsule_SetContext(module_capsule, &capsule_payload) < 0) {
        return -1;
    }
    return PyModule_AddObjectRef(module, "PAYLOAD", module_capsule);
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
