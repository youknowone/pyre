/* A type whose instances this runtime's allocator never handed out.

   `_cffi_backend.c allocate_owning_object` says it plainly: "objects with
   &CDataOwning_Type are always allocated with either a plain malloc() or
   calloc(), and freed with free()".  A block like that is unknown to whatever
   census the object allocator keeps, and the deallocator's `free` leaves that
   census unchanged -- so nothing may be read out of the block once the
   deallocator has returned.

   Instances come from a pool here rather than `malloc` so that the words a
   released block held are still there to be read: the deallocator fills its
   slot with a pattern, and a write that reaches the block afterwards shows up
   as a byte that is no longer the pattern. */

#include <Python.h>
#include <string.h>

#define SLOT_BYTES 128
#define SLOTS 8
#define PATTERN 0xA5

typedef struct {
    PyObject_HEAD
    long tag;
} Foreign;

typedef union {
    Foreign object;
    char bytes[SLOT_BYTES];
    double alignment;
} Slot;

static Slot pool[SLOTS];
static int taken;
static char *released;

static void foreign_dealloc(PyObject *self)
{
    released = (char *)self;
    memset(released, PATTERN, SLOT_BYTES);
}

static PyTypeObject ForeignType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "cpyext_foreign_block.Foreign",
    .tp_basicsize = sizeof(Foreign),
    .tp_dealloc = foreign_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

/* One instance, built on a block of this module's own and let go of again.

   `fill` is what the slot holds when `PyObject_Init` is handed it: zero is
   what `calloc` leaves, and anything else is what `malloc` leaves --
   `_cffi_backend.c allocate_owning_object` uses whichever the allocator it
   was given asks for, and its `should_clear_after_alloc=False` case is the
   second. */
static PyObject *make(PyObject *self, PyObject *arg)
{
    Slot *slot;
    long fill;

    (void)self;
    fill = PyLong_AsLong(arg);
    if (fill == -1 && PyErr_Occurred()) {
        return NULL;
    }
    if (taken >= SLOTS) {
        PyErr_SetString(PyExc_RuntimeError, "the pool is spent");
        return NULL;
    }
    slot = &pool[taken++];
    memset(slot, (int)(unsigned char)fill, sizeof(*slot));
    if (PyObject_Init((PyObject *)slot, &ForeignType) == NULL) {
        return NULL;
    }
    slot->object.tag = 1234;
    Py_DECREF((PyObject *)slot);
    Py_RETURN_NONE;
}

/* Whether the block the deallocator gave back still reads as it left it. */
static PyObject *released_intact(PyObject *self, PyObject *noargs)
{
    size_t i;

    (void)self;
    (void)noargs;
    if (released == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "no block has been released");
        return NULL;
    }
    for (i = 0; i < SLOT_BYTES; i++) {
        if ((unsigned char)released[i] != PATTERN) {
            return PyLong_FromSsize_t((Py_ssize_t)i);
        }
    }
    return PyLong_FromSsize_t(-1);
}

static PyObject *released_yet(PyObject *self, PyObject *noargs)
{
    (void)self;
    (void)noargs;
    return PyBool_FromLong(released != NULL);
}

static PyMethodDef methods[] = {
    {"make", make, METH_O, NULL},
    {"released_intact", released_intact, METH_NOARGS, NULL},
    {"released_yet", released_yet, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "cpyext_foreign_block", NULL, -1, methods,
};

PyMODINIT_FUNC PyInit_cpyext_foreign_block(void)
{
    if (PyType_Ready(&ForeignType) < 0) {
        return NULL;
    }
    return PyModule_Create(&module);
}
