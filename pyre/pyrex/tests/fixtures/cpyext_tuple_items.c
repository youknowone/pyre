/* The tuple item array an extension reads and writes without a call.

   `PyTuple_GET_ITEM` is an lvalue -- cffi's `realize_c_type.c` writes
   `(CTypeDescrObject **)&PyTuple_GET_ITEM(fargs, 0)` -- and `PyTuple_SET_ITEM`
   beside it is the plain assignment: it overwrites a filled slot, takes the
   value's reference, and gives back nothing.  `ffi_obj.c
   _ffi_callback_decorator` leans on all of that at once, which is the shape
   `borrow_swap` below is.

   Every expectation was taken from CPython 3.14.6 running the same script
   against this same fixture. */

#include <Python.h>

/* Read slot 0 through the address of the slot, which is what an extension
   takes when it wants to pass the slot along by reference. */
static PyObject *first_through_address(PyObject *self, PyObject *tuple)
{
    PyObject **slot;

    (void)self;
    if (!PyTuple_Check(tuple)) {
        PyErr_SetString(PyExc_TypeError, "tuple expected");
        return NULL;
    }
    if (PyTuple_GET_SIZE(tuple) < 1) {
        PyErr_SetString(PyExc_IndexError, "empty");
        return NULL;
    }
    slot = &PyTuple_GET_ITEM(tuple, 0);
    return Py_NewRef(*slot);
}

/* Every slot, read one at a time, as a list. */
static PyObject *items(PyObject *self, PyObject *tuple)
{
    Py_ssize_t i;
    PyObject *out;

    (void)self;
    out = PyList_New(0);
    if (out == NULL) {
        return NULL;
    }
    for (i = 0; i < PyTuple_GET_SIZE(tuple); i++) {
        if (PyList_Append(out, PyTuple_GET_ITEM(tuple, i)) < 0) {
            Py_DECREF(out);
            return NULL;
        }
    }
    return out;
}

/* `_ffi_callback_decorator`'s shape: slot 1 is read, a borrowed object is put
   in its place for the length of one call, and the old value is put back.
   Neither write owns what it stores.

   Answers with what the reader saw and what slot 1 read from C while the
   substitute was in place; the caller checks the tuple afterwards. */
static PyObject *borrow_swap(PyObject *self, PyObject *args)
{
    PyObject *outer, *fn, *reader, *old, *seen, *result;

    (void)self;
    if (!PyArg_ParseTuple(args, "OOO", &outer, &fn, &reader)) {
        return NULL;
    }
    old = PyTuple_GET_ITEM(outer, 1);
    PyTuple_SET_ITEM(outer, 1, fn);
    seen = PyObject_CallFunctionObjArgs(reader, outer, NULL);
    result = seen == NULL ? NULL
                          : Py_BuildValue("(OO)", seen, PyTuple_GET_ITEM(outer, 1));
    Py_XDECREF(seen);
    PyTuple_SET_ITEM(outer, 1, old);
    return result;
}

/* `PyTuple_SetItem` over a slot that already holds something: the new value's
   reference is taken and the old one given back. */
static PyObject *set_twice(PyObject *self, PyObject *args)
{
    PyObject *first, *second, *tuple;

    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &first, &second)) {
        return NULL;
    }
    tuple = PyTuple_New(1);
    if (tuple == NULL) {
        return NULL;
    }
    if (PyTuple_SetItem(tuple, 0, Py_NewRef(first)) < 0) {
        Py_DECREF(tuple);
        return NULL;
    }
    if (PyTuple_SetItem(tuple, 0, Py_NewRef(second)) < 0) {
        Py_DECREF(tuple);
        return NULL;
    }
    return tuple;
}

/* A slot written after C has taken the array's address is read through it. */
static PyObject *address_then_set(PyObject *self, PyObject *value)
{
    PyObject **slot, *tuple, *read;

    (void)self;
    tuple = PyTuple_New(1);
    if (tuple == NULL) {
        return NULL;
    }
    slot = &PyTuple_GET_ITEM(tuple, 0);
    if (PyTuple_SetItem(tuple, 0, Py_NewRef(value)) < 0) {
        Py_DECREF(tuple);
        return NULL;
    }
    read = Py_NewRef(*slot);
    Py_DECREF(tuple);
    return read;
}

static PyMethodDef methods[] = {
    {"first_through_address", first_through_address, METH_O, NULL},
    {"items", items, METH_O, NULL},
    {"borrow_swap", borrow_swap, METH_VARARGS, NULL},
    {"set_twice", set_twice, METH_VARARGS, NULL},
    {"address_then_set", address_then_set, METH_O, NULL},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "cpyext_tuple_items", NULL, -1, methods,
};

PyMODINIT_FUNC PyInit_cpyext_tuple_items(void)
{
    return PyModule_Create(&module);
}
