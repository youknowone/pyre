/* What an extension holds while it works: its own mutex, a lock it allocated,
   a buffer it resizes, and the error it chains onto one already on its way
   out. */

#include <Python.h>

#include <string.h>

/* The pending exception's class name and message, taken and cleared. */
static PyObject *pending(void)
{
    PyObject *value = PyErr_GetRaisedException();
    if (value == NULL) {
        Py_RETURN_NONE;
    }
    PyObject *text = PyObject_Str(value);
    PyObject *pair = Py_BuildValue("(sO)", Py_TYPE(value)->tp_name,
                                   text == NULL ? Py_None : text);
    Py_XDECREF(text);
    Py_DECREF(value);
    return pair;
}

/* ── the buffer an extension fills then cuts down ─────────────────────── */

/* The pattern a caller with an upper bound uses: take a buffer that big,
   write what it turns out to need, then resize to that. */
static PyObject *build_then_shrink(PyObject *self, PyObject *args)
{
    (void)self;
    const char *text;
    Py_ssize_t length;
    Py_ssize_t reserve;
    Py_ssize_t final;
    if (!PyArg_ParseTuple(args, "y#nn", &text, &length, &reserve, &final)) {
        return NULL;
    }
    PyObject *value = PyBytes_FromStringAndSize(NULL, reserve);
    if (value == NULL) {
        return NULL;
    }
    memcpy(PyBytes_AS_STRING(value), text, (size_t)length);
    if (_PyBytes_Resize(&value, final) < 0) {
        return Py_BuildValue("(OO)", Py_None, pending());
    }
    /* Read through the object, which is what decides the contents. */
    PyObject *row = Py_BuildValue("(nO)", PyBytes_GET_SIZE(value), value);
    Py_DECREF(value);
    return row;
}

/* The same, but the buffer is written after the resize -- which is what makes
   the answer say whether the address is still usable. */
static PyObject *shrink_then_fill(PyObject *self, PyObject *args)
{
    (void)self;
    Py_ssize_t reserve;
    Py_ssize_t final;
    if (!PyArg_ParseTuple(args, "nn", &reserve, &final)) {
        return NULL;
    }
    PyObject *value = PyBytes_FromStringAndSize(NULL, reserve);
    if (value == NULL) {
        return NULL;
    }
    if (_PyBytes_Resize(&value, final) < 0) {
        return Py_BuildValue("(OO)", Py_None, pending());
    }
    char *buffer = PyBytes_AS_STRING(value);
    for (Py_ssize_t index = 0; index < final; index++) {
        buffer[index] = (char)('a' + (index % 26));
    }
    PyObject *row = Py_BuildValue("(nO)", PyBytes_GET_SIZE(value), value);
    Py_DECREF(value);
    return row;
}

/* A `bytes` that already exists: the object is immutable, so the answer is a
   different one and the prefix is what carries over. */
static PyObject *resize_existing(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *source;
    Py_ssize_t final;
    if (!PyArg_ParseTuple(args, "On", &source, &final)) {
        return NULL;
    }
    PyObject *value = Py_NewRef(source);
    if (_PyBytes_Resize(&value, final) < 0) {
        return Py_BuildValue("(OOO)", Py_None, Py_False, pending());
    }
    Py_ssize_t kept = final < PyBytes_GET_SIZE(source) ? final : PyBytes_GET_SIZE(source);
    /* Only the prefix: what a grown buffer holds past the old length is the
       allocator's business until the caller writes it. */
    PyObject *prefix = PyBytes_FromStringAndSize(PyBytes_AS_STRING(value), kept);
    PyObject *row = Py_BuildValue("(nOO)", PyBytes_GET_SIZE(value), prefix,
                                  pending());
    Py_XDECREF(prefix);
    Py_DECREF(value);
    return row;
}

/* The refusals, which give the reference up on the way out. */
static PyObject *resize_refused(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *source;
    Py_ssize_t final;
    if (!PyArg_ParseTuple(args, "On", &source, &final)) {
        return NULL;
    }
    PyObject *value = Py_NewRef(source);
    int answer = _PyBytes_Resize(&value, final);
    PyObject *row = Py_BuildValue("(iOO)", answer, value == NULL ? Py_True : Py_False,
                                  pending());
    Py_XDECREF(value);
    return row;
}

/* ── the mutex an extension embeds ────────────────────────────────────── */

static PyMutex embedded;

static PyObject *mutex_states(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    (void)self;
    int before = PyMutex_IsLocked(&embedded);
    PyMutex_Lock(&embedded);
    int held = PyMutex_IsLocked(&embedded);
    PyMutex_Unlock(&embedded);
    int after = PyMutex_IsLocked(&embedded);
    /* Again, to say the byte really went back to where it started. */
    PyMutex_Lock(&embedded);
    int again = PyMutex_IsLocked(&embedded);
    PyMutex_Unlock(&embedded);
    return Py_BuildValue("(iiii)", before, held, after, again);
}

/* A mutex of the caller's own, zero-initialized as the header says. */
static PyObject *mutex_local(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    (void)self;
    PyMutex local = (PyMutex){0};
    int before = PyMutex_IsLocked(&local);
    PyMutex_Lock(&local);
    int held = PyMutex_IsLocked(&local);
    PyMutex_Unlock(&local);
    return Py_BuildValue("(iii)", before, held, PyMutex_IsLocked(&local));
}

/* ── the lock an extension allocates ──────────────────────────────────── */

static PyObject *thread_lock(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    (void)self;
    PyThread_type_lock lock = PyThread_allocate_lock();
    if (lock == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "no lock");
        return NULL;
    }
    int first = PyThread_acquire_lock(lock, NOWAIT_LOCK);
    int again = PyThread_acquire_lock(lock, NOWAIT_LOCK);
    PyThread_release_lock(lock);
    int after = PyThread_acquire_lock(lock, NOWAIT_LOCK);
    /* Held, so a wait with a deadline gives up rather than blocking for good. */
    PyLockStatus timed = PyThread_acquire_lock_timed(lock, 1000, 0);
    PyThread_release_lock(lock);
    /* Free, so a blocking wait takes it at once. */
    PyLockStatus waited = PyThread_acquire_lock_timed(lock, -1, 0);
    PyThread_release_lock(lock);
    PyThread_free_lock(lock);
    return Py_BuildValue("(iiiii)", first, again, after, (int)timed, (int)waited);
}

static PyObject *thread_ident(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    (void)self;
    return PyLong_FromUnsignedLong(PyThread_get_thread_ident());
}

/* ── chaining onto what was already on its way out ────────────────────── */

/* Raise `first` (or nothing), then chain `second` onto it. */
static PyObject *chain(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *first;
    PyObject *second;
    if (!PyArg_ParseTuple(args, "OO", &first, &second)) {
        return NULL;
    }
    if (first != Py_None) {
        PyErr_SetObject((PyObject *)Py_TYPE(first), first);
    }
    _PyErr_ChainExceptions1(second == Py_None ? NULL : Py_NewRef(second));
    PyObject *raised = PyErr_GetRaisedException();
    if (raised == NULL) {
        Py_RETURN_NONE;
    }
    PyObject *context = PyException_GetContext(raised);
    PyObject *row = Py_BuildValue("(OO)", raised, context == NULL ? Py_None : context);
    Py_XDECREF(context);
    Py_DECREF(raised);
    return row;
}

/* ── the mistake an extension reports about itself ────────────────────── */

/* The macro names the caller's own file and line, so both sides say the same
   thing about the same call. */
static PyObject *bad_internal_call(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    (void)self;
    PyErr_BadInternalCall();
    return pending();
}

static PyMethodDef methods[] = {
    {"build_then_shrink", build_then_shrink, METH_VARARGS, NULL},
    {"shrink_then_fill", shrink_then_fill, METH_VARARGS, NULL},
    {"resize_existing", resize_existing, METH_VARARGS, NULL},
    {"resize_refused", resize_refused, METH_VARARGS, NULL},
    {"mutex_states", mutex_states, METH_NOARGS, NULL},
    {"mutex_local", mutex_local, METH_NOARGS, NULL},
    {"thread_lock", thread_lock, METH_NOARGS, NULL},
    {"thread_ident", thread_ident, METH_NOARGS, NULL},
    {"chain", chain, METH_VARARGS, NULL},
    {"bad_internal_call", bad_internal_call, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef def = {PyModuleDef_HEAD_INIT, "cpyext_locks", NULL, -1,
                                 methods};

PyMODINIT_FUNC PyInit_cpyext_locks(void)
{
    return PyModule_Create(&def);
}
