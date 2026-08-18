/* The exception-object accessors and the two error-indicator families.

   Each function answers the observable outcome, so a Python-side comparison
   against CPython running the same code says whether the two agree. */

#include <Python.h>

/* The pending exception's class name and message, taken and cleared. */
static PyObject *refused(void)
{
    PyObject *value = PyErr_GetRaisedException();
    PyObject *text = value == NULL ? PyUnicode_FromString("") : PyObject_Str(value);
    PyObject *name = PyUnicode_FromString(
        value == NULL ? "?" : Py_TYPE(value)->tp_name);
    PyObject *pair = Py_BuildValue("(OO)", name, text);
    Py_XDECREF(value);
    Py_XDECREF(text);
    Py_XDECREF(name);
    return pair;
}

/* ── PyException_* ────────────────────────────────────────────────────── */

/* `__traceback__` read back through the accessor, `None` standing for the
   NULL the accessor answers when the slot is empty. */
static PyObject *exc_get_traceback(PyObject *self, PyObject *exc)
{
    (void)self;
    PyObject *traceback = PyException_GetTraceback(exc);
    if (traceback == NULL) {
        Py_RETURN_NONE;
    }
    return traceback;
}

/* Set the slot from `value`, then answer what the accessor and the attribute
   both say, so a setter that wrote somewhere else is visible. */
static PyObject *exc_set_traceback(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *exc, *value;
    if (!PyArg_ParseTuple(args, "OO", &exc, &value)) {
        return NULL;
    }
    if (PyException_SetTraceback(exc, value) < 0) {
        return refused();
    }
    PyObject *through = PyException_GetTraceback(exc);
    PyObject *attribute = PyObject_GetAttrString(exc, "__traceback__");
    if (attribute == NULL) {
        Py_XDECREF(through);
        return refused();
    }
    PyObject *answer = Py_BuildValue(
        "(OO)", through == NULL ? Py_None : through,
        (through == NULL ? Py_None : through) == attribute ? Py_True : Py_False);
    Py_XDECREF(through);
    Py_DECREF(attribute);
    return answer;
}

static PyObject *exc_get_cause(PyObject *self, PyObject *exc)
{
    (void)self;
    PyObject *cause = PyException_GetCause(exc);
    if (cause == NULL) {
        Py_RETURN_NONE;
    }
    return cause;
}

/* The setter steals, so the reference handed in is one this function owns. */
static PyObject *exc_set_cause(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *exc, *value;
    if (!PyArg_ParseTuple(args, "OO", &exc, &value)) {
        return NULL;
    }
    PyObject *stolen = value == Py_None ? NULL : Py_NewRef(value);
    PyException_SetCause(exc, stolen);
    PyObject *through = PyException_GetCause(exc);
    PyObject *suppress = PyObject_GetAttrString(exc, "__suppress_context__");
    if (suppress == NULL) {
        Py_XDECREF(through);
        return refused();
    }
    PyObject *answer = Py_BuildValue("(OO)", through == NULL ? Py_None : through, suppress);
    Py_XDECREF(through);
    Py_DECREF(suppress);
    return answer;
}

static PyObject *exc_get_context(PyObject *self, PyObject *exc)
{
    (void)self;
    PyObject *context = PyException_GetContext(exc);
    if (context == NULL) {
        Py_RETURN_NONE;
    }
    return context;
}

static PyObject *exc_set_context(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *exc, *value;
    if (!PyArg_ParseTuple(args, "OO", &exc, &value)) {
        return NULL;
    }
    PyObject *stolen = value == Py_None ? NULL : Py_NewRef(value);
    PyException_SetContext(exc, stolen);
    PyObject *through = PyException_GetContext(exc);
    PyObject *suppress = PyObject_GetAttrString(exc, "__suppress_context__");
    if (suppress == NULL) {
        Py_XDECREF(through);
        return refused();
    }
    PyObject *answer = Py_BuildValue("(OO)", through == NULL ? Py_None : through, suppress);
    Py_XDECREF(through);
    Py_DECREF(suppress);
    return answer;
}

static PyObject *exc_get_args(PyObject *self, PyObject *exc)
{
    (void)self;
    return PyException_GetArgs(exc);
}

/* The setter increfs rather than stealing, so the caller's reference survives:
   the row carries the value read back and whether it is the same object. */
static PyObject *exc_set_args(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *exc, *value;
    if (!PyArg_ParseTuple(args, "OO", &exc, &value)) {
        return NULL;
    }
    PyException_SetArgs(exc, value);
    PyObject *through = PyException_GetArgs(exc);
    PyObject *attribute = PyObject_GetAttrString(exc, "args");
    if (through == NULL || attribute == NULL) {
        Py_XDECREF(through);
        Py_XDECREF(attribute);
        return refused();
    }
    PyObject *answer = Py_BuildValue("(OO)", through, attribute);
    Py_DECREF(through);
    Py_DECREF(attribute);
    return answer;
}

/* The three classification spellings, answered together. */
static PyObject *exc_classify(PyObject *self, PyObject *object)
{
    (void)self;
    int is_class = PyExceptionClass_Check(object);
    int is_instance = PyExceptionInstance_Check(object);
    PyObject *class_of = (PyObject *)PyExceptionInstance_Class(object);
    const char *named = is_class ? PyExceptionClass_Name(object) : "";
    return Py_BuildValue("(OOsO)", is_class ? Py_True : Py_False,
                         is_instance ? Py_True : Py_False, named, class_of);
}

/* ── the raised indicator ─────────────────────────────────────────────── */

/* Set one, take it back, and answer what came out along with whether the
   indicator was left clear. */
static PyObject *raised_round_trip(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *type, *value;
    if (!PyArg_ParseTuple(args, "OO", &type, &value)) {
        return NULL;
    }
    PyErr_SetObject(type, value);
    PyObject *before = PyErr_Occurred();
    PyObject *taken = PyErr_GetRaisedException();
    PyObject *after = PyErr_Occurred();
    PyObject *answer = Py_BuildValue(
        "(OOO)", before == NULL ? Py_None : before, taken == NULL ? Py_None : taken,
        after == NULL ? Py_None : after);
    Py_XDECREF(taken);
    return answer;
}

/* Nothing pending: the answer is NULL and the indicator stays clear. */
static PyObject *raised_when_clear(PyObject *self, PyObject *ignored)
{
    (void)self;
    (void)ignored;
    PyObject *taken = PyErr_GetRaisedException();
    PyObject *answer = Py_BuildValue("(OO)", taken == NULL ? Py_None : taken,
                                     PyErr_Occurred() == NULL ? Py_None : PyErr_Occurred());
    Py_XDECREF(taken);
    return answer;
}

/* The setter steals, so the round trip hands back the identical object. */
static PyObject *raised_set(PyObject *self, PyObject *exc)
{
    (void)self;
    PyErr_SetRaisedException(Py_NewRef(exc));
    PyObject *occurred = PyErr_Occurred();
    PyObject *taken = PyErr_GetRaisedException();
    PyObject *answer = Py_BuildValue("(OOO)", occurred == NULL ? Py_None : occurred,
                                     taken == NULL ? Py_None : taken,
                                     taken == exc ? Py_True : Py_False);
    Py_XDECREF(taken);
    return answer;
}

/* A NULL argument is the clear spelling, and the exception it displaces is
   released rather than leaked. */
static PyObject *raised_set_null(PyObject *self, PyObject *exc)
{
    (void)self;
    PyErr_SetRaisedException(Py_NewRef(exc));
    Py_ssize_t while_pending = Py_REFCNT(exc);
    PyErr_SetRaisedException(NULL);
    Py_ssize_t after_clear = Py_REFCNT(exc);
    return Py_BuildValue("(OnO)", PyErr_Occurred() == NULL ? Py_None : PyErr_Occurred(),
                         while_pending - after_clear,
                         PyErr_GetRaisedException() == NULL ? Py_True : Py_False);
}

/* Replacing a pending exception releases the one displaced. */
static PyObject *raised_replace(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *first, *second;
    if (!PyArg_ParseTuple(args, "OO", &first, &second)) {
        return NULL;
    }
    PyErr_SetRaisedException(Py_NewRef(first));
    Py_ssize_t while_pending = Py_REFCNT(first);
    PyErr_SetRaisedException(Py_NewRef(second));
    Py_ssize_t after_replace = Py_REFCNT(first);
    PyObject *taken = PyErr_GetRaisedException();
    PyObject *answer = Py_BuildValue("(nO)", while_pending - after_replace,
                                     taken == NULL ? Py_None : taken);
    Py_XDECREF(taken);
    return answer;
}

/* The triple `PyErr_Fetch` hands back for an exception this call raises, the
   traceback slot included. */
static PyObject *fetch_triple(PyObject *self, PyObject *exc)
{
    (void)self;
    PyObject *type, *value, *traceback;
    if (exc != Py_None) {
        PyErr_SetRaisedException(Py_NewRef(exc));
    }
    PyErr_Fetch(&type, &value, &traceback);
    PyObject *answer = Py_BuildValue(
        "(OOOO)", type == NULL ? Py_None : type, value == NULL ? Py_None : value,
        traceback == NULL ? Py_None : traceback,
        PyErr_Occurred() == NULL ? Py_None : PyErr_Occurred());
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
    return answer;
}

/* Fetch the triple and hand it straight back, which is the save-and-restore an
   extension does around work of its own.  Nothing may be lost by the pair. */
static PyObject *fetch_and_restore(PyObject *self, PyObject *exc)
{
    (void)self;
    PyObject *type, *value, *traceback;
    PyErr_SetRaisedException(Py_NewRef(exc));
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_Restore(type, value, traceback);
    PyObject *back = PyErr_GetRaisedException();
    if (back == NULL) {
        Py_RETURN_NONE;
    }
    PyObject *own = PyException_GetTraceback(back);
    PyObject *answer = Py_BuildValue("(OOO)", back, own == NULL ? Py_None : own,
                                     back == exc ? Py_True : Py_False);
    Py_XDECREF(own);
    Py_DECREF(back);
    return answer;
}

/* `PyErr_SetObject` chains onto the exception being handled, the way a raise
   from inside an `except` block does. */
static PyObject *set_object_context(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *type, *value;
    if (!PyArg_ParseTuple(args, "OO", &type, &value)) {
        return NULL;
    }
    PyErr_SetObject(type, value);
    PyObject *raised = PyErr_GetRaisedException();
    if (raised == NULL) {
        Py_RETURN_NONE;
    }
    PyObject *context = PyException_GetContext(raised);
    PyObject *answer = Py_BuildValue("(OO)", raised, context == NULL ? Py_None : context);
    Py_XDECREF(context);
    Py_DECREF(raised);
    return answer;
}

/* ── the handled exception ────────────────────────────────────────────── */

static PyObject *handled_get(PyObject *self, PyObject *ignored)
{
    (void)self;
    (void)ignored;
    PyObject *handled = PyErr_GetHandledException();
    if (handled == NULL) {
        Py_RETURN_NONE;
    }
    return handled;
}

/* Two consecutive reads: the same object comes back and nothing is cleared. */
static PyObject *handled_twice(PyObject *self, PyObject *ignored)
{
    (void)self;
    (void)ignored;
    PyObject *first = PyErr_GetHandledException();
    PyObject *second = PyErr_GetHandledException();
    PyObject *answer = Py_BuildValue("(OO)", first == NULL ? Py_None : first,
                                     first == second ? Py_True : Py_False);
    Py_XDECREF(first);
    Py_XDECREF(second);
    return answer;
}

/* The setter borrows rather than stealing, so a caller that hands over a
   reference of its own must still release it -- and the slot has to survive
   that release.  The row is what the slot reads back afterwards. */
static PyObject *handled_set(PyObject *self, PyObject *exc)
{
    (void)self;
    Py_INCREF(exc);
    PyErr_SetHandledException(exc == Py_None ? NULL : exc);
    Py_DECREF(exc);
    PyObject *through = PyErr_GetHandledException();
    PyObject *answer = Py_BuildValue("(OO)", through == NULL ? Py_None : through,
                                     through == exc ? Py_True : Py_False);
    Py_XDECREF(through);
    return answer;
}

/* `sys.exc_info()`'s triple as `PyErr_GetExcInfo` spells it: the empty state
   is the row that tells the two functions apart. */
static PyObject *exc_info(PyObject *self, PyObject *ignored)
{
    (void)self;
    (void)ignored;
    PyObject *type, *value, *traceback;
    PyErr_GetExcInfo(&type, &value, &traceback);
    PyObject *answer = Py_BuildValue(
        "(OOO)", type == NULL ? Py_None : type, value == NULL ? Py_None : value,
        traceback == NULL ? Py_None : traceback);
    /* The NULL the empty state writes into the value slot, told apart from a
       `None` that was really stored. */
    PyObject *row = Py_BuildValue("(OO)", answer, value == NULL ? Py_True : Py_False);
    Py_DECREF(answer);
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
    return row;
}

/* Are the type and traceback slots derived from the value? */
static PyObject *exc_info_derived(PyObject *self, PyObject *ignored)
{
    (void)self;
    (void)ignored;
    PyObject *type, *value, *traceback;
    PyErr_GetExcInfo(&type, &value, &traceback);
    if (value == NULL) {
        Py_XDECREF(type);
        Py_XDECREF(traceback);
        Py_RETURN_NONE;
    }
    PyObject *own = PyException_GetTraceback(value);
    PyObject *answer = Py_BuildValue(
        "(OO)", type == (PyObject *)Py_TYPE(value) ? Py_True : Py_False,
        traceback == (own == NULL ? Py_None : own) ? Py_True : Py_False);
    Py_XDECREF(own);
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
    return answer;
}

/* All three references are stolen, and only the value is stored. */
static PyObject *exc_info_set(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *type, *value, *traceback;
    if (!PyArg_ParseTuple(args, "OOO", &type, &value, &traceback)) {
        return NULL;
    }
    PyErr_SetExcInfo(type == Py_None ? NULL : Py_NewRef(type),
                     value == Py_None ? NULL : Py_NewRef(value),
                     traceback == Py_None ? NULL : Py_NewRef(traceback));
    PyObject *read_type, *read_value, *read_traceback;
    PyErr_GetExcInfo(&read_type, &read_value, &read_traceback);
    PyObject *answer = Py_BuildValue(
        "(OOO)", read_type == NULL ? Py_None : read_type,
        read_value == NULL ? Py_None : read_value,
        read_traceback == NULL ? Py_None : read_traceback);
    Py_XDECREF(read_type);
    Py_XDECREF(read_value);
    Py_XDECREF(read_traceback);
    return answer;
}

/* ── ImportError ──────────────────────────────────────────────────────── */

/* The four fields the built exception carries, plus the return value's own
   convention: this entry point always answers NULL. */
static PyObject *import_error(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *message, *name, *path;
    if (!PyArg_ParseTuple(args, "OOO", &message, &name, &path)) {
        return NULL;
    }
    PyObject *returned = PyErr_SetImportError(message == Py_None ? NULL : message,
                                              name == Py_None ? NULL : name,
                                              path == Py_None ? NULL : path);
    PyObject *raised = PyErr_GetRaisedException();
    if (raised == NULL) {
        Py_RETURN_NONE;
    }
    PyObject *text = PyObject_Str(raised);
    /* The three keyword-set fields, read back off the instance; a class that
       does not carry one answers with the absence rather than failing. */
    static const char *const named[] = {"name", "path", "name_from"};
    PyObject *fields[3];
    for (int i = 0; i < 3; i++) {
        fields[i] = PyObject_GetAttrString(raised, named[i]);
        if (fields[i] == NULL) {
            PyErr_Clear();
            fields[i] = PyUnicode_FromString("<absent>");
        }
    }
    PyObject *answer = Py_BuildValue(
        "(sOOOOO)", Py_TYPE(raised)->tp_name, text == NULL ? Py_None : text,
        returned == NULL ? Py_True : Py_False, fields[0], fields[1], fields[2]);
    Py_XDECREF(text);
    for (int i = 0; i < 3; i++) {
        Py_XDECREF(fields[i]);
    }
    Py_DECREF(raised);
    return answer;
}

static PyObject *import_error_subclass(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *type, *message, *name, *path;
    if (!PyArg_ParseTuple(args, "OOOO", &type, &message, &name, &path)) {
        return NULL;
    }
    PyObject *returned = PyErr_SetImportErrorSubclass(
        type, message == Py_None ? NULL : message, name == Py_None ? NULL : name,
        path == Py_None ? NULL : path);
    PyObject *raised = PyErr_GetRaisedException();
    if (raised == NULL) {
        Py_RETURN_NONE;
    }
    PyObject *text = PyObject_Str(raised);
    PyObject *answer = Py_BuildValue("(sOO)", Py_TYPE(raised)->tp_name,
                                     text == NULL ? Py_None : text,
                                     returned == NULL ? Py_True : Py_False);
    Py_XDECREF(text);
    Py_DECREF(raised);
    return answer;
}

/* The implicit chaining a freshly set ImportError picks up. */
static PyObject *import_error_context(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *message;
    if (!PyArg_ParseTuple(args, "O", &message)) {
        return NULL;
    }
    PyErr_SetImportError(message, NULL, NULL);
    PyObject *raised = PyErr_GetRaisedException();
    if (raised == NULL) {
        Py_RETURN_NONE;
    }
    PyObject *context = PyException_GetContext(raised);
    PyObject *cause = PyException_GetCause(raised);
    PyObject *answer = Py_BuildValue("(OO)", context == NULL ? Py_None : context,
                                     cause == NULL ? Py_None : cause);
    Py_XDECREF(context);
    Py_XDECREF(cause);
    Py_DECREF(raised);
    return answer;
}

static PyMethodDef methods[] = {
    {"exc_get_traceback", exc_get_traceback, METH_O, NULL},
    {"exc_set_traceback", exc_set_traceback, METH_VARARGS, NULL},
    {"exc_get_cause", exc_get_cause, METH_O, NULL},
    {"exc_set_cause", exc_set_cause, METH_VARARGS, NULL},
    {"exc_get_context", exc_get_context, METH_O, NULL},
    {"exc_set_context", exc_set_context, METH_VARARGS, NULL},
    {"exc_get_args", exc_get_args, METH_O, NULL},
    {"exc_set_args", exc_set_args, METH_VARARGS, NULL},
    {"exc_classify", exc_classify, METH_O, NULL},
    {"raised_round_trip", raised_round_trip, METH_VARARGS, NULL},
    {"raised_when_clear", raised_when_clear, METH_NOARGS, NULL},
    {"raised_set", raised_set, METH_O, NULL},
    {"raised_set_null", raised_set_null, METH_O, NULL},
    {"raised_replace", raised_replace, METH_VARARGS, NULL},
    {"fetch_triple", fetch_triple, METH_O, NULL},
    {"fetch_and_restore", fetch_and_restore, METH_O, NULL},
    {"set_object_context", set_object_context, METH_VARARGS, NULL},
    {"handled_get", handled_get, METH_NOARGS, NULL},
    {"handled_twice", handled_twice, METH_NOARGS, NULL},
    {"handled_set", handled_set, METH_O, NULL},
    {"exc_info", exc_info, METH_NOARGS, NULL},
    {"exc_info_derived", exc_info_derived, METH_NOARGS, NULL},
    {"exc_info_set", exc_info_set, METH_VARARGS, NULL},
    {"import_error", import_error, METH_VARARGS, NULL},
    {"import_error_subclass", import_error_subclass, METH_VARARGS, NULL},
    {"import_error_context", import_error_context, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef def = {PyModuleDef_HEAD_INIT, "cpyext_exceptions", NULL, -1,
                                 methods};

PyMODINIT_FUNC PyInit_cpyext_exceptions(void)
{
    return PyModule_Create(&def);
}
