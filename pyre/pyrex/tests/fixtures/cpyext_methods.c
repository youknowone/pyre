/* A multi-phase (PEP 489) extension exercising the method calling
   conventions, the argument parsers, the object constructors and the
   exception indicator. */

#include <Python.h>
#include <string.h>

typedef struct {
    long calls;
} methods_state;

static struct PyModuleDef moduledef;

static methods_state *state_of(PyObject *module)
{
    methods_state *state = (methods_state *)PyModule_GetState(module);
    if (state == NULL && !PyErr_Occurred()) {
        PyErr_SetString(PyExc_SystemError, "module state is missing");
    }
    return state;
}

/* METH_NOARGS: the second argument is NULL and `self` is the module. */
static PyObject *m_bump(PyObject *self, PyObject *unused)
{
    if (unused != NULL) {
        PyErr_SetString(PyExc_SystemError, "METH_NOARGS was passed an argument");
        return NULL;
    }
    methods_state *state = state_of(self);
    if (state == NULL) {
        return NULL;
    }
    state->calls++;
    return PyLong_FromLong(state->calls);
}

/* METH_O */
static PyObject *m_wrap(PyObject *self, PyObject *arg)
{
    (void)self;
    return Py_BuildValue("(Os)", arg, "seen");
}

/* METH_VARARGS with an optional unit. */
static PyObject *m_add(PyObject *self, PyObject *args)
{
    (void)self;
    long left = 0;
    long right = 10;
    if (!PyArg_ParseTuple(args, "l|l:add", &left, &right)) {
        return NULL;
    }
    return PyLong_FromLong(left + right);
}

/* METH_VARARGS | METH_KEYWORDS */
static PyObject *m_greet(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    static char *keywords[] = {"name", "punct", NULL};
    const char *name = NULL;
    const char *punct = "!";
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|s:greet", keywords,
                                     &name, &punct)) {
        return NULL;
    }
    char buffer[256];
    snprintf(buffer, sizeof(buffer), "hello %s%s", name, punct);
    return PyUnicode_FromString(buffer);
}

/* METH_FASTCALL */
static PyObject *m_total(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    (void)self;
    long total = 0;
    for (Py_ssize_t index = 0; index < nargs; index++) {
        long value = PyLong_AsLong(args[index]);
        if (value == -1 && PyErr_Occurred()) {
            return NULL;
        }
        total += value;
    }
    return PyLong_FromLong(total);
}

/* METH_FASTCALL | METH_KEYWORDS: the keyword values follow the positional
   ones in `args`, and `kwnames` names them in order. */
static PyObject *m_layout(PyObject *self, PyObject *const *args,
                          Py_ssize_t nargs, PyObject *kwnames)
{
    (void)self;
    PyObject *positional = PyList_New(0);
    if (positional == NULL) {
        return NULL;
    }
    for (Py_ssize_t index = 0; index < nargs; index++) {
        if (PyList_Append(positional, args[index]) < 0) {
            Py_DECREF(positional);
            return NULL;
        }
    }
    PyObject *named = PyList_New(0);
    if (named == NULL) {
        Py_DECREF(positional);
        return NULL;
    }
    Py_ssize_t count = kwnames == NULL ? 0 : PyTuple_Size(kwnames);
    for (Py_ssize_t index = 0; index < count; index++) {
        PyObject *pair = Py_BuildValue("(OO)", PyTuple_GetItem(kwnames, index),
                                       args[nargs + index]);
        if (pair == NULL || PyList_Append(named, pair) < 0) {
            Py_XDECREF(pair);
            Py_DECREF(named);
            Py_DECREF(positional);
            return NULL;
        }
        Py_DECREF(pair);
    }
    PyObject *result = Py_BuildValue("(NN)", positional, named);
    if (result == NULL) {
        Py_DECREF(named);
        Py_DECREF(positional);
    }
    return result;
}

/* PyArg_UnpackTuple and the object protocol. */
static PyObject *m_apply(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *callable = NULL;
    PyObject *argument = NULL;
    if (!PyArg_UnpackTuple(args, "apply", 2, 2, &callable, &argument)) {
        return NULL;
    }
    if (!PyCallable_Check(callable)) {
        PyErr_SetString(PyExc_TypeError, "apply() needs a callable");
        return NULL;
    }
    PyObject *call_args = Py_BuildValue("(O)", argument);
    if (call_args == NULL) {
        return NULL;
    }
    PyObject *result = PyObject_CallObject(callable, call_args);
    Py_DECREF(call_args);
    return result;
}

/* Attribute, item and stringification entry points. */
static PyObject *m_inspect(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *target = NULL;
    const char *name = NULL;
    if (!PyArg_ParseTuple(args, "Os:inspect", &target, &name)) {
        return NULL;
    }
    int present = PyObject_HasAttrString(target, name);
    PyObject *text = PyObject_Str(target);
    if (text == NULL) {
        return NULL;
    }
    PyObject *shown = PyObject_Repr(target);
    if (shown == NULL) {
        Py_DECREF(text);
        return NULL;
    }
    Py_ssize_t size = PyObject_Size(target);
    if (size < 0) {
        PyErr_Clear();
        size = -1;
    }
    PyObject *result = Py_BuildValue("(iNNni)", present, text, shown, size,
                                     PyObject_IsTrue(target));
    return result;
}

/* The primitive constructors and accessors. */
static PyObject *m_build(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    return Py_BuildValue("{s:i,s:l,s:d,s:s,s:y,s:[i,i,i],s:(s,i)}",
                         "int", 3,
                         "long", 4L,
                         "float", 1.5,
                         "str", "text",
                         "bytes", "raw",
                         "list", 1, 2, 3,
                         "tuple", "pair", 9);
}

static PyObject *m_roundtrip(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *text = NULL;
    PyObject *raw = NULL;
    if (!PyArg_ParseTuple(args, "US:roundtrip", &text, &raw)) {
        return NULL;
    }
    Py_ssize_t text_size = 0;
    const char *utf8 = PyUnicode_AsUTF8AndSize(text, &text_size);
    if (utf8 == NULL) {
        return NULL;
    }
    char *bytes = NULL;
    Py_ssize_t bytes_size = 0;
    if (PyBytes_AsStringAndSize(raw, &bytes, &bytes_size) < 0) {
        return NULL;
    }
    return Py_BuildValue("(s#y#nn)", utf8, text_size, bytes, bytes_size,
                         PyUnicode_GetLength(text), PyBytes_Size(raw));
}

static PyObject *m_numbers(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *value = NULL;
    if (!PyArg_ParseTuple(args, "O:numbers", &value)) {
        return NULL;
    }
    long as_long = PyLong_AsLong(value);
    if (as_long == -1 && PyErr_Occurred()) {
        return NULL;
    }
    double as_double = PyLong_AsDouble(value);
    if (as_double == -1.0 && PyErr_Occurred()) {
        return NULL;
    }
    PyObject *parsed = PyLong_FromString("-0x2a", NULL, 0);
    if (parsed == NULL) {
        return NULL;
    }
    PyObject *quarter = PyFloat_FromDouble(0.25);
    if (quarter == NULL) {
        Py_DECREF(parsed);
        return NULL;
    }
    double quarter_value = PyFloat_AsDouble(quarter);
    Py_DECREF(quarter);
    return Py_BuildValue("(ldNiid)", as_long, as_double, parsed,
                         PyLong_Check(value), PyFloat_Check(value),
                         quarter_value);
}

static PyObject *m_dict_ops(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    PyObject *mapping = PyDict_New();
    if (mapping == NULL) {
        return NULL;
    }
    PyObject *value = PyLong_FromLong(7);
    if (value == NULL || PyDict_SetItemString(mapping, "seven", value) < 0) {
        Py_XDECREF(value);
        Py_DECREF(mapping);
        return NULL;
    }
    Py_DECREF(value);
    PyObject *key = PyUnicode_FromString("seven");
    if (key == NULL) {
        Py_DECREF(mapping);
        return NULL;
    }
    PyObject *found = PyDict_GetItem(mapping, key);
    long seven = found == NULL ? -1 : PyLong_AsLong(found);
    int contains = PyDict_Contains(mapping, key);
    Py_ssize_t before = PyDict_Size(mapping);
    if (PyDict_DelItem(mapping, key) < 0) {
        Py_DECREF(key);
        Py_DECREF(mapping);
        return NULL;
    }
    Py_ssize_t after = PyDict_Size(mapping);
    PyObject *missing = PyDict_GetItemString(mapping, "seven");
    Py_DECREF(key);
    Py_DECREF(mapping);
    return Py_BuildValue("(linni)", seven, contains, before, after,
                         missing == NULL);
}

static PyObject *m_sequences(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    PyObject *tuple = PyTuple_New(2);
    if (tuple == NULL) {
        return NULL;
    }
    PyObject *first = PyLong_FromLong(1);
    if (first == NULL || PyTuple_SetItem(tuple, 0, first) < 0) {
        Py_DECREF(tuple);
        return NULL;
    }
    PyObject *second = PyUnicode_FromString("two");
    if (second == NULL || PyTuple_SetItem(tuple, 1, second) < 0) {
        Py_DECREF(tuple);
        return NULL;
    }
    PyObject *list = PyList_New(1);
    if (list == NULL) {
        Py_DECREF(tuple);
        return NULL;
    }
    PyObject *only = PyLong_FromLong(5);
    if (only == NULL || PyList_SetItem(list, 0, only) < 0) {
        Py_DECREF(list);
        Py_DECREF(tuple);
        return NULL;
    }
    PyObject *extra = PyBytes_FromStringAndSize("ab\0c", 4);
    if (extra == NULL || PyList_Append(list, extra) < 0) {
        Py_XDECREF(extra);
        Py_DECREF(list);
        Py_DECREF(tuple);
        return NULL;
    }
    Py_DECREF(extra);
    return Py_BuildValue("(NNnn)", tuple, list, PyTuple_Size(tuple),
                         PyList_Size(list));
}

/* The singletons, so the caller can check identity from Python. */
static PyObject *m_singletons(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    return Py_BuildValue("(OOOOO)", Py_None, Py_True, Py_False, Py_Ellipsis,
                         Py_NotImplemented);
}

static PyObject *m_predicates(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *value = NULL;
    if (!PyArg_ParseTuple(args, "O:predicates", &value)) {
        return NULL;
    }
    return Py_BuildValue("(iiiiiii)", Py_IsNone(value), Py_IsTrue(value),
                         PyUnicode_Check(value), PyBytes_Check(value),
                         PyTuple_Check(value), PyList_Check(value),
                         PyDict_Check(value));
}

/* The exception indicator. */
static PyObject *m_fail(PyObject *self, PyObject *args)
{
    (void)self;
    const char *kind = NULL;
    if (!PyArg_ParseTuple(args, "s:fail", &kind)) {
        return NULL;
    }
    if (strcmp(kind, "value") == 0) {
        PyErr_SetString(PyExc_ValueError, "a value complaint");
        return NULL;
    }
    if (strcmp(kind, "format") == 0) {
        PyErr_Format(PyExc_TypeError, "formatted %s and %d", "message", 7);
        return NULL;
    }
    if (strcmp(kind, "object") == 0) {
        PyObject *value = PyUnicode_FromString("payload");
        if (value == NULL) {
            return NULL;
        }
        PyErr_SetObject(PyExc_KeyError, value);
        Py_DECREF(value);
        return NULL;
    }
    if (strcmp(kind, "none") == 0) {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }
    if (strcmp(kind, "memory") == 0) {
        return PyErr_NoMemory();
    }
    if (strcmp(kind, "argument") == 0) {
        PyErr_BadArgument();
        return NULL;
    }
    PyErr_SetString(PyExc_RuntimeError, "unknown failure kind");
    return NULL;
}

/* Catch a raised exception inside C: set, observe, match and clear. */
static PyObject *m_caught(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    PyErr_SetString(PyExc_IndexError, "swallowed");
    PyObject *occurred = PyErr_Occurred();
    int is_lookup = PyErr_ExceptionMatches(PyExc_LookupError);
    int is_type = PyErr_ExceptionMatches(PyExc_TypeError);
    PyObject *type = NULL;
    PyObject *value = NULL;
    PyObject *traceback = NULL;
    PyErr_Fetch(&type, &value, &traceback);
    int cleared = PyErr_Occurred() == NULL;
    PyObject *text = value == NULL ? PyUnicode_FromString("") : PyObject_Str(value);
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
    if (text == NULL) {
        return NULL;
    }
    return Py_BuildValue("(iiiiN)", occurred != NULL, is_lookup, is_type,
                         cleared, text);
}

/* The version macros, as an extension expands them: the string ones inside a
   banner, the numeric ones as values. A mismatch against what the runtime
   reports is a stale header. */
static PyObject *m_version_macros(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    static const char banner[] = "python " PY_VERSION " / pyre " PYRE_VERSION;
    return Py_BuildValue("(siiiiisi)", banner, PY_MAJOR_VERSION, PY_MINOR_VERSION,
                         PY_MICRO_VERSION, PY_RELEASE_LEVEL, PY_RELEASE_SERIAL,
                         PY_VERSION, PY_VERSION_HEX);
}

/* `PyErr_Restore` puts back what `PyErr_Fetch` took, and its three degenerate
   inputs: a NULL class clears whatever was set, a NULL value is built into a
   bare instance, and a fetched pair goes back unchanged. */
static PyObject *m_restore(PyObject *self, PyObject *unused)
{
    (void)self;
    (void)unused;
    /* A NULL class clears an indicator that is already set. */
    PyErr_SetString(PyExc_IndexError, "dropped");
    PyErr_Restore(NULL, NULL, NULL);
    int cleared = PyErr_Occurred() == NULL;

    /* A class with no value becomes a bare instance of that class. */
    PyErr_Restore(Py_NewRef(PyExc_KeyError), NULL, NULL);
    int bare_is_key = PyErr_ExceptionMatches(PyExc_KeyError);
    PyObject *type = NULL, *value = NULL, *traceback = NULL;
    PyErr_Fetch(&type, &value, &traceback);
    int bare_instance = value != NULL && PyObject_IsInstance(value, PyExc_KeyError) == 1;

    /* The fetched pair restores to the same indicator it came from. */
    PyErr_Restore(type, value, traceback);
    int round_trip = PyErr_ExceptionMatches(PyExc_KeyError);
    PyErr_Clear();

    /* A NULL class clears even when a value is handed along with it. */
    PyErr_SetString(PyExc_TypeError, "also dropped");
    PyErr_Restore(NULL, PyUnicode_FromString("orphan"), NULL);
    int cleared_with_value = PyErr_Occurred() == NULL;

    return Py_BuildValue("(iiiii)", cleared, bare_is_key, bare_instance,
                         round_trip, cleared_with_value);
}

/* Drive every call entry point at one callable and hand back what each
   returned, so the test can compare them against the same call made from
   Python.  `args` is (callable, one_arg, keyword_value). */
static PyObject *m_call_surface(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *callable, *one, *kwvalue;
    if (!PyArg_ParseTuple(args, "OOO", &callable, &one, &kwvalue)) {
        return NULL;
    }

    PyObject *vector[3];
    PyObject *kwnames = NULL;
    PyObject *noargs = NULL, *onearg = NULL, *vec = NULL, *vec_kw = NULL;
    PyObject *objargs = NULL, *fmt = NULL, *meth_obj = NULL, *meth_fmt = NULL;
    PyObject *name = NULL, *meth_no = NULL, *meth_one = NULL, *result = NULL;
    PyObject *text = NULL;

    noargs = PyObject_CallNoArgs(callable);
    if (noargs == NULL) goto done;
    onearg = PyObject_CallOneArg(callable, one);
    if (onearg == NULL) goto done;

    vector[0] = one;
    vector[1] = one;
    vec = PyObject_Vectorcall(callable, vector, 2, NULL);
    if (vec == NULL) goto done;

    /* One positional and one named, the name in a `kwnames` tuple. */
    kwnames = PyTuple_New(1);
    if (kwnames == NULL) goto done;
    {
        PyObject *key = PyUnicode_FromString("kw");
        if (key == NULL) goto done;
        PyTuple_SetItem(kwnames, 0, key);
    }
    vector[0] = one;
    vector[1] = kwvalue;
    vec_kw = PyObject_Vectorcall(callable, vector, 1, kwnames);
    if (vec_kw == NULL) goto done;

    objargs = PyObject_CallFunctionObjArgs(callable, one, one, one, NULL);
    if (objargs == NULL) goto done;
    fmt = PyObject_CallFunction(callable, "OO", one, one);
    if (fmt == NULL) goto done;

    /* The method spellings, against a str the test can predict. */
    text = PyUnicode_FromString("abc");
    if (text == NULL) goto done;
    name = PyUnicode_FromString("upper");
    if (name == NULL) goto done;
    meth_no = PyObject_CallMethodNoArgs(text, name);
    if (meth_no == NULL) goto done;
    meth_one = PyObject_CallMethodOneArg(text, name, text);
    /* `upper` takes no argument, so that call must have failed. */
    if (meth_one != NULL) {
        PyErr_SetString(PyExc_AssertionError, "upper() accepted an argument");
        goto done;
    }
    PyErr_Clear();
    meth_obj = PyObject_CallMethodObjArgs(text, name, NULL);
    if (meth_obj == NULL) goto done;
    meth_fmt = PyObject_CallMethod(text, "count", "s", "b");
    if (meth_fmt == NULL) goto done;

    result = Py_BuildValue("(OOOOOOOO)", noargs, onearg, vec, vec_kw,
                           objargs, fmt, meth_no, meth_fmt);

done:
    Py_XDECREF(noargs);
    Py_XDECREF(onearg);
    Py_XDECREF(vec);
    Py_XDECREF(vec_kw);
    Py_XDECREF(kwnames);
    Py_XDECREF(objargs);
    Py_XDECREF(fmt);
    Py_XDECREF(meth_no);
    Py_XDECREF(meth_one);
    Py_XDECREF(meth_obj);
    Py_XDECREF(meth_fmt);
    Py_XDECREF(name);
    Py_XDECREF(text);
    return result;
}

/* Drive the set protocol and the raw allocators.  `args` is (iterable, key). */
static PyObject *m_set_ops(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *iterable, *key;
    if (!PyArg_ParseTuple(args, "OO", &iterable, &key)) {
        return NULL;
    }

    PyObject *empty = PySet_New(NULL);
    PyObject *built = PySet_New(iterable);
    PyObject *frozen = PyFrozenSet_New(iterable);
    PyObject *popped = NULL, *result = NULL;
    if (empty == NULL || built == NULL || frozen == NULL) {
        goto done;
    }

    /* A key added once is present; adding it again does not grow the set. */
    if (PySet_Add(built, key) < 0) goto done;
    Py_ssize_t after_add = PySet_Size(built);
    if (PySet_Add(built, key) < 0) goto done;
    Py_ssize_t after_readd = PySet_Size(built);
    int has_key = PySet_Contains(built, key);

    /* Discard answers 1 the first time and 0 the second, without raising. */
    int first = PySet_Discard(built, key);
    int second = PySet_Discard(built, key);
    if (first < 0 || second < 0) goto done;

    popped = PySet_Pop(built);
    if (popped == NULL) goto done;
    Py_ssize_t after_pop = PySet_Size(built);
    if (PySet_Clear(built) < 0) goto done;

    /* The raw allocators: write a pattern, grow, and read it back. */
    unsigned char *block = (unsigned char *)PyMem_Calloc(8, 1);
    if (block == NULL) goto done;
    int was_zero = 1;
    for (int i = 0; i < 8; i++) { was_zero = was_zero && block[i] == 0; }
    for (int i = 0; i < 8; i++) { block[i] = (unsigned char)(i + 1); }
    block = (unsigned char *)PyMem_Realloc(block, 64);
    if (block == NULL) goto done;
    int kept = 1;
    for (int i = 0; i < 8; i++) { kept = kept && block[i] == (unsigned char)(i + 1); }
    PyMem_Free(block);

    long *typed = PyMem_New(long, 4);
    if (typed == NULL) goto done;
    typed[3] = 1234;
    long typed_read = typed[3];
    PyMem_Del(typed);

    result = Py_BuildValue(
        "(nnniiinniiil)",
        PySet_Size(empty), after_add, after_readd, has_key,
        first, second, after_pop, PySet_Size(built),
        PySet_Check(built), PyFrozenSet_Check(frozen), PyAnySet_Check(frozen),
        was_zero && kept ? typed_read : -1);

done:
    Py_XDECREF(empty);
    Py_XDECREF(built);
    Py_XDECREF(frozen);
    Py_XDECREF(popped);
    return result;
}

/* Drive the dict entry points.  `args` is (dict, key, other_mapping). */
static PyObject *m_dict_more(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *d, *key, *other;
    if (!PyArg_ParseTuple(args, "OOO", &d, &key, &other)) {
        return NULL;
    }

    PyObject *copy = NULL, *merged = NULL, *keys = NULL, *values = NULL, *items = NULL;
    PyObject *walked = NULL, *result = NULL, *got = NULL;
    int ref_hit = -2, ref_miss = -2, str_hit = -2;
    PyObject *ref_out = NULL, *miss_out = (PyObject *)1, *str_out = NULL;

    copy = PyDict_Copy(d);
    merged = PyDict_Copy(d);
    keys = PyDict_Keys(d);
    values = PyDict_Values(d);
    items = PyDict_Items(d);
    if (copy == NULL || merged == NULL || keys == NULL || values == NULL ||
        items == NULL) goto done;

    /* GetItemRef hands back a strong reference and says whether it was there. */
    ref_hit = PyDict_GetItemRef(d, key, &ref_out);
    PyObject *absent = PyUnicode_FromString("no-such-key");
    if (absent == NULL) goto done;
    ref_miss = PyDict_GetItemRef(d, absent, &miss_out);
    /* A miss must not leave an exception behind. */
    int miss_clean = PyErr_Occurred() == NULL;
    Py_DECREF(absent);
    str_hit = PyDict_GetItemStringRef(d, "a", &str_out);

    /* GetItemWithError: NULL and no exception for an absent key. */
    got = PyDict_GetItemWithError(d, key);   /* borrowed */
    int with_error_clean = PyErr_Occurred() == NULL;

    /* Walk every pair, building the list of keys PyDict_Next reported. */
    walked = PyList_New(0);
    if (walked == NULL) goto done;
    {
        Py_ssize_t pos = 0;
        PyObject *k, *v;
        while (PyDict_Next(d, &pos, &k, &v)) {
            PyObject *pair = Py_BuildValue("(OO)", k, v);
            if (pair == NULL) goto done;
            if (PyList_Append(walked, pair) < 0) { Py_DECREF(pair); goto done; }
            Py_DECREF(pair);
        }
    }

    /* Merge without override leaves existing keys alone; Update replaces. */
    if (PyDict_Merge(merged, other, 0) < 0) goto done;
    Py_ssize_t after_merge = PyDict_Size(merged);
    PyObject *kept = PyDict_GetItemString(merged, "b");   /* borrowed */
    long kept_value = kept == NULL ? -1 : PyLong_AsLong(kept);
    if (PyDict_Update(merged, other) < 0) goto done;
    Py_ssize_t after_update = PyDict_Size(merged);
    PyObject *replaced = PyDict_GetItemString(merged, "b");
    long replaced_value = replaced == NULL ? -1 : PyLong_AsLong(replaced);

    result = Py_BuildValue("(OOOOOiiiiiOnnll)", copy, keys, values, items, walked,
                           ref_hit, ref_miss, str_hit, miss_clean,
                           with_error_clean,
                           got == NULL ? Py_None : got,
                           after_merge, after_update, kept_value, replaced_value);

done:
    Py_XDECREF(copy);
    Py_XDECREF(merged);
    Py_XDECREF(keys);
    Py_XDECREF(values);
    Py_XDECREF(items);
    Py_XDECREF(walked);
    Py_XDECREF(ref_out);
    Py_XDECREF(str_out);
    return result;
}

/* Drive the list entry points, PyTuple_Pack and PyTuple_GetSlice.
   `args` is (items,). */
static PyObject *m_list_ops(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *items;
    if (!PyArg_ParseTuple(args, "O", &items)) {
        return NULL;
    }

    PyObject *built = NULL, *ref = NULL, *as_tuple = NULL, *sliced = NULL;
    PyObject *packed = NULL, *tuple_slice = NULL, *result = NULL;
    long head_value = -1, after_reverse = -1, after_sort = -1;

    built = PySequence_List(items);
    if (built == NULL) goto done;

    /* Insert at the front, then read the new head back strongly. */
    PyObject *head = PyLong_FromLong(99);
    if (head == NULL) goto done;
    int inserted = PyList_Insert(built, 0, head);
    Py_DECREF(head);
    if (inserted < 0) goto done;
    ref = PyList_GetItemRef(built, 0);
    if (ref == NULL) goto done;
    head_value = PyLong_AsLong(ref);

    /* Reverse, then sort: the sorted order is the one that survives. */
    if (PyList_Reverse(built) < 0) goto done;
    after_reverse = PyLong_AsLong(PyList_GetItem(built, 0));
    if (PyList_Sort(built) < 0) goto done;
    after_sort = PyLong_AsLong(PyList_GetItem(built, 0));

    as_tuple = PyList_AsTuple(built);
    sliced = PyList_GetSlice(built, 1, 3);
    if (as_tuple == NULL || sliced == NULL) goto done;

    /* Replace [0:2] with a two-item list, then delete [0:1]. */
    PyObject *replacement = Py_BuildValue("[ii]", -1, -2);
    if (replacement == NULL) goto done;
    int replaced = PyList_SetSlice(built, 0, 2, replacement);
    Py_DECREF(replacement);
    if (replaced < 0) goto done;
    if (PyList_SetSlice(built, 0, 1, NULL) < 0) goto done;

    packed = PyTuple_Pack(3, Py_None, Py_True, Py_False);
    if (packed == NULL) goto done;
    tuple_slice = PyTuple_GetSlice(packed, 1, 3);
    if (tuple_slice == NULL) goto done;

    result = Py_BuildValue("(lllOOOOO)", head_value, after_reverse, after_sort,
                           as_tuple, sliced, built, packed, tuple_slice);

done:
    Py_XDECREF(built);
    Py_XDECREF(ref);
    Py_XDECREF(as_tuple);
    Py_XDECREF(sliced);
    Py_XDECREF(packed);
    Py_XDECREF(tuple_slice);
    return result;
}

/* Drive the slice entry points.  `args` is (slice, length). */
static PyObject *m_slice_ops(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *sl;
    Py_ssize_t length;
    if (!PyArg_ParseTuple(args, "On", &sl, &length)) {
        return NULL;
    }

    Py_ssize_t u_start, u_stop, u_step;
    if (PySlice_Unpack(sl, &u_start, &u_stop, &u_step) < 0) {
        return NULL;
    }

    /* AdjustIndices clamps in place and answers the slice length. */
    Py_ssize_t a_start = u_start, a_stop = u_stop;
    Py_ssize_t a_len = PySlice_AdjustIndices(length, &a_start, &a_stop, u_step);

    Py_ssize_t g_start, g_stop, g_step;
    if (PySlice_GetIndices(sl, length, &g_start, &g_stop, &g_step) < 0) {
        return NULL;
    }

    Py_ssize_t e_start, e_stop, e_step, e_len;
    if (PySlice_GetIndicesEx(sl, length, &e_start, &e_stop, &e_step, &e_len) < 0) {
        return NULL;
    }

    /* A NULL bound reads as None. */
    PyObject *one = PyLong_FromLong(1);
    if (one == NULL) return NULL;
    PyObject *made = PySlice_New(one, NULL, one);
    Py_DECREF(one);
    if (made == NULL) return NULL;

    PyObject *result = Py_BuildValue(
        "(nnn nnn nnn nnnn Oii)",
        u_start, u_stop, u_step,
        a_start, a_stop, a_len,
        g_start, g_stop, g_step,
        e_start, e_stop, e_step, e_len,
        made, PySlice_Check(made), PySlice_Check(Py_None));
    Py_DECREF(made);
    return result;
}

/* Drive the rest of the sequence protocol, PyIter_NextItem and the two
   remaining number spellings.  `args` is (sequence, value). */
static PyObject *m_seq_more(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *sequence, *value;
    if (!PyArg_ParseTuple(args, "OO", &sequence, &value)) {
        return NULL;
    }

    PyObject *fast = NULL, *fast_items = NULL, *iterator = NULL, *drained = NULL;
    PyObject *owned = NULL, *based = NULL, *powered = NULL, *result = NULL;
    int clean_after_drain = 0, fast_is_self = 0;
    Py_ssize_t fast_size = -1;

    Py_ssize_t count = PySequence_Count(sequence, value);
    int contained = PySequence_In(sequence, value);
    if (count < 0 || contained < 0) goto done;

    /* Fast hands back the sequence itself when it is already a list or a
       tuple, and a list of its items otherwise. */
    fast = PySequence_Fast(sequence, "seq_more() wants a sequence");
    if (fast == NULL) goto done;
    fast_is_self = fast == sequence;
    fast_size = PySequence_Fast_GET_SIZE(fast);
    fast_items = PyList_New(0);
    if (fast_items == NULL) goto done;
    for (Py_ssize_t i = 0; i < fast_size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(fast, i);   /* borrowed */
        if (item == NULL) goto done;
        if (PyList_Append(fast_items, item) < 0) goto done;
    }

    /* Drain an iterator: 1 with an item each step, then 0 with no exception. */
    iterator = PyObject_GetIter(sequence);
    if (iterator == NULL) goto done;
    drained = PyList_New(0);
    if (drained == NULL) goto done;
    for (;;) {
        PyObject *item = NULL;
        int stepped = PyIter_NextItem(iterator, &item);
        if (stepped < 0) goto done;
        if (stepped == 0) break;
        int appended = PyList_Append(drained, item);
        Py_DECREF(item);
        if (appended < 0) goto done;
    }
    clean_after_drain = PyErr_Occurred() == NULL;

    /* Slice assignment and deletion, on a list this call owns. */
    owned = PySequence_List(sequence);
    if (owned == NULL) goto done;
    PyObject *replacement = Py_BuildValue("[i]", 7);
    if (replacement == NULL) goto done;
    int assigned = PySequence_SetSlice(owned, 0, 1, replacement);
    Py_DECREF(replacement);
    if (assigned < 0) goto done;
    if (PySequence_DelSlice(owned, 1, 2) < 0) goto done;

    based = PyNumber_ToBase(value, 16);
    if (based == NULL) goto done;

    powered = PyLong_FromLong(3);
    if (powered == NULL) goto done;
    PyObject *exponent = PyLong_FromLong(4);
    if (exponent == NULL) goto done;
    PyObject *raised = PyNumber_InPlacePower(powered, exponent, Py_None);
    Py_DECREF(exponent);
    if (raised == NULL) goto done;
    Py_SETREF(powered, raised);

    result = Py_BuildValue("(niOniOiOOO)", count, contained, fast_items,
                           fast_size, fast_is_self, drained, clean_after_drain,
                           owned, based, powered);

done:
    Py_XDECREF(fast);
    Py_XDECREF(fast_items);
    Py_XDECREF(iterator);
    Py_XDECREF(drained);
    Py_XDECREF(owned);
    Py_XDECREF(based);
    Py_XDECREF(powered);
    return result;
}

/* PyNumber_ToBase on its own, so that the base rejection is reachable.
   `args` is (n, base). */
static PyObject *m_to_base(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *n;
    int base;
    if (!PyArg_ParseTuple(args, "Oi", &n, &base)) {
        return NULL;
    }
    return PyNumber_ToBase(n, base);
}

/* Index a list this call has just built, many times over, so that the entry
   point runs under sustained allocation pressure rather than once.  The answer
   is the running total, so a misread item shows up as a wrong sum rather than
   only as a crash.  `args` is (rounds,). */
static PyObject *m_gc_window(PyObject *self, PyObject *args)
{
    (void)self;
    long rounds;
    if (!PyArg_ParseTuple(args, "l", &rounds)) {
        return NULL;
    }

    PyObject *fresh = PyList_New(0);
    if (fresh == NULL) return NULL;
    for (int i = 0; i < 8; i++) {
        PyObject *value = PyLong_FromLong(i);
        if (value == NULL || PyList_Append(fresh, value) < 0) {
            Py_XDECREF(value);
            Py_DECREF(fresh);
            return NULL;
        }
        Py_DECREF(value);
    }

    long total = 0;
    for (long round = 0; round < rounds; round++) {
        PyObject *item = PySequence_GetItem(fresh, round % 8);
        if (item == NULL) {
            Py_DECREF(fresh);
            return NULL;
        }
        long value = PyLong_AsLong(item);
        Py_DECREF(item);
        if (value == -1 && PyErr_Occurred()) {
            Py_DECREF(fresh);
            return NULL;
        }
        total += value;
    }
    Py_DECREF(fresh);
    return PyLong_FromLong(total);
}

/* A table added to a module after it was created, which is what
   `PyModule_AddFunctions` exists for. */
static PyObject *m_added(PyObject *self, PyObject *unused)
{
    (void)unused;
    return PyModule_GetNameObject(self);
}

static PyMethodDef added_methods[] = {
    {"added", m_added, METH_NOARGS, "the name of the module this was added to"},
    {NULL, NULL, 0, NULL},
};

/* The module constructors and accessors, driven from a module that already
   exists so the definition and the state block are known. */
static PyObject *m_module_ops(PyObject *self, PyObject *unused)
{
    (void)unused;
    PyObject *fresh = PyModule_New("cpyext_methods.fresh");
    if (fresh == NULL) {
        return NULL;
    }
    if (!PyModule_Check(fresh) || !PyModule_CheckExact(fresh)) {
        PyErr_SetString(PyExc_SystemError, "PyModule_New did not make a module");
        Py_DECREF(fresh);
        return NULL;
    }
    const char *fresh_name = PyModule_GetName(fresh);
    if (fresh_name == NULL) {
        Py_DECREF(fresh);
        return NULL;
    }
    if (PyModule_SetDocString(fresh, "a module built from C") < 0) {
        Py_DECREF(fresh);
        return NULL;
    }
    /* `PyModule_Add` releases its argument whether or not the store worked,
       so the count it is handed is the only one it consumes. */
    PyObject *value = PyLong_FromLong(7);
    if (value == NULL || PyModule_Add(fresh, "SEVEN", value) < 0) {
        Py_DECREF(fresh);
        return NULL;
    }
    if (PyModule_AddFunctions(fresh, added_methods) < 0) {
        Py_DECREF(fresh);
        return NULL;
    }

    PyObject *name = PyUnicode_FromString("cpyext_methods.named");
    if (name == NULL) {
        Py_DECREF(fresh);
        return NULL;
    }
    PyObject *named = PyModule_NewObject(name);
    Py_DECREF(name);
    if (named == NULL) {
        Py_DECREF(fresh);
        return NULL;
    }
    PyObject *named_name = PyModule_GetNameObject(named);
    Py_DECREF(named);
    if (named_name == NULL) {
        Py_DECREF(fresh);
        return NULL;
    }

    /* This module was imported from a file, so it has both spellings of the
       name it was loaded from. */
    PyObject *own_file = PyModule_GetFilenameObject(self);
    if (own_file == NULL) {
        Py_DECREF(fresh);
        Py_DECREF(named_name);
        return NULL;
    }
    const char *own_file_utf8 = PyModule_GetFilename(self);
    if (own_file_utf8 == NULL) {
        Py_DECREF(fresh);
        Py_DECREF(named_name);
        Py_DECREF(own_file);
        return NULL;
    }
    const char *own_file_text = PyUnicode_AsUTF8(own_file);
    if (own_file_text == NULL) {
        Py_DECREF(fresh);
        Py_DECREF(named_name);
        Py_DECREF(own_file);
        return NULL;
    }
    int same_file = strcmp(own_file_text, own_file_utf8) == 0;
    Py_DECREF(own_file);

    PyObject *result = Py_BuildValue(
        "(NsNi)", fresh, fresh_name, named_name, same_file);
    return result;
}

/* A module with no `__file__` reports the absence as an error rather than
   handing back a missing entry. */
static PyObject *m_module_no_file(PyObject *self, PyObject *unused)
{
    (void)self; (void)unused;
    PyObject *fresh = PyModule_New("cpyext_methods.bare");
    if (fresh == NULL) {
        return NULL;
    }
    PyObject *missing = PyModule_GetFilenameObject(fresh);
    Py_DECREF(fresh);
    if (missing == NULL) {
        return NULL;
    }
    Py_DECREF(missing);
    Py_RETURN_NONE;
}

/* The definition this fixture already carries, run a second time through the
   entry points import would otherwise drive. */
static PyObject *m_module_from_def(PyObject *self, PyObject *spec)
{
    PyObject *made = PyModule_FromDefAndSpec(&moduledef, spec);
    if (made == NULL) {
        return NULL;
    }
    if (PyModule_ExecDef(made, &moduledef) < 0) {
        Py_DECREF(made);
        return NULL;
    }
    return made;
}

/* The attribute half of the object protocol: the generic terminals, the
   optional lookup that reports an absence instead of raising, and the
   deletions.  `args` is `(object, name)` where `object` carries a `__dict__`
   and already has the named attribute. */
static PyObject *m_object_attrs(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *object;
    PyObject *name;
    if (!PyArg_ParseTuple(args, "OO", &object, &name)) {
        return NULL;
    }
    PyObject *got = PyObject_GenericGetAttr(object, name);
    if (got == NULL) {
        return NULL;
    }
    Py_DECREF(got);

    PyObject *marker = PyUnicode_FromString("set-generically");
    if (marker == NULL) {
        return NULL;
    }
    int stored = PyObject_GenericSetAttr(object, name, marker);
    Py_DECREF(marker);
    if (stored < 0) {
        return NULL;
    }

    /* The optional lookup answers 1/0 without leaving an indicator behind. */
    PyObject *found = NULL;
    int present = PyObject_GetOptionalAttr(object, name, &found);
    Py_XDECREF(found);
    if (present < 0) {
        return NULL;
    }
    PyObject *absent = NULL;
    int missing = PyObject_GetOptionalAttrString(object, "no_such_attribute", &absent);
    if (missing < 0 || absent != NULL) {
        Py_XDECREF(absent);
        PyErr_SetString(PyExc_SystemError, "an absent attribute was reported present");
        return NULL;
    }
    int has = PyObject_HasAttrWithError(object, name);
    int has_string = PyObject_HasAttrStringWithError(object, "no_such_attribute");
    int has_swallowed = PyObject_HasAttr(object, name);
    if (has < 0 || has_string < 0) {
        return NULL;
    }

    PyObject *dict = PyObject_GenericGetDict(object, NULL);
    if (dict == NULL) {
        return NULL;
    }
    Py_ssize_t entries = PyDict_Size(dict);
    Py_DECREF(dict);

    if (PyObject_DelAttr(object, name) < 0) {
        return NULL;
    }
    int gone = PyObject_HasAttrWithError(object, name);
    if (gone < 0) {
        return NULL;
    }

    return Py_BuildValue("(iiiini)", present, missing, has, has_string,
                         entries, has_swallowed && gone == 0);
}

/* Comparison, hashing and the string/format conversions.  `args` is
   `(left, right, mapping, key)`. */
static PyObject *m_object_values(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *left;
    PyObject *right;
    PyObject *mapping;
    PyObject *key;
    if (!PyArg_ParseTuple(args, "OOOO", &left, &right, &mapping, &key)) {
        return NULL;
    }
    PyObject *less = PyObject_RichCompare(left, right, Py_LT);
    if (less == NULL) {
        return NULL;
    }
    int same = PyObject_RichCompareBool(left, left, Py_EQ);
    int differs = PyObject_RichCompareBool(left, right, Py_NE);
    if (same < 0 || differs < 0) {
        Py_DECREF(less);
        return NULL;
    }
    Py_hash_t hash = PyObject_Hash(left);
    if (hash == -1 && PyErr_Occurred()) {
        Py_DECREF(less);
        return NULL;
    }
    PyObject *ascii_form = PyObject_ASCII(left);
    PyObject *formatted = PyObject_Format(left, NULL);
    PyObject *names = PyObject_Dir(left);
    if (ascii_form == NULL || formatted == NULL || names == NULL) {
        Py_DECREF(less);
        Py_XDECREF(ascii_form);
        Py_XDECREF(formatted);
        Py_XDECREF(names);
        return NULL;
    }
    Py_ssize_t name_count = PyObject_Length(names);
    Py_DECREF(names);

    if (PyObject_DelItem(mapping, key) < 0) {
        Py_DECREF(less);
        Py_DECREF(ascii_form);
        Py_DECREF(formatted);
        return NULL;
    }
    Py_ssize_t left_over = PyObject_Length(mapping);

    int subclass = PyObject_IsSubclass((PyObject *)Py_TYPE(left), (PyObject *)Py_TYPE(left));
    if (subclass < 0) {
        Py_DECREF(less);
        Py_DECREF(ascii_form);
        Py_DECREF(formatted);
        return NULL;
    }
    return Py_BuildValue("(NiiiNNnni)", less, same, differs, hash != -1,
                         ascii_form, formatted, name_count, left_over, subclass);
}

/* `bytes(object)` — a `__bytes__` override first, then the buffer and the
   iterable. */
static PyObject *m_object_bytes(PyObject *self, PyObject *arg)
{
    (void)self;
    return PyObject_Bytes(arg);
}

/* The same conversion without the `__bytes__` step, which is the one
   difference between the two entry points. */
static PyObject *m_bytes_from(PyObject *self, PyObject *arg)
{
    (void)self;
    return PyBytes_FromObject(arg);
}

/* The allocate-then-fill shape `PyBytes_FromStringAndSize(NULL, size)` exists
   for: the caller writes through `PyBytes_AS_STRING` and the object is a
   `bytes` by the time it is handed back. */
static PyObject *m_bytes_fill(PyObject *self, PyObject *arg)
{
    (void)self;
    Py_ssize_t size = PyLong_AsSsize_t(arg);
    if (size < 0 && PyErr_Occurred()) {
        return NULL;
    }
    PyObject *out = PyBytes_FromStringAndSize(NULL, size);
    if (out == NULL) {
        return NULL;
    }
    /* The type tests and the size are answered while the buffer is still being
       written, so asking them does not decide the contents early. */
    if (!PyBytes_Check(out) || !PyBytes_CheckExact(out) || PyBytes_GET_SIZE(out) != size) {
        Py_DECREF(out);
        PyErr_SetString(PyExc_SystemError, "a bytes being filled reads as something else");
        return NULL;
    }
    char *data = PyBytes_AS_STRING(out);
    if (data == NULL) {
        Py_DECREF(out);
        return NULL;
    }
    for (Py_ssize_t index = 0; index < size; index++) {
        data[index] = (char)('a' + (index % 26));
    }
    return out;
}

/* The same buffer handed to another entry point instead of being returned. */
static PyObject *m_bytes_pairs(PyObject *self, PyObject *unused)
{
    (void)self; (void)unused;
    PyObject *dict = PyDict_New();
    if (dict == NULL) {
        return NULL;
    }
    PyObject *key = PyBytes_FromStringAndSize(NULL, 2);
    PyObject *value = PyBytes_FromStringAndSize(NULL, 3);
    if (key == NULL || value == NULL) {
        Py_XDECREF(key);
        Py_XDECREF(value);
        Py_DECREF(dict);
        return NULL;
    }
    memcpy(PyBytes_AS_STRING(key), "kk", 2);
    memcpy(PyBytes_AS_STRING(value), "vv\0", 3);
    int failed = PyDict_SetItem(dict, key, value);
    Py_DECREF(key);
    Py_DECREF(value);
    if (failed < 0) {
        Py_DECREF(dict);
        return NULL;
    }
    return dict;
}

/* An empty allocation, and the checked accessors reading the buffer back. */
static PyObject *m_bytes_empty(PyObject *self, PyObject *unused)
{
    (void)self; (void)unused;
    PyObject *out = PyBytes_FromStringAndSize(NULL, 0);
    if (out == NULL) {
        return NULL;
    }
    char *data = NULL;
    Py_ssize_t size = -1;
    if (PyBytes_AsStringAndSize(out, &data, &size) < 0) {
        Py_DECREF(out);
        return NULL;
    }
    if (size != 0 || data == NULL || data[0] != '\0' || PyBytes_Size(out) != 0
        || PyBytes_AsString(out) != data) {
        Py_DECREF(out);
        PyErr_SetString(PyExc_SystemError, "an empty allocation reads back wrong");
        return NULL;
    }
    /* A buffer released without ever having been read as a value, so its
       mirror is freed with no `bytes` behind it. */
    PyObject *dropped = PyBytes_FromStringAndSize(NULL, 4);
    if (dropped == NULL) {
        Py_DECREF(out);
        return NULL;
    }
    memcpy(PyBytes_AS_STRING(dropped), "abcd", 4);
    Py_DECREF(dropped);
    if (PyBytes_FromStringAndSize(NULL, -1) != NULL || !PyErr_Occurred()) {
        Py_DECREF(out);
        PyErr_SetString(PyExc_SystemError, "a negative size was accepted");
        return NULL;
    }
    PyErr_Clear();
    return out;
}

/* `Py_NewRef` and `Py_XNewRef` -- a reference taken and handed back in one
   expression, including the NULL the `X` spelling admits. */
static PyObject *m_new_ref(PyObject *self, PyObject *arg)
{
    (void)self;
    if (Py_XNewRef(NULL) != NULL) {
        PyErr_SetString(PyExc_SystemError, "Py_XNewRef(NULL) is not NULL");
        return NULL;
    }
    /* `N` takes over each reference, so both are handed straight on. */
    return Py_BuildValue("(NN)", Py_NewRef(Py_True), Py_XNewRef(arg));
}

/* The type mirror behind `Py_TYPE`: its name, and whether it is a heap type.
   A class written in Python is one and a built-in is not, which is what
   decides whose storage the mirror is and whether this block holds a
   reference to it. */
static PyObject *m_type_mirror(PyObject *self, PyObject *arg)
{
    (void)self;
    PyTypeObject *tp = Py_TYPE(arg);
    return Py_BuildValue("(si)", tp->tp_name,
                         PyType_HasFeature(tp, Py_TPFLAGS_HEAPTYPE) ? 1 : 0);
}

/* The object allocator, whose blocks go back through `PyObject_Free`. */
static PyObject *m_object_blocks(PyObject *self, PyObject *unused)
{
    (void)self; (void)unused;
    char *block = (char *)PyObject_Malloc(64);
    if (block == NULL) {
        return PyErr_NoMemory();
    }
    memset(block, 'a', 64);
    char *grown = (char *)PyObject_Realloc(block, 256);
    if (grown == NULL) {
        PyObject_Free(block);
        return PyErr_NoMemory();
    }
    int kept = grown[0] == 'a' && grown[63] == 'a';
    PyObject_Free(grown);

    int *zeroed = (int *)PyObject_Calloc(16, sizeof(int));
    if (zeroed == NULL) {
        return PyErr_NoMemory();
    }
    int clear = 1;
    for (int index = 0; index < 16; index++) {
        clear = clear && zeroed[index] == 0;
    }
    PyObject_Free(zeroed);

    /* A count that would wrap is refused rather than under-allocated. */
    void *refused = PyObject_Calloc((size_t)-1 / 2, 4);
    if (refused != NULL) {
        PyObject_Free(refused);
        PyErr_SetString(PyExc_SystemError, "PyObject_Calloc accepted a wrapping product");
        return NULL;
    }
    return Py_BuildValue("(iii)", kept, clear, PyObject_GC_IsTracked(self));
}

/* The int conversions: the fixed widths, the overflow reports, the masks and
   the pointer and byte-buffer round trips.  `arg` is an int too large for a
   C `long`. */
static PyObject *m_int_convert(PyObject *self, PyObject *arg)
{
    (void)self;
    PyObject *from_small = PyLong_FromInt32(-7);
    PyObject *from_wide = PyLong_FromUInt64((uint64_t)-1);
    if (from_small == NULL || from_wide == NULL) {
        Py_XDECREF(from_small);
        Py_XDECREF(from_wide);
        return NULL;
    }

    int32_t narrow = 0;
    uint64_t wide = 0;
    if (PyLong_AsInt32(from_small, &narrow) < 0 ||
        PyLong_AsUInt64(from_wide, &wide) < 0) {
        Py_DECREF(from_small);
        Py_DECREF(from_wide);
        return NULL;
    }
    /* A value that does not fit reports the overflow rather than truncating. */
    int32_t refused = 0;
    int narrowed = PyLong_AsInt32(from_wide, &refused);
    if (narrowed == 0 || !PyErr_ExceptionMatches(PyExc_OverflowError)) {
        Py_DECREF(from_small);
        Py_DECREF(from_wide);
        PyErr_SetString(PyExc_SystemError, "PyLong_AsInt32 accepted an out-of-range value");
        return NULL;
    }
    PyErr_Clear();

    int overflow = 0;
    long fits = PyLong_AsLongAndOverflow(from_small, &overflow);
    if (fits == -1 && PyErr_Occurred()) {
        Py_DECREF(from_small);
        Py_DECREF(from_wide);
        return NULL;
    }
    int too_big = 0;
    (void)PyLong_AsLongLongAndOverflow(arg, &too_big);
    if (PyErr_Occurred()) {
        Py_DECREF(from_small);
        Py_DECREF(from_wide);
        return NULL;
    }

    unsigned long masked = PyLong_AsUnsignedLongMask(from_small);
    unsigned long long masked_wide = PyLong_AsUnsignedLongLongMask(from_small);
    if (masked != (unsigned long)-7 || masked_wide != (unsigned long long)-7) {
        Py_DECREF(from_small);
        Py_DECREF(from_wide);
        PyErr_SetString(PyExc_SystemError, "a mask conversion dropped the low bits");
        return NULL;
    }

    /* A pointer survives the round trip whether or not it sets the top bit. */
    PyObject *address = PyLong_FromVoidPtr((void *)&moduledef);
    if (address == NULL) {
        Py_DECREF(from_small);
        Py_DECREF(from_wide);
        return NULL;
    }
    int same_pointer = PyLong_AsVoidPtr(address) == (void *)&moduledef;
    Py_DECREF(address);

    /* The bytes of a C variable, in and out. */
    unsigned char buffer[8];
    Py_ssize_t needed = PyLong_AsNativeBytes(from_small, buffer, sizeof(buffer),
                                             Py_ASNATIVEBYTES_DEFAULTS);
    if (needed < 0) {
        Py_DECREF(from_small);
        Py_DECREF(from_wide);
        return NULL;
    }
    PyObject *restored = PyLong_FromNativeBytes(buffer, sizeof(buffer),
                                                Py_ASNATIVEBYTES_DEFAULTS);
    PyObject *unsigned_restored = PyLong_FromUnsignedNativeBytes(
        buffer, sizeof(buffer), Py_ASNATIVEBYTES_DEFAULTS);
    PyObject *info = PyLong_GetInfo();
    if (restored == NULL || unsigned_restored == NULL || info == NULL) {
        Py_DECREF(from_small);
        Py_DECREF(from_wide);
        Py_XDECREF(restored);
        Py_XDECREF(unsigned_restored);
        Py_XDECREF(info);
        return NULL;
    }
    PyObject *bits = PyObject_GetAttrString(info, "bits_per_digit");
    Py_DECREF(info);
    if (bits == NULL) {
        Py_DECREF(from_small);
        Py_DECREF(from_wide);
        Py_DECREF(restored);
        Py_DECREF(unsigned_restored);
        return NULL;
    }
    int digit_bits = PyLong_AsInt(bits);
    Py_DECREF(bits);
    if (digit_bits == -1 && PyErr_Occurred()) {
        Py_DECREF(from_small);
        Py_DECREF(from_wide);
        Py_DECREF(restored);
        Py_DECREF(unsigned_restored);
        return NULL;
    }

    return Py_BuildValue("(NNiKiinNNi)", from_small, from_wide, (int)narrow,
                         (unsigned long long)wide, overflow, too_big, needed,
                         restored, unsigned_restored, digit_bits);
}

/* `_PyLong_FromByteArray` / `_PyLong_AsByteArray`, including the two ways the
   write half fails.  `Py_UNUSED` names the argument neither half reads. */
static PyObject *m_byte_arrays(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    (void)self;
    const unsigned char be[4] = {0x01, 0x02, 0x03, 0x04};
    PyObject *big = _PyLong_FromByteArray(be, sizeof(be), 0, 0);
    PyObject *little = _PyLong_FromByteArray(be, sizeof(be), 1, 0);
    PyObject *negative = _PyLong_FromByteArray(be, sizeof(be), 0, 1);
    if (big == NULL || little == NULL || negative == NULL) {
        Py_XDECREF(big);
        Py_XDECREF(little);
        Py_XDECREF(negative);
        return NULL;
    }

    unsigned char out[4] = {0, 0, 0, 0};
    if (_PyLong_AsByteArray((PyLongObject *)big, out, sizeof(out), 0, 0, 1) < 0) {
        Py_DECREF(big);
        Py_DECREF(little);
        Py_DECREF(negative);
        return NULL;
    }
    int roundtrip = memcmp(out, be, sizeof(out)) == 0;

    /* A value too wide for the destination still fills it with the low bytes,
       which is what PyLong_AsNativeBytes reads back out. */
    unsigned char low[2] = {0, 0};
    int truncated = _PyLong_AsByteArray((PyLongObject *)big, low, sizeof(low), 1, 0, 0);
    int kept_low = truncated == -1 && !PyErr_Occurred() && low[0] == 0x04 && low[1] == 0x03;

    /* A negative value asked for as unsigned writes nothing and raises. */
    PyObject *below = PyLong_FromLong(-1);
    if (below == NULL) {
        Py_DECREF(big);
        Py_DECREF(little);
        Py_DECREF(negative);
        return NULL;
    }
    unsigned char untouched[2] = {0xAA, 0xAA};
    int refused = _PyLong_AsByteArray((PyLongObject *)below, untouched, sizeof(untouched),
                                      1, 0, 1);
    int raised = refused == -1 && PyErr_ExceptionMatches(PyExc_OverflowError) &&
                 untouched[0] == 0xAA && untouched[1] == 0xAA;
    PyErr_Clear();
    Py_DECREF(below);

    return Py_BuildValue("(NNNiii)", big, little, negative, roundtrip, kept_low, raised);
}

/* The unchecked accessors, which are calls here rather than field reads. */
static PyObject *m_fast_accessors(PyObject *self, PyObject *arg)
{
    (void)self;
    PyObject *text = PyUnicode_FromString("na\303\257ve");
    if (text == NULL) {
        return NULL;
    }
    Py_ssize_t points = PyUnicode_GET_LENGTH(text);
    Py_DECREF(text);

    const char *raw = PyBytes_AS_STRING(arg);
    if (raw == NULL) {
        return NULL;
    }
    Py_ssize_t size = PyBytes_GET_SIZE(arg);
    return Py_BuildValue("(ny#)", points, raw, size);
}

/* The argument formats that fill a `Py_buffer` the callee releases. */
static PyObject *m_buffer_formats(PyObject *self, PyObject *args, PyObject *keywords)
{
    (void)self;
    static char *names[] = {"text", "data", "maybe", NULL};
    Py_buffer text;
    Py_buffer data;
    Py_buffer maybe;
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "s*y*z*", names,
                                     &text, &data, &maybe)) {
        return NULL;
    }
    PyObject *result = Py_BuildValue("(y#y#nii)", (const char *)text.buf, text.len,
                                     (const char *)data.buf, data.len, maybe.len,
                                     maybe.buf == NULL, text.readonly);
    PyBuffer_Release(&text);
    PyBuffer_Release(&data);
    PyBuffer_Release(&maybe);
    return result;
}

/* The argument formats that hand over a pointer rather than a view.  The
   answer names what came back so a Python-side comparison can say whether the
   two agree; a refusal answers with the exception's own class name, the
   message being the part that differs. */
static PyObject *m_pointer_formats(PyObject *self, PyObject *args)
{
    (void)self;
    const char *format;
    PyObject *value;
    if (!PyArg_ParseTuple(args, "sO", &format, &value)) {
        return NULL;
    }
    PyObject *one = PyTuple_Pack(1, value);
    if (one == NULL) {
        return NULL;
    }
    const char *text = NULL;
    Py_ssize_t length = 0;
    int parsed = format[1] == '#'
                     ? PyArg_ParseTuple(one, format, &text, &length)
                     : PyArg_ParseTuple(one, format, &text);
    Py_DECREF(one);
    if (!parsed) {
        PyObject *kind = PyErr_Occurred();
        const char *name = kind == NULL ? "(none)" : ((PyTypeObject *)kind)->tp_name;
        PyErr_Clear();
        return Py_BuildValue("(ss)", "refused", name);
    }
    if (text == NULL) {
        return PyUnicode_FromString("null");
    }
    /* Without a size the pointer is a C string, and its own length is what the
       caller would read through it. */
    return Py_BuildValue("y#", text, format[1] == '#' ? length : (Py_ssize_t)strlen(text));
}

/* `w*` asks for a writable view.  An interpreter object only ever exports a
   read-only snapshot, so the request is refused rather than handing back a
   pointer whose writes would go nowhere. */
static PyObject *m_writable_buffer(PyObject *self, PyObject *args)
{
    (void)self;
    Py_buffer target;
    if (PyArg_ParseTuple(args, "w*", &target)) {
        PyBuffer_Release(&target);
        Py_RETURN_NONE;
    }
    if (!PyErr_ExceptionMatches(PyExc_BufferError)) {
        return NULL;
    }
    PyErr_Clear();
    return PyUnicode_FromString("read-only");
}

static PyMethodDef methods[] = {
    {"byte_arrays", m_byte_arrays, METH_NOARGS, "the private byte-array conversions"},
    {"fast_accessors", m_fast_accessors, METH_O, "the unchecked bytes and str accessors"},
    {"buffer_formats", (PyCFunction)(void (*)(void))m_buffer_formats,
     METH_VARARGS | METH_KEYWORDS, "the buffer-filling argument formats"},
    {"pointer_formats", m_pointer_formats, METH_VARARGS, "the 's', 'z' and 'y' argument formats"},
    {"writable_buffer", m_writable_buffer, METH_VARARGS, "the 'w*' argument format"},
    {"object_attrs", m_object_attrs, METH_VARARGS, "the generic attribute protocol"},
    {"object_values", m_object_values, METH_VARARGS, "comparison, hashing and formatting"},
    {"object_bytes", m_object_bytes, METH_O, "PyObject_Bytes"},
    {"bytes_from", m_bytes_from, METH_O, "PyBytes_FromObject"},
    {"bytes_fill", m_bytes_fill, METH_O, "PyBytes_FromStringAndSize(NULL, size)"},
    {"bytes_pairs", m_bytes_pairs, METH_NOARGS, "a filled buffer as a dict key and value"},
    {"bytes_empty", m_bytes_empty, METH_NOARGS, "the empty allocation and the size check"},
    {"new_ref", m_new_ref, METH_O, "Py_NewRef and Py_XNewRef"},
    {"type_mirror", m_type_mirror, METH_O, "Py_TYPE's name and heap-type flag"},
    {"object_blocks", m_object_blocks, METH_NOARGS, "the object allocator"},
    {"int_convert", m_int_convert, METH_O, "the int conversions"},
    {"module_ops", m_module_ops, METH_NOARGS, "the module constructors"},
    {"module_no_file", m_module_no_file, METH_NOARGS, "a module with no __file__"},
    {"module_from_def", m_module_from_def, METH_O, "PyModule_FromDefAndSpec"},
    {"bump", (PyCFunction)m_bump, METH_NOARGS, "bump the module counter"},
    {"wrap", (PyCFunction)m_wrap, METH_O, NULL},
    {"add", (PyCFunction)m_add, METH_VARARGS, NULL},
    {"greet", (PyCFunction)(void (*)(void))m_greet,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"total", (PyCFunction)(void (*)(void))m_total, METH_FASTCALL, NULL},
    {"layout", (PyCFunction)(void (*)(void))m_layout,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"apply", (PyCFunction)m_apply, METH_VARARGS, NULL},
    {"inspect", (PyCFunction)m_inspect, METH_VARARGS, NULL},
    {"build", (PyCFunction)m_build, METH_NOARGS, NULL},
    {"roundtrip", (PyCFunction)m_roundtrip, METH_VARARGS, NULL},
    {"numbers", (PyCFunction)m_numbers, METH_VARARGS, NULL},
    {"dict_ops", (PyCFunction)m_dict_ops, METH_NOARGS, NULL},
    {"sequences", (PyCFunction)m_sequences, METH_NOARGS, NULL},
    {"singletons", (PyCFunction)m_singletons, METH_NOARGS, NULL},
    {"predicates", (PyCFunction)m_predicates, METH_VARARGS, NULL},
    {"fail", (PyCFunction)m_fail, METH_VARARGS, NULL},
    {"caught", (PyCFunction)m_caught, METH_NOARGS, NULL},
    {"restore", (PyCFunction)m_restore, METH_NOARGS, "PyErr_Restore and its degenerate inputs"},
    {"version_macros", (PyCFunction)m_version_macros, METH_NOARGS, "the patchlevel.h macros"},
    {"call_surface", (PyCFunction)m_call_surface, METH_VARARGS, NULL},
    {"set_ops", (PyCFunction)m_set_ops, METH_VARARGS, NULL},
    {"dict_more", (PyCFunction)m_dict_more, METH_VARARGS, NULL},
    {"list_ops", (PyCFunction)m_list_ops, METH_VARARGS, NULL},
    {"slice_ops", (PyCFunction)m_slice_ops, METH_VARARGS, NULL},
    {"seq_more", (PyCFunction)m_seq_more, METH_VARARGS, NULL},
    {"to_base", (PyCFunction)m_to_base, METH_VARARGS, NULL},
    {"gc_window", (PyCFunction)m_gc_window, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL},
};

static int methods_exec(PyObject *module)
{
    methods_state *state = state_of(module);
    if (state == NULL) {
        return -1;
    }
    state->calls = 0;
    if (PyModule_GetDef(module) != &moduledef) {
        PyErr_SetString(PyExc_SystemError, "PyModule_GetDef returned another definition");
        return -1;
    }
    if (PyModule_AddIntConstant(module, "ANSWER", 42) < 0) {
        return -1;
    }
    if (PyModule_AddStringConstant(module, "GREETING", "hi") < 0) {
        return -1;
    }
    PyObject *owned = PyLong_FromLong(11);
    if (owned == NULL) {
        return -1;
    }
    if (PyModule_AddObject(module, "OWNED", owned) < 0) {
        Py_DECREF(owned);
        return -1;
    }
    PyObject *shared = PyLong_FromLong(12);
    if (shared == NULL) {
        return -1;
    }
    int added = PyModule_AddObjectRef(module, "SHARED", shared);
    Py_DECREF(shared);
    if (added < 0) {
        return -1;
    }
    PyObject *namespace_ = PyModule_GetDict(module);
    if (namespace_ == NULL) {
        return -1;
    }
    PyObject *marker = PyUnicode_FromString("through-dict");
    if (marker == NULL) {
        return -1;
    }
    added = PyDict_SetItemString(namespace_, "VIA_DICT", marker);
    Py_DECREF(marker);
    return added < 0 ? -1 : 0;
}

static PyModuleDef_Slot slots[] = {
    {Py_mod_exec, (void *)methods_exec},
    {0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cpyext_methods",
    "pyre cpyext method module",
    sizeof(methods_state),
    methods,
    slots,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC
PyInit_cpyext_methods(void)
{
    return PyModuleDef_Init(&moduledef);
}
