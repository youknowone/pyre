/* Every `PyDict_*` entry point, plus the keyword mapping of `PyObject_Call`,
   applied to whatever the caller passes.

   Each function answers with the observable outcome rather than with whether an
   error was set: `PyDict_Clear` reports the size that follows it, `set` reads
   the key back.  A predicate that only asked "did this raise" would pass for an
   operation that quietly did nothing. */

#include <Python.h>

static PyObject *failed(const char *what)
{
    PyErr_Clear();
    return PyUnicode_FromString(what);
}

/* ── the two predicates, and the protocol ones that consult them ───────── */
static PyObject *c_check(PyObject *s, PyObject *o)
{ (void)s; return PyBool_FromLong(PyDict_Check(o)); }
static PyObject *c_check_exact(PyObject *s, PyObject *o)
{ (void)s; return PyBool_FromLong(PyDict_CheckExact(o)); }
static PyObject *c_mapping_check(PyObject *s, PyObject *o)
{ (void)s; return PyBool_FromLong(PyMapping_Check(o)); }
static PyObject *c_sequence_check(PyObject *s, PyObject *o)
{ (void)s; return PyBool_FromLong(PySequence_Check(o)); }
static PyObject *c_iter_check(PyObject *s, PyObject *o)
{ (void)s; return PyBool_FromLong(PyIter_Check(o)); }

/* ── reads ────────────────────────────────────────────────────────────── */
static PyObject *c_size(PyObject *s, PyObject *o)
{ (void)s; Py_ssize_t n = PyDict_Size(o); PyErr_Clear(); return PyLong_FromSsize_t(n); }

static PyObject *c_getitem(PyObject *s, PyObject *o)
{
    (void)s;
    PyObject *key = PyUnicode_FromString("k");
    if (key == NULL) return NULL;
    PyObject *value = PyDict_GetItem(o, key);   /* borrowed */
    Py_DECREF(key);
    if (value == NULL) return failed("getitem-missing");
    return Py_NewRef(value);
}

static PyObject *c_getitem_string(PyObject *s, PyObject *o)
{
    (void)s;
    PyObject *value = PyDict_GetItemString(o, "k");   /* borrowed */
    if (value == NULL) return failed("getitemstring-missing");
    return Py_NewRef(value);
}

static PyObject *c_getitem_ref(PyObject *s, PyObject *o)
{
    (void)s;
    PyObject *key = PyUnicode_FromString("k");
    if (key == NULL) return NULL;
    PyObject *value = NULL;
    int found = PyDict_GetItemRef(o, key, &value);
    Py_DECREF(key);
    if (found != 1) return failed("getitemref-missing");
    return value;
}

static PyObject *c_getitem_string_ref(PyObject *s, PyObject *o)
{
    (void)s;
    PyObject *value = NULL;
    if (PyDict_GetItemStringRef(o, "k", &value) != 1) return failed("getitemstringref-missing");
    return value;
}

static PyObject *c_getitem_with_error(PyObject *s, PyObject *o)
{
    (void)s;
    PyObject *key = PyUnicode_FromString("k");
    if (key == NULL) return NULL;
    PyObject *value = PyDict_GetItemWithError(o, key);   /* borrowed */
    Py_DECREF(key);
    if (value == NULL) return failed("getitemwitherror-missing");
    return Py_NewRef(value);
}

static PyObject *c_contains(PyObject *s, PyObject *o)
{
    (void)s;
    PyObject *key = PyUnicode_FromString("k");
    if (key == NULL) return NULL;
    int has = PyDict_Contains(o, key);
    Py_DECREF(key);
    if (has < 0) return failed("contains-failed");
    return PyBool_FromLong(has);
}

static PyObject *c_keys(PyObject *s, PyObject *o)
{ (void)s; PyObject *r = PyDict_Keys(o); return r ? r : failed("keys-failed"); }
static PyObject *c_values(PyObject *s, PyObject *o)
{ (void)s; PyObject *r = PyDict_Values(o); return r ? r : failed("values-failed"); }
static PyObject *c_items(PyObject *s, PyObject *o)
{ (void)s; PyObject *r = PyDict_Items(o); return r ? r : failed("items-failed"); }
static PyObject *c_copy(PyObject *s, PyObject *o)
{ (void)s; PyObject *r = PyDict_Copy(o); return r ? r : failed("copy-failed"); }

/* Walk the whole mapping with `PyDict_Next` and report the pairs in order. */
static PyObject *c_next_items(PyObject *s, PyObject *o)
{
    (void)s;
    PyObject *out = PyList_New(0);
    if (out == NULL) return NULL;
    Py_ssize_t pos = 0;
    PyObject *key, *value;
    while (PyDict_Next(o, &pos, &key, &value)) {
        PyObject *pair = PyTuple_Pack(2, key, value);
        if (pair == NULL || PyList_Append(out, pair) < 0) {
            Py_XDECREF(pair);
            Py_DECREF(out);
            return failed("next-failed");
        }
        Py_DECREF(pair);
    }
    if (PyErr_Occurred()) { Py_DECREF(out); return failed("next-raised"); }
    return out;
}

/* ── writes, each followed by a read of what it was supposed to do ────── */
static PyObject *c_clear_then_size(PyObject *s, PyObject *o)
{
    (void)s;
    PyDict_Clear(o);
    if (PyErr_Occurred()) return failed("clear-raised");
    Py_ssize_t size = PyDict_Size(o);
    if (size < 0) return failed("clear-size-failed");
    return PyLong_FromSsize_t(size);
}

static PyObject *c_setitem_then_read(PyObject *s, PyObject *o)
{
    (void)s;
    PyObject *key = PyUnicode_FromString("added");
    if (key == NULL) return NULL;
    int set = PyDict_SetItem(o, key, Py_None);
    Py_DECREF(key);
    if (set < 0) return failed("setitem-failed");
    PyObject *back = PyDict_GetItemString(o, "added");
    if (back == NULL) return failed("setitem-readback-missing");
    return Py_NewRef(back);
}

static PyObject *c_setitem_string_then_read(PyObject *s, PyObject *o)
{
    (void)s;
    if (PyDict_SetItemString(o, "added", Py_None) < 0) return failed("setitemstring-failed");
    PyObject *back = PyDict_GetItemString(o, "added");
    if (back == NULL) return failed("setitemstring-readback-missing");
    return Py_NewRef(back);
}

static PyObject *c_delitem_then_contains(PyObject *s, PyObject *o)
{
    (void)s;
    PyObject *key = PyUnicode_FromString("k");
    if (key == NULL) return NULL;
    int gone = PyDict_DelItem(o, key);
    if (gone < 0) { Py_DECREF(key); return failed("delitem-failed"); }
    int has = PyDict_Contains(o, key);
    Py_DECREF(key);
    if (has < 0) return failed("delitem-contains-failed");
    return PyBool_FromLong(has);
}

static PyObject *c_delitem_string_then_contains(PyObject *s, PyObject *o)
{
    (void)s;
    if (PyDict_DelItemString(o, "k") < 0) return failed("delitemstring-failed");
    PyObject *value = PyDict_GetItemString(o, "k");
    if (value != NULL) return failed("delitemstring-still-there");
    PyErr_Clear();
    Py_RETURN_FALSE;
}

/* ── merge, in both directions ────────────────────────────────────────── */
static PyObject *c_merge_from(PyObject *s, PyObject *o)
{
    (void)s;
    PyObject *target = PyDict_New();
    if (target == NULL) return NULL;
    if (PyDict_Merge(target, o, 1) < 0) { Py_DECREF(target); return failed("merge-failed"); }
    return target;
}

static PyObject *c_update_from(PyObject *s, PyObject *o)
{
    (void)s;
    PyObject *target = PyDict_New();
    if (target == NULL) return NULL;
    if (PyDict_Update(target, o) < 0) { Py_DECREF(target); return failed("update-failed"); }
    return target;
}

/* Merge into the caller's mapping, so the write has to land where Python can
   see it rather than in a copy. */
static PyObject *c_merge_into(PyObject *s, PyObject *args)
{
    (void)s;
    PyObject *target, *source;
    int override;
    if (!PyArg_ParseTuple(args, "OOi", &target, &source, &override)) return NULL;
    if (PyDict_Merge(target, source, override) < 0) return failed("merge-failed");
    Py_RETURN_NONE;
}

static PyObject *c_merge_from_seq2(PyObject *s, PyObject *seq)
{
    (void)s;
    PyObject *target = PyDict_New();
    if (target == NULL) return NULL;
    if (PyDict_MergeFromSeq2(target, seq, 1) < 0) { Py_DECREF(target); return failed("seq2-failed"); }
    return target;
}

/* ── the keyword mapping of a call ────────────────────────────────────── */
static PyObject *c_call_kwargs(PyObject *s, PyObject *args)
{
    (void)s;
    PyObject *callable, *kwargs;
    if (!PyArg_ParseTuple(args, "OO", &callable, &kwargs)) return NULL;
    PyObject *empty = PyTuple_New(0);
    if (empty == NULL) return NULL;
    PyObject *result = PyObject_Call(callable, empty, kwargs);
    Py_DECREF(empty);
    if (result == NULL) return failed("call-failed");
    return result;
}

#define M(name, fn) {name, fn, METH_O, NULL}
static PyMethodDef methods[] = {
    M("check", c_check), M("check_exact", c_check_exact),
    M("mapping_check", c_mapping_check), M("sequence_check", c_sequence_check),
    M("iter_check", c_iter_check),

    M("size", c_size), M("getitem", c_getitem),
    M("getitem_string", c_getitem_string), M("getitem_ref", c_getitem_ref),
    M("getitem_string_ref", c_getitem_string_ref),
    M("getitem_with_error", c_getitem_with_error), M("contains", c_contains),
    M("keys", c_keys), M("values", c_values), M("items", c_items),
    M("copy", c_copy), M("next_items", c_next_items),

    M("clear_then_size", c_clear_then_size),
    M("setitem_then_read", c_setitem_then_read),
    M("setitem_string_then_read", c_setitem_string_then_read),
    M("delitem_then_contains", c_delitem_then_contains),
    M("delitem_string_then_contains", c_delitem_string_then_contains),

    M("merge_from", c_merge_from), M("update_from", c_update_from),
    M("merge_from_seq2", c_merge_from_seq2),
    {"merge_into", c_merge_into, METH_VARARGS, NULL},
    {"call_kwargs", c_call_kwargs, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef def = {
    PyModuleDef_HEAD_INIT, "cpyext_dict_subclass", NULL, -1, methods};

PyMODINIT_FUNC PyInit_cpyext_dict_subclass(void)
{
    return PyModule_Create(&def);
}
