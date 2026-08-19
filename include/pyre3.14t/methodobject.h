/* `PyMethodDef` and the calling conventions it selects.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_METHODOBJECT_H
#define PYRE_METHODOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
struct PyMethodDef {
    const char *ml_name;
    PyCFunction ml_meth;
    int ml_flags;
    const char *ml_doc;
};

#define METH_VARARGS 0x0001
#define METH_KEYWORDS 0x0002
#define METH_NOARGS 0x0004
#define METH_O 0x0008
#define METH_CLASS 0x0010
#define METH_STATIC 0x0020
#define METH_COEXIST 0x0040
#define METH_FASTCALL 0x0080
/* A method that is handed the class defining it beside its receiver, and so
   carries a `PyCMethodObject` rather than a `PyCFunctionObject`. */
#define METH_METHOD 0x0200

/* The two layouts a C function carries.
 *
 * Nothing pyre makes is read through these: what a `PyMethodDef`-backed
 * callable holds is reached through the entry points below, which is why they
 * are calls where the reference header spells them as field reads.  The fields
 * are declared because an extension defining a type derived from
 * `PyCFunction_Type` embeds one of these in its own instance layout and writes
 * them itself -- the block a C-defined type's instance gets is `tp_basicsize`
 * bytes, so those fields are that instance's own storage. */
typedef struct {
    PyObject_HEAD
    PyMethodDef *m_ml;
    PyObject *m_self;
    PyObject *m_module;
    PyObject *m_weakreflist;
    vectorcallfunc vectorcall;
} PyCFunctionObject;

typedef struct {
    PyCFunctionObject func;
    PyTypeObject *mm_class;
} PyCMethodObject;

/* The type an extension's own `PyMethodDef` becomes, and so the base a type
   derived from it names.  An interpreter builtin such as `len` is a different
   `builtin_function_or_method` that no symbol here names, and
   `PyCFunction_Check` answers no for it: it carries no `PyMethodDef`, so
   `PyCFunction_GetFunction` would have nothing to hand back. */
PyAPI_DATA(PyTypeObject) PyCFunction_Type;

#define PyCFunction_Check(op) PyObject_TypeCheck((op), &PyCFunction_Type)
#define PyCFunction_CheckExact(op) Py_IS_TYPE((op), &PyCFunction_Type)
#define PyCMethod_CheckExact(op) Py_IS_TYPE((op), &PyCMethod_Type)
#define PyCMethod_Check(op) PyObject_TypeCheck((op), &PyCMethod_Type)

#define PyCFunction_GET_FUNCTION(func) PyCFunction_GetFunction((PyObject *)(func))
#define PyCFunction_GET_SELF(func) PyCFunction_GetSelf((PyObject *)(func))
#define PyCFunction_GET_FLAGS(func) PyCFunction_GetFlags((PyObject *)(func))
#define PyCFunction_GET_CLASS(func) PyCFunction_GetClass((PyObject *)(func))

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_METHODOBJECT_H */
