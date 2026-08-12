#ifndef PYRE_PYTHON_H
#define PYRE_PYTHON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PY_MAJOR_VERSION 3
#define PY_MINOR_VERSION 14
#define PY_VERSION_HEX 0x030E0000
#define PYTHON_API_VERSION 1013

#if defined(_WIN32)
#  define PyAPI_FUNC(RTYPE) __declspec(dllimport) RTYPE
#  define PyMODINIT_FUNC __declspec(dllexport) PyObject *
#elif defined(__cplusplus)
#  define PyAPI_FUNC(RTYPE) __attribute__((visibility("default"))) RTYPE
#  define PyMODINIT_FUNC extern "C" __attribute__((visibility("default"))) PyObject *
#else
#  define PyAPI_FUNC(RTYPE) __attribute__((visibility("default"))) RTYPE
#  define PyMODINIT_FUNC __attribute__((visibility("default"))) PyObject *
#endif

typedef intptr_t Py_ssize_t;
typedef struct _typeobject PyTypeObject;
typedef struct _object {
    Py_ssize_t ob_refcnt;
    Py_ssize_t ob_pyre_link;
    PyTypeObject *ob_type;
} PyObject;

#define PyObject_HEAD PyObject ob_base;
#define Py_REFCNT(ob) (((PyObject *)(ob))->ob_refcnt)
#define Py_TYPE(ob) (((PyObject *)(ob))->ob_type)
PyAPI_FUNC(void) Py_IncRef(PyObject *ob);
PyAPI_FUNC(void) Py_DecRef(PyObject *ob);
#define Py_INCREF(ob) Py_IncRef((PyObject *)(ob))
#define Py_DECREF(ob) Py_DecRef((PyObject *)(ob))
#define Py_XINCREF(ob) do { if ((ob) != NULL) Py_INCREF(ob); } while (0)
#define Py_XDECREF(ob) do { if ((ob) != NULL) Py_DECREF(ob); } while (0)

typedef PyObject *(*PyCFunction)(PyObject *, PyObject *);
typedef int (*visitproc)(PyObject *, void *);
typedef int (*traverseproc)(PyObject *, visitproc, void *);
typedef int (*inquiry)(PyObject *);
typedef void (*freefunc)(void *);
typedef struct PyMethodDef {
    const char *ml_name;
    PyCFunction ml_meth;
    int ml_flags;
    const char *ml_doc;
} PyMethodDef;

#define METH_VARARGS 0x0001
#define METH_KEYWORDS 0x0002
#define METH_NOARGS 0x0004
#define METH_O 0x0008

typedef struct PyModuleDef_Base {
    PyObject ob_base;
    PyObject *(*m_init)(void);
    Py_ssize_t m_index;
    PyObject *m_copy;
} PyModuleDef_Base;

typedef struct PyModuleDef_Slot {
    int slot;
    void *value;
} PyModuleDef_Slot;

typedef struct PyModuleDef {
    PyModuleDef_Base m_base;
    const char *m_name;
    const char *m_doc;
    Py_ssize_t m_size;
    PyMethodDef *m_methods;
    PyModuleDef_Slot *m_slots;
    traverseproc m_traverse;
    inquiry m_clear;
    freefunc m_free;
} PyModuleDef;

#define PyModuleDef_HEAD_INIT \
    { { 1, 0, NULL }, NULL, 0, NULL }

PyAPI_FUNC(PyObject *) PyModuleDef_Init(PyModuleDef *def);
PyAPI_FUNC(PyObject *) PyModule_Create2(PyModuleDef *def, int api_version);
#define PyModule_Create(module) PyModule_Create2((module), PYTHON_API_VERSION)

#define PyDoc_STRVAR(name, str) static const char name[] = str

#ifdef __cplusplus
}
#endif
#endif
