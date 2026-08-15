#ifndef PYRE_PYTHON_H
#define PYRE_PYTHON_H

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PY_MAJOR_VERSION 3
#define PY_MINOR_VERSION 14
#define PY_MICRO_VERSION 6
/* 3.14.6 final, matching sys.hexversion. The release-level nibble is 0xF for a
   final release, so a value ending in 0x00 would put every `#if PY_VERSION_HEX
   >= 0x030E00F0` extension on its pre-release branch. */
#define PY_VERSION_HEX 0x030E06F0
#define PYTHON_API_VERSION 1013
#define PYTHON_ABI_VERSION 3

#if defined(_WIN32)
#  define PyAPI_FUNC(RTYPE) __declspec(dllimport) RTYPE
#  define PyAPI_DATA(RTYPE) extern __declspec(dllimport) RTYPE
#  define PyMODINIT_FUNC __declspec(dllexport) PyObject *
#elif defined(__cplusplus)
#  define PyAPI_FUNC(RTYPE) __attribute__((visibility("default"))) RTYPE
#  define PyAPI_DATA(RTYPE) extern __attribute__((visibility("default"))) RTYPE
#  define PyMODINIT_FUNC extern "C" __attribute__((visibility("default"))) PyObject *
#else
#  define PyAPI_FUNC(RTYPE) __attribute__((visibility("default"))) RTYPE
#  define PyAPI_DATA(RTYPE) extern __attribute__((visibility("default"))) RTYPE
#  define PyMODINIT_FUNC __attribute__((visibility("default"))) PyObject *
#endif

typedef intptr_t Py_ssize_t;
#define PY_SSIZE_T_MAX ((Py_ssize_t)(((size_t)-1) >> 1))
#define PY_SSIZE_T_MIN (-PY_SSIZE_T_MAX - 1)
typedef Py_ssize_t Py_hash_t;
typedef size_t Py_uhash_t;

typedef struct _typeobject PyTypeObject;

/* `ob_pyre_link` is the interpreter object this mirror stands for.  It is
   opaque to C: the interpreter moves its objects, and only the runtime may
   read or write the slot. */
typedef struct _object {
    Py_ssize_t ob_refcnt;
    Py_ssize_t ob_pyre_link;
    PyTypeObject *ob_type;
} PyObject;

#define PyObject_HEAD PyObject ob_base;
#define PyObject_HEAD_INIT(type) { 0, 0, type },
#define Py_REFCNT(ob) (((PyObject *)(ob))->ob_refcnt)
#define Py_TYPE(ob) (((PyObject *)(ob))->ob_type)
PyAPI_FUNC(void) Py_IncRef(PyObject *ob);
PyAPI_FUNC(void) Py_DecRef(PyObject *ob);
#define Py_INCREF(ob) Py_IncRef((PyObject *)(ob))
#define Py_DECREF(ob) Py_DecRef((PyObject *)(ob))
#define Py_XINCREF(ob) do { if ((ob) != NULL) Py_INCREF(ob); } while (0)
#define Py_XDECREF(ob) do { if ((ob) != NULL) Py_DECREF(ob); } while (0)
#define Py_CLEAR(ob) do { PyObject *_tmp = (PyObject *)(ob); (ob) = NULL; Py_XDECREF(_tmp); } while (0)
#define Py_SETREF(ob, value) do { PyObject *_old = (PyObject *)(ob); (ob) = (value); Py_XDECREF(_old); } while (0)

/* The singletons.  Each is a mirror the runtime binds to its interpreter
   object before the first `PyInit_*` runs, so pointer comparison against them
   is the identity test C code expects it to be. */
PyAPI_DATA(PyObject) _Py_NoneStruct;
PyAPI_DATA(PyObject) _Py_TrueStruct;
PyAPI_DATA(PyObject) _Py_FalseStruct;
PyAPI_DATA(PyObject) _Py_NotImplementedStruct;
PyAPI_DATA(PyObject) _Py_EllipsisObject;
#define Py_None (&_Py_NoneStruct)
#define Py_True (&_Py_TrueStruct)
#define Py_False (&_Py_FalseStruct)
#define Py_NotImplemented (&_Py_NotImplementedStruct)
#define Py_Ellipsis (&_Py_EllipsisObject)
#define Py_RETURN_NONE do { Py_INCREF(Py_None); return Py_None; } while (0)
#define Py_RETURN_TRUE do { Py_INCREF(Py_True); return Py_True; } while (0)
#define Py_RETURN_FALSE do { Py_INCREF(Py_False); return Py_False; } while (0)
#define Py_RETURN_NOTIMPLEMENTED \
    do { Py_INCREF(Py_NotImplemented); return Py_NotImplemented; } while (0)
#define Py_Is(x, y) ((x) == (y))
#define Py_IsNone(x) Py_Is((x), Py_None)
#define Py_IsTrue(x) Py_Is((x), Py_True)
#define Py_IsFalse(x) Py_Is((x), Py_False)

typedef PyObject *(*PyCFunction)(PyObject *, PyObject *);
typedef PyObject *(*PyCFunctionWithKeywords)(PyObject *, PyObject *, PyObject *);
typedef PyObject *(*_PyCFunctionFast)(PyObject *, PyObject *const *, Py_ssize_t);
typedef PyObject *(*_PyCFunctionFastWithKeywords)(PyObject *, PyObject *const *,
                                                  Py_ssize_t, PyObject *);
typedef int (*visitproc)(PyObject *, void *);
typedef int (*traverseproc)(PyObject *, visitproc, void *);
typedef int (*inquiry)(PyObject *);
typedef void (*freefunc)(void *);
typedef int (*converter)(PyObject *, void *);

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
#define METH_CLASS 0x0010
#define METH_STATIC 0x0020
#define METH_COEXIST 0x0040
#define METH_FASTCALL 0x0080

/* Types. */
typedef struct {
    PyObject ob_base;
    Py_ssize_t ob_size;
} PyVarObject;

#define PyObject_VAR_HEAD PyVarObject ob_base;
#define PyVarObject_HEAD_INIT(type, size) { PyObject_HEAD_INIT(type) size },
#define Py_SIZE(ob) (((PyVarObject *)(ob))->ob_size)

typedef void (*destructor)(PyObject *);
typedef PyObject *(*unaryfunc)(PyObject *);
typedef PyObject *(*binaryfunc)(PyObject *, PyObject *);
typedef PyObject *(*ternaryfunc)(PyObject *, PyObject *, PyObject *);
typedef PyObject *(*reprfunc)(PyObject *);
typedef Py_hash_t (*hashfunc)(PyObject *);
typedef PyObject *(*getattrfunc)(PyObject *, char *);
typedef int (*setattrfunc)(PyObject *, char *, PyObject *);
typedef PyObject *(*getattrofunc)(PyObject *, PyObject *);
typedef int (*setattrofunc)(PyObject *, PyObject *, PyObject *);
typedef PyObject *(*richcmpfunc)(PyObject *, PyObject *, int);
typedef PyObject *(*getiterfunc)(PyObject *);
typedef PyObject *(*iternextfunc)(PyObject *);
typedef PyObject *(*descrgetfunc)(PyObject *, PyObject *, PyObject *);
typedef int (*descrsetfunc)(PyObject *, PyObject *, PyObject *);
typedef int (*initproc)(PyObject *, PyObject *, PyObject *);
typedef PyObject *(*newfunc)(PyTypeObject *, PyObject *, PyObject *);
typedef PyObject *(*allocfunc)(PyTypeObject *, Py_ssize_t);
typedef PyObject *(*getter)(PyObject *, void *);
typedef int (*setter)(PyObject *, PyObject *, void *);
typedef PyObject *(*vectorcallfunc)(PyObject *, PyObject *const *, size_t, PyObject *);

typedef Py_ssize_t (*lenfunc)(PyObject *);
typedef PyObject *(*ssizeargfunc)(PyObject *, Py_ssize_t);
typedef int (*ssizeobjargproc)(PyObject *, Py_ssize_t, PyObject *);
typedef int (*objobjproc)(PyObject *, PyObject *);
typedef int (*objobjargproc)(PyObject *, PyObject *, PyObject *);

typedef struct {
    binaryfunc nb_add;
    binaryfunc nb_subtract;
    binaryfunc nb_multiply;
    binaryfunc nb_remainder;
    binaryfunc nb_divmod;
    ternaryfunc nb_power;
    unaryfunc nb_negative;
    unaryfunc nb_positive;
    unaryfunc nb_absolute;
    inquiry nb_bool;
    unaryfunc nb_invert;
    binaryfunc nb_lshift;
    binaryfunc nb_rshift;
    binaryfunc nb_and;
    binaryfunc nb_xor;
    binaryfunc nb_or;
    unaryfunc nb_int;
    void *nb_reserved;
    unaryfunc nb_float;
    binaryfunc nb_inplace_add;
    binaryfunc nb_inplace_subtract;
    binaryfunc nb_inplace_multiply;
    binaryfunc nb_inplace_remainder;
    ternaryfunc nb_inplace_power;
    binaryfunc nb_inplace_lshift;
    binaryfunc nb_inplace_rshift;
    binaryfunc nb_inplace_and;
    binaryfunc nb_inplace_xor;
    binaryfunc nb_inplace_or;
    binaryfunc nb_floor_divide;
    binaryfunc nb_true_divide;
    binaryfunc nb_inplace_floor_divide;
    binaryfunc nb_inplace_true_divide;
    unaryfunc nb_index;
    binaryfunc nb_matrix_multiply;
    binaryfunc nb_inplace_matrix_multiply;
} PyNumberMethods;

typedef struct {
    lenfunc sq_length;
    binaryfunc sq_concat;
    ssizeargfunc sq_repeat;
    ssizeargfunc sq_item;
    void *was_sq_slice;
    ssizeobjargproc sq_ass_item;
    void *was_sq_ass_slice;
    objobjproc sq_contains;
    binaryfunc sq_inplace_concat;
    ssizeargfunc sq_inplace_repeat;
} PySequenceMethods;

typedef struct {
    lenfunc mp_length;
    binaryfunc mp_subscript;
    objobjargproc mp_ass_subscript;
} PyMappingMethods;

typedef enum {
    PYGEN_RETURN = 0,
    PYGEN_ERROR = -1,
    PYGEN_NEXT = 1
} PySendResult;

typedef PySendResult (*sendfunc)(PyObject *, PyObject *, PyObject **);

typedef struct {
    unaryfunc am_await;
    unaryfunc am_aiter;
    unaryfunc am_anext;
    sendfunc am_send;
} PyAsyncMethods;

typedef struct {
    void *buf;
    PyObject *obj;
    Py_ssize_t len;
    Py_ssize_t itemsize;
    int readonly;
    int ndim;
    char *format;
    Py_ssize_t *shape;
    Py_ssize_t *strides;
    Py_ssize_t *suboffsets;
    void *internal;
} Py_buffer;

#define PyBUF_MAX_NDIM 64

#define PyBUF_SIMPLE 0
#define PyBUF_WRITABLE 0x0001
#define PyBUF_WRITEABLE PyBUF_WRITABLE
#define PyBUF_FORMAT 0x0004
#define PyBUF_ND 0x0008
#define PyBUF_STRIDES (0x0010 | PyBUF_ND)
#define PyBUF_C_CONTIGUOUS (0x0020 | PyBUF_STRIDES)
#define PyBUF_F_CONTIGUOUS (0x0040 | PyBUF_STRIDES)
#define PyBUF_ANY_CONTIGUOUS (0x0080 | PyBUF_STRIDES)
#define PyBUF_INDIRECT (0x0100 | PyBUF_STRIDES)

#define PyBUF_CONTIG (PyBUF_ND | PyBUF_WRITABLE)
#define PyBUF_CONTIG_RO (PyBUF_ND)
#define PyBUF_STRIDED (PyBUF_STRIDES | PyBUF_WRITABLE)
#define PyBUF_STRIDED_RO (PyBUF_STRIDES)
#define PyBUF_RECORDS (PyBUF_STRIDES | PyBUF_WRITABLE | PyBUF_FORMAT)
#define PyBUF_RECORDS_RO (PyBUF_STRIDES | PyBUF_FORMAT)
#define PyBUF_FULL (PyBUF_INDIRECT | PyBUF_WRITABLE | PyBUF_FORMAT)
#define PyBUF_FULL_RO (PyBUF_INDIRECT | PyBUF_FORMAT)

#define PyBUF_READ 0x100
#define PyBUF_WRITE 0x200

typedef int (*getbufferproc)(PyObject *, Py_buffer *, int);
typedef void (*releasebufferproc)(PyObject *, Py_buffer *);

typedef struct {
    getbufferproc bf_getbuffer;
    releasebufferproc bf_releasebuffer;
} PyBufferProcs;

typedef struct PyMemberDef {
    const char *name;
    int type;
    Py_ssize_t offset;
    int flags;
    const char *doc;
} PyMemberDef;

#define Py_T_SHORT 0
#define Py_T_INT 1
#define Py_T_LONG 2
#define Py_T_FLOAT 3
#define Py_T_DOUBLE 4
#define Py_T_STRING 5
#define Py_T_OBJECT_EX 16
#define Py_T_CHAR 7
#define Py_T_BYTE 8
#define Py_T_UBYTE 9
#define Py_T_USHORT 10
#define Py_T_UINT 11
#define Py_T_ULONG 12
#define Py_T_BOOL 14
#define Py_T_LONGLONG 17
#define Py_T_ULONGLONG 18
#define Py_T_PYSSIZET 19
#define Py_READONLY 1
/* The `structmember.h` spellings, which are what most sources still use. */
#define T_SHORT Py_T_SHORT
#define T_INT Py_T_INT
#define T_LONG Py_T_LONG
#define T_FLOAT Py_T_FLOAT
#define T_DOUBLE Py_T_DOUBLE
#define T_STRING Py_T_STRING
#define T_OBJECT 6
#define T_OBJECT_EX Py_T_OBJECT_EX
#define T_CHAR Py_T_CHAR
#define T_BYTE Py_T_BYTE
#define T_UBYTE Py_T_UBYTE
#define T_USHORT Py_T_USHORT
#define T_UINT Py_T_UINT
#define T_ULONG Py_T_ULONG
#define T_BOOL Py_T_BOOL
#define T_LONGLONG Py_T_LONGLONG
#define T_ULONGLONG Py_T_ULONGLONG
#define T_PYSSIZET Py_T_PYSSIZET
#define T_NONE 20
#define READONLY Py_READONLY

typedef struct PyGetSetDef {
    const char *name;
    getter get;
    setter set;
    const char *doc;
    void *closure;
} PyGetSetDef;

struct _typeobject {
    PyObject_VAR_HEAD
    const char *tp_name;
    Py_ssize_t tp_basicsize;
    Py_ssize_t tp_itemsize;
    destructor tp_dealloc;
    Py_ssize_t tp_vectorcall_offset;
    getattrfunc tp_getattr;
    setattrfunc tp_setattr;
    PyAsyncMethods *tp_as_async;
    reprfunc tp_repr;
    PyNumberMethods *tp_as_number;
    PySequenceMethods *tp_as_sequence;
    PyMappingMethods *tp_as_mapping;
    hashfunc tp_hash;
    ternaryfunc tp_call;
    reprfunc tp_str;
    getattrofunc tp_getattro;
    setattrofunc tp_setattro;
    PyBufferProcs *tp_as_buffer;
    unsigned long tp_flags;
    const char *tp_doc;
    traverseproc tp_traverse;
    inquiry tp_clear;
    richcmpfunc tp_richcompare;
    Py_ssize_t tp_weaklistoffset;
    getiterfunc tp_iter;
    iternextfunc tp_iternext;
    PyMethodDef *tp_methods;
    PyMemberDef *tp_members;
    PyGetSetDef *tp_getset;
    PyTypeObject *tp_base;
    PyObject *tp_dict;
    descrgetfunc tp_descr_get;
    descrsetfunc tp_descr_set;
    Py_ssize_t tp_dictoffset;
    initproc tp_init;
    allocfunc tp_alloc;
    newfunc tp_new;
    freefunc tp_free;
    inquiry tp_is_gc;
    PyObject *tp_bases;
    PyObject *tp_mro;
    PyObject *tp_cache;
    void *tp_subclasses;
    PyObject *tp_weaklist;
    destructor tp_del;
    unsigned int tp_version_tag;
    destructor tp_finalize;
    vectorcallfunc tp_vectorcall;
    unsigned char tp_watched;
    uint16_t tp_versions_used;
};

/* Type flags.  `Py_TPFLAGS_DEFAULT` is 0 here as it is upstream, where it
   reduces to `Py_TPFLAGS_HAVE_STACKLESS_EXTENSION | 0` and that half is 0 off
   Stackless.

   Only the eight `*_SUBCLASS` flags and the three pyre sets itself -- READY,
   READYING and HEAPTYPE -- are ever filled in by pyre. The rest are declared
   so that an extension naming one compiles, and carry whatever the extension
   put there; a flag describing an object layout pyre does not have (the
   MANAGED/PREHEADER/INLINE_VALUES/ITEMS_AT_END group) is never acted on. */
#define Py_TPFLAGS_DEFAULT 0UL
#define Py_TPFLAGS_MANAGED_WEAKREF (1UL << 3)
#define Py_TPFLAGS_MANAGED_DICT (1UL << 4)
#define Py_TPFLAGS_PREHEADER (Py_TPFLAGS_MANAGED_WEAKREF | Py_TPFLAGS_MANAGED_DICT)
#define Py_TPFLAGS_SEQUENCE (1UL << 5)
#define Py_TPFLAGS_MAPPING (1UL << 6)
#define Py_TPFLAGS_DISALLOW_INSTANTIATION (1UL << 7)
#define Py_TPFLAGS_IMMUTABLETYPE (1UL << 8)
#define Py_TPFLAGS_HEAPTYPE (1UL << 9)
#define Py_TPFLAGS_BASETYPE (1UL << 10)
#define Py_TPFLAGS_HAVE_VECTORCALL (1UL << 11)
#define Py_TPFLAGS_READY (1UL << 12)
#define Py_TPFLAGS_READYING (1UL << 13)
#define Py_TPFLAGS_HAVE_GC (1UL << 14)
#define Py_TPFLAGS_HAVE_STACKLESS_EXTENSION 0UL
#define Py_TPFLAGS_METHOD_DESCRIPTOR (1UL << 17)
#define Py_TPFLAGS_HAVE_VERSION_TAG (1UL << 18)
#define Py_TPFLAGS_VALID_VERSION_TAG (1UL << 19)
#define Py_TPFLAGS_IS_ABSTRACT (1UL << 20)
#define Py_TPFLAGS_INLINE_VALUES (1UL << 2)
#define Py_TPFLAGS_ITEMS_AT_END (1UL << 23)
#define Py_TPFLAGS_LONG_SUBCLASS (1UL << 24)
#define Py_TPFLAGS_LIST_SUBCLASS (1UL << 25)
#define Py_TPFLAGS_TUPLE_SUBCLASS (1UL << 26)
#define Py_TPFLAGS_BYTES_SUBCLASS (1UL << 27)
#define Py_TPFLAGS_UNICODE_SUBCLASS (1UL << 28)
#define Py_TPFLAGS_DICT_SUBCLASS (1UL << 29)
#define Py_TPFLAGS_BASE_EXC_SUBCLASS (1UL << 30)
#define Py_TPFLAGS_TYPE_SUBCLASS (1UL << 31)
#define Py_TPFLAGS_HAVE_FINALIZE (1UL << 0)

#define PyType_HasFeature(type, feature) ((PyType_GetFlags(type) & (feature)) != 0)
#define PyType_FastSubclass(type, flag) PyType_HasFeature(type, flag)

/* `tp_richcompare` operations. */
#define Py_LT 0
#define Py_LE 1
#define Py_EQ 2
#define Py_NE 3
#define Py_GT 4
#define Py_GE 5

PyAPI_FUNC(int) PyType_Ready(PyTypeObject *type);
PyAPI_FUNC(int) PyType_Check(PyObject *object);
PyAPI_FUNC(int) PyType_IsSubtype(PyTypeObject *subtype, PyTypeObject *supertype);
PyAPI_FUNC(unsigned long) PyType_GetFlags(PyTypeObject *type);
PyAPI_FUNC(PyObject *) PyType_GenericAlloc(PyTypeObject *type, Py_ssize_t nitems);
PyAPI_FUNC(PyObject *) PyType_GenericNew(PyTypeObject *type, PyObject *args, PyObject *kwds);
PyAPI_FUNC(PyObject *) PyObject_Init(PyObject *object, PyTypeObject *type);

/* Heap types.  Only single inheritance is supported: a spec naming more than
   one base is rejected rather than silently losing the rest. */
typedef struct PyType_Slot {
    int slot;
    void *pfunc;
} PyType_Slot;

typedef struct PyType_Spec {
    const char *name;
    int basicsize;
    int itemsize;
    unsigned int flags;
    PyType_Slot *slots;
} PyType_Spec;

PyAPI_FUNC(PyObject *) PyType_FromSpec(PyType_Spec *spec);
PyAPI_FUNC(PyObject *) PyType_FromSpecWithBases(PyType_Spec *spec, PyObject *bases);
PyAPI_FUNC(PyObject *) PyType_FromModuleAndSpec(PyObject *module, PyType_Spec *spec,
                                                PyObject *bases);
PyAPI_FUNC(PyObject *) PyType_FromMetaclass(PyTypeObject *metaclass, PyObject *module,
                                            PyType_Spec *spec, PyObject *bases);
PyAPI_FUNC(void *) PyType_GetSlot(PyTypeObject *type, int slot);
PyAPI_FUNC(PyObject *) PyType_GetName(PyTypeObject *type);
PyAPI_FUNC(PyObject *) PyType_GetQualName(PyTypeObject *type);
PyAPI_FUNC(PyObject *) PyType_GetModuleName(PyTypeObject *type);
PyAPI_FUNC(PyObject *) PyType_GetFullyQualifiedName(PyTypeObject *type);
PyAPI_FUNC(PyObject *) PyType_GetModule(PyTypeObject *type);
PyAPI_FUNC(void *) PyType_GetModuleState(PyTypeObject *type);
/* `PyType_GetModuleByDef` needs `PyModuleDef`, so it is declared with it. */
PyAPI_FUNC(int) PyType_GetBaseByToken(PyTypeObject *type, void *token, PyTypeObject **result);
PyAPI_FUNC(Py_ssize_t) PyType_GetTypeDataSize(PyTypeObject *type);
PyAPI_FUNC(void *) PyObject_GetTypeData(PyObject *object, PyTypeObject *type);
PyAPI_FUNC(void *) PyObject_GetItemData(PyObject *object);
PyAPI_FUNC(void) PyType_Modified(PyTypeObject *type);
PyAPI_FUNC(unsigned int) PyType_ClearCache(void);
PyAPI_FUNC(int) PyType_Freeze(PyTypeObject *type);

#define Py_bf_getbuffer 1
#define Py_bf_releasebuffer 2
#define Py_mp_ass_subscript 3
#define Py_mp_length 4
#define Py_mp_subscript 5
#define Py_nb_absolute 6
#define Py_nb_add 7
#define Py_nb_and 8
#define Py_nb_bool 9
#define Py_nb_divmod 10
#define Py_nb_float 11
#define Py_nb_floor_divide 12
#define Py_nb_index 13
#define Py_nb_inplace_add 14
#define Py_nb_inplace_and 15
#define Py_nb_inplace_floor_divide 16
#define Py_nb_inplace_lshift 17
#define Py_nb_inplace_multiply 18
#define Py_nb_inplace_or 19
#define Py_nb_inplace_power 20
#define Py_nb_inplace_remainder 21
#define Py_nb_inplace_rshift 22
#define Py_nb_inplace_subtract 23
#define Py_nb_inplace_true_divide 24
#define Py_nb_inplace_xor 25
#define Py_nb_int 26
#define Py_nb_invert 27
#define Py_nb_lshift 28
#define Py_nb_multiply 29
#define Py_nb_negative 30
#define Py_nb_or 31
#define Py_nb_positive 32
#define Py_nb_power 33
#define Py_nb_remainder 34
#define Py_nb_rshift 35
#define Py_nb_subtract 36
#define Py_nb_true_divide 37
#define Py_nb_xor 38
#define Py_sq_ass_item 39
#define Py_sq_concat 40
#define Py_sq_contains 41
#define Py_sq_inplace_concat 42
#define Py_sq_inplace_repeat 43
#define Py_sq_item 44
#define Py_sq_length 45
#define Py_sq_repeat 46
#define Py_tp_alloc 47
#define Py_tp_base 48
#define Py_tp_bases 49
#define Py_tp_call 50
#define Py_tp_clear 51
#define Py_tp_dealloc 52
#define Py_tp_del 53
#define Py_tp_descr_get 54
#define Py_tp_descr_set 55
#define Py_tp_doc 56
#define Py_tp_getattr 57
#define Py_tp_getattro 58
#define Py_tp_hash 59
#define Py_tp_init 60
#define Py_tp_is_gc 61
#define Py_tp_iter 62
#define Py_tp_iternext 63
#define Py_tp_methods 64
#define Py_tp_new 65
#define Py_tp_repr 66
#define Py_tp_richcompare 67
#define Py_tp_setattr 68
#define Py_tp_setattro 69
#define Py_tp_str 70
#define Py_tp_traverse 71
#define Py_tp_members 72
#define Py_tp_getset 73
#define Py_tp_free 74
#define Py_nb_matrix_multiply 75
#define Py_nb_inplace_matrix_multiply 76
#define Py_am_await 77
#define Py_am_aiter 78
#define Py_am_anext 79
#define Py_tp_finalize 80
#define Py_am_send 81
#define Py_tp_vectorcall 82
#define Py_tp_token 83
/* The `Py_tp_token` value asking for the spec's own address as the token. */
#define Py_TP_USE_SPEC NULL
#define PyObject_TypeCheck(ob, type) \
    (Py_TYPE(ob) == (type) || PyType_IsSubtype(Py_TYPE(ob), (type)))
#define PyObject_New(type, tp) ((type *)PyType_GenericAlloc((tp), 0))
#define PyObject_GC_New(type, tp) PyObject_New(type, tp)
PyAPI_FUNC(void) PyObject_Free(void *ob);

/* The raw allocators.  The `PyMem_*` and `PyMem_Raw*` halves are the same
   functions, as they are upstream (`cpyext/src/pymem.c:57`).  Memory from
   these never holds an interpreter object. */
PyAPI_FUNC(void *) PyMem_Malloc(size_t size);
PyAPI_FUNC(void *) PyMem_Calloc(size_t nelem, size_t elsize);
PyAPI_FUNC(void *) PyMem_Realloc(void *ptr, size_t size);
PyAPI_FUNC(void) PyMem_Free(void *ptr);
PyAPI_FUNC(void *) PyMem_RawMalloc(size_t size);
PyAPI_FUNC(void *) PyMem_RawCalloc(size_t nelem, size_t elsize);
PyAPI_FUNC(void *) PyMem_RawRealloc(void *ptr, size_t size);
PyAPI_FUNC(void) PyMem_RawFree(void *ptr);

#define PyMem_New(type, n) \
    ((type *)((size_t)(n) > (size_t)PY_SSIZE_T_MAX / sizeof(type) \
              ? NULL : PyMem_Malloc((n) * sizeof(type))))
#define PyMem_Resize(p, type, n) \
    ((p) = ((size_t)(n) > (size_t)PY_SSIZE_T_MAX / sizeof(type) \
            ? NULL : (type *)PyMem_Realloc((p), (n) * sizeof(type))))
#define PyMem_Del PyMem_Free
#define PyMem_DEL PyMem_Free
#define PyMem_MALLOC PyMem_Malloc
#define PyMem_REALLOC PyMem_Realloc
#define PyMem_FREE PyMem_Free
/* No `PyObject_Malloc` alias: `PyObject_Free` releases an object block, which
   is not what these hand out, so the two would not pair. */
PyAPI_FUNC(void) PyObject_Del(void *ob);
#define PyObject_GC_Del(ob) PyObject_Del(ob)
#define PyObject_GC_Track(ob) ((void)(ob))
#define PyObject_GC_UnTrack(ob) ((void)(ob))

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

#define Py_mod_create 1
#define Py_mod_exec 2

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
PyAPI_FUNC(PyObject *) PyModule_FromDefAndSpec2(PyModuleDef *def, PyObject *spec,
                                                int api_version);
#define PyModule_FromDefAndSpec(module, spec) \
    PyModule_FromDefAndSpec2((module), (spec), PYTHON_API_VERSION)
PyAPI_FUNC(int) PyModule_ExecDef(PyObject *module, PyModuleDef *def);
PyAPI_FUNC(PyObject *) PyType_GetModuleByDef(PyTypeObject *type, PyModuleDef *def);
PyAPI_FUNC(PyObject *) PyModule_New(const char *name);
PyAPI_FUNC(PyObject *) PyModule_NewObject(PyObject *name);
PyAPI_FUNC(int) PyModule_Check(PyObject *object);
PyAPI_FUNC(int) PyModule_CheckExact(PyObject *object);
PyAPI_FUNC(PyObject *) PyModule_GetDict(PyObject *module);
PyAPI_FUNC(void *) PyModule_GetState(PyObject *module);
PyAPI_FUNC(PyModuleDef *) PyModule_GetDef(PyObject *module);
PyAPI_FUNC(const char *) PyModule_GetName(PyObject *module);
PyAPI_FUNC(PyObject *) PyModule_GetNameObject(PyObject *module);
PyAPI_FUNC(const char *) PyModule_GetFilename(PyObject *module);
PyAPI_FUNC(PyObject *) PyModule_GetFilenameObject(PyObject *module);
PyAPI_FUNC(int) PyModule_SetDocString(PyObject *module, const char *doc);
PyAPI_FUNC(int) PyModule_AddFunctions(PyObject *module, PyMethodDef *functions);
PyAPI_FUNC(int) PyModule_Add(PyObject *module, const char *name, PyObject *value);
PyAPI_FUNC(int) PyModule_AddObject(PyObject *module, const char *name, PyObject *value);
PyAPI_FUNC(int) PyModule_AddObjectRef(PyObject *module, const char *name, PyObject *value);
PyAPI_FUNC(int) PyModule_AddIntConstant(PyObject *module, const char *name, Py_ssize_t value);
PyAPI_FUNC(int) PyModule_AddStringConstant(PyObject *module, const char *name, const char *value);
#define PyModule_AddIntMacro(module, macro) \
    PyModule_AddIntConstant((module), #macro, (Py_ssize_t)(macro))
#define PyModule_AddStringMacro(module, macro) \
    PyModule_AddStringConstant((module), #macro, (macro))

/* Exceptions. */
PyAPI_DATA(PyObject *) PyExc_BaseException;
PyAPI_DATA(PyObject *) PyExc_Exception;
PyAPI_DATA(PyObject *) PyExc_ArithmeticError;
PyAPI_DATA(PyObject *) PyExc_AssertionError;
PyAPI_DATA(PyObject *) PyExc_AttributeError;
PyAPI_DATA(PyObject *) PyExc_BufferError;
PyAPI_DATA(PyObject *) PyExc_EOFError;
PyAPI_DATA(PyObject *) PyExc_FileNotFoundError;
PyAPI_DATA(PyObject *) PyExc_FloatingPointError;
PyAPI_DATA(PyObject *) PyExc_GeneratorExit;
PyAPI_DATA(PyObject *) PyExc_ImportError;
PyAPI_DATA(PyObject *) PyExc_IndexError;
PyAPI_DATA(PyObject *) PyExc_KeyError;
PyAPI_DATA(PyObject *) PyExc_KeyboardInterrupt;
PyAPI_DATA(PyObject *) PyExc_LookupError;
PyAPI_DATA(PyObject *) PyExc_MemoryError;
PyAPI_DATA(PyObject *) PyExc_ModuleNotFoundError;
PyAPI_DATA(PyObject *) PyExc_NameError;
PyAPI_DATA(PyObject *) PyExc_NotImplementedError;
PyAPI_DATA(PyObject *) PyExc_OSError;
PyAPI_DATA(PyObject *) PyExc_OverflowError;
PyAPI_DATA(PyObject *) PyExc_RecursionError;
PyAPI_DATA(PyObject *) PyExc_ReferenceError;
PyAPI_DATA(PyObject *) PyExc_RuntimeError;
PyAPI_DATA(PyObject *) PyExc_StopAsyncIteration;
PyAPI_DATA(PyObject *) PyExc_StopIteration;
PyAPI_DATA(PyObject *) PyExc_SyntaxError;
PyAPI_DATA(PyObject *) PyExc_SystemError;
PyAPI_DATA(PyObject *) PyExc_SystemExit;
PyAPI_DATA(PyObject *) PyExc_TypeError;
PyAPI_DATA(PyObject *) PyExc_UnboundLocalError;
PyAPI_DATA(PyObject *) PyExc_UnicodeDecodeError;
PyAPI_DATA(PyObject *) PyExc_UnicodeEncodeError;
PyAPI_DATA(PyObject *) PyExc_UnicodeError;
PyAPI_DATA(PyObject *) PyExc_UnicodeTranslateError;
PyAPI_DATA(PyObject *) PyExc_ValueError;
PyAPI_DATA(PyObject *) PyExc_ZeroDivisionError;

PyAPI_FUNC(void) PyErr_SetObject(PyObject *type, PyObject *value);
PyAPI_FUNC(void) PyErr_SetString(PyObject *type, const char *message);
PyAPI_FUNC(void) PyErr_SetNone(PyObject *type);
PyAPI_FUNC(PyObject *) PyErr_Occurred(void);
PyAPI_FUNC(void) PyErr_Clear(void);
PyAPI_FUNC(PyObject *) PyErr_NoMemory(void);
PyAPI_FUNC(int) PyErr_BadArgument(void);
PyAPI_FUNC(void) PyErr_BadInternalCall(void);
PyAPI_FUNC(int) PyErr_ExceptionMatches(PyObject *expected);
PyAPI_FUNC(int) PyErr_GivenExceptionMatches(PyObject *given, PyObject *expected);
PyAPI_FUNC(void) PyErr_Fetch(PyObject **type, PyObject **value, PyObject **traceback);
PyAPI_FUNC(void) PyErr_Restore(PyObject *type, PyObject *value, PyObject *traceback);
PyAPI_FUNC(void) PyErr_NormalizeException(PyObject **type, PyObject **value, PyObject **traceback);
PyAPI_FUNC(PyObject *) PyErr_NewException(const char *name, PyObject *base, PyObject *dict);
PyAPI_FUNC(PyObject *) PyErr_NewExceptionWithDoc(const char *name, const char *doc,
                                                 PyObject *base, PyObject *dict);
PyAPI_FUNC(PyObject *) _PyPyre_ErrFormatted(PyObject *type, const char *message);

/* Objects. */
PyAPI_FUNC(PyObject *) PyObject_GetAttrString(PyObject *object, const char *name);
PyAPI_FUNC(int) PyObject_SetAttrString(PyObject *object, const char *name, PyObject *value);
PyAPI_FUNC(int) PyObject_HasAttrString(PyObject *object, const char *name);
PyAPI_FUNC(PyObject *) PyObject_GetAttr(PyObject *object, PyObject *name);
PyAPI_FUNC(int) PyObject_SetAttr(PyObject *object, PyObject *name, PyObject *value);
PyAPI_FUNC(PyObject *) PyObject_Str(PyObject *object);
PyAPI_FUNC(PyObject *) PyObject_Repr(PyObject *object);
PyAPI_FUNC(int) PyObject_IsTrue(PyObject *object);
PyAPI_FUNC(int) PyObject_Not(PyObject *object);
PyAPI_FUNC(Py_ssize_t) PyObject_Size(PyObject *object);
#define PyObject_Length PyObject_Size
PyAPI_FUNC(PyObject *) PyObject_GetItem(PyObject *object, PyObject *key);
PyAPI_FUNC(int) PyObject_SetItem(PyObject *object, PyObject *key, PyObject *value);
PyAPI_FUNC(PyObject *) PyObject_Call(PyObject *callable, PyObject *args, PyObject *kwargs);
PyAPI_FUNC(PyObject *) PyObject_CallObject(PyObject *callable, PyObject *args);
PyAPI_FUNC(int) PyCallable_Check(PyObject *object);

/* The vectorcall protocol.  `args[0 .. nargs)` are positional and the rest are
   the values for the names in `kwnames`, in order.  The high bit of `nargsf`
   says the caller left a spare slot before args[0]; pyre only reads the array,
   so the bit just has to be stripped from the count. */
#define PY_VECTORCALL_ARGUMENTS_OFFSET ((size_t)1 << (8 * sizeof(size_t) - 1))

PyAPI_FUNC(PyObject *) PyObject_Vectorcall(PyObject *callable,
                                           PyObject *const *args,
                                           size_t nargsf, PyObject *kwnames);
PyAPI_FUNC(PyObject *) PyObject_VectorcallMethod(PyObject *name,
                                                 PyObject *const *args,
                                                 size_t nargsf,
                                                 PyObject *kwnames);
PyAPI_FUNC(PyObject *) PyVectorcall_Call(PyObject *callable, PyObject *args,
                                         PyObject *kwargs);
PyAPI_FUNC(PyObject *) PyObject_CallNoArgs(PyObject *callable);
PyAPI_FUNC(PyObject *) PyObject_CallOneArg(PyObject *callable, PyObject *arg);

static inline Py_ssize_t PyVectorcall_NARGS(size_t n)
{
    return (Py_ssize_t)(n & ~PY_VECTORCALL_ARGUMENTS_OFFSET);
}

static inline PyObject *PyObject_CallMethodNoArgs(PyObject *self, PyObject *name)
{
    PyObject *args[1];
    args[0] = self;
    return PyObject_VectorcallMethod(name, args, 1, NULL);
}

static inline PyObject *PyObject_CallMethodOneArg(PyObject *self, PyObject *name,
                                                  PyObject *arg)
{
    PyObject *args[2];
    args[0] = self;
    args[1] = arg;
    return PyObject_VectorcallMethod(name, args, 2, NULL);
}
PyAPI_FUNC(int) PyObject_IsInstance(PyObject *object, PyObject *class_);
PyAPI_FUNC(PyObject *) PyObject_Type(PyObject *object);

/* Capsules.  The destructor is recorded but never runs: pyre has no object
   deallocation path to call it from. */
typedef void (*PyCapsule_Destructor)(PyObject *);
PyAPI_FUNC(PyObject *) PyCapsule_New(void *pointer, const char *name,
                                     PyCapsule_Destructor destructor);
PyAPI_FUNC(void *) PyCapsule_GetPointer(PyObject *capsule, const char *name);
PyAPI_FUNC(int) PyCapsule_SetPointer(PyObject *capsule, void *pointer);
PyAPI_FUNC(const char *) PyCapsule_GetName(PyObject *capsule);
PyAPI_FUNC(int) PyCapsule_SetName(PyObject *capsule, const char *name);
PyAPI_FUNC(void *) PyCapsule_GetContext(PyObject *capsule);
PyAPI_FUNC(int) PyCapsule_SetContext(PyObject *capsule, void *context);
PyAPI_FUNC(PyCapsule_Destructor) PyCapsule_GetDestructor(PyObject *capsule);
PyAPI_FUNC(int) PyCapsule_SetDestructor(PyObject *capsule, PyCapsule_Destructor destructor);
PyAPI_FUNC(int) PyCapsule_IsValid(PyObject *capsule, const char *name);
PyAPI_FUNC(void *) PyCapsule_Import(const char *name, int no_block);
PyAPI_FUNC(int) PyCapsule_CheckExact(PyObject *object);

/* Imports.  The borrowed-reference `PyImport_AddModule` and
   `PyImport_GetModuleDict` are absent: pyre has no container to hang the
   borrow on, so only the strong-reference forms exist. */
PyAPI_FUNC(PyObject *) PyImport_ImportModule(const char *name);
PyAPI_FUNC(PyObject *) PyImport_ImportModuleNoBlock(const char *name);
PyAPI_FUNC(PyObject *) PyImport_Import(PyObject *name);
PyAPI_FUNC(PyObject *) PyImport_AddModuleRef(const char *name);
PyAPI_FUNC(PyObject *) PyImport_GetModule(PyObject *name);

/* The number protocol. */
PyAPI_FUNC(PyObject *) PyNumber_Add(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_Subtract(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_Multiply(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_MatrixMultiply(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_FloorDivide(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_TrueDivide(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_Remainder(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_Divmod(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_Lshift(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_Rshift(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_And(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_Xor(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_Or(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_Power(PyObject *base, PyObject *exponent, PyObject *modulus);
PyAPI_FUNC(PyObject *) PyNumber_InPlacePower(PyObject *base, PyObject *exponent, PyObject *modulus);
PyAPI_FUNC(PyObject *) PyNumber_ToBase(PyObject *object, int base);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceAdd(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceSubtract(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceMultiply(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceMatrixMultiply(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceFloorDivide(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceTrueDivide(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceRemainder(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceLshift(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceRshift(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceAnd(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceXor(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceOr(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PyNumber_Negative(PyObject *object);
PyAPI_FUNC(PyObject *) PyNumber_Positive(PyObject *object);
PyAPI_FUNC(PyObject *) PyNumber_Absolute(PyObject *object);
PyAPI_FUNC(PyObject *) PyNumber_Invert(PyObject *object);
PyAPI_FUNC(PyObject *) PyNumber_Index(PyObject *object);
PyAPI_FUNC(PyObject *) PyNumber_Float(PyObject *object);
PyAPI_FUNC(int) PyNumber_Check(PyObject *object);
PyAPI_FUNC(Py_ssize_t) PyNumber_AsSsize_t(PyObject *object, PyObject *exc);

/* The sequence protocol. */
PyAPI_FUNC(int) PySequence_Check(PyObject *object);
PyAPI_FUNC(Py_ssize_t) PySequence_Size(PyObject *object);
PyAPI_FUNC(Py_ssize_t) PySequence_Length(PyObject *object);
PyAPI_FUNC(PyObject *) PySequence_Concat(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PySequence_InPlaceConcat(PyObject *left, PyObject *right);
PyAPI_FUNC(PyObject *) PySequence_Repeat(PyObject *object, Py_ssize_t count);
PyAPI_FUNC(PyObject *) PySequence_InPlaceRepeat(PyObject *object, Py_ssize_t count);
PyAPI_FUNC(PyObject *) PySequence_GetItem(PyObject *object, Py_ssize_t index);
PyAPI_FUNC(int) PySequence_SetItem(PyObject *object, Py_ssize_t index, PyObject *value);
PyAPI_FUNC(int) PySequence_DelItem(PyObject *object, Py_ssize_t index);
PyAPI_FUNC(int) PySequence_Contains(PyObject *object, PyObject *value);
PyAPI_FUNC(Py_ssize_t) PySequence_Index(PyObject *object, PyObject *value);
PyAPI_FUNC(PyObject *) PySequence_List(PyObject *object);
PyAPI_FUNC(PyObject *) PySequence_Tuple(PyObject *object);
PyAPI_FUNC(PyObject *) PySequence_GetSlice(PyObject *object, Py_ssize_t start, Py_ssize_t stop);
PyAPI_FUNC(int) PySequence_SetSlice(PyObject *object, Py_ssize_t low, Py_ssize_t high, PyObject *value);
PyAPI_FUNC(int) PySequence_DelSlice(PyObject *object, Py_ssize_t low, Py_ssize_t high);
PyAPI_FUNC(int) PySequence_In(PyObject *object, PyObject *value);
PyAPI_FUNC(Py_ssize_t) PySequence_Count(PyObject *object, PyObject *value);
PyAPI_FUNC(PyObject *) PySequence_Fast(PyObject *object, const char *message);
/* Functions rather than the macros the reference header spells: a mirror has
   no item array of its own, so the length and the items come from the
   interpreter object behind it. */
PyAPI_FUNC(Py_ssize_t) PySequence_Fast_GET_SIZE(PyObject *object);
PyAPI_FUNC(PyObject *) PySequence_Fast_GET_ITEM(PyObject *object, Py_ssize_t index);

/* The mapping protocol. */
PyAPI_FUNC(int) PyMapping_Check(PyObject *object);
PyAPI_FUNC(Py_ssize_t) PyMapping_Size(PyObject *object);
PyAPI_FUNC(Py_ssize_t) PyMapping_Length(PyObject *object);
PyAPI_FUNC(PyObject *) PyMapping_GetItemString(PyObject *object, const char *key);
PyAPI_FUNC(int) PyMapping_SetItemString(PyObject *object, const char *key, PyObject *value);
PyAPI_FUNC(int) PyMapping_DelItemString(PyObject *object, const char *key);
PyAPI_FUNC(int) PyMapping_HasKey(PyObject *object, PyObject *key);
PyAPI_FUNC(int) PyMapping_HasKeyString(PyObject *object, const char *key);
PyAPI_FUNC(PyObject *) PyMapping_Keys(PyObject *object);
PyAPI_FUNC(PyObject *) PyMapping_Values(PyObject *object);
PyAPI_FUNC(PyObject *) PyMapping_Items(PyObject *object);

/* The iterator protocol. */
PyAPI_FUNC(PyObject *) PyObject_GetIter(PyObject *object);
PyAPI_FUNC(PyObject *) PyObject_SelfIter(PyObject *object);
PyAPI_FUNC(int) PyIter_Check(PyObject *object);
PyAPI_FUNC(PyObject *) PyIter_Next(PyObject *object);
PyAPI_FUNC(int) PyIter_NextItem(PyObject *iterator, PyObject **item);
PyAPI_FUNC(PySendResult) PyIter_Send(PyObject *iterator, PyObject *value, PyObject **result);
PyAPI_FUNC(PyObject *) PyObject_GetAIter(PyObject *object);
PyAPI_FUNC(int) PyAiter_Check(PyObject *object);

/* The buffer protocol. */
PyAPI_FUNC(int) PyObject_CheckBuffer(PyObject *object);
PyAPI_FUNC(int) PyObject_GetBuffer(PyObject *object, Py_buffer *view, int flags);
PyAPI_FUNC(void) PyBuffer_Release(Py_buffer *view);
PyAPI_FUNC(int) PyBuffer_FillInfo(Py_buffer *view, PyObject *object, void *buf,
                                  Py_ssize_t length, int readonly, int flags);
PyAPI_FUNC(int) PyBuffer_IsContiguous(const Py_buffer *view, char order);
PyAPI_FUNC(Py_ssize_t) PyBuffer_SizeFromFormat(const char *format);
PyAPI_FUNC(int) PyObject_CopyData(PyObject *destination, PyObject *source);
PyAPI_FUNC(int) PyBuffer_ToContiguous(void *buf, const Py_buffer *view,
                                      Py_ssize_t length, char order);
PyAPI_FUNC(int) PyBuffer_FromContiguous(const Py_buffer *view, const void *buf,
                                        Py_ssize_t length, char order);

/* memoryview. */
/* The set protocol (`cpyext/setobject.py`).  The `Check` spellings are
   functions here, as pyre's other type checks are. */
PyAPI_FUNC(PyObject *) PySet_New(PyObject *iterable);
PyAPI_FUNC(PyObject *) PyFrozenSet_New(PyObject *iterable);
PyAPI_FUNC(int) PySet_Add(PyObject *set, PyObject *key);
PyAPI_FUNC(int) PySet_Discard(PyObject *set, PyObject *key);
PyAPI_FUNC(PyObject *) PySet_Pop(PyObject *set);
PyAPI_FUNC(int) PySet_Clear(PyObject *set);
PyAPI_FUNC(Py_ssize_t) PySet_Size(PyObject *set);
PyAPI_FUNC(int) PySet_Contains(PyObject *set, PyObject *key);
PyAPI_FUNC(int) PySet_Check(PyObject *object);
PyAPI_FUNC(int) PyFrozenSet_Check(PyObject *object);
PyAPI_FUNC(int) PyAnySet_Check(PyObject *object);
#define PySet_GET_SIZE(ob) PySet_Size((PyObject *)(ob))

PyAPI_FUNC(int) PyMemoryView_Check(PyObject *object);
PyAPI_FUNC(PyObject *) PyMemoryView_FromObject(PyObject *object);
PyAPI_FUNC(PyObject *) PyMemoryView_FromMemory(char *memory, Py_ssize_t size, int flags);
PyAPI_FUNC(PyObject *) PyMemoryView_FromBuffer(const Py_buffer *view);

/* int / bool. */
PyAPI_FUNC(PyObject *) PyLong_FromLong(long value);
PyAPI_FUNC(PyObject *) PyLong_FromLongLong(long long value);
PyAPI_FUNC(PyObject *) PyLong_FromSsize_t(Py_ssize_t value);
PyAPI_FUNC(PyObject *) PyLong_FromUnsignedLong(unsigned long value);
PyAPI_FUNC(PyObject *) PyLong_FromUnsignedLongLong(unsigned long long value);
PyAPI_FUNC(PyObject *) PyLong_FromSize_t(size_t value);
PyAPI_FUNC(PyObject *) PyLong_FromDouble(double value);
PyAPI_FUNC(PyObject *) PyLong_FromString(const char *text, char **end, int base);
PyAPI_FUNC(long) PyLong_AsLong(PyObject *object);
PyAPI_FUNC(long long) PyLong_AsLongLong(PyObject *object);
PyAPI_FUNC(Py_ssize_t) PyLong_AsSsize_t(PyObject *object);
PyAPI_FUNC(unsigned long) PyLong_AsUnsignedLong(PyObject *object);
PyAPI_FUNC(unsigned long long) PyLong_AsUnsignedLongLong(PyObject *object);
PyAPI_FUNC(size_t) PyLong_AsSize_t(PyObject *object);
PyAPI_FUNC(double) PyLong_AsDouble(PyObject *object);
PyAPI_FUNC(int) PyLong_Check(PyObject *object);
PyAPI_FUNC(int) PyLong_CheckExact(PyObject *object);
PyAPI_FUNC(PyObject *) PyNumber_Long(PyObject *object);
PyAPI_FUNC(PyObject *) PyBool_FromLong(long value);

/* float. */
PyAPI_FUNC(PyObject *) PyFloat_FromDouble(double value);
PyAPI_FUNC(double) PyFloat_AsDouble(PyObject *object);
PyAPI_FUNC(int) PyFloat_Check(PyObject *object);
PyAPI_FUNC(int) PyFloat_CheckExact(PyObject *object);

/* str. */
PyAPI_FUNC(PyObject *) PyUnicode_FromString(const char *text);
PyAPI_FUNC(PyObject *) PyUnicode_FromStringAndSize(const char *text, Py_ssize_t size);
PyAPI_FUNC(const char *) PyUnicode_AsUTF8(PyObject *object);
PyAPI_FUNC(const char *) PyUnicode_AsUTF8AndSize(PyObject *object, Py_ssize_t *size);
PyAPI_FUNC(Py_ssize_t) PyUnicode_GetLength(PyObject *object);
PyAPI_FUNC(int) PyUnicode_Check(PyObject *object);
PyAPI_FUNC(int) PyUnicode_CheckExact(PyObject *object);

/* bytes. */
PyAPI_FUNC(PyObject *) PyBytes_FromString(const char *text);
PyAPI_FUNC(PyObject *) PyBytes_FromStringAndSize(const char *text, Py_ssize_t size);
PyAPI_FUNC(char *) PyBytes_AsString(PyObject *object);
PyAPI_FUNC(int) PyBytes_AsStringAndSize(PyObject *object, char **buffer, Py_ssize_t *size);
PyAPI_FUNC(Py_ssize_t) PyBytes_Size(PyObject *object);
PyAPI_FUNC(int) PyBytes_Check(PyObject *object);
PyAPI_FUNC(int) PyBytes_CheckExact(PyObject *object);

/* tuple. */
PyAPI_FUNC(PyObject *) PyTuple_New(Py_ssize_t size);
PyAPI_FUNC(Py_ssize_t) PyTuple_Size(PyObject *object);
PyAPI_FUNC(PyObject *) PyTuple_GetItem(PyObject *object, Py_ssize_t index);
PyAPI_FUNC(int) PyTuple_SetItem(PyObject *object, Py_ssize_t index, PyObject *item);
PyAPI_FUNC(PyObject *) PyTuple_GetSlice(PyObject *object, Py_ssize_t low, Py_ssize_t high);
PyAPI_FUNC(int) PyTuple_Check(PyObject *object);
PyAPI_FUNC(int) PyTuple_CheckExact(PyObject *object);
#define PyTuple_GET_SIZE(ob) PyTuple_Size((PyObject *)(ob))
#define PyTuple_GET_ITEM(ob, i) PyTuple_GetItem((PyObject *)(ob), (i))
#define PyTuple_SET_ITEM(ob, i, v) ((void)PyTuple_SetItem((PyObject *)(ob), (i), (v)))

/* Variadic, so it is built here out of the non-variadic exports. */
static inline PyObject *PyTuple_Pack(Py_ssize_t n, ...)
{
    PyObject *result = PyTuple_New(n);
    va_list vargs;
    Py_ssize_t i;

    if (result == NULL) {
        return NULL;
    }
    va_start(vargs, n);
    for (i = 0; i < n; i++) {
        PyObject *item = va_arg(vargs, PyObject *);
        Py_XINCREF(item);
        if (PyTuple_SetItem(result, i, item) < 0) {
            va_end(vargs);
            Py_DECREF(result);
            return NULL;
        }
    }
    va_end(vargs);
    return result;
}

/* list. */
PyAPI_FUNC(PyObject *) PyList_New(Py_ssize_t size);
PyAPI_FUNC(Py_ssize_t) PyList_Size(PyObject *object);
PyAPI_FUNC(PyObject *) PyList_GetItem(PyObject *object, Py_ssize_t index);
PyAPI_FUNC(int) PyList_SetItem(PyObject *object, Py_ssize_t index, PyObject *item);
PyAPI_FUNC(int) PyList_Append(PyObject *object, PyObject *item);
PyAPI_FUNC(PyObject *) PyList_GetItemRef(PyObject *object, Py_ssize_t index);
PyAPI_FUNC(int) PyList_Insert(PyObject *object, Py_ssize_t index, PyObject *item);
PyAPI_FUNC(int) PyList_Sort(PyObject *object);
PyAPI_FUNC(int) PyList_Reverse(PyObject *object);
PyAPI_FUNC(PyObject *) PyList_AsTuple(PyObject *object);
PyAPI_FUNC(PyObject *) PyList_GetSlice(PyObject *object, Py_ssize_t low, Py_ssize_t high);
PyAPI_FUNC(int) PyList_SetSlice(PyObject *object, Py_ssize_t low, Py_ssize_t high, PyObject *itemlist);
PyAPI_FUNC(int) PyList_Check(PyObject *object);
PyAPI_FUNC(int) PyList_CheckExact(PyObject *object);
#define PyList_GET_SIZE(ob) PyList_Size((PyObject *)(ob))
#define PyList_GET_ITEM(ob, i) PyList_GetItem((PyObject *)(ob), (i))
#define PyList_SET_ITEM(ob, i, v) ((void)PyList_SetItem((PyObject *)(ob), (i), (v)))

/* slice. */
PyAPI_FUNC(PyObject *) PySlice_New(PyObject *start, PyObject *stop, PyObject *step);
PyAPI_FUNC(int) PySlice_Check(PyObject *object);
PyAPI_FUNC(int) PySlice_Unpack(PyObject *slice, Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step);
PyAPI_FUNC(Py_ssize_t) PySlice_AdjustIndices(Py_ssize_t length, Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t step);
PyAPI_FUNC(int) PySlice_GetIndices(PyObject *slice, Py_ssize_t length, Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step);
PyAPI_FUNC(int) PySlice_GetIndicesEx(PyObject *slice, Py_ssize_t length, Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step, Py_ssize_t *slicelength);
/* The spelling every extension compiles against.  The exported function is the
   same composition, so a caller reaching either gets the same answer. */
#define PySlice_GetIndicesEx(slice, length, start, stop, step, slicelen) (      \
    PySlice_Unpack((slice), (start), (stop), (step)) < 0 ?                      \
    ((*(slicelen) = 0), -1) :                                                   \
    ((*(slicelen) = PySlice_AdjustIndices((length), (start), (stop), *(step))), \
     0))

/* dict. */
PyAPI_FUNC(PyObject *) PyDict_New(void);
PyAPI_FUNC(int) PyDict_SetItem(PyObject *object, PyObject *key, PyObject *value);
PyAPI_FUNC(int) PyDict_SetItemString(PyObject *object, const char *key, PyObject *value);
PyAPI_FUNC(PyObject *) PyDict_GetItem(PyObject *object, PyObject *key);
PyAPI_FUNC(PyObject *) PyDict_GetItemString(PyObject *object, const char *key);
PyAPI_FUNC(int) PyDict_DelItem(PyObject *object, PyObject *key);
PyAPI_FUNC(Py_ssize_t) PyDict_Size(PyObject *object);
PyAPI_FUNC(int) PyDict_Contains(PyObject *object, PyObject *key);
PyAPI_FUNC(void) PyDict_Clear(PyObject *object);
PyAPI_FUNC(PyObject *) PyDict_Copy(PyObject *object);
PyAPI_FUNC(int) PyDict_DelItemString(PyObject *object, const char *key);
PyAPI_FUNC(PyObject *) PyDict_GetItemWithError(PyObject *object, PyObject *key);
PyAPI_FUNC(int) PyDict_GetItemRef(PyObject *object, PyObject *key, PyObject **result);
PyAPI_FUNC(int) PyDict_GetItemStringRef(PyObject *object, const char *key, PyObject **result);
PyAPI_FUNC(PyObject *) PyDict_Keys(PyObject *object);
PyAPI_FUNC(PyObject *) PyDict_Values(PyObject *object);
PyAPI_FUNC(PyObject *) PyDict_Items(PyObject *object);
PyAPI_FUNC(int) PyDict_Merge(PyObject *into, PyObject *from, int override_);
PyAPI_FUNC(int) PyDict_Update(PyObject *into, PyObject *from);
PyAPI_FUNC(int) PyDict_MergeFromSeq2(PyObject *into, PyObject *seq2, int override_);
PyAPI_FUNC(int) PyDict_Next(PyObject *object, Py_ssize_t *pos, PyObject **key, PyObject **value);
PyAPI_FUNC(int) PyDict_Check(PyObject *object);
PyAPI_FUNC(int) PyDict_CheckExact(PyObject *object);

#define PyDoc_STRVAR(name, str) static const char name[] = str
#define PyDoc_STR(str) str

/* `PyModule_AddType` readies the type and binds it under the part of
   `tp_name` after the last dot. */
static inline int PyModule_AddType(PyObject *module, PyTypeObject *type)
{
    const char *name;
    const char *dot;
    if (PyType_Ready(type) < 0) {
        return -1;
    }
    name = type->tp_name;
    dot = strrchr(name, '.');
    if (dot != NULL) {
        name = dot + 1;
    }
    Py_INCREF(type);
    if (PyModule_AddObject(module, name, (PyObject *)type) < 0) {
        Py_DECREF(type);
        return -1;
    }
    return 0;
}

/* ── The variadic entry points ─────────────────────────────────────────
 *
 * `PyArg_ParseTuple`, `Py_BuildValue` and `PyErr_Format` are C functions with
 * C bodies, as they are upstream (`pypy/module/cpyext/src/getargs.c`).  Pyre
 * ships no companion library, so they are `static inline` here and every
 * extension compiles its own copy; each one is built entirely out of the
 * non-variadic entry points declared above.
 */

static inline void _PyPyre_ArgError(const char *fname, const char *message)
{
    char buffer[256];
    snprintf(buffer, sizeof(buffer), "%s() %s", fname ? fname : "function", message);
    PyErr_SetString(PyExc_TypeError, buffer);
}

/* Convert one argument according to *format, advancing it past the unit. */
static inline int _PyPyre_ArgConvert(PyObject *arg, const char **format,
                                     va_list *va, const char *fname)
{
    char code = **format;
    (*format)++;
    switch (code) {
    case 'b': case 'B': case 'h': case 'H': case 'i': case 'I':
    case 'l': case 'k': case 'L': case 'K': case 'n': case 'p': {
        long long value;
        if (code == 'p') {
            int truth = PyObject_IsTrue(arg);
            if (truth < 0) {
                return 0;
            }
            value = truth;
        } else {
            value = PyLong_AsLongLong(arg);
            if (value == -1 && PyErr_Occurred()) {
                return 0;
            }
        }
        switch (code) {
        case 'b': *va_arg(*va, char *) = (char)value; break;
        case 'B': *va_arg(*va, unsigned char *) = (unsigned char)value; break;
        case 'h': *va_arg(*va, short *) = (short)value; break;
        case 'H': *va_arg(*va, unsigned short *) = (unsigned short)value; break;
        case 'i': case 'p': *va_arg(*va, int *) = (int)value; break;
        case 'I': *va_arg(*va, unsigned int *) = (unsigned int)value; break;
        case 'l': *va_arg(*va, long *) = (long)value; break;
        case 'k': *va_arg(*va, unsigned long *) = (unsigned long)value; break;
        case 'L': *va_arg(*va, long long *) = value; break;
        case 'K': *va_arg(*va, unsigned long long *) = (unsigned long long)value; break;
        case 'n': *va_arg(*va, Py_ssize_t *) = (Py_ssize_t)value; break;
        default: break;
        }
        return 1;
    }
    case 'f': case 'd': {
        double value = PyFloat_AsDouble(arg);
        if (value == -1.0 && PyErr_Occurred()) {
            return 0;
        }
        if (code == 'f') {
            *va_arg(*va, float *) = (float)value;
        } else {
            *va_arg(*va, double *) = value;
        }
        return 1;
    }
    case 's': case 'z': case 'y': {
        int with_size = (**format == '#');
        if (with_size) {
            (*format)++;
        }
        const char *text = NULL;
        Py_ssize_t length = 0;
        if (code == 'z' && Py_IsNone(arg)) {
            text = NULL;
        } else if (code == 'y') {
            char *buffer = NULL;
            if (PyBytes_AsStringAndSize(arg, &buffer, &length) < 0) {
                return 0;
            }
            text = buffer;
        } else {
            text = PyUnicode_AsUTF8AndSize(arg, &length);
            if (text == NULL) {
                return 0;
            }
        }
        *va_arg(*va, const char **) = text;
        if (with_size) {
            *va_arg(*va, Py_ssize_t *) = length;
        }
        return 1;
    }
    case 'c': {
        char *buffer = NULL;
        Py_ssize_t length = 0;
        if (PyBytes_AsStringAndSize(arg, &buffer, &length) < 0) {
            return 0;
        }
        if (length != 1) {
            _PyPyre_ArgError(fname, "argument must be a byte string of length 1");
            return 0;
        }
        *va_arg(*va, char *) = buffer[0];
        return 1;
    }
    case 'C': {
        Py_ssize_t length = PyUnicode_GetLength(arg);
        const char *text = PyUnicode_AsUTF8(arg);
        if (text == NULL) {
            return 0;
        }
        if (length != 1) {
            _PyPyre_ArgError(fname, "argument must be a string of length 1");
            return 0;
        }
        *va_arg(*va, int *) = (unsigned char)text[0];
        return 1;
    }
    case 'S': {
        if (!PyBytes_Check(arg)) {
            _PyPyre_ArgError(fname, "argument must be bytes");
            return 0;
        }
        *va_arg(*va, PyObject **) = arg;
        return 1;
    }
    case 'U': {
        if (!PyUnicode_Check(arg)) {
            _PyPyre_ArgError(fname, "argument must be str");
            return 0;
        }
        *va_arg(*va, PyObject **) = arg;
        return 1;
    }
    case 'O': {
        if (**format == '!') {
            (*format)++;
            PyObject *expected = (PyObject *)va_arg(*va, PyTypeObject *);
            int matched = PyObject_IsInstance(arg, expected);
            if (matched < 0) {
                return 0;
            }
            if (!matched) {
                _PyPyre_ArgError(fname, "argument has the wrong type");
                return 0;
            }
            *va_arg(*va, PyObject **) = arg;
            return 1;
        }
        if (**format == '&') {
            (*format)++;
            converter convert = (converter)va_arg(*va, converter);
            void *target = va_arg(*va, void *);
            return convert(arg, target) != 0;
        }
        *va_arg(*va, PyObject **) = arg;
        return 1;
    }
    default:
        _PyPyre_ArgError(fname, "uses an argument format this build does not support");
        return 0;
    }
}

/* Step over one format unit, consuming its destination pointers without
   writing them -- `skipitem()` in getargs.c.  The pointers still have to leave
   the `va_list`, or the unit after an absent optional reads the slot the
   absent one would have written. */
static inline void _PyPyre_ArgSkip(const char **format, va_list *va)
{
    char code = **format;
    (*format)++;
    switch (code) {
    case 's': case 'z': case 'y':
        (void)va_arg(*va, void *);
        if (**format == '#') {
            (*format)++;
            (void)va_arg(*va, void *);
        }
        break;
    case 'O':
        if (**format == '!' || **format == '&') {
            (*format)++;
            (void)va_arg(*va, void *);
        }
        (void)va_arg(*va, void *);
        break;
    default:
        (void)va_arg(*va, void *);
        break;
    }
}

/* How many argument units the format string describes, and where the optional
   tail begins.  `|`, `$`, `:` and `;` are markers rather than units. */
static inline void _PyPyre_ArgCount(const char *format, Py_ssize_t *total,
                                    Py_ssize_t *required, const char **fname)
{
    Py_ssize_t count = 0;
    Py_ssize_t minimum = -1;
    for (const char *cursor = format; *cursor; cursor++) {
        switch (*cursor) {
        case '|': case '$':
            if (minimum < 0) {
                minimum = count;
            }
            break;
        case ':': case ';':
            if (*cursor == ':' && fname != NULL) {
                *fname = cursor + 1;
            }
            goto done;
        case '#': case '!': case '&':
            break;
        default:
            count++;
            break;
        }
    }
done:
    *total = count;
    *required = minimum < 0 ? count : minimum;
}

static inline int _PyPyre_VaParse(PyObject *args, PyObject *kwargs,
                                  const char *format, char **keywords,
                                  va_list *va, const char *fname)
{
    Py_ssize_t total = 0;
    Py_ssize_t required = 0;
    _PyPyre_ArgCount(format, &total, &required, &fname);
    Py_ssize_t given = args == NULL ? 0 : PyTuple_Size(args);
    if (given < 0) {
        return 0;
    }
    if (given > total) {
        char buffer[128];
        snprintf(buffer, sizeof(buffer),
                 "takes at most %zd arguments (%zd given)", total, given);
        _PyPyre_ArgError(fname, buffer);
        return 0;
    }
    const char *cursor = format;
    for (Py_ssize_t index = 0; index < total; index++) {
        while (*cursor == '|' || *cursor == '$') {
            cursor++;
        }
        if (*cursor == '\0' || *cursor == ':' || *cursor == ';') {
            break;
        }
        PyObject *arg = NULL;
        if (index < given) {
            arg = PyTuple_GetItem(args, index);
            if (arg == NULL) {
                return 0;
            }
        } else if (kwargs != NULL && keywords != NULL && keywords[index] != NULL) {
            arg = PyDict_GetItemString(kwargs, keywords[index]);
        }
        if (arg == NULL) {
            if (index < required) {
                char buffer[128];
                snprintf(buffer, sizeof(buffer),
                         "requires at least %zd arguments (%zd given)", required, given);
                _PyPyre_ArgError(fname, buffer);
                return 0;
            }
            /* Absent optional: the destination keeps whatever the caller left
               in it, but its pointer still leaves the `va_list`. */
            _PyPyre_ArgSkip(&cursor, va);
            continue;
        }
        if (!_PyPyre_ArgConvert(arg, &cursor, va, fname)) {
            return 0;
        }
    }
    if (kwargs != NULL && keywords != NULL) {
        /* An unexpected keyword is an error, which is the only reason
           `PyArg_ParseTupleAndKeywords` needs the name list at all once the
           positional mapping above is done. */
        Py_ssize_t size = PyDict_Size(kwargs);
        Py_ssize_t matched = 0;
        for (Py_ssize_t index = 0; index < total && keywords[index] != NULL; index++) {
            if (index >= given && PyDict_GetItemString(kwargs, keywords[index]) != NULL) {
                matched++;
            }
        }
        if (matched != size) {
            _PyPyre_ArgError(fname, "got an unexpected keyword argument");
            return 0;
        }
    }
    return 1;
}

static inline int PyArg_ParseTuple(PyObject *args, const char *format, ...)
{
    va_list va;
    va_start(va, format);
    int parsed = _PyPyre_VaParse(args, NULL, format, NULL, &va, NULL);
    va_end(va);
    return parsed;
}

static inline int PyArg_ParseTupleAndKeywords(PyObject *args, PyObject *kwargs,
                                              const char *format, char **keywords, ...)
{
    va_list va;
    va_start(va, keywords);
    int parsed = _PyPyre_VaParse(args, kwargs, format, keywords, &va, NULL);
    va_end(va);
    return parsed;
}

static inline int PyArg_UnpackTuple(PyObject *args, const char *name,
                                    Py_ssize_t least, Py_ssize_t most, ...)
{
    Py_ssize_t given = PyTuple_Size(args);
    if (given < 0) {
        return 0;
    }
    if (given < least || given > most) {
        char buffer[128];
        snprintf(buffer, sizeof(buffer),
                 "expected %zd-%zd arguments (%zd given)", least, most, given);
        _PyPyre_ArgError(name, buffer);
        return 0;
    }
    va_list va;
    va_start(va, most);
    for (Py_ssize_t index = 0; index < most; index++) {
        PyObject **target = va_arg(va, PyObject **);
        *target = index < given ? PyTuple_GetItem(args, index) : NULL;
    }
    va_end(va);
    return 1;
}

static inline PyObject *_PyPyre_BuildValue(const char **format, va_list *va);

/* One `Py_BuildValue` unit.  Containers recurse until their closing bracket. */
static inline PyObject *_PyPyre_BuildOne(const char **format, va_list *va)
{
    char code = **format;
    (*format)++;
    switch (code) {
    case 'i': case 'b': case 'h':
        return PyLong_FromLong((long)va_arg(*va, int));
    case 'B': case 'H': case 'I':
        return PyLong_FromUnsignedLong((unsigned long)va_arg(*va, unsigned int));
    case 'l':
        return PyLong_FromLong(va_arg(*va, long));
    case 'k':
        return PyLong_FromUnsignedLong(va_arg(*va, unsigned long));
    case 'L':
        return PyLong_FromLongLong(va_arg(*va, long long));
    case 'K':
        return PyLong_FromUnsignedLongLong(va_arg(*va, unsigned long long));
    case 'n':
        return PyLong_FromSsize_t(va_arg(*va, Py_ssize_t));
    case 'f': case 'd':
        return PyFloat_FromDouble(va_arg(*va, double));
    case 'c': {
        char value = (char)va_arg(*va, int);
        return PyBytes_FromStringAndSize(&value, 1);
    }
    case 'C': {
        char value = (char)va_arg(*va, int);
        return PyUnicode_FromStringAndSize(&value, 1);
    }
    case 's': case 'z': {
        const char *text = va_arg(*va, const char *);
        Py_ssize_t length = -1;
        if (**format == '#') {
            (*format)++;
            length = va_arg(*va, Py_ssize_t);
        }
        if (text == NULL) {
            Py_INCREF(Py_None);
            return Py_None;
        }
        return length < 0 ? PyUnicode_FromString(text)
                          : PyUnicode_FromStringAndSize(text, length);
    }
    case 'y': {
        const char *text = va_arg(*va, const char *);
        Py_ssize_t length = -1;
        if (**format == '#') {
            (*format)++;
            length = va_arg(*va, Py_ssize_t);
        }
        if (text == NULL) {
            Py_INCREF(Py_None);
            return Py_None;
        }
        return length < 0 ? PyBytes_FromString(text)
                          : PyBytes_FromStringAndSize(text, length);
    }
    case 'O': case 'S': case 'N': {
        PyObject *value = va_arg(*va, PyObject *);
        if (value == NULL) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_SystemError,
                                "Py_BuildValue: NULL object passed to O format");
            }
            return NULL;
        }
        if (code != 'N') {
            Py_INCREF(value);
        }
        return value;
    }
    case '(': case '[': {
        char closing = code == '(' ? ')' : ']';
        PyObject *items = PyList_New(0);
        if (items == NULL) {
            return NULL;
        }
        while (**format && **format != closing) {
            if (**format == ',' || **format == ' ') {
                (*format)++;
                continue;
            }
            PyObject *item = _PyPyre_BuildOne(format, va);
            if (item == NULL || PyList_Append(items, item) < 0) {
                Py_XDECREF(item);
                Py_DECREF(items);
                return NULL;
            }
            Py_DECREF(item);
        }
        if (**format == closing) {
            (*format)++;
        }
        if (closing == ']') {
            return items;
        }
        Py_ssize_t size = PyList_Size(items);
        PyObject *tuple = PyTuple_New(size);
        if (tuple == NULL) {
            Py_DECREF(items);
            return NULL;
        }
        for (Py_ssize_t index = 0; index < size; index++) {
            PyObject *item = PyList_GetItem(items, index);
            Py_INCREF(item);
            PyTuple_SetItem(tuple, index, item);
        }
        Py_DECREF(items);
        return tuple;
    }
    case '{': {
        PyObject *mapping = PyDict_New();
        if (mapping == NULL) {
            return NULL;
        }
        while (**format && **format != '}') {
            if (**format == ',' || **format == ' ' || **format == ':') {
                (*format)++;
                continue;
            }
            PyObject *key = _PyPyre_BuildOne(format, va);
            if (key == NULL) {
                Py_DECREF(mapping);
                return NULL;
            }
            while (**format == ',' || **format == ' ' || **format == ':') {
                (*format)++;
            }
            PyObject *value = _PyPyre_BuildOne(format, va);
            if (value == NULL || PyDict_SetItem(mapping, key, value) < 0) {
                Py_DECREF(key);
                Py_XDECREF(value);
                Py_DECREF(mapping);
                return NULL;
            }
            Py_DECREF(key);
            Py_DECREF(value);
        }
        if (**format == '}') {
            (*format)++;
        }
        return mapping;
    }
    default:
        PyErr_SetString(PyExc_SystemError, "Py_BuildValue: unsupported format");
        return NULL;
    }
}

static inline PyObject *_PyPyre_BuildValue(const char **format, va_list *va)
{
    while (**format == ' ' || **format == ',') {
        (*format)++;
    }
    if (**format == '\0') {
        Py_INCREF(Py_None);
        return Py_None;
    }
    const char *lookahead = *format;
    PyObject *first = _PyPyre_BuildOne(format, va);
    if (first == NULL) {
        return NULL;
    }
    while (**format == ' ' || **format == ',') {
        (*format)++;
    }
    if (**format == '\0') {
        return first;
    }
    (void)lookahead;
    /* More than one unit builds a tuple, exactly as `Py_BuildValue`
       documents. */
    PyObject *items = PyList_New(0);
    if (items == NULL || PyList_Append(items, first) < 0) {
        Py_DECREF(first);
        Py_XDECREF(items);
        return NULL;
    }
    Py_DECREF(first);
    while (**format) {
        if (**format == ' ' || **format == ',') {
            (*format)++;
            continue;
        }
        PyObject *item = _PyPyre_BuildOne(format, va);
        if (item == NULL || PyList_Append(items, item) < 0) {
            Py_XDECREF(item);
            Py_DECREF(items);
            return NULL;
        }
        Py_DECREF(item);
    }
    Py_ssize_t size = PyList_Size(items);
    PyObject *tuple = PyTuple_New(size);
    if (tuple == NULL) {
        Py_DECREF(items);
        return NULL;
    }
    for (Py_ssize_t index = 0; index < size; index++) {
        PyObject *item = PyList_GetItem(items, index);
        Py_INCREF(item);
        PyTuple_SetItem(tuple, index, item);
    }
    Py_DECREF(items);
    return tuple;
}

static inline PyObject *Py_BuildValue(const char *format, ...)
{
    va_list va;
    va_start(va, format);
    const char *cursor = format;
    PyObject *value = _PyPyre_BuildValue(&cursor, &va);
    va_end(va);
    return value;
}

/* The `ObjArgs` pair take a NULL-terminated argument list.  The list is walked
   twice -- once to count and once to fill a tuple -- rather than collected into
   a fixed buffer, so an arity no buffer anticipated still works. */
static inline PyObject *_PyPyre_ObjArgsTuple(va_list count_va, va_list fill_va)
{
    Py_ssize_t size = 0;
    while (va_arg(count_va, PyObject *) != NULL) {
        size++;
    }
    PyObject *tuple = PyTuple_New(size);
    if (tuple == NULL) {
        return NULL;
    }
    for (Py_ssize_t index = 0; index < size; index++) {
        PyObject *item = va_arg(fill_va, PyObject *);
        Py_XINCREF(item);
        PyTuple_SetItem(tuple, index, item);
    }
    return tuple;
}

static inline PyObject *PyObject_CallFunctionObjArgs(PyObject *callable, ...)
{
    va_list count_va, fill_va;
    va_start(count_va, callable);
    va_start(fill_va, callable);
    PyObject *args = _PyPyre_ObjArgsTuple(count_va, fill_va);
    va_end(count_va);
    va_end(fill_va);
    if (args == NULL) {
        return NULL;
    }
    PyObject *result = PyObject_Call(callable, args, NULL);
    Py_DECREF(args);
    return result;
}

static inline PyObject *PyObject_CallMethodObjArgs(PyObject *self, PyObject *name, ...)
{
    PyObject *method = PyObject_GetAttr(self, name);
    if (method == NULL) {
        return NULL;
    }
    va_list count_va, fill_va;
    va_start(count_va, name);
    va_start(fill_va, name);
    PyObject *args = _PyPyre_ObjArgsTuple(count_va, fill_va);
    va_end(count_va);
    va_end(fill_va);
    if (args == NULL) {
        Py_DECREF(method);
        return NULL;
    }
    PyObject *result = PyObject_Call(method, args, NULL);
    Py_DECREF(args);
    Py_DECREF(method);
    return result;
}

/* `PyObject_CallFunction` and `PyObject_CallMethod` take a `Py_BuildValue`
   format.  A format building exactly one value that is already a tuple is the
   argument list itself; anything else becomes a one-element list. */
static inline PyObject *_PyPyre_CallWithFormat(PyObject *callable, PyObject *built)
{
    if (built == NULL) {
        return NULL;
    }
    PyObject *args = built;
    if (!PyTuple_Check(built)) {
        args = PyTuple_New(1);
        if (args == NULL) {
            Py_DECREF(built);
            return NULL;
        }
        PyTuple_SetItem(args, 0, built);   /* steals `built` */
    }
    PyObject *result = PyObject_Call(callable, args, NULL);
    Py_DECREF(args);
    return result;
}

static inline PyObject *PyObject_CallFunction(PyObject *callable, const char *format, ...)
{
    if (format == NULL || *format == '\0') {
        return PyObject_CallNoArgs(callable);
    }
    va_list va;
    va_start(va, format);
    const char *cursor = format;
    PyObject *built = _PyPyre_BuildValue(&cursor, &va);
    va_end(va);
    return _PyPyre_CallWithFormat(callable, built);
}

static inline PyObject *PyObject_CallMethod(PyObject *self, const char *name,
                                            const char *format, ...)
{
    PyObject *method = PyObject_GetAttrString(self, name);
    if (method == NULL) {
        return NULL;
    }
    if (format == NULL || *format == '\0') {
        PyObject *result = PyObject_CallNoArgs(method);
        Py_DECREF(method);
        return result;
    }
    va_list va;
    va_start(va, format);
    const char *cursor = format;
    PyObject *built = _PyPyre_BuildValue(&cursor, &va);
    va_end(va);
    PyObject *result = _PyPyre_CallWithFormat(method, built);
    Py_DECREF(method);
    return result;
}

/* `%S`, `%R`, `%U` and `%A` take a `PyObject *`; everything else is handed to
   `snprintf` one conversion at a time. */
static inline PyObject *PyErr_Format(PyObject *type, const char *format, ...)
{
    char message[1024];
    size_t filled = 0;
    va_list va;
    va_start(va, format);
    for (const char *cursor = format; *cursor && filled + 1 < sizeof(message);) {
        if (*cursor != '%') {
            message[filled++] = *cursor++;
            continue;
        }
        const char *start = cursor++;
        while (*cursor && strchr("0123456789.-+ #lzhj", *cursor) != NULL) {
            cursor++;
        }
        char code = *cursor;
        if (code == '\0') {
            break;
        }
        cursor++;
        char spec[32];
        size_t spec_length = (size_t)(cursor - start);
        if (spec_length >= sizeof(spec)) {
            spec_length = sizeof(spec) - 1;
        }
        memcpy(spec, start, spec_length);
        spec[spec_length] = '\0';
        size_t room = sizeof(message) - filled;
        int written = 0;
        switch (code) {
        case '%':
            message[filled++] = '%';
            continue;
        case 'S': case 'R': case 'A': case 'U': case 'V': {
            PyObject *object = va_arg(va, PyObject *);
            PyObject *text = (code == 'R' || code == 'A') ? PyObject_Repr(object)
                                                          : PyObject_Str(object);
            const char *utf8 = text == NULL ? "<unprintable>" : PyUnicode_AsUTF8(text);
            written = snprintf(message + filled, room, "%s", utf8 ? utf8 : "<unprintable>");
            Py_XDECREF(text);
            break;
        }
        case 's':
            written = snprintf(message + filled, room, spec, va_arg(va, const char *));
            break;
        case 'p':
            written = snprintf(message + filled, room, spec, va_arg(va, void *));
            break;
        case 'f': case 'g': case 'e':
            written = snprintf(message + filled, room, spec, va_arg(va, double));
            break;
        case 'c':
            written = snprintf(message + filled, room, spec, va_arg(va, int));
            break;
        default:
            if (strstr(spec, "ll") != NULL) {
                written = snprintf(message + filled, room, spec, va_arg(va, long long));
            } else if (strchr(spec, 'l') != NULL || strchr(spec, 'z') != NULL) {
                written = snprintf(message + filled, room, spec, va_arg(va, long));
            } else {
                written = snprintf(message + filled, room, spec, va_arg(va, int));
            }
            break;
        }
        if (written < 0) {
            break;
        }
        filled += (size_t)written < room ? (size_t)written : room - 1;
    }
    va_end(va);
    message[filled < sizeof(message) ? filled : sizeof(message) - 1] = '\0';
    return _PyPyre_ErrFormatted(type, message);
}

#ifdef __cplusplus
}
#endif
#endif
