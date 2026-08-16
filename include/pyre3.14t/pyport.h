/* The compiler-facing spellings: linkage, and the integer types.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_PYPORT_H
#define PYRE_PYPORT_H

#ifdef __cplusplus
extern "C" {
#endif
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
/* C++ has no implicit conversion from `const char *[]` to `char **`, so the
 * keyword list a C++ extension can build is only accepted when the parameter
 * carries this qualifier.  C keeps it unqualified, where an extension may pass
 * a non-const array. */
#ifndef PY_CXX_CONST
#  ifdef __cplusplus
#    define PY_CXX_CONST const
#  else
#    define PY_CXX_CONST
#  endif
#endif

typedef intptr_t Py_ssize_t;
#define PY_SSIZE_T_MAX ((Py_ssize_t)(((size_t)-1) >> 1))
#define PY_SSIZE_T_MIN (-PY_SSIZE_T_MAX - 1)
typedef Py_ssize_t Py_hash_t;
typedef size_t Py_uhash_t;
typedef uint32_t Py_UCS4;
typedef uint16_t Py_UCS2;
typedef uint8_t Py_UCS1;

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_PYPORT_H */
