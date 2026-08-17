/* Thread state and the GIL.
 *
 * A thread holds the GIL for as long as it runs pyre code, so an extension
 * needs the same two spellings it uses on CPython: one to give the GIL up
 * around a blocking call, one to take it before calling in from a thread of
 * its own.
 *
 * `PyThreadState` is opaque here. An extension receives one from
 * `PyEval_SaveThread` and hands it back to `PyEval_RestoreThread`; everything
 * upstream keeps inside the struct is reached through an entry point instead.
 */
#ifndef PYRE_PYSTATE_H
#define PYRE_PYSTATE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _ts PyThreadState;

typedef enum { PyGILState_LOCKED, PyGILState_UNLOCKED } PyGILState_STATE;

/* Declared here rather than in the generated `pyre_decl.h` because they are
   not implemented in the `cpyext` layer: a build without it exports these two
   as well, for cffi's embedding header. */
PyAPI_FUNC(PyGILState_STATE) PyGILState_Ensure(void);
PyAPI_FUNC(void) PyGILState_Release(PyGILState_STATE);

/* The block an extension wraps a blocking call in. `_save` is the extension's
   own local, so the two halves have to appear in one scope, which is what the
   braces enforce. */
#define Py_BEGIN_ALLOW_THREADS                                                 \
    {                                                                          \
        PyThreadState *_save;                                                  \
        _save = PyEval_SaveThread();
#define Py_BLOCK_THREADS PyEval_RestoreThread(_save);
#define Py_UNBLOCK_THREADS _save = PyEval_SaveThread();
#define Py_END_ALLOW_THREADS                                                   \
    PyEval_RestoreThread(_save);                                               \
    }

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_PYSTATE_H */
