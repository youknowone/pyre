/* PyMutex -- a mutex that occupies one byte, which the caller embeds and
   zero-initializes.

   The two low bits are the state; only the lowest is ever written here.  The
   fast paths below are inline so that an uncontended take is a compare-exchange
   and nothing else, exactly as the reference header spells it. */

#ifndef Py_LOCK_H
#define Py_LOCK_H
#ifdef __cplusplus
extern "C" {
#endif

#define _Py_UNLOCKED    0
#define _Py_LOCKED      1

typedef struct PyMutex {
    uint8_t _bits;  /* (private) */
} PyMutex;

PyAPI_FUNC(void) PyMutex_Lock(PyMutex *m);
PyAPI_FUNC(void) PyMutex_Unlock(PyMutex *m);
PyAPI_FUNC(int) PyMutex_IsLocked(PyMutex *m);

/* Takes the mutex, parking the calling thread until it is free.  A parked
   thread gives the GIL up, so whoever holds the mutex can run. */
static inline void
_PyMutex_Lock(PyMutex *m)
{
    uint8_t expected = _Py_UNLOCKED;
    if (!__atomic_compare_exchange_n(&m->_bits, &expected, _Py_LOCKED, 0,
                                     __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
        PyMutex_Lock(m);
    }
}
#define PyMutex_Lock _PyMutex_Lock

static inline void
_PyMutex_Unlock(PyMutex *m)
{
    uint8_t expected = _Py_LOCKED;
    if (!__atomic_compare_exchange_n(&m->_bits, &expected, _Py_UNLOCKED, 0,
                                     __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
        PyMutex_Unlock(m);
    }
}
#define PyMutex_Unlock _PyMutex_Unlock

static inline int
_PyMutex_IsLocked(PyMutex *m)
{
    return __atomic_load_n(&m->_bits, __ATOMIC_ACQUIRE) & _Py_LOCKED;
}
#define PyMutex_IsLocked _PyMutex_IsLocked

#ifdef __cplusplus
}
#endif
#endif /* !Py_LOCK_H */
