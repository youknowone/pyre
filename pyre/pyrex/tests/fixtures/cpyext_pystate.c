/* The GIL as an extension hands it back and forth: the block it wraps a
   blocking call in, and the pair it takes the GIL with on a thread of its own.

   `PyGILState_Check` is what makes these observable without timing anything --
   it answers False exactly where the GIL was given up.  `sleep_holding` is the
   same sleep with the release removed, so a concurrency measurement has the
   control that says the released one is why another thread ran. */

#include <Python.h>
#include <pthread.h>
#include <time.h>

static void nap(long ms)
{
    struct timespec span = {ms / 1000, (ms % 1000) * 1000000L};
    nanosleep(&span, NULL);
}

/* ── the block, and the same sleep without it ─────────────────────────── */
static PyObject *sleep_released(PyObject *self, PyObject *arg)
{
    (void)self;
    long ms = PyLong_AsLong(arg);
    if (ms == -1 && PyErr_Occurred()) return NULL;
    Py_BEGIN_ALLOW_THREADS
    nap(ms);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyObject *sleep_holding(PyObject *self, PyObject *arg)
{
    (void)self;
    long ms = PyLong_AsLong(arg);
    if (ms == -1 && PyErr_Occurred()) return NULL;
    nap(ms);
    Py_RETURN_NONE;
}

/* ── where the GIL is, at each depth ──────────────────────────────────── */
static PyObject *gil_held(PyObject *self, PyObject *args)
{ (void)self; (void)args; return PyBool_FromLong(PyGILState_Check()); }

/* Held outside, given up inside, and held again after.

   The block is not nested here: releasing what is already released detaches a
   thread state CPython then reads without checking, so a second one inside the
   first is undefined there and there would be no reference answer. */
static PyObject *gil_around_block(PyObject *self, PyObject *args)
{
    (void)self; (void)args;
    int before = PyGILState_Check();
    int inside;
    Py_BEGIN_ALLOW_THREADS
    inside = PyGILState_Check();
    Py_END_ALLOW_THREADS
    int after = PyGILState_Check();
    return Py_BuildValue("(OOO)", before ? Py_True : Py_False,
                         inside ? Py_True : Py_False, after ? Py_True : Py_False);
}

/* `Py_BLOCK_THREADS` takes the GIL back for a stretch in the middle. */
static PyObject *gil_around_block_threads(PyObject *self, PyObject *args)
{
    (void)self; (void)args;
    int inside, blocked, unblocked;
    Py_BEGIN_ALLOW_THREADS
    inside = PyGILState_Check();
    Py_BLOCK_THREADS
    blocked = PyGILState_Check();
    Py_UNBLOCK_THREADS
    unblocked = PyGILState_Check();
    Py_END_ALLOW_THREADS
    return Py_BuildValue("(OOO)", inside ? Py_True : Py_False,
                         blocked ? Py_True : Py_False,
                         unblocked ? Py_True : Py_False);
}

/* The explicit spelling the macros expand to. */
static PyObject *save_restore(PyObject *self, PyObject *args)
{
    (void)self; (void)args;
    PyThreadState *saved = PyEval_SaveThread();
    int released = !PyGILState_Check();
    int named = saved != NULL;
    PyEval_RestoreThread(saved);
    return Py_BuildValue("(OO)", named ? Py_True : Py_False,
                         released ? Py_True : Py_False);
}

/* Ensure/Release reports which state the thread was already in, and nests. */
static PyObject *ensure_states(PyObject *self, PyObject *args)
{
    (void)self; (void)args;
    PyGILState_STATE outer = PyGILState_Ensure();
    PyGILState_STATE inner = PyGILState_Ensure();
    int held = PyGILState_Check();
    PyGILState_Release(inner);
    PyGILState_Release(outer);
    /* Both are LOCKED: this thread already held the GIL both times. */
    return Py_BuildValue("(iiO)", (int)outer, (int)inner, held ? Py_True : Py_False);
}

static PyObject *thread_state_identity(PyObject *self, PyObject *args)
{
    (void)self; (void)args;
    PyThreadState *first = PyThreadState_Get();
    PyThreadState *again = PyThreadState_Get();
    PyThreadState *previous = PyThreadState_Swap(NULL);
    PyThreadState *none = PyThreadState_Swap(first);
    return Py_BuildValue("(OOO)",
                         (first != NULL && first == again) ? Py_True : Py_False,
                         (previous == first) ? Py_True : Py_False,
                         (none == NULL) ? Py_True : Py_False);
}

/* ── a callback delivered from a thread pyre never created ────────────── */
struct job { PyObject *callable; PyObject *result; int state_was; };

static void *worker(void *raw)
{
    struct job *job = raw;
    PyGILState_STATE state = PyGILState_Ensure();
    job->state_was = (int)state;
    job->result = PyObject_CallNoArgs(job->callable);
    if (job->result == NULL) PyErr_Clear();
    PyGILState_Release(state);
    return NULL;
}

static PyObject *call_from_foreign_thread(PyObject *self, PyObject *callable)
{
    (void)self;
    struct job job = {callable, NULL, -1};
    pthread_t thread;
    int started;
    /* The GIL has to be off this thread, or the worker could never take it. */
    Py_BEGIN_ALLOW_THREADS
    started = pthread_create(&thread, NULL, worker, &job) == 0;
    if (started) pthread_join(thread, NULL);
    Py_END_ALLOW_THREADS
    if (!started) return PyUnicode_FromString("thread-failed");
    if (job.result == NULL) return PyUnicode_FromString("callback-failed");
    /* The foreign thread did not hold the GIL, so its Ensure had to take it. */
    PyObject *pair = Py_BuildValue("(iO)", job.state_was, job.result);
    Py_DECREF(job.result);
    return pair;
}

static PyMethodDef methods[] = {
    {"sleep_released", sleep_released, METH_O, NULL},
    {"sleep_holding", sleep_holding, METH_O, NULL},
    {"gil_held", gil_held, METH_NOARGS, NULL},
    {"gil_around_block", gil_around_block, METH_NOARGS, NULL},
    {"gil_around_block_threads", gil_around_block_threads, METH_NOARGS, NULL},
    {"save_restore", save_restore, METH_NOARGS, NULL},
    {"ensure_states", ensure_states, METH_NOARGS, NULL},
    {"thread_state_identity", thread_state_identity, METH_NOARGS, NULL},
    {"call_from_foreign_thread", call_from_foreign_thread, METH_O, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef def = {
    PyModuleDef_HEAD_INIT, "cpyext_pystate", NULL, -1, methods};

PyMODINIT_FUNC PyInit_cpyext_pystate(void)
{
    return PyModule_Create(&def);
}
