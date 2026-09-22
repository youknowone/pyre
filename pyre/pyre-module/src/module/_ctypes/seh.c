/* A foreign call can raise a structured exception — an access violation is
   the usual one — and a structured exception is not a Rust panic or a C++
   throw.  Nothing written in Rust reaches it; only a frame-based `__except`
   filter does, and only a compiler that speaks Microsoft SEH emits one.
   `_ctypes_callproc` wraps `ffi_call` in exactly this (`callproc.c:950-959`),
   and without it a bad address handed to a foreign function takes the process
   down where it should have raised OSError.

   The guarded call is written in Rust, so it is passed in rather than
   inlined here.  `__except` unwinds every frame above this one before the
   handler runs, so the caller's own frame — and the cell it lends through
   `arg` — is still standing when `pyre_seh_guard` returns 1. */

#include <windows.h>

typedef struct {
    unsigned long code;
    unsigned long long info[2];
    unsigned long ninfo;
} pyre_seh_record;

/* `HandleException` (`callproc.c:442-453`): record what was raised and take
   it, except for a breakpoint — that is how a debugger attaches to a running
   process, and the handler that wants it is further out. */
static int pyre_seh_filter(EXCEPTION_POINTERS *ptrs, pyre_seh_record *out)
{
    const EXCEPTION_RECORD *rec = ptrs->ExceptionRecord;
    out->code = rec->ExceptionCode;
    out->ninfo = (unsigned long)rec->NumberParameters;
    out->info[0] = rec->NumberParameters > 0
                 ? (unsigned long long)rec->ExceptionInformation[0] : 0;
    out->info[1] = rec->NumberParameters > 1
                 ? (unsigned long long)rec->ExceptionInformation[1] : 0;
    if (rec->ExceptionCode == EXCEPTION_BREAKPOINT)
        return EXCEPTION_CONTINUE_SEARCH;
    return EXCEPTION_EXECUTE_HANDLER;
}

int pyre_seh_guard(void (*body)(void *), void *arg, pyre_seh_record *out)
{
    out->code = 0;
    out->ninfo = 0;
    out->info[0] = 0;
    out->info[1] = 0;
    __try {
        body(arg);
        return 0;
    }
    __except (pyre_seh_filter(GetExceptionInformation(), out)) {
        return 1;
    }
}
