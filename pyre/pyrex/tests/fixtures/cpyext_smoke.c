#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

static void touch_marker(const char *env_name)
{
    const char *path = getenv(env_name);
    if (path != NULL) {
        FILE *marker = fopen(path, "wx");
        if (marker != NULL) {
            fclose(marker);
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((destructor))
static void cpyext_smoke_unloaded(void)
{
    touch_marker("PYRE_CPYEXT_UNLOAD_MARKER");
}
#endif

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cpyext_smoke",
    "pyre cpyext smoke module",
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC
PyInit_cpyext_smoke(void)
{
    const char *path = getenv("PYRE_CPYEXT_INIT_MARKER");
    FILE *marker = path == NULL ? NULL : fopen(path, "wx");
    if (path != NULL && marker == NULL) {
        return NULL;
    }
    if (marker != NULL) {
        fclose(marker);
    }
    return PyModule_Create(&moduledef);
}
