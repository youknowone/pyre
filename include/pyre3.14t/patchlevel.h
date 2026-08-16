/* The version this ABI is.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_PATCHLEVEL_H
#define PYRE_PATCHLEVEL_H

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

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_PATCHLEVEL_H */
