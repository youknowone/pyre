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
/* Values for PY_RELEASE_LEVEL */
#define PY_RELEASE_LEVEL_ALPHA 0xA
#define PY_RELEASE_LEVEL_BETA 0xB
#define PY_RELEASE_LEVEL_GAMMA 0xC /* For release candidates */
#define PY_RELEASE_LEVEL_FINAL 0xF /* Serial should be 0 here */

/* Version parsed out into numeric values */
#define PY_MAJOR_VERSION 3
#define PY_MINOR_VERSION 14
#define PY_MICRO_VERSION 6
#define PY_RELEASE_LEVEL PY_RELEASE_LEVEL_FINAL
#define PY_RELEASE_SERIAL 0

/* Version as a string */
#define PY_VERSION "3.14.6"

/* Pyre version as a string. Keep in sync with the `pyrex` package version,
   which is what `sys.pyre_version_info` reports; `cpyext_patchlevel.rs`
   compares the two. */
#define PYRE_VERSION "0.0.2"
#define PYRE_VERSION_NUM 0x00000200

/* Version as a single 4-byte hex number, e.g. 0x030E06F0 == 3.14.6 final.
   Use this for numeric comparisons, e.g. #if PY_VERSION_HEX >= ...
   The release-level nibble is 0xF for a final release, so a value ending in
   0x00 would put every `#if PY_VERSION_HEX >= 0x030E00F0` extension on its
   pre-release branch. */
#define PY_VERSION_HEX ((PY_MAJOR_VERSION << 24) | \
                        (PY_MINOR_VERSION << 16) | \
                        (PY_MICRO_VERSION <<  8) | \
                        (PY_RELEASE_LEVEL <<  4) | \
                        (PY_RELEASE_SERIAL << 0))

#define PYTHON_API_VERSION 1013
#define PYTHON_ABI_VERSION 3

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_PATCHLEVEL_H */
