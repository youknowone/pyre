#!/usr/bin/env python3
"""The C declarations pyre's extension ABI publishes, and where they come from.

Three commands over one shared model:

    snapshot <cpython-Include-dir>   rewrite the recorded CPython declarations
    check                            every export against that record
    generate                         write `pyre_decl.h` from both

`check` is the gate. An extension is compiled against CPython's own headers on
every other implementation, so a parameter pyre declares differently is not a
style difference -- it is a calling-convention mismatch that no test in this
repo would otherwise see. The failure is silent and platform-dependent: a
`long` third argument passed where the callee reads a `Py_ssize_t` agrees on
LP64 and disagrees on Windows.

The record is a file rather than a path into a CPython checkout because CI has
no CPython source. Regenerating it is how the target version is bumped, and the
diff is then the ABI change that bump makes.
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CPYEXT = ROOT / "pyre/pyre-interpreter/src/cpyext"
HEADER_DIR = ROOT / "include/pyre3.14t"
RECORD = pathlib.Path(__file__).resolve().parent / "cpython-abi.txt"


# ── reading C ──────────────────────────────────────────────────────────────

def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def tidy(text):
    """One spelling for one type: `PyObject**` and `PyObject * *` both fold."""
    text = re.sub(r"\bregister\b", " ", text)
    text = re.sub(r"\s*\*\s*", " * ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Re-join the stars a declarator wrote apart, keeping `T *const *` readable.
    text = re.sub(r"\* (?=\*)", "*", text)
    return re.sub(r"\* const", "*const", text).strip()


def split_commas(text):
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
            continue
        depth += ch in "(["
        depth -= ch in ")]"
        cur += ch
    if cur.strip():
        out.append(cur)
    return out


NAMED = re.compile(r"^(.*?[\w\]\)\*])\s+([A-Za-z_]\w*)$")
KEYWORDS = {"void", "int", "char", "long", "short", "unsigned", "signed",
            "float", "double", "const"}


def param_type(text):
    """A parameter's type with its name dropped, since a declaration needs none."""
    text = tidy(text)
    if text in ("", "void"):
        return "void"
    if "(" in text:  # a function pointer spelled inline
        # Only the declarator -- `(*name)` -- holds a name to drop.  An
        # unanchored match would take the first identifier before any `)`,
        # which is a type in `void (*)(int, int)` and a trailing attribute
        # clause in `va_list) Py_GCC_ATTRIBUTE((format(printf, 1, 0))`.
        return re.sub(r"\(\s*(\*+)\s*[A-Za-z_]\w*\s*\)", r"(\1)", text, count=1)
    text = re.sub(r"\[\s*\]", " *", text)
    named = NAMED.match(text)
    if named and named.group(2) not in KEYWORDS:
        return tidy(named.group(1))
    return text


DECLARATION = re.compile(
    r"PyAPI_FUNC\s*\(\s*(?P<ret>[^;]*?)\s*\)\s*(?P<stars>\**)\s*"
    r"(?:_Py_NO_RETURN\s*)?(?P<name>[A-Za-z_]\w*)\s*\((?P<params>.*?)\)\s*;",
    re.S)


def read_declarations(paths):
    found = {}
    for path in paths:
        text = strip_comments(path.read_text(errors="replace"))
        for match in DECLARATION.finditer(text):
            params = [param_type(p) for p in split_commas(match.group("params"))]
            found.setdefault(match.group("name"),
                             (params or ["void"], tidy(match.group("ret") + match.group("stars"))))
    return found


# ── reading Rust ───────────────────────────────────────────────────────────

EXPORT = re.compile(
    r'#\[unsafe\(no_mangle\)\]\s*pub\s+(?:unsafe\s+)?extern\s+"C"\s+fn\s+'
    r"(?P<name>\w+)\s*\((?P<params>.*?)\)\s*(?:->\s*(?P<ret>[^{]+?))?\s*\{",
    re.S)

SCALARS = {
    "c_char": "char", "c_int": "int", "c_uint": "unsigned int",
    "c_long": "long", "c_ulong": "unsigned long",
    "c_longlong": "long long", "c_ulonglong": "unsigned long long",
    "c_double": "double", "c_float": "float",
    "isize": "Py_ssize_t", "usize": "size_t",
    "u8": "unsigned char", "i8": "signed char",
    "i32": "int32_t", "u32": "uint32_t", "i64": "int64_t", "u64": "uint64_t",
    # Named aliases the Rust side spells the same way the header does, so the
    # comparison is against the reference name rather than its width.
    "Py_UCS4": "Py_UCS4",
}

STRUCTS = {
    "CPyObject": "PyObject", "CPyVarObject": "PyVarObject",
    "CPyTypeObject": "PyTypeObject", "CPyBuffer": "Py_buffer",
    "CPyModuleDef": "PyModuleDef", "CPyTypeSpec": "PyType_Spec",
    "CPyMethodDef": "PyMethodDef", "CPyMemberDef": "PyMemberDef",
    "CPyGetSetDef": "PyGetSetDef", "c_void": "void",
}


class Unmapped(Exception):
    """A Rust type with no recorded C spelling, which must not be guessed."""


def rust_to_c(rust):
    """`*mut CPyObject` is `PyObject *`; `*const *mut T` is `T *const *`."""
    rust = " ".join(rust.split())
    if rust in ("", "()"):
        return "void"
    for prefix, qualified in (("*mut ", False), ("*const ", True)):
        if rust.startswith(prefix):
            inner = rust_to_c(rust[len(prefix):])
            if not qualified:
                return inner + ("*" if inner.endswith("*") else " *")
            return inner + "const *" if inner.endswith("*") else "const " + inner + " *"
    bare = rust.split("::")[-1]
    for table in (SCALARS, STRUCTS):
        if bare in table:
            return table[bare]
    raise Unmapped(rust)


def read_exports():
    """(module, name, [C param types], C return type) for every cpyext export."""
    for path in sorted(CPYEXT.glob("*.rs")):
        for match in EXPORT.finditer(path.read_text()):
            params = []
            for piece in split_commas(match.group("params")):
                if piece.strip():
                    params.append(rust_to_c(piece.split(":", 1)[1]))
            ret = match.group("ret")
            yield (path.stem, match.group("name"), params or ["void"],
                   rust_to_c(ret) if ret else "void")


# ── comparing the two ──────────────────────────────────────────────────────

FUNCTION_POINTER = re.compile(r"typedef\s+[^;]*?\(\s*\*\s*([A-Za-z_]\w*)\s*\)\s*\(", re.S)
ENUM = re.compile(r"typedef\s+enum\b[^;{]*\{.*?\}\s*([A-Za-z_]\w*)\s*;", re.S)
ALIAS = re.compile(r"typedef\s+([A-Za-z_][\w \t]*?)\s+([A-Za-z_]\w*)\s*;")


INLINE = re.compile(
    r"^static\s+inline\s+(?P<ret>[A-Za-z_][\w\s]*?)\s*(?P<stars>\**)\s*"
    r"(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^;{]*?)\)\s*\{",
    re.M | re.S)


def read_header_inlines():
    """The entry points implemented in the header rather than exported.

    `PyArg_ParseTuple`, `Py_BuildValue` and the other variadics are C in
    `Python.h`, because pyre ships no companion library and rustc's
    `c_variadic` is unstable. They are entry points an extension calls by the
    same name and the same convention as any export, so they are checked
    against the record the same way -- otherwise the gate reports "every export
    matches" while the half of the ABI it cannot see drifts.
    """
    for path in sorted(HEADER_DIR.glob("*.h")):
        text = strip_comments(path.read_text(errors="replace"))
        for match in INLINE.finditer(text):
            params = [param_type(p) for p in split_commas(match.group("params"))]
            ret = tidy(match.group("ret") + " " + match.group("stars"))
            yield path.name, match.group("name"), params or ["void"], ret


def read_typedefs(paths):
    """name -> what it stands for, so an alias is not read as a distinct type."""
    table = {}
    for path in paths:
        text = strip_comments(path.read_text(errors="replace"))
        for match in FUNCTION_POINTER.finditer(text):
            table[match.group(1)] = "void *"
        for match in ENUM.finditer(text):
            table[match.group(1)] = "int"
        for match in ALIAS.finditer(text):
            base = " ".join(match.group(1).split())
            if not base.startswith(("struct", "enum", "union")):
                table.setdefault(match.group(2), base)
    return table


# Every pointer occupies one slot and is passed the same way, so the pointee is
# not part of the calling convention. Integer widths are.
POINTER_SLOT = "a pointer"
WIDTH_ALIASES = {"ssize_t": "Py_ssize_t", "intptr_t": "Py_ssize_t"}


def abi_slot(c_type, typedefs):
    text = tidy(c_type)
    if text.endswith("*") or text.endswith("const"):
        return POINTER_SLOT
    text = text.replace("const ", "").strip()
    for _ in range(8):
        if text not in typedefs or typedefs[text] == text:
            break
        text = typedefs[text]
        if text.endswith("*"):
            return POINTER_SLOT
    return WIDTH_ALIASES.get(text, text)


def load_record():
    if not RECORD.exists():
        sys.exit(f"{RECORD} is missing; run `{sys.argv[0]} snapshot <Include dir>` first")
    declarations, typedefs, section = {}, {}, None
    for line in RECORD.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            if line.startswith("# ["):
                section = line[3:].rstrip("]\n")
            continue
        name, _, rest = line.partition(" :: ")
        if section == "typedefs":
            typedefs[name] = rest
            continue
        args, _, ret = rest.partition(" -> ")
        # The record was written with `split_commas`, so a parameter that
        # carries a comma of its own -- a function pointer's own list -- must
        # be read back with the same depth-aware split.
        declarations[name] = ([a.strip() for a in split_commas(args)], ret.strip())
    return declarations, typedefs


# ── commands ───────────────────────────────────────────────────────────────

def command_snapshot(args):
    include = pathlib.Path(args.include)
    headers = sorted(include.glob("*.h")) + sorted(include.glob("cpython/*.h"))
    if not headers:
        sys.exit(f"no headers under {include}")
    declarations = read_declarations(headers)
    typedefs = read_typedefs(sorted(include.rglob("*.h")))
    lines = [
        "# The declarations pyre's extension ABI is measured against.",
        "# Written by scripts/cpyext-abi.py snapshot; do not edit by hand.",
        f"# source: CPython {args.version} Include/*.h and Include/cpython/*.h",
        "",
        "# [declarations]",
    ]
    lines += [f"{n} :: {', '.join(a)} -> {r}" for n, (a, r) in sorted(declarations.items())]
    lines += ["", "# [typedefs]"]
    lines += [f"{n} :: {t}" for n, t in sorted(typedefs.items())]
    RECORD.write_text("\n".join(lines) + "\n")
    print(f"{len(declarations)} declarations, {len(typedefs)} typedefs -> {RECORD}")
    return 0


def command_check(args):
    declarations, typedefs = load_record()
    disagree, converted = [], []
    checked = {"export": 0, "header inline": 0}
    entry_points = [("export", f"cpyext/{m}.rs", n, p, r)
                    for m, n, p, r in read_exports()]
    entry_points += [("header inline", f"{HEADER_DIR.name}/{h}", n, p, r)
                     for h, n, p, r in read_header_inlines()]
    for kind, where_defined, name, params, ret in entry_points:
        if name not in declarations:
            converted.append((where_defined, name))
            continue
        theirs, their_ret = declarations[name]
        checked[kind] += 1
        ours = [abi_slot(p, typedefs) for p in params]
        wanted = [abi_slot(p, typedefs) for p in theirs]
        if ours != wanted:
            disagree.append((where_defined, name, params, theirs, "arguments"))
        elif abi_slot(ret, typedefs) != abi_slot(their_ret, typedefs):
            disagree.append((where_defined, name, [ret], [their_ret], "return"))
    print(f"{checked['export']} exports and {checked['header inline']} header "
          f"inlines checked against the recorded CPython ABI; "
          f"{len(converted)} have no CPython declaration")
    for where_defined, name, ours, theirs, where in disagree:
        print(f"\n{name} disagrees on its {where}  [{where_defined}]")
        print(f"    pyre    ({', '.join(ours)})")
        print(f"    cpython ({', '.join(theirs)})")
    if disagree:
        print(f"\n{len(disagree)} entry point(s) do not match the C declaration "
              f"an extension is compiled against.")
        return 1
    print("every entry point matches.")
    return 0


def command_generate(args):
    declarations, _ = load_record()
    by_module = {}
    for module, name, params, ret in read_exports():
        # CPython's own spelling wherever it has one: `check` has already
        # established the two describe the same call, and the reference
        # spelling is the one an extension's own prototypes agree with.
        if name in declarations:
            theirs, their_ret = declarations[name]
            params, ret = theirs, their_ret
        by_module.setdefault(module, []).append((name, params, ret))

    out = [
        "/* The exported entry points, one block per `cpyext` module.",
        " *",
        " * Written by scripts/cpyext-abi.py generate; do not edit by hand.",
        " * A declaration here is CPython's own where CPython has one, so an",
        " * extension's prototypes and pyre's agree by construction.",
        " */",
        "#ifndef PYRE_DECL_H",
        "#define PYRE_DECL_H",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
    ]
    for module in sorted(by_module):
        out.append(f"/* cpyext/{module}.rs */")
        for name, params, ret in sorted(by_module[module]):
            out.append(f"PyAPI_FUNC({ret}) {name}({', '.join(params)});")
        out.append("")
    out += ["#ifdef __cplusplus", "}", "#endif", "", "#endif /* !PYRE_DECL_H */"]

    text = "\n".join(out) + "\n"
    target = HEADER_DIR / "pyre_decl.h"
    if args.check:
        if not target.exists() or target.read_text() != text:
            print(f"{target} is not what `generate` produces; re-run without --check")
            return 1
        print(f"{target} is up to date")
        return 0
    target.write_text(text)
    print(f"{sum(len(v) for v in by_module.values())} declarations -> {target}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    snapshot = sub.add_parser("snapshot", help="rewrite the recorded CPython declarations")
    snapshot.add_argument("include", help="a CPython checkout's Include directory")
    snapshot.add_argument("--version", default="3.14", help="what to record as the source")
    snapshot.set_defaults(run=command_snapshot)

    check = sub.add_parser("check", help="every export against the record")
    check.set_defaults(run=command_check)

    generate = sub.add_parser("generate", help="write pyre_decl.h")
    generate.add_argument("--check", action="store_true",
                          help="fail if the checked-in file is not what this produces")
    generate.set_defaults(run=command_generate)

    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
