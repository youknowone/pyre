#!/usr/bin/env python3
"""Report RPython/PyPy module-name parity gaps in the Rust port.

This is an audit helper for actionable module-name gaps.  It normalizes package
entry points (`__init__.py` in Python, `mod.rs`/`lib.rs` in Rust) so the report
focuses on real module names rather than language-specific filesystem
conventions.  Pyre-local Rust boundaries and permanently-unused PyPy layers
are reported separately as ignored entries, with reasons, so they do not drive
blind ports of code pyre will not use.

With `--symbols`, the helper also compares top-level Python class names with
top-level Rust public type names, and top-level Python function names with
top-level Rust public function names, for already-matched modules.  Thin Rust
reexport wrappers are classified separately so shared implementation crates
such as `majit_ir` and `majit_trace` do not turn into false positives.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModulePair:
    label: str
    python_dir: Path
    rust_dir: Path


@dataclass(frozen=True)
class StringSetPair:
    label: str
    python_path: Path
    python_symbol: str
    rust_path: Path
    rust_function: str


DEFAULT_PAIRS = [
    ModulePair(
        "rpython/annotator",
        Path("rpython/annotator"),
        Path("majit/majit-translate/src/annotator"),
    ),
    ModulePair(
        "rpython/config",
        Path("rpython/config"),
        Path("majit/majit-translate/src/config"),
    ),
    ModulePair(
        "rpython/flowspace",
        Path("rpython/flowspace"),
        Path("majit/majit-translate/src/flowspace"),
    ),
    ModulePair(
        "rpython/jit/codewriter",
        Path("rpython/jit/codewriter"),
        Path("majit/majit-translate/src/codewriter"),
    ),
    ModulePair(
        "rpython/jit/metainterp",
        Path("rpython/jit/metainterp"),
        Path("majit/majit-metainterp/src"),
    ),
    ModulePair(
        "rpython/jit/metainterp/ruleopt",
        Path("rpython/jit/metainterp/ruleopt"),
        Path("majit/majit-metainterp/src/ruleopt"),
    ),
    ModulePair(
        "rpython/jit/metainterp/optimizeopt",
        Path("rpython/jit/metainterp/optimizeopt"),
        Path("majit/majit-metainterp/src/optimizeopt"),
    ),
    ModulePair(
        "rpython/rtyper",
        Path("rpython/rtyper"),
        Path("majit/majit-translate/src/translator/rtyper"),
    ),
    ModulePair(
        "rpython/rtyper/lltypesystem",
        Path("rpython/rtyper/lltypesystem"),
        Path("majit/majit-translate/src/translator/rtyper/lltypesystem"),
    ),
    ModulePair(
        "rpython/rtyper/lltypesystem/module",
        Path("rpython/rtyper/lltypesystem/module"),
        Path("majit/majit-translate/src/translator/rtyper/lltypesystem/module"),
    ),
    ModulePair(
        "rpython/rtyper/tool",
        Path("rpython/rtyper/tool"),
        Path("majit/majit-translate/src/translator/rtyper/tool"),
    ),
    ModulePair(
        "rpython/tool/algo",
        Path("rpython/tool/algo"),
        Path("majit/majit-translate/src/tool/algo"),
    ),
    ModulePair(
        "rpython/translator",
        Path("rpython/translator"),
        Path("majit/majit-translate/src/translator"),
    ),
]

DEFAULT_STRING_SET_PAIRS = [
    StringSetPair(
        "codewriter USE_C_FORM",
        Path("rpython/jit/codewriter/assembler.py"),
        "USE_C_FORM",
        Path("majit/majit-translate/src/codewriter/assembler.rs"),
        "use_c_form",
    ),
    StringSetPair(
        "runtime USE_C_FORM",
        Path("rpython/jit/codewriter/assembler.py"),
        "USE_C_FORM",
        Path("pyre/pyre-jit/src/jit/assembler.rs"),
        "use_c_form",
    ),
]

DEFAULT_EXCLUDES = {"test", "__pycache__"}
PACKAGE_ENTRY = "mod"

INTENTIONAL_MISSING: dict[str, dict[str, str]] = {
    "rpython/rtyper/lltypesystem": {
        "ll2ctypes": "permanently unused: pyre never simulates lltype programs through ctypes",
        "llarena": "permanently unused: pyre does not port RPython moving-GC arena simulation",
    },
    "rpython/rtyper/tool": {
        "rffi_platform": "permanently unused: pyre uses Rust/Charon layouts instead of C probing",
    },
    "rpython/translator": {
        "c": "permanently unused: pyre must not grow a local translator/c backend tree",
        "exceptiontransform": "represented in Rust Result/? lowering, not a standalone module",
    },
}

INTENTIONAL_EXTRA: dict[str, dict[str, str]] = {
    "rpython/jit/codewriter": {
        "annotation_state": "local Rust boundary for temporary ValueType/SomeValue projection",
        "insns": "local stable byte table derived from assembler.py's dynamic insns table",
        "jtransform_opname": "local transducer for rtyped helper graphs into jtransform shape",
        "jtransform_shadow": "env-gated diagnostic, never production path",
        "transform_profile": "env-gated drain profiler with no upstream runtime effect",
        "type_state": "local concretetype projection boundary during rtyper cutover",
    },
    "rpython/jit/metainterp": {
        "box_trace": "boxed primitive trace helper shared by pyre-jit and pyre-jit-trace",
        "call_descr": "runtime call-descr boundary for codewriter/backend descriptor surfaces",
        "io_buffer": "compiled-loop stdout buffer; RPython interpreter writes directly",
        "jit": "runtime half of rpython/rlib/jit.py; translator half lives under rlib",
        "jit_state": "Rust trait abstraction for interpreter state",
        "jitcode": "runtime ABI boundary around canonical translate-side jitcode.py port",
        "parity": "test-only trace comparison utilities",
        "recorder": "runtime Trace boundary around opencoder/history recording roles",
        "trace_ctx": "Rust tracing context split across history/compile roles",
    },
    "rpython/rtyper": {
        "cutover": "transitional bridge between legacy and orthodox graph paths",
        "flowspace_adapter": "transitional bridge from pyre graph model to flowspace graph model",
        "legacy_annotator": "temporary legacy graph adapter for cutover",
        "legacy_resolve": "temporary legacy call resolution adapter for cutover",
        "pairtype": "Rust carrier for rtyper-side __extend__(pairtype(...)) blocks",
        "pyre_call_registry": "symbolic FunctionPath registration in place of CPython callable identity",
        "unit_variant_fold": "Rust unit-variant PBC pre-folding before jtransform",
    },
    "rpython/translator": {
        "backend": "intentional non-c module for minimal CBuilder-shaped driver shells",
        "rtyper": "crate-local nesting; upstream rtyper remains compared separately",
        "targetspec": "typed carrier for driver.py from_targetspec's open Python dict",
    },
}

INTENTIONAL_SYMBOL_EXTRA: dict[tuple[str, str], dict[str, dict[str, str]]] = {
    ("rpython/config", "config"): {
        "types": {
            "Child": "Rust enum for OptionDescription._children entries",
            "ConfigValue": "Rust carrier for dynamic __getattr__ return values",
            "DependencyEdge": "Rust carrier for upstream requires/suggests tuple pairs",
            "OptionValue": "Rust carrier for upstream Any-typed option values",
            "Owner": "Rust enum for upstream value-owner strings",
        },
    },
    ("rpython/config", "support"): {
        "functions": {
            "detect_number_of_processors_with_path": "test fixture injection for upstream's filename_or_file parameter",
            "detect_pax_with_path": "test fixture injection for upstream's /proc/self/status read",
        },
    },
    ("rpython/annotator", "description"): {
        "types": {
            "CallTableRow": "Rust alias for upstream's Desc-identity-keyed calltable row dict",
            "DescEntry": "Rust discriminated carrier for upstream Desc subclass instances",
            "DescKey": "Rust identity handle for upstream's Desc object keys",
            "FuncDescEntry": "Rust carrier preserving FunctionDesc/MemoDesc identity under one Desc entry",
            "SpecializeResult": "Rust typed carrier for upstream specializers returning graph-or-annotation",
        },
    },
    ("rpython/annotator", "bookkeeper"): {
        "types": {
            "PositionKey": "Rust identity carrier for upstream's position_key tuple",
        },
    },
    ("rpython/annotator", "builtin"): {
        "types": {
            "BuiltinAnalyzer": "Rust function-pointer carrier for upstream analyzer_for registry entries",
        },
    },
    ("rpython/annotator", "model"): {
        "types": {
            "AnnotatorException": "Rust enum carrier for upstream AnnotatorError/UnionError/HarmlesslyBlocked exception variants",
            "DescKind": "Rust enum for upstream Desc subclass identity returned by SomePBC.getKind()",
            "ExitCaseKey": "Rust map key for upstream knowntypedata exit-case tuples",
            "KnownType": "Rust enum carrier for upstream live Python type objects stored in knowntype",
            "KnownTypeData": "Rust alias for upstream knowntypedata nested dict shape",
            "SandboxingPayload": "Rust typed payload carried by SomeBuiltin in place of a dynamic analyser attribute",
            "SomeObjectTrait": "Rust trait surface for methods inherited from upstream SomeObject through Python MRO",
            "SomeValue": "Rust closed enum for the upstream SomeObject subclass lattice",
            "SomeValueTag": "Rust discriminant helper for the upstream SomeObject subclass lattice",
        },
    },
    ("rpython/annotator", "policy"): {
        "types": {
            "PolicyError": "Rust error carrier for upstream get_specializer AttributeError/Exception paths",
            "PolicyHandle": "Rust trait-object handle for upstream policy instances and subclass dispatch",
            "PolicyOps": "Rust trait carrier for upstream policy instance methods overridden by subclasses",
            "Specializer": "Rust enum carrier for upstream specialize.py function objects returned by get_specializer",
        },
    },
    ("rpython/annotator", "signature"): {
        "types": {
            "AnnotationSpec": "Rust enum carrier for upstream annotation(t)'s polymorphic Python input value",
            "ParamType": "Rust enum carrier for upstream enforce_signature_args paramtype shapes",
            "SigArgType": "Rust enum carrier for upstream Sig.argtypes dynamic callable/None/Void/NOT_CONSTANT/type cases",
            "TypeMarker": "Rust enum carrier for upstream rlib.types SelfTypeMarker/AnyTypeMarker classes",
        },
    },
    ("rpython/flowspace", "argument"): {
        "types": {
            "CallShape": "Rust carrier for upstream CallSpec._rawshape()'s anonymous (shape_cnt, shape_keys, shape_star) tuple",
        },
    },
    ("rpython/flowspace", "framestate"): {
        "types": {
            "MergeCell": "Rust carrier for upstream FrameState.mergeable cells, which may be Variable, Constant, or None",
            "StackElem": "Rust carrier for upstream FrameState.stack cells, which may be Variable, Constant, or FlowSignal",
        },
    },
    ("rpython/flowspace", "flowcontext"): {
        "types": {
            "FlowContextError": "Rust error carrier for upstream FlowingError/StopFlowing/FlowSignal/BytecodeCorruption exception unwinds",
            "FlowSignalTag": "Rust discriminant for upstream FlowSignal subclass identity used by rebuild_with_args",
            "FrameBlockKind": "Rust discriminant for upstream FrameBlock subclass identity stored on FrameBlock",
            "PendingBlock": "Rust carrier for upstream pendingblocks list containing SpamBlock or EggBlock instances",
        },
    },
    ("rpython/flowspace", "operation"): {
        "types": {
            "OpKind": "Rust enum replacing upstream's op namespace plus HLOperationMeta-generated per-op classes",
        },
    },
    ("rpython/annotator", "specialize"): {
        "types": {
            "MemoFamily": "Rust carrier for upstream Bookkeeper.all_specializations UnionFind plus host-call error latch",
        },
    },
    ("rpython/rtyper/lltypesystem", "lltype"): {
        "types": {
            "ParentIndex": "Rust enum carrier for upstream's field-name-or-item-index parent tuple element",
        },
    },
    ("rpython/rtyper/lltypesystem", "llheap"): {
        "functions": {
            "_is_pinned": "Rust public testable surface for upstream's private `_is_pinned` helper",
            "free": "Rust function surface for upstream's `from lltype import free` alias",
            "setfield": "Rust function surface for upstream's `setfield = setattr` alias",
        },
    },
    ("rpython/rtyper/lltypesystem", "rbuilder"): {
        "functions": {
            "_ll_append": "Rust public placeholder for upstream's private `_ll_append` helper",
            "_ll_append_multiple_char": "Rust public placeholder for upstream's private `_ll_append_multiple_char` helper",
            "stringbuilder_repr": "Rust accessor for upstream's `stringbuilder_repr = StringBuilderRepr()` singleton",
            "unicodebuilder_repr": "Rust accessor for upstream's `unicodebuilder_repr = UnicodeBuilderRepr()` singleton",
        },
    },
    ("rpython/rtyper/lltypesystem", "rbytearray"): {
        "functions": {
            "_empty_bytearray": "Rust public surface for upstream's private `_empty_bytearray` helper",
            "bytearray_repr": "Rust accessor for upstream's `bytearray_repr = ByteArrayRepr()` singleton",
            "empty": "Rust accessor for upstream's `empty = lltype.malloc(BYTEARRAY, 0, immortal=True)` singleton",
        },
    },
    ("rpython/translator", "simplify"): {
        "types": {
            "GraphKeyForSeen": "Rust graph-identity carrier for upstream's seen dict keys",
        },
    },
    ("rpython/tool/algo", "graphlib"): {
        "types": {
            "DfsEvent": "Rust enum carrier for upstream depth_first_search ('start'/'stop') event strings",
            "EdgeDict": "Rust alias for upstream's official edges dict shape",
            "VertexSet": "Rust trait carrier for upstream's set-or-dict vertices protocol",
        },
    },
    ("rpython/tool/algo", "sparsemat"): {
        "types": {
            "SparseMatError": "Rust error carrier for upstream sparse matrix ValueError/IndexError paths",
        },
    },
    ("rpython/tool/algo", "unionfind"): {
        "types": {
            "UnionFindInfo": "Rust trait carrier for upstream dynamic info.absorb(other_info) root payloads",
        },
    },
}

INTENTIONAL_SYMBOL_MISSING: dict[tuple[str, str], dict[str, dict[str, str]]] = {
    ("rpython/annotator", "argument"): {
        "types": {
            "ArgErrCount": "represented by ArgErr::Count enum variant",
            "ArgErrMultipleValues": "represented by ArgErr::MultipleValues enum variant",
            "ArgErrUnknownKwds": "represented by ArgErr::UnknownKwds enum variant",
        },
    },
    ("rpython/annotator", "classdesc"): {
        "types": {
            "Sample": "CPython member-descriptor probe for MemberDescriptorTypes; pyre uses typed HostObject/classdict entries instead",
        },
    },
    ("rpython/config", "config"): {
        "types": {
            "BoolConfigUpdate": "deferred with optparse integration until CLI driver code lands",
            "ConfigUpdate": "deferred with optparse integration until CLI driver code lands",
            "ConflictConfigError": "represented by ConfigError::Conflict instead of a separate Rust exception type",
            "OptHelpFormatter": "deferred with optparse integration until CLI driver code lands",
        },
        "functions": {
            "make_dict": "deferred with optparse/config dump integration until a consumer lands",
            "to_optparse": "deferred with optparse integration until CLI driver code lands",
        },
    },
    ("rpython/config", "translationoption"): {
        "functions": {
            "get_platform": "deferred with translator.platform pick_platform until platform compile integration is ported",
            "set_platform": "deferred with translator.platform set_platform until platform compile integration is ported",
        },
    },
    ("rpython/annotator", "specialize"): {
        "types": {
            "AccessDirect": "represented by GraphCacheKey::AccessDirect instead of a standalone marker class",
        },
    },
    ("rpython/flowspace", "flowcontext"): {
        "types": {
            "Break": "represented by FlowSignal::Break rather than a standalone subclass",
            "Continue": "represented by FlowSignal::Continue rather than a standalone subclass",
            "ExceptBlock": "represented by FrameBlockKind::Except on FrameBlock rather than a standalone subclass",
            "FinallyBlock": "represented by FrameBlockKind::Finally on FrameBlock rather than a standalone subclass",
            "IterBlock": "represented by FrameBlockKind::Iter on FrameBlock rather than a standalone subclass",
            "LoopBlock": "represented by FrameBlockKind::Loop on FrameBlock rather than a standalone subclass",
            "Raise": "represented by FlowSignal::Raise rather than a standalone subclass",
            "RaiseImplicit": "represented by FlowSignal::RaiseImplicit rather than a standalone subclass",
            "Return": "represented by FlowSignal::Return rather than a standalone subclass",
            "WithBlock": "represented by FrameBlockKind::With on FrameBlock rather than a standalone subclass",
        },
        "functions": {
            "binaryoperation": "Python opcode-method factory is represented inline by FlowContext::handle_bytecode dispatch",
            "unaryoperation": "Python opcode-method factory is represented inline by FlowContext::handle_bytecode dispatch",
            "unsupportedoperation": "Python opcode-method factory is represented inline by FlowContext::handle_bytecode dispatch",
        },
    },
    ("rpython/flowspace", "model"): {
        "types": {
            "ConstException": "represented by FSException carrying Constant-wrapped type/value rather than Python multiple inheritance",
            "UnwrapException": "Rust typed Hlvalue APIs do not unwrap Variables through Python exceptions",
            "WrapException": "Rust constant wrapping returns typed Result/Option fallbacks at the call sites instead of raising a marker exception",
        },
        "functions": {
            "flattenobj": "Python dynamic recursive tuple/list flattener is unnecessary because Rust graph walkers traverse typed fields directly",
        },
    },
    ("rpython/flowspace", "operation"): {
        "types": {
            "CallArgs": "represented by OpKind::CallArgs and HLOperation dispatch rather than a standalone Rust subclass",
            "CallOp": "represented by OpKind call-family canraise logic instead of a standalone base class",
            "Contains": "represented by OpKind::Contains and unaryop contains registrations rather than a standalone Rust subclass",
            "DoubleDispatchMixin": "represented by OpKind::dispatch plus the crate-local _REGISTRY_DOUBLE table",
            "GetAttr": "represented by OpKind::GetAttr plus HLOperation::constfold_getattr",
            "HLOperationMeta": "represented by the static OpKind enum and registry tables instead of a runtime metaclass",
            "Iter": "represented by OpKind::Iter and HLOperation constfold/eval dispatch",
            "NewDict": "represented by OpKind::NewDict and HLOperation::consider",
            "NewList": "represented by OpKind::NewList and HLOperation::consider",
            "NewSlice": "represented by OpKind::NewSlice and HLOperation::consider",
            "NewTuple": "represented by OpKind::NewTuple, pyfunc tuple folding, and HLOperation::consider",
            "Next": "represented by OpKind::Next and flowcontext next handling",
            "OverflowingOperation": "represented by OpKind::can_overflow and OpKind::ovf_variant",
            "Pow": "represented by OpKind::Pow and pyfunc pow folding",
            "PureOperation": "represented by OpKind::pure and HLOperation::constfold",
            "PureOperation1": "represented by OpKind::arity plus HLOperation::constfold",
            "PureOperation2": "represented by OpKind::arity plus HLOperation::constfold",
            "SimpleCall": "represented by OpKind::SimpleCall and specialcase lookup in flowcontext",
            "SingleDispatchMixin": "represented by OpKind::dispatch plus the crate-local _REGISTRY_SINGLE table",
        },
        "functions": {
            "add_operator": "upstream dynamic class factory is represented by the static OpKind table",
            "delete": "represented by OpKind::Delete pyfunc/canraise metadata",
            "do_delslice": "represented by OpKind::DelSlice pyfunc/canraise metadata",
            "do_float": "represented by OpKind::Float pyfunc folding",
            "do_getslice": "represented by OpKind::GetSlice pyfunc/canraise metadata",
            "do_index": "represented by OpKind::Index pyfunc folding",
            "do_int": "represented by OpKind::Int pyfunc folding",
            "do_long": "represented by OpKind::Long pyfunc folding",
            "do_setslice": "represented by OpKind::SetSlice pyfunc/canraise metadata",
            "get": "represented by OpKind::Get pyfunc/canraise metadata",
            "inplace_add": "represented by OpKind::InplaceAdd metadata",
            "inplace_and": "represented by OpKind::InplaceAnd metadata",
            "inplace_div": "represented by OpKind::InplaceDiv metadata",
            "inplace_floordiv": "represented by OpKind::InplaceFloorDiv metadata",
            "inplace_lshift": "represented by OpKind::InplaceLShift metadata",
            "inplace_mod": "represented by OpKind::InplaceMod metadata",
            "inplace_mul": "represented by OpKind::InplaceMul metadata",
            "inplace_or": "represented by OpKind::InplaceOr metadata",
            "inplace_pow": "represented by OpKind::InplacePow metadata",
            "inplace_rshift": "represented by OpKind::InplaceRShift metadata",
            "inplace_sub": "represented by OpKind::InplaceSub metadata",
            "inplace_truediv": "represented by OpKind::InplaceTrueDiv metadata",
            "inplace_xor": "represented by OpKind::InplaceXor metadata",
            "new_style_type": "represented by OpKind::Type pyfunc folding",
            "next": "represented by OpKind::Next and flowcontext next handling",
            "set": "represented by OpKind::Set pyfunc/canraise metadata",
            "unsupported": "represented by OpKind::Format/Trunc/Buffer fallback metadata",
            "userdel": "represented by OpKind::UserDel metadata",
        },
    },
    ("rpython/flowspace", "specialcase"): {
        "types": {
            "StdOutBuffer": "Pyre records rpython_print_* as HostObject call targets and does not execute the print buffer in flowspace",
        },
        "functions": {
            "redirect_function": "Python import-time registry mutator is represented by the static SPECIAL_CASES LazyLock table",
            "register_flow_sc": "Python decorator registry mutator is represented by the static SPECIAL_CASES LazyLock table",
            "rpython_print_end": "represented as a HOST_ENV builtin call target emitted by FlowContext::handle_print_function",
            "rpython_print_item": "represented as a HOST_ENV builtin call target emitted by FlowContext::handle_print_function",
            "rpython_print_newline": "represented as a HOST_ENV builtin call target emitted by FlowContext::handle_print_function",
        },
    },
    ("rpython/jit/codewriter", "format"): {
        "functions": {
            "unformat_assembler": "reverse text-to-SSA test parser is deferred until Rust SSARepr stores parseable FlatOp operands instead of typed pipeline-only SpaceOperations",
        },
    },
    ("rpython/tool/algo", "graphlib"): {
        "functions": {
            "break_cycles": "upstream immediately skips this obsolete edge-cutting helper; pyre keeps break_cycles_v only",
            "show_graph": "GraphPage/DotGen GUI debug helper with no pyre translation-time consumer",
        },
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def python_modules(path: Path, excludes: set[str]) -> set[str]:
    modules = set()
    for child in path.iterdir():
        if child.name in excludes:
            continue
        if child.is_file() and child.suffix == ".py":
            modules.add("mod" if child.stem == "__init__" else child.stem)
        elif child.is_dir() and (child / "__init__.py").is_file():
            modules.add(child.name)
    return modules


def rust_file_module_name(child: Path) -> str:
    if child.stem == "lib":
        return "mod"
    return child.stem


def rust_modules(path: Path, excludes: set[str]) -> set[str]:
    modules = set()
    for child in path.iterdir():
        if child.name in excludes:
            continue
        if child.is_file() and child.suffix == ".rs":
            modules.add(rust_file_module_name(child))
        elif child.is_dir() and (child / "mod.rs").is_file():
            modules.add(child.name)
    return modules


def python_module_path(path: Path, module: str) -> Path:
    if module == PACKAGE_ENTRY:
        return path / "__init__.py"
    file_path = path / f"{module}.py"
    if file_path.is_file():
        return file_path
    return path / module / "__init__.py"


def rust_module_path(path: Path, module: str) -> Path:
    if module == PACKAGE_ENTRY:
        lib_path = path / "lib.rs"
        if lib_path.is_file():
            return lib_path
        return path / "mod.rs"
    file_path = path / f"{module}.rs"
    if file_path.is_file():
        return file_path
    return path / module / "mod.rs"


PYTHON_TOP_LEVEL_SYMBOL = re.compile(r"^(class|def)\s+([A-Za-z_][A-Za-z0-9_]*)\b")
PYTHON_BLOCK_START = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\b")
PYTHON_MODULE_CONTROL_BLOCKS = {
    "else",
    "elif",
    "except",
    "finally",
    "for",
    "if",
    "try",
    "while",
    "with",
}


def python_top_level_symbols(path: Path) -> dict[str, set[str]]:
    symbols = {"types": set(), "functions": set()}
    block_stack: list[tuple[int, str]] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()

        while block_stack and indent <= block_stack[-1][0]:
            block_stack.pop()

        in_class_or_def = any(kind in {"class", "def"} for _, kind in block_stack)
        symbol_match = PYTHON_TOP_LEVEL_SYMBOL.match(stripped)
        if symbol_match:
            kind, name = symbol_match.groups()
            if not in_class_or_def and not name.startswith("_"):
                if kind == "class":
                    symbols["types"].add(name)
                else:
                    symbols["functions"].add(name)
            # A `def`/`class` block can span multiple physical lines, e.g.
            # `def to_optparse(...,\n                extra_usage=None):`.
            # Treat it as a block immediately so nested helpers in the body do
            # not get misclassified as module-level symbols.
            block_stack.append((indent, kind))
            continue

        block_match = PYTHON_BLOCK_START.match(stripped)
        if (
            block_match
            and stripped.endswith(":")
            and block_match.group(1) in PYTHON_MODULE_CONTROL_BLOCKS
        ):
            block_stack.append((indent, "control"))
    return symbols


RUST_PUB_ITEM = re.compile(
    r"^pub\s+(?:unsafe\s+)?(?:extern\s+(?:\"[^\"]+\"\s+)?)?"
    r"(struct|enum|trait|type|fn)\s+(?:r#)?([A-Za-z_][A-Za-z0-9_]*)\b"
)
RUST_TOP_LEVEL_ITEM = re.compile(
    r"^(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?(?:extern\s+(?:\"[^\"]+\"\s+)?)?"
    r"(struct|enum|trait|type|fn)\s+(?:r#)?([A-Za-z_][A-Za-z0-9_]*)\b"
)
RUST_PUB_REEXPORT = re.compile(r"^pub\s+use\s+")
RUST_ITEM_START = re.compile(
    r"^(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?(?:extern\s+(?:\"[^\"]+\"\s+)?)?"
    r"(struct|enum|trait|type|fn|const|static|impl|mod)\b"
)


def _strip_rust_line(line: str) -> str:
    line = line.strip()
    if line.startswith("//"):
        return ""
    if line.startswith("#["):
        return ""
    return line


def _split_top_level_commas(text: str) -> list[str]:
    parts = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_rust_reexport_names(statement: str) -> set[str]:
    statement = statement.strip().removesuffix(";").strip()
    if not statement.startswith("pub use "):
        return set()
    path = statement[len("pub use ") :].strip()
    if "*" in path:
        return set()
    if "{" not in path:
        leaf = path.rsplit("::", 1)[-1].strip()
        if " as " in leaf:
            leaf = leaf.rsplit(" as ", 1)[-1].strip()
        return {leaf} if leaf and leaf not in {"crate", "self", "super"} else set()

    start = path.find("{")
    end = path.rfind("}")
    if end < start:
        return set()
    names = set()
    for item in _split_top_level_commas(path[start + 1 : end]):
        if not item:
            continue
        if "{" in item:
            names.update(_extract_rust_reexport_names(f"pub use {item};"))
            continue
        if " as " in item:
            item = item.rsplit(" as ", 1)[-1].strip()
        elif "::" in item:
            item = item.rsplit("::", 1)[-1].strip()
        if item and item not in {"crate", "self", "super"}:
            names.add(item)
    return names


def rust_top_level_symbols(
    path: Path,
) -> tuple[dict[str, set[str]], dict[str, set[str]], set[str], bool]:
    symbols = {"types": set(), "functions": set()}
    nonpub_symbols = {"types": set(), "functions": set()}
    reexports: set[str] = set()
    has_pub_reexport = False
    has_direct_item = False
    depth = 0
    in_block_comment = False
    reexport_lines: list[str] | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line
        if in_block_comment:
            if "*/" in line:
                line = line.split("*/", 1)[1]
                in_block_comment = False
            else:
                continue
        while "/*" in line:
            before, after = line.split("/*", 1)
            if "*/" in after:
                after = after.split("*/", 1)[1]
                line = before + after
            else:
                line = before
                in_block_comment = True
                break

        candidate = _strip_rust_line(line)
        if reexport_lines is not None:
            if candidate:
                reexport_lines.append(candidate)
            if ";" in candidate:
                statement = " ".join(reexport_lines)
                reexports.update(_extract_rust_reexport_names(statement))
                reexport_lines = None
            continue

        if depth == 0 and candidate:
            pub_match = RUST_PUB_ITEM.match(candidate)
            if pub_match:
                kind = pub_match.group(1)
                bucket = "functions" if kind == "fn" else "types"
                symbols[bucket].add(pub_match.group(2))
                has_direct_item = True
            elif item_match := RUST_TOP_LEVEL_ITEM.match(candidate):
                kind = item_match.group(1)
                bucket = "functions" if kind == "fn" else "types"
                nonpub_symbols[bucket].add(item_match.group(2))
                has_direct_item = True
            elif RUST_PUB_REEXPORT.match(candidate):
                has_pub_reexport = True
                if ";" in candidate:
                    reexports.update(_extract_rust_reexport_names(candidate))
                else:
                    reexport_lines = [candidate]
                continue
            elif RUST_ITEM_START.match(candidate):
                if not re.match(r"mod\s+tests\b", candidate):
                    has_direct_item = True

        depth += line.count("{") - line.count("}")
        if depth < 0:
            depth = 0

    return symbols, nonpub_symbols, reexports, has_pub_reexport and not has_direct_item and not reexports


def _strings_from_ast_collection(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "set":
        if len(node.args) != 1 or node.keywords:
            raise ValueError("expected set([...]) with one positional argument")
        node = node.args[0]
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        raise ValueError("expected a list, tuple, or set literal")

    values = set()
    for item in node.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            raise ValueError("expected string-only collection literal")
        values.add(item.value)
    return values


def python_string_set(path: Path, symbol: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol:
                    return _strings_from_ast_collection(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == symbol
            and node.value is not None
        ):
            return _strings_from_ast_collection(node.value)
    raise ValueError(f"{symbol} not found in {path}")


def _rust_function_body(text: str, function: str) -> str:
    match = re.search(rf"\bfn\s+{re.escape(function)}\s*\([^)]*\)\s*->\s*bool\s*\{{", text)
    if not match:
        raise ValueError(f"{function} function not found")
    start = match.end() - 1
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    raise ValueError(f"{function} body is not closed")


def rust_string_literals_in_bool_function(path: Path, function: str) -> set[str]:
    body = _rust_function_body(path.read_text(encoding="utf-8"), function)
    strings = set()
    for match in re.finditer(r'"(?:\\.|[^"\\])*"', body):
        value = ast.literal_eval(match.group(0))
        if isinstance(value, str):
            strings.add(value)
    return strings


def compare_string_sets(root: Path, pairs: list[StringSetPair]) -> list[dict[str, object]]:
    results = []
    for pair in pairs:
        py_path = root / pair.python_path
        rs_path = root / pair.rust_path
        py_values = python_string_set(py_path, pair.python_symbol)
        rs_values = rust_string_literals_in_bool_function(rs_path, pair.rust_function)
        results.append(
            {
                "label": pair.label,
                "python_path": pair.python_path.as_posix(),
                "python_symbol": pair.python_symbol,
                "rust_path": pair.rust_path.as_posix(),
                "rust_function": pair.rust_function,
                "matched": sorted(py_values & rs_values),
                "missing_in_rust": sorted(py_values - rs_values),
                "extra_in_rust": sorted(rs_values - py_values),
            }
        )
    return results


def compare_symbols_for_pair(
    root: Path, pair: ModulePair, matched: list[str]
) -> list[dict[str, object]]:
    python_dir = root / pair.python_dir
    rust_dir = root / pair.rust_dir
    results = []

    for module in matched:
        if module == PACKAGE_ENTRY:
            continue
        py_path = python_module_path(python_dir, module)
        rs_path = rust_module_path(rust_dir, module)
        if not py_path.is_file() or not rs_path.is_file():
            continue

        py_symbols = python_top_level_symbols(py_path)
        rs_symbols, rs_nonpub_symbols, rs_reexports, is_reexport = rust_top_level_symbols(rs_path)
        rs_type_names = rs_symbols["types"] | rs_reexports
        rs_function_names = rs_symbols["functions"] | rs_reexports
        rs_implemented_function_names = rs_function_names | rs_nonpub_symbols["functions"]
        raw_missing_types = py_symbols["types"] - rs_type_names
        raw_missing_functions = py_symbols["functions"] - rs_implemented_function_names
        raw_extra_types = rs_symbols["types"] - py_symbols["types"]
        raw_extra_functions = rs_symbols["functions"] - py_symbols["functions"]
        implemented_private_functions = (
            py_symbols["functions"] & rs_nonpub_symbols["functions"] - rs_function_names
        )
        intentional_missing = INTENTIONAL_SYMBOL_MISSING.get((pair.label, module), {})
        intentional_extra = INTENTIONAL_SYMBOL_EXTRA.get((pair.label, module), {})
        ignored_missing_types = {
            name: reason
            for name, reason in intentional_missing.get("types", {}).items()
            if name in raw_missing_types
        }
        ignored_missing_functions = {
            name: reason
            for name, reason in intentional_missing.get("functions", {}).items()
            if name in raw_missing_functions
        }
        ignored_extra_types = {
            name: reason
            for name, reason in intentional_extra.get("types", {}).items()
            if name in raw_extra_types
        }
        ignored_extra_functions = {
            name: reason
            for name, reason in intentional_extra.get("functions", {}).items()
            if name in raw_extra_functions
        }
        result = {
            "module": module,
            "python_path": py_path.relative_to(root).as_posix(),
            "rust_path": rs_path.relative_to(root).as_posix(),
            "types": {
                "matched": sorted(py_symbols["types"] & rs_type_names),
                "missing": sorted(raw_missing_types - ignored_missing_types.keys()),
                "ignored_missing": dict(sorted(ignored_missing_types.items())),
                "extra": sorted(raw_extra_types - ignored_extra_types.keys()),
                "ignored_extra": dict(sorted(ignored_extra_types.items())),
            },
            "functions": {
                "matched": sorted(py_symbols["functions"] & rs_function_names),
                "implemented_private": sorted(implemented_private_functions),
                "missing": sorted(raw_missing_functions - ignored_missing_functions.keys()),
                "ignored_missing": dict(sorted(ignored_missing_functions.items())),
                "extra": sorted(raw_extra_functions - ignored_extra_functions.keys()),
                "ignored_extra": dict(sorted(ignored_extra_functions.items())),
            },
            "skipped_reexport": is_reexport,
        }
        results.append(result)
    return results


def compare_pair(root: Path, pair: ModulePair, excludes: set[str]) -> dict[str, object]:
    python_dir = root / pair.python_dir
    rust_dir = root / pair.rust_dir
    if not python_dir.is_dir():
        raise SystemExit(f"missing Python directory: {pair.python_dir}")
    if not rust_dir.is_dir():
        raise SystemExit(f"missing Rust directory: {pair.rust_dir}")

    py_modules = python_modules(python_dir, excludes)
    rs_modules = rust_modules(rust_dir, excludes)
    raw_missing = py_modules - rs_modules
    raw_extra = rs_modules - py_modules
    ignored_missing = {
        name: reason
        for name, reason in INTENTIONAL_MISSING.get(pair.label, {}).items()
        if name in raw_missing
    }
    ignored_extra = {
        name: reason
        for name, reason in INTENTIONAL_EXTRA.get(pair.label, {}).items()
        if name in raw_extra
    }
    missing = sorted(raw_missing - ignored_missing.keys())
    extra = sorted(raw_extra - ignored_extra.keys())
    matched = sorted(py_modules & rs_modules)
    return {
        "label": pair.label,
        "python_dir": pair.python_dir.as_posix(),
        "rust_dir": pair.rust_dir.as_posix(),
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "ignored_missing": dict(sorted(ignored_missing.items())),
        "ignored_extra": dict(sorted(ignored_extra.items())),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare immediate RPython/PyPy module names with their Rust "
            "port directories."
        )
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="include Python test packages in module comparison",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of text",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when any missing or extra module is found",
    )
    parser.add_argument(
        "--symbols",
        action="store_true",
        help="also compare top-level class/function names with Rust pub item names",
    )
    parser.add_argument(
        "--strict-symbols",
        action="store_true",
        help="exit non-zero when --symbols finds any non-reexport symbol gap",
    )
    parser.add_argument(
        "--jit-strings",
        action="store_true",
        help="also compare selected JIT/codewriter string-name tables",
    )
    parser.add_argument(
        "--strict-jit-strings",
        action="store_true",
        help="exit non-zero when --jit-strings finds a string-name gap",
    )
    return parser.parse_args(argv)


def print_text(results: list[dict[str, object]], show_symbols: bool) -> None:
    for result in results:
        print(f"## {result['label']} -> {result['rust_dir']}")
        missing = result["missing"]
        extra = result["extra"]
        if missing:
            print("missing: " + ", ".join(missing))
        else:
            print("missing: <none>")
        if extra:
            print("extra: " + ", ".join(extra))
        else:
            print("extra: <none>")
        ignored_missing = result["ignored_missing"]
        ignored_extra = result["ignored_extra"]
        if ignored_missing:
            print(
                "ignored missing: "
                + "; ".join(f"{name} ({reason})" for name, reason in ignored_missing.items())
            )
        if ignored_extra:
            print(
                "ignored extra: "
                + "; ".join(f"{name} ({reason})" for name, reason in ignored_extra.items())
            )
        if show_symbols:
            symbol_results = result["symbols"]
            symbol_gaps = [
                item
                for item in symbol_results
                if item["types"]["missing"]
                or item["types"]["extra"]
                or item["types"]["ignored_missing"]
                or item["types"]["ignored_extra"]
                or item["functions"]["missing"]
                or item["functions"]["extra"]
                or item["functions"]["implemented_private"]
                or item["functions"]["ignored_missing"]
                or item["functions"]["ignored_extra"]
                or item["skipped_reexport"]
            ]
            if not symbol_gaps:
                print("symbols: <none>")
            else:
                print("symbols:")
                for item in symbol_gaps:
                    if item["skipped_reexport"]:
                        print(
                            f"  {item['module']}: skipped reexport wrapper "
                            f"({item['rust_path']})"
                        )
                    else:
                        details = []
                        if item["types"]["missing"]:
                            details.append(
                                "missing types " + ", ".join(item["types"]["missing"])
                            )
                        if item["types"]["ignored_missing"]:
                            details.append(
                                "ignored missing types "
                                + "; ".join(
                                    f"{name} ({reason})"
                                    for name, reason in item["types"][
                                        "ignored_missing"
                                    ].items()
                                )
                            )
                        if item["types"]["extra"]:
                            details.append(
                                "extra types " + ", ".join(item["types"]["extra"])
                            )
                        if item["types"]["ignored_extra"]:
                            details.append(
                                "ignored extra types "
                                + "; ".join(
                                    f"{name} ({reason})"
                                    for name, reason in item["types"][
                                        "ignored_extra"
                                    ].items()
                                )
                            )
                        if item["functions"]["missing"]:
                            details.append(
                                "missing functions "
                                + ", ".join(item["functions"]["missing"])
                            )
                        if item["functions"]["implemented_private"]:
                            details.append(
                                "implemented private functions "
                                + ", ".join(item["functions"]["implemented_private"])
                            )
                        if item["functions"]["ignored_missing"]:
                            details.append(
                                "ignored missing functions "
                                + "; ".join(
                                    f"{name} ({reason})"
                                    for name, reason in item["functions"][
                                        "ignored_missing"
                                    ].items()
                                )
                            )
                        if item["functions"]["extra"]:
                            details.append(
                                "extra functions "
                                + ", ".join(item["functions"]["extra"])
                            )
                        if item["functions"]["ignored_extra"]:
                            details.append(
                                "ignored extra functions "
                                + "; ".join(
                                    f"{name} ({reason})"
                                    for name, reason in item["functions"][
                                        "ignored_extra"
                                    ].items()
                                )
                            )
                        print(f"  {item['module']}: " + "; ".join(details))
        print()


def print_string_set_text(results: list[dict[str, object]]) -> None:
    print("## JIT string parity")
    for result in results:
        details = []
        if result["missing_in_rust"]:
            details.append("missing in Rust " + ", ".join(result["missing_in_rust"]))
        if result["extra_in_rust"]:
            details.append("extra in Rust " + ", ".join(result["extra_in_rust"]))
        if not details:
            details.append("<none>")
        print(f"{result['label']}: " + "; ".join(details))
    print()


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = repo_root()
    excludes = set(DEFAULT_EXCLUDES)
    if args.include_tests:
        excludes.discard("test")

    results = [compare_pair(root, pair, excludes) for pair in DEFAULT_PAIRS]
    show_symbols = args.symbols or args.strict_symbols
    show_jit_strings = args.jit_strings or args.strict_jit_strings
    if show_symbols:
        for pair, result in zip(DEFAULT_PAIRS, results):
            result["symbols"] = compare_symbols_for_pair(root, pair, result["matched"])
    string_set_results = (
        compare_string_sets(root, DEFAULT_STRING_SET_PAIRS) if show_jit_strings else []
    )
    if args.json:
        if show_jit_strings:
            payload = {"modules": results, "jit_strings": string_set_results}
        else:
            payload = results
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_text(results, show_symbols)
        if show_jit_strings:
            print_string_set_text(string_set_results)

    has_gap = any(result["missing"] or result["extra"] for result in results)
    has_symbol_gap = False
    if show_symbols:
        has_symbol_gap = any(
            (
                item["types"]["missing"]
                or item["types"]["extra"]
                or item["functions"]["missing"]
                or item["functions"]["extra"]
            )
            and not item["skipped_reexport"]
            for result in results
            for item in result["symbols"]
        )
    has_string_set_gap = any(
        result["missing_in_rust"] or result["extra_in_rust"]
        for result in string_set_results
    )
    if args.strict and has_gap:
        return 1
    if args.strict_symbols and has_symbol_gap:
        return 1
    if args.strict_jit_strings and has_string_set_gap:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
