//! Call control — inline vs residual decision for function calls.
//!
//! RPython equivalent: `rpython/jit/codewriter/call.py` class `CallControl`.
//!
//! Decides which functions should be inlined into JitCode ("regular") and
//! which should remain as opaque calls ("residual").  Also handles builtin
//! (oopspec) and recursive (portal) call classification.

use std::collections::{HashMap, HashSet};

use majit_ir::descr::{EffectInfo, ExtraEffect, OopSpecIndex};
use majit_ir::value::Type;
use serde::{Deserialize, Serialize};

use crate::model::{CallTarget, FunctionGraph, OpKind, Terminator};
use crate::parse::CallPath;

// ── Graph-based analyzers (RPython effectinfo.py + canraise.py) ────
//
// RPython uses BoolGraphAnalyzer subclasses that traverse call graphs
// transitively. Each analyzer checks for specific operations:
//   - RaiseAnalyzer: Abort terminators (canraise.py)
//   - VirtualizableAnalyzer: jit_force_virtualizable/jit_force_virtual ops
//   - QuasiImmutAnalyzer: jit_force_quasi_immutable ops
//   - RandomEffectsAnalyzer: unanalyzable external calls

/// RPython: canraise.py — result of raise analysis.
///
/// `_canraise()` returns True, False, or "mem" (only MemoryError).
/// call.py:337-355.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanRaise {
    /// Function cannot raise any exception.
    No,
    /// Function can only raise MemoryError.
    MemoryErrorOnly,
    /// Function can raise arbitrary exceptions.
    Yes,
}

/// RPython: DependencyTracker equivalent — caches transitive analysis results.
///
/// Each analyzer in RPython has its own `seen` set (via `analyze_direct_call`).
/// We cache the final result per CallPath so repeated queries are O(1).
#[derive(Default)]
pub struct AnalysisCache {
    can_raise: HashMap<CallPath, CanRaise>,
    forces_virtualizable: HashMap<CallPath, bool>,
    random_effects: HashMap<CallPath, bool>,
    can_invalidate: HashMap<CallPath, bool>,
    /// RPython: collect_analyzer (collectanalyze.py) — can this call trigger GC?
    can_collect: HashMap<CallPath, bool>,
    /// RPython: `cpu.fielddescrof(T, fieldname)` / `cpu.arraydescrof(ARRAY)`.
    /// Assigns sequential, collision-free ei_index values for bitstrings.
    pub descr_indices: DescrIndexRegistry,
}

/// RPython: readwrite_analyzer.analyze(op) return value.
///
/// Represents the set of read/write effects collected from graph traversal.
/// RPython uses a set of tuples like ("struct", T, fieldname); we use bitsets.
/// This is passed as the first argument to effectinfo_from_writeanalyze(),
/// matching RPython's `effects` parameter (effectinfo.py:276).
pub struct WriteAnalysis {
    pub read_fields: u64,
    pub write_fields: u64,
    pub read_arrays: u64,
    pub write_arrays: u64,
    pub read_interiorfields: u64,
    pub write_interiorfields: u64,
    pub array_write_descrs: Vec<majit_ir::descr::DescrRef>,
    /// RPython: `effects is top_set` — unanalyzable (random effects).
    pub is_top: bool,
}

/// Call descriptor — associates a call target with its effect info.
///
/// RPython equivalent: the combination of `CallDescr` + `EffectInfo`
/// stored on call operations by `CallControl.getcalldescr()`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallDescriptor {
    pub target: CallTarget,
    pub effect_info: EffectInfo,
}

impl CallDescriptor {
    pub fn known(target: CallTarget, effect_info: EffectInfo) -> Self {
        Self {
            target,
            effect_info,
        }
    }

    pub fn override_effect(target: CallTarget, effect_info: EffectInfo) -> Self {
        Self {
            target,
            effect_info,
        }
    }

    pub fn effect_info(&self) -> EffectInfo {
        self.effect_info.clone()
    }
}

/// Call classification — RPython `guess_call_kind()` return values.
///
/// RPython: the string literals `'regular'`, `'residual'`, `'builtin'`,
/// `'recursive'` returned by `CallControl.guess_call_kind()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallKind {
    /// Inline this call — callee graph is available and is a candidate.
    /// RPython: `'regular'` → produces `inline_call_*` jitcode instruction.
    Regular,
    /// Leave as a residual call in the trace.
    /// RPython: `'residual'` → produces `residual_call_*` jitcode instruction.
    Residual,
    /// Built-in operation with oopspec semantics (list ops, string ops, etc.)
    /// RPython: `'builtin'` → special handling per oopspec name.
    Builtin,
    /// Recursive call back to the portal (JIT entry point).
    /// RPython: `'recursive'` → produces `recursive_call_*` jitcode instruction.
    Recursive,
}

/// RPython: `JitDriverStaticData` — per-jitdriver metadata.
///
/// RPython `metainterp/jitdriver.py`: stores green/red variable names,
/// virtualizable info, portal graph reference, etc.
#[derive(Debug, Clone)]
pub struct JitDriverStaticData {
    /// RPython: `jitdriver_sd.index`
    pub index: usize,
    /// RPython: `jitdriver.greens` — loop-invariant variable names.
    pub greens: Vec<String>,
    /// RPython: `jitdriver.reds` — loop-variant variable names.
    pub reds: Vec<String>,
    /// Portal graph path.
    pub portal_graph: CallPath,
    /// RPython: `jd.mainjitcode` (call.py:147) — JitCode index for the portal.
    /// Set by `grab_initial_jitcodes()`.
    pub mainjitcode: Option<usize>,
}

/// Call control — decides inline vs residual for each call target.
///
/// RPython: `call.py::CallControl`.
///
/// In RPython, `CallControl` discovers all candidate graphs by traversing
/// from the portal graph, then for each `direct_call` operation it classifies
/// the call as regular/residual/builtin/recursive.
///
/// In majit-codewriter, we don't have RPython's function pointer linkage.
/// Instead, callee graphs are collected from parsed Rust source files
/// (free functions via `collect_function_graphs` and trait impl methods
/// via `extract_trait_impls`).
pub struct CallControl {
    /// Free function graphs: CallPath → FunctionGraph.
    /// RPython: `funcptr._obj.graph` linkage.
    function_graphs: HashMap<CallPath, FunctionGraph>,

    /// Trait impl method graphs: (method_name, impl_type) → FunctionGraph.
    /// Used for resolving `handler.method_name()` calls.
    trait_method_graphs: HashMap<(String, String), FunctionGraph>,

    /// Trait bindings: method_name → Vec<impl_type>.
    /// Tracks which types implement a given method.
    trait_method_impls: HashMap<String, Vec<String>>,

    /// Candidate targets — graphs we will inline.
    /// RPython: `CallControl.candidate_graphs`.
    candidate_graphs: HashSet<CallPath>,

    /// Portal entry points (recursive call detection).
    /// RPython: `CallControl.jitdrivers_sd`.
    portal_targets: HashSet<CallPath>,

    /// RPython: `JitDriverStaticData` — metadata for each jitdriver.
    /// `jitdrivers_sd[i]` holds the green/red arg layout for driver i.
    jitdrivers_sd: Vec<JitDriverStaticData>,

    /// Builtin targets (oopspec operations).
    /// RPython: detected via `funcobj.graph.func.oopspec`.
    builtin_targets: HashSet<CallPath>,

    /// RPython: `CallControl.jitcodes` — map {graph_key: jitcode_index}.
    /// Tracks which graphs have been assigned JitCode objects.
    /// The index is assigned sequentially and used by InlineCall ops.
    jitcodes: HashMap<CallPath, usize>,

    /// RPython: `CallControl.unfinished_graphs` — graphs pending assembly.
    unfinished_graphs: Vec<CallPath>,

    /// RPython: `CallControl.callinfocollection` (call.py:31).
    /// Stores oopspec function info for builtin call handling.
    pub callinfocollection: majit_ir::descr::CallInfoCollection,

    /// Next JitCode index to assign.
    next_jitcode_index: usize,

    /// RPython: `getattr(func, "_elidable_function_", False)` (call.py:239).
    /// Targets known to be elidable (pure, no side effects).
    elidable_targets: HashSet<CallPath>,

    /// RPython: `getattr(func, "_jit_loop_invariant_", False)` (call.py:240).
    /// Targets known to be loop-invariant (call once per loop).
    loopinvariant_targets: HashSet<CallPath>,

    /// RPython: known struct types for `get_type_flag(ARRAY.OF)` → FLAG_STRUCT.
    /// If an array's element type is in this set, the array descriptor gets
    /// `ArrayFlag::Struct` (like RPython's `isinstance(TYPE, lltype.Struct)`).
    known_struct_names: HashSet<String>,

    /// RPython: struct field type info — maps struct_name → [(field_name, type_string)].
    /// Used by `resolve_array_identity` to determine the ARRAY element type
    /// when the base of an array access comes from a FieldRead.
    /// Equivalent to `op.args[0].concretetype.TO` in RPython's rtyped graph.
    struct_fields: crate::front::StructFieldRegistry,

    /// RPython: `op.result.concretetype` — function return type strings.
    /// Maps CallPath → full return type string (e.g. "Vec<Point>").
    /// Used by `resolve_array_identity` for Call result array identity.
    pub return_types: HashMap<CallPath, String>,

    /// RPython: `symbolic.get_array_token(ARRAY, tsc)[0]` — array base size.
    /// Offset from the array object pointer to the first element.
    /// RPython GcArray layout: `[length (WORD)] [items...]`, so
    /// `basesize = carray.items.offset = sizeof(Signed) = WORD`.
    /// Default: WORD (8 on 64-bit) matching RPython's standard GcArray.
    pub array_header_size: usize,

    /// RPython: `symbolic.get_field_token(STRUCT, fieldname, tsc)` / `symbolic.get_size()`.
    /// Pre-computed struct layouts from actual runtime (std::mem::offset_of! etc.).
    /// When registered, provides exact (offset, size) for struct fields,
    /// bypassing the type-string heuristic. The runtime/proc-macro populates
    /// this via `set_struct_layout()`.
    pub struct_layouts: HashMap<String, StructLayout>,
    /// RPython: collectanalyze.py:15 — _gctransformer_hint_cannot_collect_.
    /// Functions known not to trigger GC collection. The collect_analyzer
    /// returns False immediately for these.
    pub cannot_collect_targets: HashSet<CallPath>,
    /// RPython: call.py:129-134 — _gctransformer_hint_close_stack_.
    /// Functions that close the stack must never produce JitCode — they are
    /// always classified as 'residual' by guess_call_kind.
    pub close_stack_targets: HashSet<CallPath>,
    /// RPython: collectanalyze.py:21-25 — funcobj.random_effects_on_gcobjs.
    /// External functions whose calls may have random effects on GC objects.
    /// analyze_can_collect returns True immediately for these.
    pub external_gc_effects: HashSet<CallPath>,
}

/// Heuristic struct layout — NOT equivalent to RPython's `symbolic.get_field_token()`.
///
/// RPython delegates to `ll2ctypes.get_ctypes_type(STRUCT)` or `llmemory.offsetof()`
/// for actual C-level layout. This struct holds heuristic approximations computed
/// from Rust type strings via `from_type_strings()`. Offsets and sizes may diverge
/// from actual `#[repr(C)]` layout. The runtime SHOULD override via
/// `set_struct_layout()` with values from `std::mem::offset_of!()` /
/// `std::mem::size_of::<T>()` for production use.
#[derive(Debug, Clone)]
pub struct StructLayout {
    /// RPython: `symbolic.get_size(STRUCT, tsc)` — total struct size.
    pub size: usize,
    /// Per-field layout: (field_name, offset, size, type).
    /// RPython: `symbolic.get_field_token(STRUCT, name, tsc) → (offset, size)`.
    pub fields: Vec<StructFieldLayout>,
}

/// Single field within a `StructLayout`.
#[derive(Debug, Clone)]
pub struct StructFieldLayout {
    pub name: String,
    /// RPython: `cfield.offset`
    pub offset: usize,
    /// RPython: `cfield.size`
    pub size: usize,
    /// RPython: `get_type_flag(getattr(STRUCT, fieldname))`
    pub flag: majit_ir::descr::ArrayFlag,
    /// IR type classification.
    pub field_type: majit_ir::value::Type,
}

impl StructLayout {
    /// Build a StructLayout from type-string heuristic.
    /// Used at pipeline init to populate struct_layouts from struct_fields.
    /// The runtime can later override with actual layout via set_struct_layout().
    pub fn from_type_strings(
        fields: &[(String, String)],
        known_structs: &std::collections::HashSet<String>,
        known_struct_sizes: &std::collections::HashMap<String, usize>,
    ) -> Self {
        // RPython: symbolic.get_array_token() computes itemsize for ANY struct,
        // even those with nested structs. UnsupportedFieldExc only affects
        // all_interiorfielddescrs (field enumeration), NOT the struct size.
        // So we always compute the full size, but mark has_nested_struct to
        // clear interior field descriptors.
        let has_nested_struct = fields
            .iter()
            .any(|(_, type_str)| known_structs.contains(type_str.as_str()));
        let mut offset: usize = 0;
        let mut layout_fields = Vec::new();
        for (name, type_str) in fields {
            // heaptracker.py:62-67: skip Void, padding, and typeptr fields.
            // typeptr is handled separately (not enumerated by all_fielddescrs).
            if name == "typeptr" || name.starts_with("c__pad") {
                // heaptracker.py:64-67
                // typeptr is still counted for offset calculation below.
                let sz = if known_structs.contains(type_str.as_str()) {
                    known_struct_sizes
                        .get(type_str.as_str())
                        .copied()
                        .unwrap_or(std::mem::size_of::<usize>())
                } else {
                    get_type_flag(type_str).2
                };
                if sz > 0 {
                    let align = sz.min(std::mem::size_of::<usize>());
                    offset = (offset + align - 1) & !(align - 1);
                    offset += sz;
                }
                continue;
            }
            let (flag, field_type, field_size) = if known_structs.contains(type_str.as_str()) {
                // RPython: symbolic.get_field_token(STRUCT, fieldname) returns the
                // actual embedded struct size, not just a pointer size.
                let nested_size = known_struct_sizes
                    .get(type_str.as_str())
                    .copied()
                    .unwrap_or(std::mem::size_of::<usize>());
                (
                    majit_ir::descr::ArrayFlag::Struct,
                    majit_ir::value::Type::Ref,
                    nested_size,
                )
            } else {
                get_type_flag(type_str)
            };
            if field_type == majit_ir::value::Type::Void || field_size == 0 {
                continue;
            }
            // RPython: alignment is typically min(field_size, WORD).
            let align = field_size.min(std::mem::size_of::<usize>());
            offset = (offset + align - 1) & !(align - 1);
            layout_fields.push(StructFieldLayout {
                name: name.clone(),
                offset,
                size: field_size,
                flag,
                field_type,
            });
            offset += field_size;
        }
        // RPython: heaptracker.py:89-90 — if nested struct exists,
        // all_interiorfielddescrs raises UnsupportedFieldExc, so
        // interior field descriptors are not enumerable. Clear fields
        // but keep the correct size.
        if has_nested_struct {
            layout_fields.clear();
        }
        let max_align = fields
            .iter()
            .map(|(_, ty)| {
                if known_structs.contains(ty.as_str()) {
                    // Use actual nested struct size for alignment.
                    known_struct_sizes
                        .get(ty.as_str())
                        .copied()
                        .unwrap_or(std::mem::size_of::<usize>())
                        .min(std::mem::size_of::<usize>())
                } else {
                    get_type_flag(ty).2
                }
            })
            .filter(|s| *s > 0)
            .max()
            .unwrap_or(8);
        let size = if offset > 0 {
            (offset + max_align - 1) & !(max_align - 1)
        } else {
            0
        };
        StructLayout {
            size,
            fields: layout_fields,
        }
    }
}

/// Sequential descriptor index assignment — majit equivalent of
/// `cpu.fielddescrof(T, fieldname).get_ei_index()` /
/// `cpu.arraydescrof(ARRAY).get_ei_index()`.
///
/// RPython: each descriptor gets a globally unique index via
/// `compute_bitstrings()`. The bitstring module creates variable-length
/// bitfields. In majit we use fixed-width u64, so indices wrap at 64.
///
/// Guarantees: same (owner_root, field_name) always gets the same index.
/// Limitation: after 64 unique field descriptors, indices wrap and may
/// alias unrelated descriptors. This matches RPython's bitstring semantics
/// where aliased bits cause conservative over-approximation (safe but
/// imprecise). Array descriptors are keyed by `(item_ty, array_type_id)`,
/// matching RPython's `cpu.arraydescrof(ARRAY)` which distinguishes by
/// ARRAY identity (e.g. `GcArray(Signed)` vs `GcArray(Ptr(STRUCT_X))`).
#[derive(Default)]
pub struct DescrIndexRegistry {
    /// (owner_root, field_name) → bit index (0..63)
    field_indices: HashMap<(Option<String>, String), u32>,
    /// (item_ty_discriminant, array_type_id) → bit index (0..63)
    /// RPython: cpu.arraydescrof(ARRAY).get_ei_index()
    array_indices: HashMap<(u8, Option<String>), u32>,
    /// (array_type_id, field_name) → bit index (0..63)
    /// RPython: cpu.interiorfielddescrof(ARRAY, fieldname).get_ei_index()
    /// Separate from field_indices — RPython keys on (ARRAY, fieldname)
    /// not (STRUCT, fieldname).
    interiorfield_indices: HashMap<(Option<String>, String), u32>,
    next_field_index: u32,
    next_array_index: u32,
    next_interiorfield_index: u32,
}

impl DescrIndexRegistry {
    /// RPython: `cpu.fielddescrof(T, fieldname).get_ei_index()`
    pub fn field_index(&mut self, owner_root: &Option<String>, field_name: &str) -> u32 {
        let key = (owner_root.clone(), field_name.to_string());
        *self.field_indices.entry(key).or_insert_with(|| {
            let idx = self.next_field_index % 64;
            self.next_field_index += 1;
            idx
        })
    }

    /// RPython: `cpu.arraydescrof(ARRAY).get_ei_index()`
    ///
    /// Keys on `(item_ty_discriminant, array_type_id)` unconditionally.
    /// RPython always distinguishes by ARRAY lltype identity:
    /// `GcArray(Char)`, `GcArray(Signed)`, `GcArray(Bool)` are all
    /// different descriptors even though they are all "integer" arrays
    /// (effectinfo.py:307-311).
    pub fn array_index(&mut self, item_ty_discriminant: u8, array_type_id: &Option<String>) -> u32 {
        let key = (item_ty_discriminant, array_type_id.clone());
        *self.array_indices.entry(key).or_insert_with(|| {
            let idx = self.next_array_index % 64;
            self.next_array_index += 1;
            idx
        })
    }

    /// RPython: `cpu.interiorfielddescrof(ARRAY, fieldname).get_ei_index()`
    ///
    /// Keys on `(array_type_id, field_name)` — matches RPython's cache key
    /// `(ARRAY, name, arrayfieldname)` in `get_interiorfield_descr()`.
    /// Separate namespace from field_indices: two different array types
    /// with the same element struct get different interiorfield indices.
    pub fn interiorfield_index(&mut self, array_type_id: &Option<String>, field_name: &str) -> u32 {
        let key = (array_type_id.clone(), field_name.to_string());
        *self.interiorfield_indices.entry(key).or_insert_with(|| {
            let idx = self.next_interiorfield_index % 64;
            self.next_interiorfield_index += 1;
            idx
        })
    }
}

impl CallControl {
    /// RPython: `CallControl.__init__`.
    pub fn new() -> Self {
        Self {
            function_graphs: HashMap::new(),
            trait_method_graphs: HashMap::new(),
            trait_method_impls: HashMap::new(),
            candidate_graphs: HashSet::new(),
            portal_targets: HashSet::new(),
            jitdrivers_sd: Vec::new(),
            builtin_targets: HashSet::new(),
            jitcodes: HashMap::new(),
            unfinished_graphs: Vec::new(),
            callinfocollection: majit_ir::descr::CallInfoCollection::new(),
            next_jitcode_index: 0,
            elidable_targets: HashSet::new(),
            loopinvariant_targets: HashSet::new(),
            known_struct_names: HashSet::new(),
            struct_fields: crate::front::StructFieldRegistry::default(),
            return_types: HashMap::new(),
            // RPython: symbolic.get_array_token(GcArray(T))[0] = carray.items.offset
            // = sizeof(Signed) = WORD. Standard GcArray has a length field before items.
            //
            array_header_size: std::mem::size_of::<usize>(),
            struct_layouts: HashMap::new(),
            cannot_collect_targets: HashSet::new(),
            close_stack_targets: HashSet::new(),
            external_gc_effects: HashSet::new(),
        }
    }

    /// RPython: register struct type names for get_type_flag(ARRAY.OF).
    pub fn set_known_struct_names(&mut self, names: HashSet<String>) {
        self.known_struct_names = names;
    }

    /// RPython: register struct field types for op.args[0].concretetype resolution.
    pub fn set_struct_fields(&mut self, registry: crate::front::StructFieldRegistry) {
        self.struct_fields = registry;
    }

    /// RPython: isinstance(TYPE, lltype.Struct) check.
    pub fn is_known_struct(&self, name: &str) -> bool {
        self.known_struct_names.contains(name)
    }

    /// RPython: register actual struct layout from `symbolic.get_field_token()`.
    /// The runtime calls this with layouts from `std::mem::offset_of!()` etc.
    pub fn set_struct_layout(&mut self, struct_name: String, layout: StructLayout) {
        self.struct_layouts.insert(struct_name, layout);
    }

    /// RPython: resolve a struct field's type string.
    /// For `owner::field_name`, returns the full type of the field.
    pub fn field_type(&self, owner: &str, field_name: &str) -> Option<&str> {
        self.struct_fields.field_type(owner, field_name)
    }

    /// RPython: `cpu.arraydescrof(ARRAY)` — descr.py:348-378.
    ///
    /// `array_type_id`: full ARRAY type string (e.g. `"Vec<Point>"`), matching
    /// RPython's ARRAY lltype identity. The element type is extracted via
    /// `extract_element_type_from_str()` for struct checks and flag resolution.
    pub fn arraydescrof(
        &self,
        idx: u32,
        array_type_id: &Option<String>,
        ir_type: majit_ir::value::Type,
    ) -> majit_ir::descr::DescrRef {
        // RPython: ARRAY_INSIDE.OF — extract element type from full ARRAY type.
        let elem_name = array_type_id
            .as_deref()
            .and_then(|s| extract_element_type_from_str(s).or_else(|| Some(s.to_string())))
            .as_deref()
            .map(String::from);
        let elem_ref = elem_name.as_deref();
        let is_struct = elem_ref.is_some_and(|n| self.is_known_struct(n));
        // RPython: descr.py:363 — flag = get_type_flag(ARRAY_INSIDE.OF)
        // RPython: descr.py:354 — itemsize from symbolic.get_array_token().
        // descr.py:363 — flag = get_type_flag(ARRAY_INSIDE.OF).
        // descr.py:365 — ArrayDescr(basesize, itemsize, ..., flag).
        // Even for struct(struct), itemsize is correct from symbolic.
        let (flag, item_size) = if is_struct {
            (
                majit_ir::descr::ArrayFlag::Struct,
                elem_ref.map(|n| compute_struct_size(self, n)).unwrap_or(8),
            )
        } else if let Some(elem) = elem_ref {
            let (f, _, s) = get_type_flag(elem);
            (f, s)
        } else {
            (
                majit_ir::descr::ArrayFlag::from_item_type(ir_type, false),
                8,
            )
        };
        // RPython: basesize, itemsize, _ = symbolic.get_array_token(ARRAY, tsc)
        let base_size = self.array_header_size;
        let mut ad = majit_ir::descr::SimpleArrayDescr::with_flag(
            idx, base_size, item_size, 0, ir_type, flag,
        );
        // RPython: descr.py:372-375 — struct arrays get interior field descriptors.
        if is_struct {
            if let Some(struct_name) = elem_ref {
                let ad_arc = std::sync::Arc::new(ad);
                let (descrs, _) = all_interiorfielddescrs(self, struct_name, ad_arc.clone());
                if !descrs.is_empty() {
                    let mut ad_mut = (*ad_arc).clone();
                    ad_mut.set_all_interiorfielddescrs(descrs);
                    return std::sync::Arc::new(ad_mut);
                } else {
                    return ad_arc;
                }
            }
        }
        std::sync::Arc::new(ad)
    }

    /// Register a free function graph.
    /// RPython: graphs are discovered via funcptr linkage.
    pub fn register_function_graph(&mut self, path: CallPath, graph: FunctionGraph) {
        self.function_graphs.insert(path, graph);
    }

    /// Register a trait impl method graph.
    ///
    /// Also registers the graph in function_graphs under a synthetic
    /// CallPath so that BFS in find_all_graphs can discover it.
    /// RPython: method graphs are reachable through funcptr._obj.graph
    /// linkage — we emulate this by dual registration.
    pub fn register_trait_method(
        &mut self,
        method_name: &str,
        impl_type: &str,
        graph: FunctionGraph,
    ) {
        self.trait_method_graphs.insert(
            (method_name.to_string(), impl_type.to_string()),
            graph.clone(),
        );
        self.trait_method_impls
            .entry(method_name.to_string())
            .or_default()
            .push(impl_type.to_string());
        // Register in function_graphs for BFS reachability.
        // RPython: each graph has its own identity via funcptr._obj.graph.
        // We emulate this with CallPath([impl_type, method_name]) —
        // each impl gets its own distinct path, preventing name collisions
        // (e.g. PyFrame::push_value vs MIFrame::push_value).
        let qualified_path = CallPath::from_segments([impl_type, method_name]);
        self.function_graphs.entry(qualified_path).or_insert(graph);
    }

    /// Mark a target as the portal entry point.
    ///
    /// RPython: `setup_jitdriver(jitdriver_sd)` + `grab_initial_jitcodes()`.
    pub fn mark_portal(&mut self, path: CallPath) {
        self.portal_targets.insert(path);
    }

    /// Register a JitDriver with its green/red variable layout.
    ///
    /// RPython: `CodeWriter.setup_jitdriver(jitdriver_sd)` (codewriter.py:96-99).
    /// Each jitdriver gets a sequential index.
    pub fn setup_jitdriver(
        &mut self,
        portal_graph: CallPath,
        greens: Vec<String>,
        reds: Vec<String>,
    ) {
        let index = self.jitdrivers_sd.len();
        self.jitdrivers_sd.push(JitDriverStaticData {
            index,
            greens,
            reds,
            portal_graph: portal_graph.clone(),
            mainjitcode: None,
        });
        self.portal_targets.insert(portal_graph);
    }

    /// RPython: `jitdriver_sd_from_portal_runner_ptr(funcptr)`.
    /// Find the jitdriver that owns a given portal target.
    pub fn jitdriver_sd_from_portal(&self, path: &CallPath) -> Option<&JitDriverStaticData> {
        self.jitdrivers_sd
            .iter()
            .find(|sd| &sd.portal_graph == path)
    }

    /// Mark a target as a builtin (oopspec) operation.
    pub fn mark_builtin(&mut self, path: CallPath) {
        self.builtin_targets.insert(path);
    }

    /// Discover candidate graphs by BFS from portal targets.
    ///
    /// RPython: `CallControl.find_all_graphs(policy)` (call.py:49-92).
    ///
    /// Walks from portal graphs transitively: for each Call op,
    /// if the callee has a graph, add it to the candidate set.
    /// Portal must be seeded via `mark_portal()` before calling.
    pub fn find_all_graphs(&mut self) {
        assert!(
            !self.portal_targets.is_empty(),
            "find_all_graphs requires at least one portal target; \
             use find_all_graphs_for_tests() if no portal is available"
        );
        self.find_all_graphs_bfs();
    }

    /// Test-only: include all registered function graphs as candidates.
    /// Production code must use `find_all_graphs()` with portal seeded.
    #[cfg(test)]
    pub fn find_all_graphs_for_tests(&mut self) {
        if self.portal_targets.is_empty() {
            let all_paths: Vec<CallPath> = self.function_graphs.keys().cloned().collect();
            for path in all_paths {
                self.candidate_graphs.insert(path);
            }
            return;
        }
        self.find_all_graphs_bfs();
    }

    fn find_all_graphs_bfs(&mut self) {
        // RPython call.py:49-92: BFS from portal targets.
        // For each graph, scan all Call ops. If guess_call_kind would
        // return 'regular' (i.e. graphs_from returns a graph AND it's
        // a candidate), add the callee graph to candidates and continue.
        //
        // During BFS we use target_to_path + function_graphs directly
        // (not graphs_from, which checks candidate_graphs — the set
        // we're building).
        let mut todo: Vec<CallPath> = self.portal_targets.iter().cloned().collect();
        for path in &todo {
            self.candidate_graphs.insert(path.clone());
        }

        while let Some(path) = todo.pop() {
            let graph = match self.function_graphs.get(&path) {
                Some(g) => g.clone(),
                None => continue,
            };
            // RPython call.py:77-90: scan all Call ops in the graph.
            // For each call, check guess_call_kind (with BFS-aware
            // is_candidate that treats "has graph" as candidate).
            for block in &graph.blocks {
                for op in &block.operations {
                    let target = match &op.kind {
                        OpKind::Call { target, .. } => target,
                        _ => continue,
                    };
                    let callee_path = match self.target_to_path(target) {
                        Some(p) => p,
                        None => continue,
                    };
                    // RPython call.py:80: kind = self.guess_call_kind(op, is_candidate)
                    // Skip recursive (portal) and builtin calls — these are NOT
                    // followed during BFS. Only "regular" calls are followed.
                    if self.portal_targets.contains(&callee_path) {
                        continue; // recursive — don't follow
                    }
                    if self.builtin_targets.contains(&callee_path) {
                        continue; // builtin — don't follow
                    }
                    if self.candidate_graphs.contains(&callee_path) {
                        continue; // already discovered
                    }
                    // RPython call.py:84,87: callee must have a graph.
                    // is_candidate during BFS = "has a graph" (default policy).
                    if self.function_graphs.contains_key(&callee_path) {
                        self.candidate_graphs.insert(callee_path.clone());
                        todo.push(callee_path);
                    }
                }
            }
        }
    }

    /// RPython: `CallControl.is_candidate(graph)`.
    /// Used only after `find_all_graphs()`.
    pub fn is_candidate(&self, path: &CallPath) -> bool {
        self.candidate_graphs.contains(path)
    }

    /// RPython: `CallControl.get_jitcode(graph, called_from)`.
    ///
    /// Retrieve or create a JitCode index for the given graph.
    /// Returns the index that should be embedded in `InlineCall` ops
    /// so the meta-interpreter can find the callee's bytecode.
    ///
    /// RPython call.py:155-172: creates JitCode(graph.name, fnaddr, calldescr)
    /// and adds graph to unfinished_graphs for later assembly.
    pub fn get_jitcode(&mut self, path: &CallPath) -> usize {
        // RPython call.py:157-158: try: return self.jitcodes[graph]
        if let Some(&index) = self.jitcodes.get(path) {
            return index;
        }
        // RPython call.py:159-165: except KeyError:
        //   must never produce JitCode for close_stack.
        assert!(
            !self.close_stack_targets.contains(path),
            "{:?} has _gctransformer_hint_close_stack_",
            path
        );
        let index = self.next_jitcode_index;
        self.next_jitcode_index += 1;
        self.jitcodes.insert(path.clone(), index);
        self.unfinished_graphs.push(path.clone());
        index
    }

    /// RPython: `CallControl.grab_initial_jitcodes()` (call.py:145-148).
    ///
    /// ```python
    /// def grab_initial_jitcodes(self):
    ///     for jd in self.jitdrivers_sd:
    ///         jd.mainjitcode = self.get_jitcode(jd.portal_graph)
    ///         jd.mainjitcode.jitdriver_sd = jd
    /// ```
    ///
    /// Creates JitCode entries for portal graphs and sets back-references.
    pub fn grab_initial_jitcodes(&mut self) {
        // Collect portal paths first to avoid borrow conflict.
        let portals: Vec<(usize, CallPath)> = self
            .jitdrivers_sd
            .iter()
            .enumerate()
            .map(|(i, jd)| (i, jd.portal_graph.clone()))
            .collect();
        for (jd_index, portal) in portals {
            // RPython: jd.mainjitcode = self.get_jitcode(jd.portal_graph)
            let jitcode_index = self.get_jitcode(&portal);
            // RPython: jd.mainjitcode.jitdriver_sd = jd
            // (In majit, we store the jitcode index on the jitdriver.)
            self.jitdrivers_sd[jd_index].mainjitcode = Some(jitcode_index);
        }
    }

    /// RPython: `CallControl.enum_pending_graphs()` (call.py:150-153).
    ///
    /// ```python
    /// def enum_pending_graphs(self):
    ///     while self.unfinished_graphs:
    ///         graph = self.unfinished_graphs.pop()  # LIFO
    ///         yield graph, self.jitcodes[graph]
    /// ```
    ///
    /// RPython uses a generator that pops one graph at a time (LIFO).
    /// During processing, new graphs may be added to `unfinished_graphs`
    /// via `get_jitcode()`, and the generator picks them up on the next
    /// iteration. We emulate this with `pop_one_graph()`.
    pub fn pop_one_graph(&mut self) -> Option<(CallPath, usize)> {
        let path = self.unfinished_graphs.pop()?; // LIFO, matching RPython
        let index = self.jitcodes[&path];
        Some((path, index))
    }

    /// Classify a call target.
    ///
    /// RPython: `CallControl.guess_call_kind(op, is_candidate)` (call.py:116-139).
    ///
    /// Exact RPython decision logic (call.py:117-139):
    /// 1. Is portal runner → 'recursive'
    /// 2. `_gctransformer_hint_close_stack_` → 'residual'
    /// 3. Has oopspec → 'builtin'
    /// 4. `graphs_from(target) is None` → 'residual'
    /// 5. Otherwise → 'regular'
    pub fn guess_call_kind(&self, target: &CallTarget) -> CallKind {
        // Step 1: recursive (RPython call.py:119-120)
        let path = self.target_to_path(target);
        if let Some(ref p) = path {
            if self.portal_targets.contains(p) {
                return CallKind::Recursive;
            }
        }
        // RPython call.py:129-134: _gctransformer_hint_close_stack_ → 'residual'.
        // Must never produce JitCode for close_stack functions.
        // Checked BEFORE builtin (RPython checks close_stack first at line 132).
        if let Some(ref p) = path {
            if self.close_stack_targets.contains(p) {
                return CallKind::Residual;
            }
        }
        // RPython call.py:135-136: has oopspec → 'builtin'
        if let Some(ref p) = path {
            if self.builtin_targets.contains(p) {
                return CallKind::Builtin;
            }
        }
        // Step 3+4: graphs_from check (RPython call.py:137-139)
        // graphs_from returns the graph ONLY if it's a candidate.
        if self.graphs_from(target).is_none() {
            CallKind::Residual
        } else {
            CallKind::Regular
        }
    }

    /// Get the callee graph for a call target, but only if it is a candidate.
    ///
    /// RPython: `CallControl.graphs_from(op, is_candidate)` (call.py:94-114).
    ///
    /// Returns the graph only if:
    /// 1. The graph exists (via function_graphs or resolve_method)
    /// 2. The graph is a candidate (in candidate_graphs)
    ///
    /// This is the gatekeeper: if graphs_from returns None, the call
    /// becomes residual. If it returns Some, the call is regular.
    pub fn graphs_from(&self, target: &CallTarget) -> Option<&FunctionGraph> {
        let path = self.target_to_path(target)?;
        // RPython call.py:100: is_candidate(graph)
        if !self.candidate_graphs.contains(&path) {
            return None;
        }
        // RPython call.py:94-101: returns the actual target graph.
        // For FunctionPath: direct lookup in function_graphs.
        // For Method: resolve_method returns the specific impl graph,
        // NOT whatever was first registered under the synthetic path.
        match target {
            CallTarget::Method {
                name,
                receiver_root,
            } => self.resolve_method(name, receiver_root.as_deref()),
            _ => self.function_graphs.get(&path),
        }
    }

    /// Convert a CallTarget to a CallPath for lookup.
    ///
    /// FunctionPath → direct path.
    /// Method → qualified CallPath([impl_type, method_name]).
    ///
    /// RPython: graph identity is by object pointer, not name.
    /// We emulate this with qualified paths that include the impl type,
    /// so different impls of the same method get distinct paths.
    fn target_to_path(&self, target: &CallTarget) -> Option<CallPath> {
        match target {
            CallTarget::FunctionPath { segments } => {
                Some(CallPath::from_segments(segments.iter().map(String::as_str)))
            }
            CallTarget::Method {
                name,
                receiver_root,
            } => {
                // RPython: direct_call → funcobj.graph. Try qualified path first
                // so inherent methods resolve by direct graph linkage.
                if let Some(receiver) = receiver_root.as_deref() {
                    let qualified = CallPath::from_segments([receiver, name.as_str()]);
                    if self.function_graphs.contains_key(&qualified) {
                        return Some(qualified);
                    }
                }
                // Fall back to trait method resolution for polymorphic calls.
                let impl_type = self.resolve_method_impl_type(name, receiver_root.as_deref())?;
                Some(CallPath::from_segments([impl_type, name.as_str()]))
            }
            CallTarget::UnsupportedExpr => None,
        }
    }

    /// Resolve a method call to a concrete impl graph.
    ///
    /// RPython: method resolution happens at the type system level.
    /// Here we resolve through the trait impl registry. If there's
    /// exactly one impl for the method, return it. If the receiver
    /// is a generic parameter (lowercase or single uppercase letter),
    /// we try all known impls and return the unique one.
    pub fn resolve_method(
        &self,
        name: &str,
        receiver_root: Option<&str>,
    ) -> Option<&FunctionGraph> {
        let impls = self.trait_method_impls.get(name)?;

        // Filter out default trait method entries (e.g. "<default methods of LocalOpcodeHandler>")
        // — we prefer concrete impls when available.
        let concrete_impls: Vec<&String> = impls
            .iter()
            .filter(|t| !t.starts_with("<default methods of"))
            .collect();

        if concrete_impls.len() == 1 {
            // Unique concrete impl — use it regardless of receiver
            let impl_type = concrete_impls[0];
            return self
                .trait_method_graphs
                .get(&(name.to_string(), impl_type.clone()));
        }

        // Multiple concrete impls — try to match by receiver root
        if let Some(receiver) = receiver_root {
            if !is_generic_receiver(receiver) {
                // Concrete receiver — look for exact match
                return self
                    .trait_method_graphs
                    .get(&(name.to_string(), receiver.to_string()));
            }
        }

        // Generic receiver with multiple concrete impls — can't resolve uniquely.
        // Fall back to default method if available.
        if concrete_impls.is_empty() && impls.len() == 1 {
            let impl_type = &impls[0];
            return self
                .trait_method_graphs
                .get(&(name.to_string(), impl_type.clone()));
        }

        None
    }

    /// Like `resolve_method`, but returns the impl type name instead of the graph.
    /// Used by `target_to_path` to build qualified CallPaths.
    /// Like `resolve_method`, but returns the impl type name.
    /// All returned references borrow from `self`, not from `receiver_root`.
    fn resolve_method_impl_type<'b>(
        &'b self,
        name: &str,
        receiver_root: Option<&str>,
    ) -> Option<&'b str> {
        let impls = self.trait_method_impls.get(name)?;
        let concrete_impls: Vec<&String> = impls
            .iter()
            .filter(|t| !t.starts_with("<default methods of"))
            .collect();

        if concrete_impls.len() == 1 {
            return Some(concrete_impls[0]);
        }
        if let Some(receiver) = receiver_root {
            if !is_generic_receiver(receiver) {
                // Find the matching impl owned by self
                if let Some(impl_name) = impls.iter().find(|t| t.as_str() == receiver) {
                    return Some(impl_name);
                }
            }
        }
        if concrete_impls.is_empty() && impls.len() == 1 {
            return Some(&impls[0]);
        }
        None
    }

    /// Access the function graphs map (for inline pass).
    pub fn function_graphs(&self) -> &HashMap<CallPath, FunctionGraph> {
        &self.function_graphs
    }

    /// Access jitdriver static data.
    pub fn jitdrivers_sd(&self) -> &[JitDriverStaticData] {
        &self.jitdrivers_sd
    }

    // ── Elidable / loop-invariant registration ──────────────────────

    /// RPython: `getattr(func, "_elidable_function_", False)` (call.py:239).
    /// Mark a target as elidable (pure function).
    pub fn mark_elidable(&mut self, path: CallPath) {
        self.elidable_targets.insert(path);
    }

    /// RPython: `getattr(func, "_jit_loop_invariant_", False)` (call.py:240).
    /// Mark a target as loop-invariant.
    pub fn mark_loopinvariant(&mut self, path: CallPath) {
        self.loopinvariant_targets.insert(path);
    }

    /// RPython: call.py:239 — check if target has `_elidable_function_`.
    pub fn is_elidable(&self, target: &CallTarget) -> bool {
        self.target_to_path(target)
            .is_some_and(|p| self.elidable_targets.contains(&p))
    }

    /// RPython: call.py:240 — check if target has `_jit_loop_invariant_`.
    pub fn is_loopinvariant(&self, target: &CallTarget) -> bool {
        self.target_to_path(target)
            .is_some_and(|p| self.loopinvariant_targets.contains(&p))
    }

    /// RPython: call.py:129-134 — `_gctransformer_hint_close_stack_`.
    /// Mark a target as close_stack (must never produce JitCode).
    pub fn mark_close_stack(&mut self, path: CallPath) {
        self.close_stack_targets.insert(path);
    }

    /// RPython: collectanalyze.py:21 — `funcobj.random_effects_on_gcobjs`.
    /// Mark an external target as having random GC effects.
    pub fn mark_external_gc_effects(&mut self, path: CallPath) {
        self.external_gc_effects.insert(path);
    }

    /// RPython: collectanalyze.py:15 — `_gctransformer_hint_cannot_collect_`.
    /// Mark a target as known not to trigger GC collection.
    pub fn mark_cannot_collect(&mut self, path: CallPath) {
        self.cannot_collect_targets.insert(path);
    }

    // ── Graph-based analyzers (call.py:282-303) ─────────────────────

    /// RPython: RaiseAnalyzer.analyze() — transitive can-raise analysis.
    ///
    /// canraise.py:8-24: RaiseAnalyzer(BoolGraphAnalyzer)
    /// - `analyze_simple_operation`: checks `LL_OPERATIONS[op.opname].canraise`
    /// - `analyze_external_call`: `getattr(fnobj, 'canraise', True)`
    /// - `analyze_exceptblock_in_graph`: checks except blocks
    ///
    /// In majit we check per-operation canraise metadata via `op_can_raise()`,
    /// Abort terminators, and transitive Call analysis.
    fn analyze_can_raise(&self, path: &CallPath, seen: &mut HashSet<CallPath>) -> bool {
        self.analyze_can_raise_impl(path, seen, false)
    }

    /// Shared implementation for both raise analyzers.
    ///
    /// RPython has two separate RaiseAnalyzer instances (call.py:34-36):
    /// - `raise_analyzer`: normal mode
    /// - `raise_analyzer_ignore_memoryerror`: `do_ignore_memory_error()` mode
    ///
    /// `ignore_memoryerror` controls whether ops that can only raise
    /// MemoryError are treated as non-raising (canraise.py:11-17).
    fn analyze_can_raise_impl(
        &self,
        path: &CallPath,
        seen: &mut HashSet<CallPath>,
        ignore_memoryerror: bool,
    ) -> bool {
        if !seen.insert(path.clone()) {
            return false; // cycle → bottom_result
        }
        let graph = match self.function_graphs.get(path) {
            Some(g) => g,
            // RPython: analyze_external_call → getattr(fnobj, 'canraise', True)
            None => return true,
        };
        for block in &graph.blocks {
            // RPython: Abort terminator = except block path.
            // canraise.py:27-41: analyze_exceptblock_in_graph.
            if let Terminator::Abort { reason } = &block.terminator {
                if ignore_memoryerror && is_memoryerror_only(reason) {
                    // RPython: do_ignore_memory_error skips MemoryError-only
                    continue;
                }
                return true;
            }
            // RPython: analyze_simple_operation(op) per operation.
            // canraise.py:14-17: LL_OPERATIONS[op.opname].canraise
            for op in &block.operations {
                match &op.kind {
                    OpKind::Call { target, .. } => {
                        let callee_path = match self.target_to_path(target) {
                            Some(p) => p,
                            None => return true, // unresolvable → conservative
                        };
                        if self.analyze_can_raise_impl(&callee_path, seen, ignore_memoryerror) {
                            return true;
                        }
                    }
                    other => {
                        if op_can_raise(other, ignore_memoryerror) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// RPython: VirtualizableAnalyzer.analyze() (effectinfo.py:401-404).
    ///
    /// analyze_simple_operation: op.opname in ('jit_force_virtualizable',
    ///                                         'jit_force_virtual')
    fn analyze_forces_virtualizable(&self, path: &CallPath, seen: &mut HashSet<CallPath>) -> bool {
        if !seen.insert(path.clone()) {
            return false;
        }
        let graph = match self.function_graphs.get(path) {
            Some(g) => g,
            // RPython: external call → analyze_external_call → bottom_result (False).
            // VirtualizableAnalyzer does not override analyze_external_call.
            None => return false,
        };
        for block in &graph.blocks {
            for op in &block.operations {
                match &op.kind {
                    // RPython: jit_force_virtualizable / jit_force_virtual
                    OpKind::VableForce => return true,
                    OpKind::Call { target, .. } => {
                        let callee_path = match self.target_to_path(target) {
                            Some(p) => p,
                            None => continue, // external call → False
                        };
                        if self.analyze_forces_virtualizable(&callee_path, seen) {
                            return true;
                        }
                    }
                    _ => {}
                }
            }
        }
        false
    }

    /// RPython: RandomEffectsAnalyzer.analyze() (effectinfo.py:410-418).
    ///
    /// ```python
    /// class RandomEffectsAnalyzer(BoolGraphAnalyzer):
    ///     def analyze_external_call(self, funcobj, seen=None):
    ///         if funcobj.random_effects_on_gcobjs:
    ///             return True
    ///         return super().analyze_external_call(funcobj, seen)
    ///     def analyze_simple_operation(self, op, graphinfo):
    ///         return False
    /// ```
    ///
    /// Key: `analyze_simple_operation` always returns False. External calls
    /// only return True if `random_effects_on_gcobjs` is set. The default
    /// `analyze_external_call` returns `bottom_result()` = False
    /// (graphanalyze.py:60-69). "No graph" ≠ random effects in RPython.
    ///
    /// In majit: functions without graphs are external calls — returns
    /// True if in `external_gc_effects`, False otherwise.
    fn analyze_random_effects(&self, path: &CallPath, seen: &mut HashSet<CallPath>) -> bool {
        if !seen.insert(path.clone()) {
            return false; // cycle → bottom_result
        }
        let graph = match self.function_graphs.get(path) {
            Some(g) => g,
            None => {
                // RPython: analyze_external_call → bottom_result (False)
                // unless funcobj.random_effects_on_gcobjs → True.
                return self.external_gc_effects.contains(path);
            }
        };
        // RPython: analyze_simple_operation always returns False.
        // Only recursive calls into graphs can propagate random effects.
        for block in &graph.blocks {
            for op in &block.operations {
                if let OpKind::Call { target, .. } = &op.kind {
                    let callee_path = match self.target_to_path(target) {
                        Some(p) => p,
                        // Unresolvable target = external call → False
                        None => continue,
                    };
                    if self.analyze_random_effects(&callee_path, seen) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// RPython: QuasiImmutAnalyzer.analyze() (effectinfo.py).
    ///
    /// analyze_simple_operation: op.opname == 'jit_force_quasi_immutable'.
    ///
    /// In majit: we don't have quasi-immutable ops in the model yet,
    /// so this always returns false. The transitive call check is still
    /// performed for future-proofing.
    fn analyze_can_invalidate(&self, path: &CallPath, seen: &mut HashSet<CallPath>) -> bool {
        if !seen.insert(path.clone()) {
            return false;
        }
        let graph = match self.function_graphs.get(path) {
            Some(g) => g,
            None => return false, // no graph → cannot invalidate (not conservative here)
        };
        for block in &graph.blocks {
            for op in &block.operations {
                // RPython: jit_force_quasi_immutable → true
                // majit: no such op yet, but check calls transitively
                if let OpKind::Call { target, .. } = &op.kind {
                    let callee_path = match self.target_to_path(target) {
                        Some(p) => p,
                        None => continue,
                    };
                    if self.analyze_can_invalidate(&callee_path, seen) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// RPython: CollectAnalyzer (collectanalyze.py).
    ///
    /// RPython: CollectAnalyzer.analyze_direct_call(graph, seen)
    /// (collectanalyze.py + graphanalyze.py:139).
    ///
    /// Traverses graph ops with:
    /// - analyze_simple_operation (collectanalyze.py:27-33): checks malloc/
    ///   malloc_varsize with GC flavor, LL_OPERATIONS[op].canmallocgc.
    ///   In majit the codewriter graph has no LL_OPERATIONS; allocations are
    ///   only reachable transitively through calls.
    /// - analyze_direct_call: recurse into callee graphs.
    /// - analyze_external_call (graphanalyze.py:60): bottom_result() (False).
    /// - _gctransformer_hint_cannot_collect_ (collectanalyze.py:15-16):
    ///   functions in `cannot_collect_targets` are known not to collect.
    fn analyze_can_collect(&self, path: &CallPath, seen: &mut HashSet<CallPath>) -> bool {
        if !seen.insert(path.clone()) {
            return false;
        }
        // collectanalyze.py:15: _gctransformer_hint_cannot_collect_ → False
        if self.cannot_collect_targets.contains(path) {
            return false;
        }
        // collectanalyze.py:15: _gctransformer_hint_close_stack_ → True.
        // close_stack functions always can collect.
        if self.close_stack_targets.contains(path) {
            return true;
        }
        let graph = match self.function_graphs.get(path) {
            Some(g) => g,
            None => {
                // collectanalyze.py:21-25: analyze_external_call —
                // if funcobj.random_effects_on_gcobjs → True,
                // else → bottom_result() (False).
                return self.external_gc_effects.contains(path);
            }
        };
        for block in &graph.blocks {
            for op in &block.operations {
                // collectanalyze.py:27-33: analyze_simple_operation
                // RPython checks: malloc/malloc_varsize with flavor='gc' → True
                //                 LL_OPERATIONS[op.opname].canmallocgc → True
                // majit codewriter graphs have no LL_OPERATIONS; the only
                // operations that can trigger GC are transitive through calls.
                // (All other OpKind variants are pure/field/array ops.)
                if let OpKind::Call { target, .. } = &op.kind {
                    // graphanalyze.py:139-164: analyze_direct_call — recurse
                    let callee_path = match self.target_to_path(target) {
                        Some(p) => p,
                        // graphanalyze.py:60: external call → bottom_result (False)
                        None => continue,
                    };
                    if self.analyze_can_collect(&callee_path, seen) {
                        return true;
                    }
                }
            }
        }
        false
    }

    // ── Cached analyzer wrappers ────────────────────────────────────

    /// Cached version of _canraise for a CallTarget.
    ///
    /// RPython call.py:337-355 — _canraise uses two analyzers:
    /// 1. raise_analyzer.can_raise(op)
    /// 2. raise_analyzer_ignore_memoryerror.can_raise(op)
    ///
    /// If (1) is True and (2) is False → "mem" (MemoryErrorOnly).
    /// If (1) is True and (2) is True → True.
    /// If (1) is False → False.
    fn cached_can_raise(&self, target: &CallTarget, cache: &mut AnalysisCache) -> CanRaise {
        let path = match self.target_to_path(target) {
            Some(p) => p,
            None => return CanRaise::Yes,
        };
        if let Some(&result) = cache.can_raise.get(&path) {
            return result;
        }
        // RPython call.py:342: self.raise_analyzer.can_raise(op)
        let mut seen = HashSet::new();
        let can_raise = self.analyze_can_raise_impl(&path, &mut seen, false);
        let result = if !can_raise {
            // RPython call.py:348: return False
            CanRaise::No
        } else {
            // RPython call.py:343: self.raise_analyzer_ignore_memoryerror.can_raise(op)
            let mut seen2 = HashSet::new();
            let can_raise_non_memoryerror = self.analyze_can_raise_impl(&path, &mut seen2, true);
            if can_raise_non_memoryerror {
                // RPython call.py:344: return True
                CanRaise::Yes
            } else {
                // RPython: return "mem"
                CanRaise::MemoryErrorOnly
            }
        };
        cache.can_raise.insert(path, result);
        result
    }

    /// Cached version of analyze_forces_virtualizable for a CallTarget.
    /// RPython: VirtualizableAnalyzer external calls → bottom_result (False).
    fn cached_forces_virtualizable(&self, target: &CallTarget, cache: &mut AnalysisCache) -> bool {
        let path = match self.target_to_path(target) {
            Some(p) => p,
            None => return false, // external → False (RPython bottom_result)
        };
        if let Some(&result) = cache.forces_virtualizable.get(&path) {
            return result;
        }
        let mut seen = HashSet::new();
        let result = self.analyze_forces_virtualizable(&path, &mut seen);
        cache.forces_virtualizable.insert(path, result);
        result
    }

    /// Cached version of analyze_random_effects for a CallTarget.
    /// RPython: RandomEffectsAnalyzer defaults to False for external calls.
    fn cached_random_effects(&self, target: &CallTarget, cache: &mut AnalysisCache) -> bool {
        let path = match self.target_to_path(target) {
            Some(p) => p,
            None => return false, // external call → False (RPython default)
        };
        if let Some(&result) = cache.random_effects.get(&path) {
            return result;
        }
        let mut seen = HashSet::new();
        let result = self.analyze_random_effects(&path, &mut seen);
        cache.random_effects.insert(path, result);
        result
    }

    /// Cached version of analyze_can_invalidate for a CallTarget.
    fn cached_can_invalidate(&self, target: &CallTarget, cache: &mut AnalysisCache) -> bool {
        let path = match self.target_to_path(target) {
            Some(p) => p,
            None => return false,
        };
        if let Some(&result) = cache.can_invalidate.get(&path) {
            return result;
        }
        let mut seen = HashSet::new();
        let result = self.analyze_can_invalidate(&path, &mut seen);
        cache.can_invalidate.insert(path, result);
        result
    }

    /// Cached version of analyze_can_collect for a CallTarget.
    /// RPython: collect_analyzer.analyze(op, self.seen_gc) (collectanalyze.py).
    /// graphanalyze.py:60: analyze_external_call → bottom_result() (False).
    fn cached_can_collect(&self, target: &CallTarget, cache: &mut AnalysisCache) -> bool {
        let path = match self.target_to_path(target) {
            Some(p) => p,
            // graphanalyze.py:60: analyze_external_call → bottom_result() (False)
            None => return false,
        };
        if let Some(&result) = cache.can_collect.get(&path) {
            return result;
        }
        let mut seen = HashSet::new();
        let result = self.analyze_can_collect(&path, &mut seen);
        cache.can_collect.insert(path, result);
        result
    }

    // ── _canraise + getcalldescr (call.py:210-355) ──────────────────

    /// RPython: CallControl._canraise(op) (call.py:337-355).
    ///
    /// ```python
    /// def _canraise(self, op):
    ///     if op.opname == 'pseudo_call_cannot_raise':
    ///         return False
    ///     try:
    ///         if self.raise_analyzer.can_raise(op):
    ///             if self.raise_analyzer_ignore_memoryerror.can_raise(op):
    ///                 return True
    ///             else:
    ///                 return "mem"
    ///         else:
    ///             return False
    ///     except DelayedPointer:
    ///         return True
    /// ```
    pub fn canraise(&self, target: &CallTarget, cache: &mut AnalysisCache) -> CanRaise {
        self.cached_can_raise(target, cache)
    }

    /// RPython: CallControl.getcalldescr(op, oopspecindex, extraeffect, ...)
    /// (call.py:210-335).
    ///
    /// Determines the effect classification for a call target by running
    /// graph-based analyzers, then builds and returns a CallDescriptor
    /// with the computed EffectInfo.
    ///
    /// ```python
    /// def getcalldescr(self, op, oopspecindex=OS_NONE,
    ///                  extraeffect=None, extradescr=None, calling_graph=None):
    ///     ...
    ///     random_effects = self.randomeffects_analyzer.analyze(op)
    ///     if random_effects:
    ///         extraeffect = EF_RANDOM_EFFECTS
    ///     can_invalidate = random_effects or self.quasiimmut_analyzer.analyze(op)
    ///     if extraeffect is None:
    ///         if self.virtualizable_analyzer.analyze(op):
    ///             extraeffect = EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE
    ///         elif loopinvariant:
    ///             extraeffect = EF_LOOPINVARIANT
    ///         elif elidable:
    ///             cr = self._canraise(op)
    ///             ...
    ///         elif self._canraise(op):
    ///             extraeffect = EF_CAN_RAISE
    ///         else:
    ///             extraeffect = EF_CANNOT_RAISE
    ///     ...
    ///     effectinfo = effectinfo_from_writeanalyze(...)
    ///     return self.cpu.calldescrof(FUNC, NON_VOID_ARGS, RESULT, effectinfo)
    /// ```
    pub fn getcalldescr(
        &self,
        target: &CallTarget,
        arg_types: Vec<Type>,
        result_type: Type,
        oopspecindex: OopSpecIndex,
        extraeffect: Option<ExtraEffect>,
        cache: &mut AnalysisCache,
    ) -> CallDescriptor {
        // RPython call.py:239-240: extract flags
        let elidable = self.is_elidable(target);
        let loopinvariant = self.is_loopinvariant(target);

        // RPython call.py:223-234: check the number and type of arguments.
        //   FUNC = op.args[0].concretetype.TO
        //   if NON_VOID_ARGS != [T for T in FUNC.ARGS if T is not Void]:
        //       raise Exception(...)
        //   if RESULT != FUNC.RESULT:
        //       raise Exception(...)
        //
        // In majit, we validate by looking up the callee's FunctionGraph inputs.
        // RPython raises on mismatch; we warn (callee signatures are approximate
        // in static analysis, so hard errors would be too strict).
        if !arg_types.is_empty() {
            if let Some(path) = self.target_to_path(target) {
                if let Some(graph) = self.function_graphs.get(&path) {
                    let expected_arity = graph.block(graph.startblock).inputargs.len();
                    if arg_types.len() != expected_arity {
                        eprintln!(
                            "[getcalldescr] WARNING: {target} expects {expected_arity} args \
                             but got {} (NON_VOID_ARGS mismatch)",
                            arg_types.len()
                        );
                    }
                }
            }
        }

        // RPython call.py:282: random_effects = self.randomeffects_analyzer.analyze(op)
        let random_effects = self.cached_random_effects(target, cache);
        let mut extraeffect = extraeffect;
        if random_effects {
            extraeffect = Some(ExtraEffect::RandomEffects);
        }

        // RPython call.py:285: can_invalidate = random_effects or quasiimmut_analyzer
        let can_invalidate = random_effects || self.cached_can_invalidate(target, cache);

        // RPython call.py:286-303: determine extraeffect
        if extraeffect.is_none() {
            extraeffect = Some(if self.cached_forces_virtualizable(target, cache) {
                // call.py:288
                ExtraEffect::ForcesVirtualOrVirtualizable
            } else if loopinvariant {
                // call.py:290
                ExtraEffect::LoopInvariant
            } else if elidable {
                // call.py:292-298
                match self.canraise(target, cache) {
                    CanRaise::No => ExtraEffect::ElidableCannotRaise,
                    CanRaise::MemoryErrorOnly => ExtraEffect::ElidableOrMemoryError,
                    CanRaise::Yes => ExtraEffect::ElidableCanRaise,
                }
            } else if matches!(
                self.canraise(target, cache),
                CanRaise::Yes | CanRaise::MemoryErrorOnly
            ) {
                // call.py:299-300
                ExtraEffect::CanRaise
            } else {
                // call.py:302
                ExtraEffect::CannotRaise
            });
        }

        let extraeffect = extraeffect.unwrap_or(ExtraEffect::CanRaise);

        // RPython call.py:249-251: loopinvariant functions must have no args
        if loopinvariant && !arg_types.is_empty() {
            panic!(
                "getcalldescr: arguments not supported for loop-invariant \
                 function {target}"
            );
        }

        // RPython call.py:305-318: check that the result is really as expected
        if loopinvariant && extraeffect != ExtraEffect::LoopInvariant {
            panic!(
                "getcalldescr: {target} is marked loop-invariant but got \
                 extraeffect={extraeffect:?}"
            );
        }
        if elidable {
            if !matches!(
                extraeffect,
                ExtraEffect::ElidableCannotRaise
                    | ExtraEffect::ElidableOrMemoryError
                    | ExtraEffect::ElidableCanRaise
            ) {
                panic!(
                    "getcalldescr: {target} is marked elidable but got \
                     extraeffect={extraeffect:?}"
                );
            }
            // RPython call.py:315-318: elidable function must have a result
            if result_type == Type::Void {
                panic!("getcalldescr: {target} is elidable but has no result");
            }
        }

        // RPython call.py:320-324:
        // effectinfo = effectinfo_from_writeanalyze(
        //     self.readwrite_analyzer.analyze(op, self.seen_rw),
        //     self.cpu, extraeffect, oopspecindex, can_invalidate,
        //     call_release_gil_target, extradescr,
        //     self.collect_analyzer.analyze(op, self.seen_gc))
        let effects = analyze_readwrite(
            target,
            &self.function_graphs,
            self,
            &mut cache.descr_indices,
        );
        let can_collect = self.cached_can_collect(target, cache);
        let effectinfo = effectinfo_from_writeanalyze(
            effects,
            extraeffect,
            oopspecindex,
            can_invalidate,
            can_collect,
        );

        // RPython call.py:326-332: assert post-conditions
        if elidable || loopinvariant {
            assert!(
                effectinfo.extra_effect < ExtraEffect::ForcesVirtualOrVirtualizable,
                "getcalldescr: elidable/loopinvariant {target} has \
                 effect {:?} >= ForcesVirtualOrVirtualizable",
                effectinfo.extra_effect
            );
        }

        // RPython call.py:334-335: cpu.calldescrof(FUNC, NON_VOID_ARGS, RESULT, effectinfo)
        CallDescriptor {
            target: target.clone(),
            effect_info: effectinfo,
        }
    }

    /// RPython: calldescr_canraise(calldescr) (call.py:357-359).
    pub fn calldescr_canraise(&self, calldescr: &CallDescriptor) -> bool {
        calldescr.effect_info.check_can_raise(false)
    }
}

impl Default for CallControl {
    fn default() -> Self {
        Self::new()
    }
}

// ── readwrite_analyzer / collect_analyzer (effectinfo.py:276-378) ──
//
// RPython: self.readwrite_analyzer.analyze(op, self.seen_rw) → effects
// RPython: self.collect_analyzer.analyze(op, self.seen_gc) → can_collect
// Then: effectinfo_from_writeanalyze(effects, cpu, ..., can_collect)

/// RPython: readwrite_analyzer.analyze(op, self.seen_rw).
///
/// Traverses the call graph to collect read/write effects as a WriteAnalysis.
/// This is the Rust equivalent of RPython's ReadWriteAnalyzer producing a
/// set of ("struct"/"array"/"interiorfield", T, fieldname) tuples.
fn analyze_readwrite(
    target: &CallTarget,
    function_graphs: &HashMap<CallPath, FunctionGraph>,
    cc: &CallControl,
    descr_indices: &mut DescrIndexRegistry,
) -> WriteAnalysis {
    let mut analysis = WriteAnalysis {
        read_fields: 0,
        write_fields: 0,
        read_arrays: 0,
        write_arrays: 0,
        read_interiorfields: 0,
        write_interiorfields: 0,
        array_write_descrs: Vec::new(),
        is_top: false,
    };
    if let Some(path) = cc.target_to_path(target) {
        let mut seen = HashSet::new();
        collect_readwrite_effects(
            &path,
            function_graphs,
            cc,
            descr_indices,
            &mut seen,
            &mut analysis.read_fields,
            &mut analysis.write_fields,
            &mut analysis.read_arrays,
            &mut analysis.write_arrays,
            &mut analysis.read_interiorfields,
            &mut analysis.write_interiorfields,
            &mut analysis.array_write_descrs,
        );
        // RPython: top_set only occurs from gc_add_memory_pressure (writeanalyze.py:72).
        // External calls return empty_set (bottom_result), not top_set.
        // We currently don't have gc_add_memory_pressure, so is_top stays false.
    }
    analysis
}

//
// In RPython, the ReadWriteAnalyzer produces a set of tuples like:
//   ("struct", T, fieldname), ("readstruct", T, fieldname),
//   ("array", T), ("readarray", T), etc.
// These are converted to field/array descriptor bitsets.
//
// In majit, we scan the callee graph's ops directly for
// FieldRead/FieldWrite/ArrayRead/ArrayWrite and collect their
// descriptor indices into EffectInfo's bitset fields.

/// RPython: effectinfo_from_writeanalyze() (effectinfo.py:276-378).
///
/// Scans the callee's graph for field/array read/write operations
/// and populates the corresponding bitset fields in EffectInfo.
/// RPython: effectinfo_from_writeanalyze(effects, cpu, extraeffect, oopspecindex,
///     can_invalidate, call_release_gil_target, extradescr, can_collect)
/// effectinfo.py:276-378.
///
/// Takes pre-analyzed `effects` (from readwrite_analyzer) and `can_collect`
/// (from collect_analyzer) and constructs an EffectInfo.
fn effectinfo_from_writeanalyze(
    effects: WriteAnalysis,
    extraeffect: ExtraEffect,
    oopspecindex: OopSpecIndex,
    can_invalidate: bool,
    can_collect: bool,
) -> EffectInfo {
    // effectinfo.py:285: if effects is top_set or extraeffect == EF_RANDOM_EFFECTS:
    if effects.is_top || extraeffect == ExtraEffect::RandomEffects {
        return EffectInfo {
            extra_effect: ExtraEffect::RandomEffects,
            oopspec_index: oopspecindex,
            readonly_descrs_fields: !0, // all bits set = top_set (None in RPython)
            write_descrs_fields: !0,
            readonly_descrs_arrays: !0,
            write_descrs_arrays: !0,
            readonly_descrs_interiorfields: !0,
            write_descrs_interiorfields: !0,
            single_write_descr_array: None,
            can_invalidate,
            can_collect: true, // effectinfo.py:364-365: forces → can_collect = True
        };
    }

    // effectinfo.py:345-360: readonly = reads that have NO corresponding write.
    let readonly_descrs_fields = effects.read_fields & !effects.write_fields;
    let readonly_descrs_arrays = effects.read_arrays & !effects.write_arrays;
    let readonly_descrs_interiorfields =
        effects.read_interiorfields & !effects.write_interiorfields;

    let mut write_descrs_fields = effects.write_fields;
    let mut write_descrs_arrays = effects.write_arrays;
    let mut write_descrs_interiorfields = effects.write_interiorfields;
    let mut array_write_descrs = effects.array_write_descrs;

    // effectinfo.py:169-181: for elidable/loopinvariant, ignore writes.
    if matches!(
        extraeffect,
        ExtraEffect::ElidableCannotRaise
            | ExtraEffect::ElidableOrMemoryError
            | ExtraEffect::ElidableCanRaise
            | ExtraEffect::LoopInvariant
    ) {
        write_descrs_fields = 0;
        write_descrs_arrays = 0;
        write_descrs_interiorfields = 0;
        array_write_descrs.clear();
    }

    // effectinfo.py:201-206: single_write_descr_array
    let single_write_descr_array = if array_write_descrs.len() == 1 {
        Some(array_write_descrs.into_iter().next().unwrap())
    } else {
        None
    };

    // effectinfo.py:364-365: if extraeffect >= EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE:
    //     can_collect = True
    let can_collect = if extraeffect >= ExtraEffect::ForcesVirtualOrVirtualizable {
        true
    } else {
        can_collect
    };

    EffectInfo {
        extra_effect: extraeffect,
        oopspec_index: oopspecindex,
        readonly_descrs_fields,
        write_descrs_fields,
        readonly_descrs_arrays,
        write_descrs_arrays,
        readonly_descrs_interiorfields,
        write_descrs_interiorfields,
        single_write_descr_array,
        can_invalidate,
        can_collect,
    }
}

/// RPython: `op.args[0].concretetype` — resolve full ARRAY identity.
///
/// Returns the full ARRAY type string (e.g. `"Vec<Point>"`, `"Vec<i64>"`),
/// matching RPython's ARRAY lltype which is the cache key for
/// `cpu.arraydescrof(ARRAY)` (descr.py:348-351).
///
/// Resolution order:
/// 1. Parser-set `array_type_id` (full container type from variable decl)
/// 2. Producer chain trace-back for `op.args[0].concretetype`:
///    - FieldRead: field type from struct_fields (full type string)
///    - ArrayRead: propagate the array's own array_type_id
///    - Call: return type from CallControl.return_types
/// 3. Phi/link source chain (limited depth)
/// 4. None (conservative: falls back to item_ty-only keying)
fn resolve_array_identity(
    base: &crate::model::ValueId,
    op_array_type_id: &Option<String>,
    value_producers: &HashMap<crate::model::ValueId, &crate::model::OpKind>,
    phi_sources: &HashMap<crate::model::ValueId, crate::model::ValueId>,
    cc: &CallControl,
) -> Option<String> {
    // 1. Parser-set element type (from FnArg or typed let binding).
    if op_array_type_id.is_some() {
        return op_array_type_id.clone();
    }
    // 2. Trace back to producer — RPython: op.args[0].concretetype.
    if let Some(producer) = value_producers.get(base) {
        match producer {
            // FieldRead: self.array → full ARRAY type from struct registry.
            // RPython: op.args[0].concretetype is the ARRAY lltype directly.
            OpKind::FieldRead { field, .. } => {
                if let Some(owner) = &field.owner_root {
                    if let Some(field_type_str) = cc.field_type(owner, &field.name) {
                        return Some(field_type_str.to_string());
                    }
                }
            }
            // ArrayRead with known array_type_id: propagate.
            OpKind::ArrayRead { array_type_id, .. } => {
                if array_type_id.is_some() {
                    return array_type_id.clone();
                }
            }
            // Call result: RPython resolves via result.concretetype → full type.
            OpKind::Call { target, .. } => {
                if let Some(callee_path) = cc.target_to_path(target) {
                    if let Some(ret_type) = cc.return_types.get(&callee_path) {
                        return Some(ret_type.clone());
                    }
                }
            }
            OpKind::Input { .. } => {}
            _ => {}
        }
    }
    // 3. Phi/link: RPython concretetype propagates through block boundaries.
    // Follow inputarg → source value chain (limited depth to avoid cycles).
    let mut vid = *base;
    for _ in 0..4 {
        if let Some(&src) = phi_sources.get(&vid) {
            // Check if the source has a producer with array identity.
            if let Some(producer) = value_producers.get(&src) {
                match producer {
                    OpKind::FieldRead { field, .. } => {
                        if let Some(owner) = &field.owner_root {
                            if let Some(fts) = cc.field_type(owner, &field.name) {
                                return Some(fts.to_string());
                            }
                        }
                    }
                    OpKind::ArrayRead { array_type_id, .. } if array_type_id.is_some() => {
                        return array_type_id.clone();
                    }
                    OpKind::Call { target, .. } => {
                        if let Some(cp) = cc.target_to_path(target) {
                            if let Some(rt) = cc.return_types.get(&cp) {
                                return Some(rt.clone());
                            }
                        }
                    }
                    _ => {}
                }
            }
            vid = src;
        } else {
            break;
        }
    }
    None
}

/// RPython: `ARRAY.OF` — extract element type from full ARRAY type string.
///
/// Handles all Rust array/container notations:
/// - `Vec<Point>` → `"Point"` (angle brackets)
/// - `[i64]` → `"i64"` (slice)
/// - `[Point; 10]` → `"Point"` (fixed-size array)
fn extract_element_type_from_str(type_str: &str) -> Option<String> {
    let s = type_str.trim();
    // Angle brackets: Vec<T>, Box<T>, etc.
    if let (Some(start), Some(end)) = (s.find('<'), s.rfind('>')) {
        if start < end {
            return Some(s[start + 1..end].trim().to_string());
        }
    }
    // Square brackets: [T] or [T; N]
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len() - 1];
        let elem = if let Some(semi) = inner.find(';') {
            inner[..semi].trim()
        } else {
            inner.trim()
        };
        if !elem.is_empty() {
            return Some(elem.to_string());
        }
    }
    None
}

/// Transitive read/write effect collection.
///
/// RPython: ReadWriteAnalyzer.analyze() — traverses callee graphs.
/// Produces a set of tuples: ("struct"/"readstruct"/"array"/"readarray", ...).
///
/// We collect raw reads and writes separately into bitsets. The caller
/// (`effectinfo_from_writeanalyze`) then applies the RPython rule:
/// "readonly = reads & ~writes" (effectinfo.py:345-360).
fn collect_readwrite_effects(
    path: &CallPath,
    function_graphs: &HashMap<CallPath, FunctionGraph>,
    cc: &CallControl,
    descr_indices: &mut DescrIndexRegistry,
    seen: &mut HashSet<CallPath>,
    read_fields: &mut u64,
    write_fields: &mut u64,
    read_arrays: &mut u64,
    write_arrays: &mut u64,
    // effectinfo.py:313-325: interiorfield descriptor bitsets.
    read_interiorfields: &mut u64,
    write_interiorfields: &mut u64,
    // effectinfo.py:201-206: collect actual array write DescrRefs
    // for single_write_descr_array population.
    array_write_descrs: &mut Vec<majit_ir::descr::DescrRef>,
) {
    if !seen.insert(path.clone()) {
        return;
    }
    let graph = match function_graphs.get(path) {
        Some(g) => g,
        None => {
            // RPython: analyze_external_call() returns bottom_result() (empty_set),
            // NOT top_set. External calls have no KNOWN read/write effects.
            // The extraeffect (CanRaise etc.) is determined separately.
            return;
        }
    };

    // RPython: the rtyped graph gives op.args[0].concretetype directly.
    // In majit, we build a producer map to resolve ValueId → producing OpKind,
    // so we can determine the array identity from the base operand's provenance.
    let value_producers: HashMap<crate::model::ValueId, &crate::model::OpKind> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| op.result.map(|vid| (vid, &op.kind)))
        .collect();

    // RPython: phi/link args carry concretetype through block boundaries.
    // Build inputarg → source value mapping from Goto/Cond terminators so
    // resolve_array_identity can trace through control-flow merges.
    let mut phi_sources: HashMap<crate::model::ValueId, crate::model::ValueId> = HashMap::new();
    for block in &graph.blocks {
        match &block.terminator {
            crate::model::Terminator::Goto { target, args } => {
                if let Some(target_block) = graph.blocks.get(target.0) {
                    for (ia, src) in target_block.inputargs.iter().zip(args.iter()) {
                        phi_sources.insert(*ia, *src);
                    }
                }
            }
            crate::model::Terminator::Branch {
                if_true,
                true_args,
                if_false,
                false_args,
                ..
            } => {
                if let Some(tb) = graph.blocks.get(if_true.0) {
                    for (ia, src) in tb.inputargs.iter().zip(true_args.iter()) {
                        phi_sources.insert(*ia, *src);
                    }
                }
                if let Some(fb) = graph.blocks.get(if_false.0) {
                    for (ia, src) in fb.inputargs.iter().zip(false_args.iter()) {
                        phi_sources.insert(*ia, *src);
                    }
                }
            }
            _ => {}
        }
    }

    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                // RPython: ("readstruct", T, fieldname)
                OpKind::FieldRead { field, .. } => {
                    // RPython: cpu.fielddescrof(T, fieldname).get_ei_index()
                    let idx = descr_indices.field_index(&field.owner_root, &field.name);
                    *read_fields |= 1u64 << idx;
                }
                // RPython: ("struct", T, fieldname)
                OpKind::FieldWrite { field, .. } => {
                    let idx = descr_indices.field_index(&field.owner_root, &field.name);
                    *write_fields |= 1u64 << idx;
                }
                // RPython: ("readarray", T)
                OpKind::ArrayRead {
                    base,
                    item_ty,
                    array_type_id,
                    ..
                } => {
                    // RPython: op.args[0].concretetype → cpu.arraydescrof(ARRAY)
                    let resolved_id = resolve_array_identity(
                        base,
                        array_type_id,
                        &value_producers,
                        &phi_sources,
                        cc,
                    );
                    let idx =
                        descr_indices.array_index(value_type_discriminant(item_ty), &resolved_id);
                    *read_arrays |= 1u64 << idx;
                }
                // RPython: ("array", T)
                OpKind::ArrayWrite {
                    base,
                    item_ty,
                    array_type_id,
                    ..
                } => {
                    let resolved_id = resolve_array_identity(
                        base,
                        array_type_id,
                        &value_producers,
                        &phi_sources,
                        cc,
                    );
                    let idx =
                        descr_indices.array_index(value_type_discriminant(item_ty), &resolved_id);
                    *write_arrays |= 1u64 << idx;
                    // RPython: effectinfo.py:307-311 — cpu.arraydescrof(ARRAY).
                    // Dedup by descriptor index (frozenset semantics).
                    if !array_write_descrs.iter().any(|d| d.index() == idx) {
                        let ir_type = match item_ty {
                            crate::model::ValueType::Int | crate::model::ValueType::State => {
                                majit_ir::value::Type::Int
                            }
                            crate::model::ValueType::Ref | crate::model::ValueType::Unknown => {
                                majit_ir::value::Type::Ref
                            }
                            crate::model::ValueType::Float => majit_ir::value::Type::Float,
                            crate::model::ValueType::Void => majit_ir::value::Type::Void,
                        };
                        array_write_descrs.push(cc.arraydescrof(idx, &resolved_id, ir_type));
                    }
                }
                // RPython: ("readinteriorfield", T, fieldname)
                // effectinfo.py:351-354: records interiorfield descriptor.
                // effectinfo.py:327-340: ALSO implicitly records array read.
                OpKind::InteriorFieldRead {
                    base,
                    field,
                    array_type_id,
                    ..
                } => {
                    let resolved_id = resolve_array_identity(
                        base,
                        array_type_id,
                        &value_producers,
                        &phi_sources,
                        cc,
                    );
                    // Interior field bit — keyed on (ARRAY, fieldname),
                    // matching cpu.interiorfielddescrof(ARRAY, fieldname).
                    let ifield_idx = descr_indices.interiorfield_index(&resolved_id, &field.name);
                    *read_interiorfields |= 1u64 << ifield_idx;
                    // effectinfo.py:327-340: implicit array read.
                    // RPython: cpu.arraydescrof(ARRAY) uses get_type_flag(ARRAY.OF).
                    // Interior fields only exist in struct arrays → element type is Ref.
                    let arr_idx = descr_indices.array_index(
                        value_type_discriminant(&crate::model::ValueType::Ref),
                        &resolved_id,
                    );
                    *read_arrays |= 1u64 << arr_idx;
                }
                // RPython: ("interiorfield", T, fieldname)
                // effectinfo.py:349-350: records interiorfield descriptor.
                // effectinfo.py:327-340: ALSO implicitly records array write.
                OpKind::InteriorFieldWrite {
                    base,
                    field,
                    array_type_id,
                    ..
                } => {
                    let resolved_id = resolve_array_identity(
                        base,
                        array_type_id,
                        &value_producers,
                        &phi_sources,
                        cc,
                    );
                    // Interior field bit — keyed on (ARRAY, fieldname),
                    // matching cpu.interiorfielddescrof(ARRAY, fieldname).
                    let ifield_idx = descr_indices.interiorfield_index(&resolved_id, &field.name);
                    *write_interiorfields |= 1u64 << ifield_idx;
                    // effectinfo.py:327-340: implicit array write.
                    // RPython: cpu.arraydescrof(ARRAY) — struct arrays are always Ref.
                    let arr_idx = descr_indices.array_index(
                        value_type_discriminant(&crate::model::ValueType::Ref),
                        &resolved_id,
                    );
                    *write_arrays |= 1u64 << arr_idx;
                }
                // Recursive: follow calls.
                OpKind::Call { target, .. } => {
                    if let Some(callee_path) = cc.target_to_path(target) {
                        collect_readwrite_effects(
                            &callee_path,
                            function_graphs,
                            cc,
                            descr_indices,
                            seen,
                            read_fields,
                            write_fields,
                            read_arrays,
                            write_arrays,
                            read_interiorfields,
                            write_interiorfields,
                            array_write_descrs,
                        );
                    } else {
                        // RPython: analyze_external_call() → bottom_result() (empty_set).
                        // External calls have no known read/write effects.
                        // (NOT top_set — that only comes from gc_add_memory_pressure.)
                    }
                }
                _ => {}
            }
        }
    }
}

/// RPython: `heaptracker.all_interiorfielddescrs(gccache, ARRAY)`.
///
/// For an array-of-structs, iterate `STRUCT._names` and create
/// `InteriorFieldDescr(arraydescr, fielddescr)` for each field.
/// Mirrors heaptracker.py:74-92 with `get_field_descr=get_interiorfield_descr`.
///
/// Layout source priority (RPython: `symbolic.get_field_token()`):
/// 1. `cc.struct_layouts[struct_name]` — actual layout from runtime
/// 2. Type-string heuristic fallback from `get_type_flag()`
///
/// Returns `(fielddescrs, item_size)`.
fn all_interiorfielddescrs(
    cc: &CallControl,
    struct_name: &str,
    array_descr: std::sync::Arc<majit_ir::descr::SimpleArrayDescr>,
) -> (Vec<majit_ir::descr::DescrRef>, usize) {
    // Path 1: actual layout from runtime (RPython: symbolic.get_field_token)
    if let Some(layout) = cc.struct_layouts.get(struct_name) {
        let mut result = Vec::new();
        // RPython: heaptracker.get_fielddescr_index_in() — counts non-Void fields.
        let mut index_in_parent: u32 = 0;
        for fl in &layout.fields {
            if fl.field_type == majit_ir::value::Type::Void {
                continue;
            }
            if fl.flag == majit_ir::descr::ArrayFlag::Struct {
                return (Vec::new(), 0);
            }
            let is_signed = fl.flag == majit_ir::descr::ArrayFlag::Signed;
            let fd = std::sync::Arc::new(majit_ir::descr::SimpleFieldDescr::new_with_name(
                index_in_parent,
                fl.offset,
                fl.size,
                fl.field_type,
                false,
                is_signed,
                format!("{}.{}", struct_name, fl.name),
            ));
            let ifd = majit_ir::descr::SimpleInteriorFieldDescr::new(
                index_in_parent,
                array_descr.clone(),
                fd,
            );
            result.push(std::sync::Arc::new(ifd) as majit_ir::descr::DescrRef);
            index_in_parent += 1;
        }
        return (result, layout.size);
    }

    // Path 2: type-string heuristic fallback
    let fields = match cc.struct_fields.fields.get(struct_name) {
        Some(f) => f,
        None => return (Vec::new(), 0),
    };
    for (_, field_type_str) in fields.iter() {
        if cc.is_known_struct(field_type_str) {
            return (Vec::new(), 0);
        }
    }
    let mut offset: usize = 0;
    let mut result = Vec::new();
    // RPython: heaptracker.get_fielddescr_index_in() — counts non-Void fields.
    let mut index_in_parent: u32 = 0;
    for (_i, (field_name, field_type_str)) in fields.iter().enumerate() {
        let (flag, field_type, field_size) = get_type_flag(field_type_str);
        if field_type == majit_ir::value::Type::Void {
            continue;
        }
        let align = field_size.min(std::mem::size_of::<usize>());
        if align > 0 {
            offset = (offset + align - 1) & !(align - 1);
        }
        let is_signed = flag == majit_ir::descr::ArrayFlag::Signed;
        let fd = std::sync::Arc::new(majit_ir::descr::SimpleFieldDescr::new_with_name(
            index_in_parent,
            offset,
            field_size,
            field_type,
            false,
            is_signed,
            format!("{}.{}", struct_name, field_name),
        ));
        let ifd = majit_ir::descr::SimpleInteriorFieldDescr::new(
            index_in_parent,
            array_descr.clone(),
            fd,
        );
        result.push(std::sync::Arc::new(ifd) as majit_ir::descr::DescrRef);
        offset += field_size;
        index_in_parent += 1;
    }
    let max_align = fields
        .iter()
        .map(|(_, ty)| get_type_flag(ty).2)
        .filter(|s| *s > 0)
        .max()
        .unwrap_or(8);
    let item_size = if offset > 0 {
        (offset + max_align - 1) & !(max_align - 1)
    } else {
        0
    };
    (result, item_size)
}

/// RPython: `symbolic.get_array_token(ARRAY, tsc)[1]` — struct item_size.
///
/// Layout source priority:
/// 1. `cc.struct_layouts[struct_name].size` — actual layout
/// 2. Type-string heuristic fallback
fn compute_struct_size(cc: &CallControl, struct_name: &str) -> usize {
    // Path 1: actual layout from runtime (RPython: symbolic.get_size(STRUCT))
    if let Some(layout) = cc.struct_layouts.get(struct_name) {
        return layout.size;
    }
    // Path 2: heuristic fallback — RPython: symbolic always computes the full
    // struct size, even with nested structs. Nested struct sizes are looked up
    // recursively from struct_layouts.
    let fields = match cc.struct_fields.fields.get(struct_name) {
        Some(f) => f,
        None => return 0,
    };
    let mut offset: usize = 0;
    for (_, field_type_str) in fields.iter() {
        let field_size = if cc.is_known_struct(field_type_str) {
            // RPython: symbolic.get_field_token() uses actual nested struct size.
            cc.struct_layouts
                .get(field_type_str.as_str())
                .map(|l| l.size)
                .unwrap_or(std::mem::size_of::<usize>())
        } else {
            let (_, field_type, s) = get_type_flag(field_type_str);
            if field_type == majit_ir::value::Type::Void || s == 0 {
                continue;
            }
            s
        };
        let align = field_size.min(std::mem::size_of::<usize>());
        offset = (offset + align - 1) & !(align - 1);
        offset += field_size;
    }
    let max_align = fields
        .iter()
        .map(|(_, ty)| {
            if cc.is_known_struct(ty) {
                cc.struct_layouts
                    .get(ty.as_str())
                    .map(|l| l.size)
                    .unwrap_or(std::mem::size_of::<usize>())
                    .min(std::mem::size_of::<usize>())
            } else {
                get_type_flag(ty).2
            }
        })
        .filter(|s| *s > 0)
        .max()
        .unwrap_or(8);
    if offset > 0 {
        (offset + max_align - 1) & !(max_align - 1)
    } else {
        0
    }
}

/// RPython: `get_type_flag(TYPE)` (descr.py:241-254).
///
/// Returns (ArrayFlag, IR type, size in bytes).
/// The ArrayFlag encodes both category AND signedness, matching RPython:
/// - Ptr(gc) → FLAG_POINTER; Ptr(non-gc) → FLAG_UNSIGNED
/// - Struct → FLAG_STRUCT; Float → FLAG_FLOAT
/// - Bool/unsigned → FLAG_UNSIGNED; signed int → FLAG_SIGNED
fn get_type_flag(type_str: &str) -> (majit_ir::descr::ArrayFlag, majit_ir::value::Type, usize) {
    use majit_ir::descr::ArrayFlag;
    match type_str {
        // RPython: isinstance(TYPE, lltype.Ptr) and TYPE.TO._gckind == 'gc' → FLAG_POINTER
        s if s.starts_with('&')
            || s.starts_with("Box<")
            || s.starts_with("Arc<")
            || s.starts_with("Rc<")
            || s.starts_with("Vec<")
            || s.starts_with("Option<")
            || s == "String" =>
        {
            (ArrayFlag::Pointer, majit_ir::value::Type::Ref, 8)
        }
        // RPython: TYPE is lltype.Float → FLAG_FLOAT
        "f64" => (ArrayFlag::Float, majit_ir::value::Type::Float, 8),
        "f32" => (ArrayFlag::Float, majit_ir::value::Type::Float, 4),
        // RPython: rffi.cast(TYPE, -1) == -1 → FLAG_SIGNED
        "i64" | "isize" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 8),
        "i32" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 4),
        "i16" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 2),
        "i8" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 1),
        // RPython: Bool → FLAG_UNSIGNED; unsigned number → FLAG_UNSIGNED
        "u64" | "usize" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 8),
        "u32" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 4),
        "u16" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 2),
        "u8" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 1),
        "bool" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 1),
        // RPython: Void fields are skipped
        "()" => (ArrayFlag::Void, majit_ir::value::Type::Void, 0),
        // Unknown type — treat as GC pointer (conservative)
        _ => (ArrayFlag::Pointer, majit_ir::value::Type::Ref, 8),
    }
}

/// RPython: `RaiseAnalyzer.analyze_simple_operation(op)` (canraise.py:14-17).
///
/// ```python
/// canraise = LL_OPERATIONS[op.opname].canraise
/// return bool(canraise) and canraise != (self.ignore_exact_class,)
/// ```
///
/// Returns true if the operation itself (not counting transitive calls)
/// can raise an exception. When `ignore_memoryerror` is true, operations
/// that can only raise MemoryError are treated as non-raising.
fn op_can_raise(op: &OpKind, ignore_memoryerror: bool) -> bool {
    // RPython canraise.py:14-17:
    //   canraise = LL_OPERATIONS[op.opname].canraise
    //   return bool(canraise) and canraise != (self.ignore_exact_class,)
    //
    // canraise.py:18: unknown op → log.WARNING + return True
    //
    // Each op has a canraise tuple. When ignore_exact_class == MemoryError,
    // ops that can ONLY raise MemoryError are treated as non-raising.
    match op {
        // ── Known non-raising ops (canraise = ()) ─────────────────
        // RPython LL: getfield_gc, setfield_gc → cannot raise
        OpKind::FieldRead { .. } | OpKind::FieldWrite { .. } => false,
        // RPython LL: getarrayitem_gc, setarrayitem_gc → cannot raise
        OpKind::ArrayRead { .. } | OpKind::ArrayWrite { .. } => false,
        // RPython LL: getinteriorfield_gc, setinteriorfield_gc → cannot raise
        OpKind::InteriorFieldRead { .. } | OpKind::InteriorFieldWrite { .. } => false,
        // RPython LL: int_add, int_sub, int_lt, int_and, etc → cannot raise
        // (non-ovf, non-div arithmetic)
        OpKind::BinOp { op, .. }
            if !op.contains("div")
                && !op.contains("mod")
                && !op.contains("rem")
                && !op.contains("ovf") =>
        {
            false
        }
        // RPython LL: int_neg, bool_not → cannot raise
        OpKind::UnaryOp { op, .. } if !op.contains("ovf") => false,
        // RPython LL: same_as, cast_*, hint → cannot raise
        OpKind::Input { .. } | OpKind::ConstInt(_) => false,
        // JIT-specific ops that cannot raise
        OpKind::GuardTrue { .. }
        | OpKind::GuardFalse { .. }
        | OpKind::GuardValue { .. }
        | OpKind::Live => false,
        // Virtualizable field/array access (from boxes, no heap) → cannot raise
        OpKind::VableFieldRead { .. }
        | OpKind::VableFieldWrite { .. }
        | OpKind::VableArrayRead { .. }
        | OpKind::VableArrayWrite { .. } => false,
        // Post-jtransform call ops: raise is determined by their descriptor,
        // not by op_can_raise. These are not "simple operations" in RPython
        // terms — they're handled by analyze() → analyze_direct_call.
        OpKind::CallResidual { .. }
        | OpKind::CallElidable { .. }
        | OpKind::CallMayForce { .. }
        | OpKind::InlineCall { .. }
        | OpKind::RecursiveCall { .. } => false,

        // ── Known raising ops ─────────────────────────────────────
        // RPython LL: jit_force_virtualizable → canraise
        OpKind::VableForce => true,
        // RPython LL: int_floordiv, int_mod → canraise = (ZeroDivisionError,)
        OpKind::BinOp { .. } => true, // div/mod/rem/ovf (others matched above)
        // RPython LL: int_neg_ovf → canraise = (OverflowError,)
        OpKind::UnaryOp { .. } => true, // ovf (others matched above)

        // ── Calls handled by analyze() dispatch, not here ─────────
        // RPython: Call ops dispatch to analyze_direct_call/analyze_external_call.
        // op_can_raise is only for "simple operations" (non-call).
        // But if we see a Call here (shouldn't happen in normal flow),
        // be conservative.
        OpKind::Call { .. } => true,

        // ── Unknown ops: canraise.py:18 → True (conservative) ─────
        // RPython: log.WARNING("Unknown operation: %s" % op.opname)
        //          return True
        OpKind::Unknown { .. } => true,
    }
}

/// Check if an Abort reason indicates MemoryError-only.
///
/// RPython: `do_ignore_memory_error()` sets `ignore_exact_class = MemoryError`.
/// Then `canraise != (self.ignore_exact_class,)` filters it out.
///
/// In majit, Abort carries a reason string. We check for MemoryError
/// indicators. This matches:
/// - "MemoryError" — explicit MemoryError raise
/// - "alloc" / "allocation" — memory allocation failures
fn is_memoryerror_only(reason: &str) -> bool {
    let r = reason.to_lowercase();
    r.contains("memoryerror") || r.contains("out of memory")
}

/// Map ValueType to a small integer for array descriptor indexing.
fn value_type_discriminant(ty: &crate::model::ValueType) -> u8 {
    use crate::model::ValueType;
    match ty {
        ValueType::Int => 0,
        ValueType::Ref => 1,
        ValueType::Float => 2,
        ValueType::Void => 3,
        ValueType::State => 4,
        ValueType::Unknown => 5,
    }
}

/// Detect generic type parameter or variable name used as receiver.
///
/// Generic: "H", "T", "handler", "self", "executor"
/// Concrete: "PyFrame", "Code", "Vec"
///
/// Heuristic: single uppercase letter is a type parameter;
/// starts with lowercase is a variable name.
pub fn is_generic_receiver(receiver: &str) -> bool {
    let mut chars = receiver.chars();
    let first = match chars.next() {
        Some(c) => c,
        None => return false,
    };
    if first.is_lowercase() {
        return true;
    }
    // Single uppercase letter: "H", "T", "E" — type parameter
    first.is_uppercase() && chars.next().is_none()
}

// ── Builtin call effect tables ──────────────────────────────────
//
// RPython equivalent: effect classification in `call.py::getcalldescr()`
// combined with the builtin function tables.
// These tables map known function targets to their effect info,
// used by `jtransform::classify_call()` as a fallback when the
// call is not in the explicit `call_effects` config.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CallTargetPattern {
    Method {
        name: &'static str,
        receiver_root: Option<&'static str>,
    },
    FunctionPath(&'static [&'static str]),
}

impl CallTargetPattern {
    fn matches(self, target: &CallTarget) -> bool {
        match (self, target) {
            (
                CallTargetPattern::Method {
                    name,
                    receiver_root,
                },
                CallTarget::Method {
                    name: target_name,
                    receiver_root: target_root,
                },
            ) => {
                if target_name != name {
                    return false;
                }
                receiver_root.is_none_or(|root| {
                    target_root.as_deref() == Some(root)
                        || target_root.as_ref().is_some_and(|r| is_generic_receiver(r))
                })
            }
            (CallTargetPattern::FunctionPath(path), CallTarget::FunctionPath { segments }) => {
                segments.iter().map(String::as_str).eq(path.iter().copied())
            }
            _ => false,
        }
    }
}

struct CallDescriptorEntry {
    targets: &'static [CallTargetPattern],
    extra_effect: ExtraEffect,
    oopspec_index: OopSpecIndex,
}

impl CallDescriptorEntry {
    fn effect_info(&self) -> EffectInfo {
        match self.extra_effect {
            ExtraEffect::ElidableCannotRaise => EffectInfo::elidable(),
            extra_effect => EffectInfo::new(extra_effect, self.oopspec_index),
        }
    }
}

// ── Builtin call descriptor table ──
//
// RPython effectinfo.py + call.py parity: pre-classified call targets.
// The codewriter matches function names to determine effect category
// and oopspec index without graph-level analysis.

const INT_ARITH_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["w_int_add"]),
    CallTargetPattern::FunctionPath(&["w_int_sub"]),
    CallTargetPattern::FunctionPath(&["w_int_mul"]),
    CallTargetPattern::FunctionPath(&["int_add"]),
    CallTargetPattern::FunctionPath(&["int_sub"]),
    CallTargetPattern::FunctionPath(&["int_mul"]),
    CallTargetPattern::FunctionPath(&["int_bitand"]),
    CallTargetPattern::FunctionPath(&["int_bitor"]),
    CallTargetPattern::FunctionPath(&["int_bitxor"]),
    // Qualified paths (annotator uses these for type inference).
    CallTargetPattern::FunctionPath(&["crate", "math", "w_int_add"]),
    CallTargetPattern::FunctionPath(&["crate", "math", "w_int_sub"]),
    CallTargetPattern::FunctionPath(&["crate", "math", "w_int_mul"]),
];

const INT_CMP_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["int_lt"]),
    CallTargetPattern::FunctionPath(&["int_le"]),
    CallTargetPattern::FunctionPath(&["int_gt"]),
    CallTargetPattern::FunctionPath(&["int_ge"]),
    CallTargetPattern::FunctionPath(&["int_eq"]),
    CallTargetPattern::FunctionPath(&["int_ne"]),
];

const FLOAT_ARITH_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["w_float_add"]),
    CallTargetPattern::FunctionPath(&["w_float_sub"]),
    CallTargetPattern::FunctionPath(&["float_add"]),
    CallTargetPattern::FunctionPath(&["float_sub"]),
    CallTargetPattern::FunctionPath(&["float_mul"]),
    CallTargetPattern::FunctionPath(&["float_truediv"]),
];

const FLOAT_CMP_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["float_lt"]),
    CallTargetPattern::FunctionPath(&["float_le"]),
    CallTargetPattern::FunctionPath(&["float_gt"]),
    CallTargetPattern::FunctionPath(&["float_ge"]),
    CallTargetPattern::FunctionPath(&["float_eq"]),
    CallTargetPattern::FunctionPath(&["float_ne"]),
];

// effectinfo.py: EF_ELIDABLE_CAN_RAISE — may raise (e.g. ZeroDivisionError)
// int_floordiv and int_mod have distinct oopspec indices (IntPyDiv vs IntPyMod)
// because intbounds.rs optimizes them differently.
const INT_FLOORDIV_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["int_floordiv"])];

const INT_MOD_TARGETS: &[CallTargetPattern] = &[CallTargetPattern::FunctionPath(&["int_mod"])];

const FLOAT_DIV_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["float_floordiv"]),
    CallTargetPattern::FunctionPath(&["float_mod"]),
];

const INT_SHIFT_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["int_lshift"]),
    CallTargetPattern::FunctionPath(&["int_rshift"]),
];

const INT_POW_TARGETS: &[CallTargetPattern] = &[CallTargetPattern::FunctionPath(&["int_pow"])];

// effectinfo.py: OS_STR_CONCAT etc. — string operations with oopspec
const STR_CONCAT_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["str_concat"]),
    CallTargetPattern::FunctionPath(&["jit_str_concat"]),
];

const STR_CMP_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["jit_str_compare"])];

// effectinfo.py: list operations (may raise IndexError)
const LIST_GETITEM_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["jit_list_getitem"]),
    CallTargetPattern::FunctionPath(&["w_list_getitem"]),
];

const LIST_SETITEM_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["jit_list_setitem"])];

const LIST_APPEND_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["jit_list_append"])];

// effectinfo.py: tuple access (elidable, cannot raise for valid index)
const TUPLE_GETITEM_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["jit_tuple_getitem"]),
    CallTargetPattern::FunctionPath(&["w_tuple_getitem"]),
];

// effectinfo.py: constructor-like (cannot raise, elidable)
const INT_NEW_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["w_int_new"]),
    CallTargetPattern::FunctionPath(&["jit_w_int_new"]),
];

const FLOAT_NEW_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["w_float_new"]),
    CallTargetPattern::FunctionPath(&["jit_w_float_new"]),
];

const BOOL_FROM_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["w_bool_from"])];

const CALL_DESCRIPTOR_TABLE: &[CallDescriptorEntry] = &[
    // ── Pure arithmetic (elidable, cannot raise) ──
    CallDescriptorEntry {
        targets: INT_ARITH_TARGETS,
        extra_effect: ExtraEffect::ElidableCannotRaise,
        oopspec_index: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: INT_CMP_TARGETS,
        extra_effect: ExtraEffect::ElidableCannotRaise,
        oopspec_index: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: FLOAT_ARITH_TARGETS,
        extra_effect: ExtraEffect::ElidableCannotRaise,
        oopspec_index: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: FLOAT_CMP_TARGETS,
        extra_effect: ExtraEffect::ElidableCannotRaise,
        oopspec_index: OopSpecIndex::None,
    },
    // ── Elidable but may raise (ZeroDivisionError, OverflowError) ──
    CallDescriptorEntry {
        targets: INT_FLOORDIV_TARGETS,
        extra_effect: ExtraEffect::ElidableCanRaise,
        oopspec_index: OopSpecIndex::IntPyDiv,
    },
    CallDescriptorEntry {
        targets: INT_MOD_TARGETS,
        extra_effect: ExtraEffect::ElidableCanRaise,
        oopspec_index: OopSpecIndex::IntPyMod,
    },
    CallDescriptorEntry {
        targets: FLOAT_DIV_TARGETS,
        extra_effect: ExtraEffect::ElidableCanRaise,
        oopspec_index: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: INT_SHIFT_TARGETS,
        extra_effect: ExtraEffect::ElidableCanRaise,
        oopspec_index: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: INT_POW_TARGETS,
        extra_effect: ExtraEffect::ElidableCanRaise,
        oopspec_index: OopSpecIndex::None,
    },
    // ── String operations with oopspec ──
    CallDescriptorEntry {
        targets: STR_CONCAT_TARGETS,
        extra_effect: ExtraEffect::ElidableCanRaise,
        oopspec_index: OopSpecIndex::StrConcat,
    },
    CallDescriptorEntry {
        targets: STR_CMP_TARGETS,
        extra_effect: ExtraEffect::ElidableCannotRaise,
        oopspec_index: OopSpecIndex::StrCmp,
    },
    // ── List operations (may raise, side effects) ──
    CallDescriptorEntry {
        targets: LIST_GETITEM_TARGETS,
        extra_effect: ExtraEffect::CanRaise,
        oopspec_index: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: LIST_SETITEM_TARGETS,
        extra_effect: ExtraEffect::CanRaise,
        oopspec_index: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: LIST_APPEND_TARGETS,
        extra_effect: ExtraEffect::CanRaise,
        oopspec_index: OopSpecIndex::None,
    },
    // ── Tuple access (elidable for valid indices) ──
    CallDescriptorEntry {
        targets: TUPLE_GETITEM_TARGETS,
        extra_effect: ExtraEffect::ElidableCanRaise,
        oopspec_index: OopSpecIndex::None,
    },
    // ── Allocating constructors (cannot raise, but NOT elidable) ──
    // w_int_new/w_float_new allocate fresh objects — CSE would merge
    // distinct allocations, breaking Python identity (is).
    CallDescriptorEntry {
        targets: INT_NEW_TARGETS,
        extra_effect: ExtraEffect::CannotRaise,
        oopspec_index: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: FLOAT_NEW_TARGETS,
        extra_effect: ExtraEffect::CannotRaise,
        oopspec_index: OopSpecIndex::None,
    },
    // w_bool_from returns singletons (True/False) — safe to CSE.
    CallDescriptorEntry {
        targets: BOOL_FROM_TARGETS,
        extra_effect: ExtraEffect::ElidableCannotRaise,
        oopspec_index: OopSpecIndex::None,
    },
];

fn matches_any(target: &CallTarget, patterns: &[CallTargetPattern]) -> bool {
    patterns
        .iter()
        .copied()
        .any(|pattern| pattern.matches(target))
}

/// Check if a call target is a known int arithmetic function.
/// Used by annotate pass for type inference.
pub fn is_int_arithmetic_target(target: &CallTarget) -> bool {
    matches_any(target, INT_ARITH_TARGETS)
}

/// Look up a call target in the builtin effect table.
///
/// RPython: part of `CallControl.getcalldescr()` — returns effect info
/// for known functions like `w_int_add` (elidable), `w_float_sub` (elidable).
pub fn describe_call(target: &CallTarget) -> Option<CallDescriptor> {
    CALL_DESCRIPTOR_TABLE
        .iter()
        .find(|entry| matches_any(target, entry.targets))
        .map(|entry| CallDescriptor::known(target.clone(), entry.effect_info()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FunctionGraph;

    #[test]
    fn guess_call_kind_function_path() {
        let mut cc = CallControl::new();
        let graph = FunctionGraph::new("opcode_load_fast");
        let path = CallPath::from_segments(["opcode_load_fast"]);
        cc.register_function_graph(path, graph);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["opcode_load_fast"]);
        assert_eq!(cc.guess_call_kind(&target), CallKind::Regular);

        let unknown = CallTarget::function_path(["unknown_function"]);
        assert_eq!(cc.guess_call_kind(&unknown), CallKind::Residual);
    }

    #[test]
    fn guess_call_kind_portal() {
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["portal_runner"]);
        cc.mark_portal(path);

        let target = CallTarget::function_path(["portal_runner"]);
        assert_eq!(cc.guess_call_kind(&target), CallKind::Recursive);
    }

    #[test]
    fn guess_call_kind_builtin() {
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["w_int_add"]);
        cc.mark_builtin(path);

        let target = CallTarget::function_path(["w_int_add"]);
        assert_eq!(cc.guess_call_kind(&target), CallKind::Builtin);
    }

    #[test]
    fn resolve_method_unique_impl() {
        let mut cc = CallControl::new();
        let graph = FunctionGraph::new("PyFrame::load_local_value");
        cc.register_trait_method("load_local_value", "PyFrame", graph);

        // Unique impl — resolves for any receiver
        assert!(
            cc.resolve_method("load_local_value", Some("handler"))
                .is_some()
        );
        assert!(cc.resolve_method("load_local_value", Some("H")).is_some());
        assert!(cc.resolve_method("load_local_value", None).is_some());
    }

    #[test]
    fn resolve_method_multiple_impls() {
        let mut cc = CallControl::new();
        cc.register_trait_method(
            "push_value",
            "PyFrame",
            FunctionGraph::new("PyFrame::push_value"),
        );
        cc.register_trait_method(
            "push_value",
            "MIFrame",
            FunctionGraph::new("MIFrame::push_value"),
        );

        // Concrete receiver — resolves to specific impl
        assert!(cc.resolve_method("push_value", Some("PyFrame")).is_some());

        // Generic receiver — can't resolve uniquely
        assert!(cc.resolve_method("push_value", Some("handler")).is_none());
        assert!(cc.resolve_method("push_value", Some("H")).is_none());
    }

    #[test]
    fn is_generic_receiver_examples() {
        assert!(is_generic_receiver("handler"));
        assert!(is_generic_receiver("self"));
        assert!(is_generic_receiver("H"));
        assert!(is_generic_receiver("T"));
        assert!(!is_generic_receiver("PyFrame"));
        assert!(!is_generic_receiver("Code"));
        assert!(!is_generic_receiver("Vec"));
    }

    // ── getcalldescr tests ───────────────────────────���──────────────

    use crate::model::{Block, SpaceOperation, Terminator, ValueId, ValueType};

    /// Helper: create a FunctionGraph with just a return.
    fn simple_graph(name: &str) -> FunctionGraph {
        let mut g = FunctionGraph::new(name);
        g.set_terminator(g.startblock, Terminator::Return(None));
        g
    }

    /// Helper: create a FunctionGraph with an Abort terminator.
    fn raising_graph(name: &str) -> FunctionGraph {
        let mut g = FunctionGraph::new(name);
        g.set_terminator(
            g.startblock,
            Terminator::Abort {
                reason: "error".into(),
            },
        );
        g
    }

    #[test]
    fn test_getcalldescr_cannot_raise() {
        // A simple function with no Abort → CannotRaise.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["pure_add"]);
        cc.register_function_graph(path.clone(), simple_graph("pure_add"));
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["pure_add"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(
            descriptor.effect_info.extra_effect,
            ExtraEffect::CannotRaise
        );
        assert!(!descriptor.effect_info.can_invalidate);
    }

    #[test]
    fn test_getcalldescr_can_raise() {
        // A function with Abort terminator → CanRaise.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["failing_func"]);
        cc.register_function_graph(path.clone(), raising_graph("failing_func"));
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["failing_func"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(descriptor.effect_info.extra_effect, ExtraEffect::CanRaise);
    }

    #[test]
    fn test_getcalldescr_elidable() {
        // An elidable function that cannot raise → ElidableCannotRaise.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["pure_lookup"]);
        cc.register_function_graph(path.clone(), simple_graph("pure_lookup"));
        cc.mark_elidable(path);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["pure_lookup"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(
            descriptor.effect_info.extra_effect,
            ExtraEffect::ElidableCannotRaise
        );
    }

    #[test]
    fn test_getcalldescr_elidable_can_raise() {
        // An elidable function that CAN raise → ElidableCanRaise.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["elidable_raiser"]);
        cc.register_function_graph(path.clone(), raising_graph("elidable_raiser"));
        cc.mark_elidable(path);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["elidable_raiser"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(
            descriptor.effect_info.extra_effect,
            ExtraEffect::ElidableCanRaise
        );
    }

    #[test]
    fn test_getcalldescr_loopinvariant() {
        // A loop-invariant function → LoopInvariant.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["get_config"]);
        cc.register_function_graph(path.clone(), simple_graph("get_config"));
        cc.mark_loopinvariant(path);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["get_config"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(
            descriptor.effect_info.extra_effect,
            ExtraEffect::LoopInvariant
        );
    }

    #[test]
    fn test_getcalldescr_forces_virtualizable() {
        // A function with VableForce → ForcesVirtualOrVirtualizable.
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("forcer");
        graph.push_op(graph.startblock, OpKind::VableForce, false);
        graph.set_terminator(graph.startblock, Terminator::Return(None));
        let path = CallPath::from_segments(["forcer"]);
        cc.register_function_graph(path, graph);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["forcer"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(
            descriptor.effect_info.extra_effect,
            ExtraEffect::ForcesVirtualOrVirtualizable
        );
    }

    #[test]
    fn test_getcalldescr_extraeffect_override() {
        // When extraeffect is provided, it overrides the analyzers.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["func"]);
        cc.register_function_graph(path, simple_graph("func"));
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["func"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            Some(ExtraEffect::ElidableCannotRaise),
            &mut cache,
        );
        assert_eq!(
            descriptor.effect_info.extra_effect,
            ExtraEffect::ElidableCannotRaise
        );
    }

    #[test]
    fn test_getcalldescr_transitive_can_raise() {
        // A function that calls another function that raises → CanRaise.
        let mut cc = CallControl::new();

        // callee: raises
        let callee_path = CallPath::from_segments(["callee"]);
        cc.register_function_graph(callee_path, raising_graph("callee"));

        // caller: calls callee (no Abort itself)
        let mut caller = FunctionGraph::new("caller");
        caller.push_op(
            caller.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["callee"]),
                args: Vec::new(),
                result_ty: ValueType::Void,
            },
            false,
        );
        caller.set_terminator(caller.startblock, Terminator::Return(None));
        let caller_path = CallPath::from_segments(["caller"]);
        cc.register_function_graph(caller_path, caller);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["caller"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(descriptor.effect_info.extra_effect, ExtraEffect::CanRaise);
    }

    #[test]
    fn test_getcalldescr_unknown_target_can_raise() {
        // Unknown target (no graph) treated as external call.
        // RPython: RandomEffectsAnalyzer returns False for external calls
        // (only True if random_effects_on_gcobjs). RaiseAnalyzer returns
        // True (top_result) for unknown graphs → CanRaise.
        let cc = CallControl::new();
        let target = CallTarget::function_path(["unknown_extern"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(descriptor.effect_info.extra_effect, ExtraEffect::CanRaise);
        // RandomEffects is false, QuasiImmut is false → can_invalidate is false.
        assert!(!descriptor.effect_info.can_invalidate);
    }

    #[test]
    fn test_getcalldescr_readwrite_effects() {
        // A function with FieldRead/FieldWrite → bitsets populated.
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("accessor");
        let base = graph.alloc_value();
        graph.push_op(
            graph.startblock,
            OpKind::FieldRead {
                base,
                field: crate::model::FieldDescriptor::new("x", Some("Point".into())),
                ty: ValueType::Int,
            },
            true,
        );
        graph.push_op(
            graph.startblock,
            OpKind::FieldWrite {
                base,
                field: crate::model::FieldDescriptor::new("y", Some("Point".into())),
                value: base, // dummy
                ty: ValueType::Int,
            },
            false,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));
        let path = CallPath::from_segments(["accessor"]);
        cc.register_function_graph(path, graph);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["accessor"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        // Should have non-zero bitsets for field reads and writes.
        assert_ne!(descriptor.effect_info.readonly_descrs_fields, 0);
        assert_ne!(descriptor.effect_info.write_descrs_fields, 0);
    }

    #[test]
    fn test_getcalldescr_elidable_ignores_writes() {
        // Elidable function: write_descrs should be 0 even if graph has writes.
        // RPython effectinfo.py:181-186: ignore writes for elidable.
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("pure_writer");
        let base = graph.alloc_value();
        graph.push_op(
            graph.startblock,
            OpKind::FieldWrite {
                base,
                field: crate::model::FieldDescriptor::new("cache", Some("Obj".into())),
                value: base,
                ty: ValueType::Int,
            },
            false,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));
        let path = CallPath::from_segments(["pure_writer"]);
        cc.register_function_graph(path.clone(), graph);
        cc.mark_elidable(path);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["pure_writer"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(
            descriptor.effect_info.extra_effect,
            ExtraEffect::ElidableCannotRaise
        );
        // Writes should be zeroed out for elidable functions.
        assert_eq!(descriptor.effect_info.write_descrs_fields, 0);
    }

    #[test]
    fn test_canraise_cached() {
        // Verify caching: second call should reuse result.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["raiser"]);
        cc.register_function_graph(path, raising_graph("raiser"));

        let target = CallTarget::function_path(["raiser"]);
        let mut cache = AnalysisCache::default();

        let r1 = cc.canraise(&target, &mut cache);
        assert_eq!(r1, CanRaise::Yes);
        assert!(
            cache
                .can_raise
                .contains_key(&CallPath::from_segments(["raiser"]))
        );

        let r2 = cc.canraise(&target, &mut cache);
        assert_eq!(r2, CanRaise::Yes);
    }

    #[test]
    fn test_readonly_excludes_written_fields() {
        // RPython effectinfo.py:345-348: readstruct only goes to readonly
        // if there's no corresponding write ("struct") for that field.
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("rw_same_field");
        let base = graph.alloc_value();
        let field = crate::model::FieldDescriptor::new("x", Some("Point".into()));
        // Both read AND write the same field "x"
        graph.push_op(
            graph.startblock,
            OpKind::FieldRead {
                base,
                field: field.clone(),
                ty: ValueType::Int,
            },
            true,
        );
        graph.push_op(
            graph.startblock,
            OpKind::FieldWrite {
                base,
                field: field.clone(),
                value: base,
                ty: ValueType::Int,
            },
            false,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));
        let path = CallPath::from_segments(["rw_same_field"]);
        cc.register_function_graph(path, graph);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["rw_same_field"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        // Write is set, but readonly should NOT have the same bit set.
        // RPython: readonly = reads & ~writes
        assert_ne!(descriptor.effect_info.write_descrs_fields, 0);
        let overlap = descriptor.effect_info.readonly_descrs_fields
            & descriptor.effect_info.write_descrs_fields;
        assert_eq!(
            overlap, 0,
            "readonly and write should not overlap for same field"
        );
    }

    #[test]
    fn test_op_can_raise_division() {
        // Division ops can raise (ZeroDivisionError).
        // RPython: LL_OPERATIONS[int_floordiv].canraise = (ZeroDivisionError,)
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("divider");
        let a = graph.alloc_value();
        let b = graph.alloc_value();
        graph.push_op(
            graph.startblock,
            OpKind::BinOp {
                op: "int_floordiv".to_string(),
                lhs: a,
                rhs: b,
                result_ty: ValueType::Int,
            },
            true,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));
        let path = CallPath::from_segments(["divider"]);
        cc.register_function_graph(path, graph);

        let target = CallTarget::function_path(["divider"]);
        let mut cache = AnalysisCache::default();
        let result = cc.canraise(&target, &mut cache);
        assert_eq!(result, CanRaise::Yes);
    }

    #[test]
    fn test_canraise_memoryerror_only() {
        // Abort with "MemoryError" reason → MemoryErrorOnly.
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("allocator");
        graph.set_terminator(
            graph.startblock,
            Terminator::Abort {
                reason: "MemoryError: allocation failed".into(),
            },
        );
        let path = CallPath::from_segments(["allocator"]);
        cc.register_function_graph(path, graph);

        let target = CallTarget::function_path(["allocator"]);
        let mut cache = AnalysisCache::default();
        let result = cc.canraise(&target, &mut cache);
        assert_eq!(result, CanRaise::MemoryErrorOnly);
    }

    #[test]
    fn struct_layout_depth3_nested_fixed_point() {
        // A contains B, B contains C.  Fixed-point iteration must
        // produce correct sizes regardless of HashMap iteration order.
        // struct C { x: i64 }            → size 8
        // struct B { c: C, y: i64 }      → size 16
        // struct A { b: B, z: i64 }      → size 24
        let mut known_structs: HashSet<String> = HashSet::new();
        known_structs.insert("C".into());
        known_structs.insert("B".into());
        known_structs.insert("A".into());

        let fields_c: Vec<(String, String)> = vec![("x".into(), "i64".into())];
        let fields_b: Vec<(String, String)> =
            vec![("c".into(), "C".into()), ("y".into(), "i64".into())];
        let fields_a: Vec<(String, String)> =
            vec![("b".into(), "B".into()), ("z".into(), "i64".into())];

        // Fixed-point iteration (same algorithm as lib.rs).
        let mut known_sizes: HashMap<String, usize> = HashMap::new();
        let all_fields: Vec<(&str, &Vec<(String, String)>)> =
            vec![("A", &fields_a), ("B", &fields_b), ("C", &fields_c)];
        loop {
            let mut changed = false;
            for (name, fields) in &all_fields {
                let layout = StructLayout::from_type_strings(fields, &known_structs, &known_sizes);
                if known_sizes.get(*name) != Some(&layout.size) {
                    known_sizes.insert(name.to_string(), layout.size);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        assert_eq!(known_sizes["C"], 8, "C: single i64");
        assert_eq!(known_sizes["B"], 16, "B: C(8) + i64(8)");
        assert_eq!(known_sizes["A"], 24, "A: B(16) + i64(8)");
    }
}
