use majit_ir::{GcRef, GreenKey, OpCode, OpRef, Type, Value};
use majit_macros::jit_driver;
use majit_meta::{
    resume::{MaterializedValue, MaterializedVirtual, ResumeDataBuilder},
    virtualizable::VirtualizableInfo,
    DeclarativeJitDriver, DriverRunOutcome, JitDriver, JitState, PendingFieldWriteLayout,
    TraceAction,
};

#[jit_driver(greens = [pc, code], reds = [frame, stack], virtualizable = frame)]
struct DeclarativeDriver;

#[jit_driver(greens = [pc, code], reds = [frame, acc], virtualizable = frame)]
struct TypedDeclarativeDriver;

#[jit_driver(greens = [pc, code], reds = [frame, stackpos, top], virtualizable = frame)]
struct AutoVirtualizableDriver;

#[jit_driver(greens = [pc, code], reds = [frame, stackpos, top])]
struct ResumeMappedDriver;

#[jit_driver(greens = [pc, code], reds = [obj, flag])]
struct VirtualResumeDriver;

#[jit_driver(greens = [pc, code], reds = [obj, flag])]
struct PendingWriteDriver;

#[jit_driver(greens = [pc, code], reds = [frame, flag])]
struct MultiFrameDriver;

#[jit_driver(greens = [pc, code], reds = [frame, flag])]
struct GenericMultiFrameDriver;

#[jit_driver(greens = [pc, code], reds = [frame, flag])]
struct LayoutTypedFrameDriver;

#[jit_driver(greens = [pc, code], reds = [array, flag])]
struct PendingArrayWriteDriver;

#[jit_driver(greens = [pc, code], reds = [outer, inner, alias])]
struct NestedVirtualResumeDriver;

#[derive(Clone, Debug, PartialEq, Eq)]
struct TestMeta {
    header_pc: usize,
}

struct TestState {
    frame: usize,
    stack: i64,
}

impl JitState for TestState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64, self.stack]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.frame)), Value::Int(self.stack)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        vec![OpRef(0), OpRef(1)]
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0] as usize;
        self.stack = values[1];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.frame = values[0].as_ref().as_usize();
        self.stack = values[1].as_int();
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

struct BadLiveState {
    frame: usize,
}

impl JitState for BadLiveState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.frame))]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        vec![OpRef(0)]
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0] as usize;
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

struct TypedState {
    frame: usize,
    acc: f64,
}

impl JitState for TypedState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64, self.acc.to_bits() as i64]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.frame)), Value::Float(self.acc)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        vec![OpRef(0), OpRef(1)]
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0] as usize;
        self.acc = f64::from_bits(values[1] as u64);
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.frame = values[0].as_ref().as_usize();
        self.acc = values[1].as_float();
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

struct MismatchedTypedState {
    frame: i64,
    acc: f64,
}

struct TypedRestoreOnlyState {
    frame: usize,
    acc: f64,
    raw_restore_calls: usize,
    typed_restore_calls: usize,
}

impl JitState for MismatchedTypedState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame, self.acc.to_bits() as i64]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Int(self.frame), Value::Float(self.acc)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        vec![OpRef(0), OpRef(1)]
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0];
        self.acc = f64::from_bits(values[1] as u64);
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for TypedRestoreOnlyState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64, self.acc.to_bits() as i64]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.frame)), Value::Float(self.acc)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        vec![OpRef(0), OpRef(1)]
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, _values: &[i64]) {
        self.raw_restore_calls += 1;
        self.frame = 0;
        self.acc = -1.0;
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.typed_restore_calls += 1;
        self.frame = values[0].as_ref().as_usize();
        self.acc = values[1].as_float();
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![(sym[0], Type::Ref), (sym[1], Type::Float)]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

struct VirtualizableSyncState {
    frame: usize,
    stack: i64,
    before_sync_calls: usize,
    after_sync_calls: usize,
    reject_sync: bool,
}

impl JitState for VirtualizableSyncState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64, self.stack]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.frame)), Value::Int(self.stack)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        vec![OpRef(0), OpRef(1)]
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0] as usize;
        self.stack = values[1];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.frame = values[0].as_ref().as_usize();
        self.stack = values[1].as_int();
    }

    fn sync_virtualizable_before_jit(
        &mut self,
        _meta: &Self::Meta,
        virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> bool {
        assert_eq!(virtualizable, "frame");
        self.before_sync_calls += 1;
        !self.reject_sync
    }

    fn sync_virtualizable_after_jit(
        &mut self,
        _meta: &Self::Meta,
        virtualizable: &str,
        _info: &VirtualizableInfo,
    ) {
        assert_eq!(virtualizable, "frame");
        self.after_sync_calls += 1;
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

struct NamedVirtualizableSyncState {
    frame: usize,
    stack: i64,
    before_sync_calls: usize,
    after_sync_calls: usize,
    saw_info_on_before: bool,
    saw_info_on_after: bool,
}

#[repr(C)]
struct AutoVirtualizableFrame {
    token: u64,
    stackpos: i64,
    stack_ptr: *mut u8,
}

struct AutoVirtualizableState {
    frame: usize,
    stackpos: i64,
    top: i64,
}

struct ResumeMappedState {
    frame: usize,
    stackpos: i64,
    top: i64,
}

struct VirtualResumeState {
    obj: usize,
    flag: i64,
    materialized_ref: usize,
    materialize_calls: usize,
}

#[repr(C)]
struct PendingWriteCell {
    field: i64,
}

struct PendingWriteState {
    obj: usize,
    flag: i64,
}

struct PendingArrayWriteState {
    array: usize,
    flag: i64,
}

#[repr(C)]
struct PendingRefCell {
    child: usize,
}

struct PendingVirtualWriteState {
    parent: usize,
    child: usize,
    parent_ref: usize,
    child_ref: usize,
    materialize_order: Vec<usize>,
}

struct MultiFrameResumeState {
    frame: usize,
    flag: i64,
    restored_pcs: Vec<u64>,
}

struct GenericMultiFrameResumeState {
    frame: usize,
    flag: i64,
    restored_pcs: Vec<u64>,
    restored_frames: Vec<(usize, i64)>,
    materialized_ref: usize,
    materialize_calls: usize,
}

struct LayoutTypedFrameRestoreState {
    frame: usize,
    flag: i64,
    fallback_restore_calls: usize,
    layout_restore_calls: usize,
    restored_frames: Vec<(usize, i64)>,
}

struct NestedVirtualResumeState {
    outer: usize,
    inner: usize,
    alias: usize,
    outer_ref: usize,
    inner_ref: usize,
    materialize_order: Vec<usize>,
}

impl JitState for NamedVirtualizableSyncState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64, self.stack]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.frame)), Value::Int(self.stack)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        vec![OpRef(0), OpRef(1)]
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0] as usize;
        self.stack = values[1];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.frame = values[0].as_ref().as_usize();
        self.stack = values[1].as_int();
    }

    fn sync_named_virtualizable_before_jit(
        &mut self,
        _meta: &Self::Meta,
        virtualizable: &str,
        info: Option<&VirtualizableInfo>,
    ) -> bool {
        assert_eq!(virtualizable, "frame");
        self.before_sync_calls += 1;
        self.saw_info_on_before = info.is_some();
        true
    }

    fn sync_named_virtualizable_after_jit(
        &mut self,
        _meta: &Self::Meta,
        virtualizable: &str,
        info: Option<&VirtualizableInfo>,
    ) {
        assert_eq!(virtualizable, "frame");
        self.after_sync_calls += 1;
        self.saw_info_on_after = info.is_some();
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for AutoVirtualizableState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64, self.stackpos, self.top]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![
            Value::Ref(GcRef(self.frame)),
            Value::Int(self.stackpos),
            Value::Int(self.top),
        ]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        vec![OpRef(0), OpRef(1), OpRef(2)]
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0] as usize;
        self.stackpos = values[1];
        self.top = values[2];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.frame = values[0].as_ref().as_usize();
        self.stackpos = values[1].as_int();
        self.top = values[2].as_int();
    }

    fn virtualizable_heap_ptr(
        &self,
        _meta: &Self::Meta,
        virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<*mut u8> {
        assert_eq!(virtualizable, "frame");
        Some(self.frame as *mut u8)
    }

    fn virtualizable_array_lengths(
        &self,
        _meta: &Self::Meta,
        virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<Vec<usize>> {
        assert_eq!(virtualizable, "frame");
        Some(vec![1])
    }

    fn import_virtualizable_boxes(
        &mut self,
        _meta: &Self::Meta,
        virtualizable: &str,
        _info: &VirtualizableInfo,
        static_boxes: &[i64],
        array_boxes: &[Vec<i64>],
    ) -> bool {
        assert_eq!(virtualizable, "frame");
        self.stackpos = static_boxes[0];
        self.top = array_boxes[0][0];
        true
    }

    fn export_virtualizable_boxes(
        &self,
        _meta: &Self::Meta,
        virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<(Vec<i64>, Vec<Vec<i64>>)> {
        assert_eq!(virtualizable, "frame");
        Some((vec![self.stackpos], vec![vec![self.top]]))
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![
            (sym[0], Type::Ref),
            (sym[1], Type::Int),
            (sym[2], Type::Int),
        ]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for ResumeMappedState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64, self.stackpos, self.top]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![
            Value::Ref(GcRef(self.frame)),
            Value::Int(self.stackpos),
            Value::Int(self.top),
        ]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        vec![OpRef(0), OpRef(1), OpRef(2)]
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0] as usize;
        self.stackpos = values[1];
        self.top = values[2];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.frame = values[0].as_ref().as_usize();
        self.stackpos = values[1].as_int();
        self.top = values[2].as_int();
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![
            (sym[0], Type::Ref),
            (sym[1], Type::Int),
            (sym[2], Type::Int),
        ]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for VirtualResumeState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.obj as i64, self.flag]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.obj)), Value::Int(self.flag)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        vec![OpRef(0), OpRef(1)]
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.obj = values[0] as usize;
        self.flag = values[1];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.obj = values[0].as_ref().as_usize();
        self.flag = values[1].as_int();
    }

    fn materialize_virtual_ref(
        &mut self,
        _meta: &Self::Meta,
        virtual_index: usize,
        materialized: &MaterializedVirtual,
    ) -> Option<GcRef> {
        assert_eq!(virtual_index, 0);
        match materialized {
            MaterializedVirtual::Struct { descr_index, .. } => assert_eq!(*descr_index, 7),
            other => panic!("unexpected virtual materialization: {other:?}"),
        }
        self.materialize_calls += 1;
        Some(GcRef(self.materialized_ref))
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![(sym[0], Type::Ref), (sym[1], Type::Int)]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for PendingWriteState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.obj as i64, self.flag]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.obj)), Value::Int(self.flag)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        vec![OpRef(0), OpRef(1)]
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.obj = values[0] as usize;
        self.flag = values[1];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.obj = values[0].as_ref().as_usize();
        self.flag = values[1].as_int();
    }

    fn pending_field_write_layout(
        &self,
        _meta: &Self::Meta,
        descr_index: u32,
        is_array_item: bool,
    ) -> Option<PendingFieldWriteLayout> {
        if descr_index == 9 && !is_array_item {
            Some(PendingFieldWriteLayout::Field {
                offset: 0,
                value_type: Type::Int,
            })
        } else {
            None
        }
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![(sym[0], Type::Ref), (sym[1], Type::Int)]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for PendingArrayWriteState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.array as i64, self.flag]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.array)), Value::Int(self.flag)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        vec![OpRef(0), OpRef(1)]
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.array = values[0] as usize;
        self.flag = values[1];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.array = values[0].as_ref().as_usize();
        self.flag = values[1].as_int();
    }

    fn pending_field_write_layout(
        &self,
        _meta: &Self::Meta,
        descr_index: u32,
        is_array_item: bool,
    ) -> Option<PendingFieldWriteLayout> {
        if descr_index == 12 && is_array_item {
            Some(PendingFieldWriteLayout::ArrayItem {
                base_offset: 0,
                item_size: std::mem::size_of::<i64>(),
                item_type: Type::Int,
            })
        } else {
            None
        }
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![(sym[0], Type::Ref), (sym[1], Type::Int)]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for MultiFrameResumeState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64, self.flag]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.frame)), Value::Int(self.flag)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        vec![OpRef(0), OpRef(1)]
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0] as usize;
        self.flag = values[1];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.frame = values[0].as_ref().as_usize();
        self.flag = values[1].as_int();
    }

    fn restore_reconstructed_frames(
        &mut self,
        _meta: &Self::Meta,
        reconstructed_state: &majit_meta::resume::ReconstructedState,
        _materialized_virtuals: &[MaterializedVirtual],
        _exception: &majit_meta::blackhole::ExceptionState,
    ) -> bool {
        if reconstructed_state.frames.len() != 2 {
            return false;
        }
        self.restored_pcs = reconstructed_state
            .frames
            .iter()
            .map(|frame| frame.pc)
            .collect();
        let inner = reconstructed_state
            .frames
            .last()
            .expect("two frames required");
        self.frame = inner.values[0].lossy_i64() as usize;
        self.flag = inner.values[1].lossy_i64();
        true
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![(sym[0], Type::Ref), (sym[1], Type::Int)]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for GenericMultiFrameResumeState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64, self.flag]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.frame)), Value::Int(self.flag)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        vec![OpRef(0), OpRef(1)]
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0] as usize;
        self.flag = values[1];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.frame = values[0].as_ref().as_usize();
        self.flag = values[1].as_int();
    }

    fn reconstructed_frame_value_types(
        &self,
        _meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _frame_pc: u64,
    ) -> Option<Vec<Type>> {
        Some(vec![Type::Ref, Type::Int])
    }

    fn restore_reconstructed_frame_values(
        &mut self,
        _meta: &Self::Meta,
        frame_index: usize,
        total_frames: usize,
        frame_pc: u64,
        values: &[Value],
        _exception: &majit_meta::blackhole::ExceptionState,
    ) -> bool {
        self.restored_pcs.push(frame_pc);
        self.restored_frames
            .push((values[0].as_ref().as_usize(), values[1].as_int()));
        if frame_index + 1 == total_frames {
            self.frame = values[0].as_ref().as_usize();
            self.flag = values[1].as_int();
        }
        true
    }

    fn materialize_virtual_ref(
        &mut self,
        _meta: &Self::Meta,
        virtual_index: usize,
        _virtual: &MaterializedVirtual,
    ) -> Option<GcRef> {
        assert_eq!(virtual_index, 0);
        self.materialize_calls += 1;
        Some(GcRef(self.materialized_ref))
    }

    fn pending_field_write_layout(
        &self,
        _meta: &Self::Meta,
        descr_index: u32,
        is_array_item: bool,
    ) -> Option<PendingFieldWriteLayout> {
        if descr_index == 31 && !is_array_item {
            Some(PendingFieldWriteLayout::Field {
                offset: 0,
                value_type: Type::Int,
            })
        } else {
            None
        }
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![(sym[0], Type::Ref), (sym[1], Type::Int)]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for LayoutTypedFrameRestoreState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64, self.flag]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.frame)), Value::Int(self.flag)]
    }

    fn live_value_types(&self, _meta: &Self::Meta) -> Vec<Type> {
        vec![Type::Int, Type::Int]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        vec![OpRef(0), OpRef(1)]
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.fallback_restore_calls += 1;
        self.frame = values[0] as usize;
        self.flag = values[1];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.fallback_restore_calls += 1;
        self.frame = match values[0] {
            Value::Ref(r) => r.as_usize(),
            Value::Int(v) => v as usize,
            _ => 0,
        };
        self.flag = match values[1] {
            Value::Int(v) => v,
            Value::Float(v) => v as i64,
            Value::Ref(r) => r.as_usize() as i64,
            Value::Void => 0,
        };
    }

    fn restore_reconstructed_frame_values(
        &mut self,
        _meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _frame_pc: u64,
        values: &[Value],
        _exception: &majit_meta::blackhole::ExceptionState,
    ) -> bool {
        self.layout_restore_calls += 1;
        self.frame = values[0].as_ref().as_usize();
        self.flag = values[1].as_int();
        self.restored_frames.push((self.frame, self.flag));
        true
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![(sym[0], Type::Ref), (sym[1], Type::Int)]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for PendingVirtualWriteState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.parent as i64, self.child as i64]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![
            Value::Ref(GcRef(self.parent)),
            Value::Ref(GcRef(self.child)),
        ]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        vec![OpRef(0), OpRef(1)]
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.parent = values[0] as usize;
        self.child = values[1] as usize;
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.parent = values[0].as_ref().as_usize();
        self.child = values[1].as_ref().as_usize();
    }

    fn materialize_virtual_ref_with_refs(
        &mut self,
        _meta: &Self::Meta,
        virtual_index: usize,
        materialized: &MaterializedVirtual,
        _materialized_refs: &[Option<GcRef>],
    ) -> Option<GcRef> {
        self.materialize_order.push(virtual_index);
        match (virtual_index, materialized) {
            (0, MaterializedVirtual::Struct { descr_index, .. }) => {
                assert_eq!(*descr_index, 30);
                Some(GcRef(self.parent_ref))
            }
            (1, MaterializedVirtual::Struct { descr_index, .. }) => {
                assert_eq!(*descr_index, 31);
                Some(GcRef(self.child_ref))
            }
            other => panic!("unexpected pending-write virtual materialization: {other:?}"),
        }
    }

    fn pending_field_write_layout(
        &self,
        _meta: &Self::Meta,
        descr_index: u32,
        is_array_item: bool,
    ) -> Option<PendingFieldWriteLayout> {
        if descr_index == 21 && !is_array_item {
            Some(PendingFieldWriteLayout::Field {
                offset: 0,
                value_type: Type::Ref,
            })
        } else {
            None
        }
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![(sym[0], Type::Ref), (sym[1], Type::Ref)]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for NestedVirtualResumeState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.outer as i64, self.inner as i64, self.alias as i64]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![
            Value::Ref(GcRef(self.outer)),
            Value::Ref(GcRef(self.inner)),
            Value::Ref(GcRef(self.alias)),
        ]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        vec![OpRef(0), OpRef(1), OpRef(2)]
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.outer = values[0] as usize;
        self.inner = values[1] as usize;
        self.alias = values[2] as usize;
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.outer = values[0].as_ref().as_usize();
        self.inner = values[1].as_ref().as_usize();
        self.alias = values[2].as_ref().as_usize();
    }

    fn materialize_virtual_ref_with_refs(
        &mut self,
        _meta: &Self::Meta,
        virtual_index: usize,
        materialized: &MaterializedVirtual,
        materialized_refs: &[Option<GcRef>],
    ) -> Option<GcRef> {
        self.materialize_order.push(virtual_index);
        match (virtual_index, materialized) {
            (
                0,
                MaterializedVirtual::Struct {
                    descr_index,
                    fields,
                    ..
                },
            ) => {
                assert_eq!(*descr_index, 10);
                assert_eq!(fields, &vec![(0, MaterializedValue::Value(77))]);
                Some(GcRef(self.inner_ref))
            }
            (
                1,
                MaterializedVirtual::Obj {
                    descr_index,
                    fields,
                    ..
                },
            ) => {
                assert_eq!(*descr_index, 20);
                assert_eq!(fields[0], (0, MaterializedValue::VirtualRef(0)));
                assert_eq!(fields[1], (1, MaterializedValue::Value(99)));
                assert_eq!(
                    materialized_refs.first().copied().flatten(),
                    Some(GcRef(self.inner_ref))
                );
                let resolved = materialized
                    .resolve_with_refs(materialized_refs)
                    .expect("nested refs should resolve");
                match resolved {
                    MaterializedVirtual::Obj { fields, .. } => {
                        assert_eq!(
                            fields[0],
                            (0, MaterializedValue::Value(self.inner_ref as i64))
                        );
                        assert_eq!(fields[1], (1, MaterializedValue::Value(99)));
                    }
                    other => panic!("unexpected resolved virtual: {other:?}"),
                }
                Some(GcRef(self.outer_ref))
            }
            other => panic!("unexpected materialization: {other:?}"),
        }
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![
            (sym[0], Type::Ref),
            (sym[1], Type::Ref),
            (sym[2], Type::Ref),
        ]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

#[test]
fn runtime_driver_preserves_structured_green_key_and_descriptor_on_trace_start() {
    let descriptor =
        DeclarativeDriver::descriptor(&[Type::Int, Type::Int], &[Type::Ref, Type::Int])
            .expect("descriptor should build");
    let green_key = DeclarativeDriver::green_key(&[7, 99]).expect("green key should build");
    let mut driver = JitDriver::<TestState>::with_descriptor(2, descriptor.clone());
    let mut state = TestState {
        frame: 10,
        stack: 20,
    };

    assert!(!driver.back_edge_structured(green_key.clone(), 7, &mut state, &(), || {}));
    assert!(!driver.back_edge_structured(green_key.clone(), 7, &mut state, &(), || {}));
    assert!(driver.is_tracing());

    let ctx = driver
        .meta_interp_mut()
        .trace_ctx()
        .expect("trace must be active");
    assert_eq!(ctx.green_key(), green_key.hash_u64());
    assert_eq!(ctx.green_key_values(), Some(&green_key));
    let trace_descriptor = ctx
        .driver_descriptor()
        .expect("descriptor must be attached to trace");
    assert_eq!(trace_descriptor.virtualizable.as_deref(), Some("frame"));
    assert_eq!(trace_descriptor.vars, descriptor.vars);
}

#[test]
fn runtime_driver_attaches_descriptor_on_keyed_trace_start_without_structured_green_key() {
    let descriptor =
        DeclarativeDriver::descriptor(&[Type::Int, Type::Int], &[Type::Ref, Type::Int])
            .expect("descriptor should build");
    let key = GreenKey::new(vec![11, 22]).hash_u64();
    let mut driver = JitDriver::<TestState>::with_descriptor(2, descriptor.clone());
    let mut state = TestState { frame: 3, stack: 4 };

    assert!(!driver.back_edge_keyed(key, 11, &mut state, &(), || {}));
    assert!(!driver.back_edge_keyed(key, 11, &mut state, &(), || {}));
    assert!(driver.is_tracing());

    let ctx = driver
        .meta_interp_mut()
        .trace_ctx()
        .expect("trace must be active");
    assert_eq!(ctx.green_key(), key);
    assert_eq!(ctx.green_key_values(), None);
    let trace_descriptor = ctx
        .driver_descriptor()
        .expect("descriptor must be attached to trace");
    assert_eq!(
        trace_descriptor
            .virtualizable()
            .map(|var| var.name.as_str()),
        Some("frame")
    );
    assert_eq!(trace_descriptor.vars, descriptor.vars);
}

#[test]
fn declarative_driver_trait_builds_runtime_driver_without_manual_descriptor_plumbing() {
    let mut driver = JitDriver::<TestState>::with_declarative::<DeclarativeDriver>(
        2,
        &[Type::Int, Type::Int],
        &[Type::Ref, Type::Int],
    )
    .expect("declarative driver should build runtime descriptor");
    let mut state = TestState { frame: 1, stack: 2 };

    assert!(!driver
        .back_edge_declarative::<DeclarativeDriver>(&[13, 21], 13, &mut state, &(), || {})
        .expect("green key should build"));
    assert!(!driver
        .back_edge_declarative::<DeclarativeDriver>(&[13, 21], 13, &mut state, &(), || {})
        .expect("green key should build"));
    assert!(driver.is_tracing());

    let ctx = driver
        .meta_interp_mut()
        .trace_ctx()
        .expect("trace must be active");
    assert_eq!(ctx.green_key_values(), Some(&GreenKey::new(vec![13, 21])));
    assert_eq!(
        ctx.driver_descriptor()
            .and_then(|descr| descr.virtualizable().map(|var| var.name.as_str())),
        Some("frame")
    );
}

#[test]
fn declarative_driver_trait_matches_inherent_green_key_builder() {
    let via_trait = <DeclarativeDriver as DeclarativeJitDriver>::green_key(&[5, 8])
        .expect("trait green key should build");
    let via_inherent =
        DeclarativeDriver::green_key(&[5, 8]).expect("inherent green key should build");
    assert_eq!(via_trait, via_inherent);
}

#[test]
fn declarative_driver_rejects_live_value_count_mismatch() {
    let mut driver = JitDriver::<BadLiveState>::with_declarative::<DeclarativeDriver>(
        2,
        &[Type::Int, Type::Int],
        &[Type::Ref, Type::Int],
    )
    .expect("declarative driver should build runtime descriptor");
    let mut state = BadLiveState { frame: 9 };

    assert!(!driver
        .back_edge_declarative::<DeclarativeDriver>(&[1, 2], 1, &mut state, &(), || {})
        .expect("green key should build"));
    assert!(!driver
        .back_edge_declarative::<DeclarativeDriver>(&[1, 2], 1, &mut state, &(), || {})
        .expect("green key should build"));
    assert!(!driver.is_tracing());
}

#[test]
fn declarative_driver_preserves_typed_red_inputargs_on_trace_start() {
    let mut driver = JitDriver::<TypedState>::with_declarative::<TypedDeclarativeDriver>(
        2,
        &[Type::Int, Type::Int],
        &[Type::Ref, Type::Float],
    )
    .expect("typed declarative driver should build runtime descriptor");
    let mut state = TypedState {
        frame: 0x1234,
        acc: 3.5,
    };

    assert!(!driver
        .back_edge_declarative::<TypedDeclarativeDriver>(&[17, 23], 17, &mut state, &(), || {})
        .expect("green key should build"));
    assert!(!driver
        .back_edge_declarative::<TypedDeclarativeDriver>(&[17, 23], 17, &mut state, &(), || {})
        .expect("green key should build"));
    assert!(driver.is_tracing());

    let (trace, _) = driver
        .meta_interp_mut()
        .finish_trace_for_parity(&[OpRef(0), OpRef(1)])
        .expect("trace should be finishable for parity");
    assert_eq!(trace.inputargs.len(), 2);
    assert_eq!(trace.inputargs[0].tp, Type::Ref);
    assert_eq!(trace.inputargs[1].tp, Type::Float);
}

#[test]
fn declarative_driver_rejects_live_value_type_mismatch() {
    let mut driver = JitDriver::<MismatchedTypedState>::with_declarative::<TypedDeclarativeDriver>(
        2,
        &[Type::Int, Type::Int],
        &[Type::Ref, Type::Float],
    )
    .expect("typed declarative driver should build runtime descriptor");
    let mut state = MismatchedTypedState {
        frame: 7,
        acc: 1.25,
    };

    assert!(!driver
        .back_edge_declarative::<TypedDeclarativeDriver>(&[19, 29], 19, &mut state, &(), || {})
        .expect("green key should build"));
    assert!(!driver
        .back_edge_declarative::<TypedDeclarativeDriver>(&[19, 29], 19, &mut state, &(), || {})
        .expect("green key should build"));
    assert!(!driver.is_tracing());
}

#[test]
fn declarative_driver_blackhole_jump_uses_typed_restore_for_ref_and_float_reds() {
    let mut driver =
        JitDriver::<TypedRestoreOnlyState>::with_declarative::<TypedDeclarativeDriver>(
            2,
            &[Type::Int, Type::Int],
            &[Type::Ref, Type::Float],
        )
        .expect("typed declarative driver should build runtime descriptor");
    let mut state = TypedRestoreOnlyState {
        frame: 0x1234,
        acc: 3.5,
        raw_restore_calls: 0,
        typed_restore_calls: 0,
    };

    assert!(!driver
        .back_edge_declarative::<TypedDeclarativeDriver>(&[31, 41], 31, &mut state, &(), || {})
        .expect("green key should build"));
    assert!(!driver
        .back_edge_declarative::<TypedDeclarativeDriver>(&[31, 41], 31, &mut state, &(), || {})
        .expect("green key should build"));
    driver.merge_point(|ctx, sym| {
        let cond = ctx.const_int(1);
        ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[cond], 1, &[sym[0], sym[1]]);
        TraceAction::CloseLoop
    });

    match driver
        .run_compiled_with_blackhole_fallback_declarative::<TypedDeclarativeDriver>(
            &[31, 41],
            &mut state,
            || {},
        )
        .expect("green key should build")
    {
        DriverRunOutcome::Jump { .. } => {}
        other => panic!("expected Jump outcome, got {other:?}"),
    }

    assert_eq!(state.frame, 0x1234);
    assert_eq!(state.acc, 3.5);
    assert_eq!(state.raw_restore_calls, 0);
    assert_eq!(state.typed_restore_calls, 1);
}

#[test]
fn declarative_driver_invokes_virtualizable_sync_hooks_on_trace_and_compiled_exit() {
    let descriptor =
        DeclarativeDriver::descriptor(&[Type::Int, Type::Int], &[Type::Ref, Type::Int])
            .expect("descriptor should build");
    let mut driver = JitDriver::<VirtualizableSyncState>::with_descriptor(2, descriptor);
    // Only set virtualizable_info with token_offset — no static fields.
    // This test verifies sync hook invocation, not field loading.
    driver.set_virtualizable_info(VirtualizableInfo::new(0));
    let mut state = VirtualizableSyncState {
        frame: 10,
        stack: 20,
        before_sync_calls: 0,
        after_sync_calls: 0,
        reject_sync: false,
    };

    assert!(!driver.back_edge_keyed(55, 55, &mut state, &(), || {}));
    assert!(!driver.back_edge_keyed(55, 55, &mut state, &(), || {}));
    assert_eq!(state.before_sync_calls, 2);
    assert_eq!(state.after_sync_calls, 0);
    assert!(driver.is_tracing());

    driver.merge_point(|ctx, sym| {
        let cond = ctx.const_int(1);
        ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[cond], 2, &[sym[0], sym[1]]);
        TraceAction::CloseLoop
    });

    assert!(driver.back_edge_keyed(55, 55, &mut state, &(), || {}));
    assert_eq!(state.before_sync_calls, 3);
    assert_eq!(state.after_sync_calls, 1);
}

#[test]
fn declarative_driver_can_reject_virtualizable_sync_before_tracing_starts() {
    let descriptor =
        DeclarativeDriver::descriptor(&[Type::Int, Type::Int], &[Type::Ref, Type::Int])
            .expect("descriptor should build");
    let mut driver = JitDriver::<VirtualizableSyncState>::with_descriptor(2, descriptor);
    driver.set_virtualizable_info(VirtualizableInfo::new(0));
    let mut state = VirtualizableSyncState {
        frame: 1,
        stack: 2,
        before_sync_calls: 0,
        after_sync_calls: 0,
        reject_sync: true,
    };

    assert!(!driver.back_edge_keyed(77, 77, &mut state, &(), || {}));
    assert!(!driver.back_edge_keyed(77, 77, &mut state, &(), || {}));
    assert_eq!(state.before_sync_calls, 2);
    assert_eq!(state.after_sync_calls, 0);
    assert!(!driver.is_tracing());
}

#[test]
fn declarative_driver_invokes_named_virtualizable_hooks_without_info() {
    let descriptor =
        DeclarativeDriver::descriptor(&[Type::Int, Type::Int], &[Type::Ref, Type::Int])
            .expect("descriptor should build");
    let mut driver = JitDriver::<NamedVirtualizableSyncState>::with_descriptor(2, descriptor);
    let mut state = NamedVirtualizableSyncState {
        frame: 5,
        stack: 9,
        before_sync_calls: 0,
        after_sync_calls: 0,
        saw_info_on_before: true,
        saw_info_on_after: true,
    };

    assert!(!driver.back_edge_keyed(91, 91, &mut state, &(), || {}));
    assert!(!driver.back_edge_keyed(91, 91, &mut state, &(), || {}));
    assert!(driver.is_tracing());
    assert_eq!(state.before_sync_calls, 2);
    assert!(!state.saw_info_on_before);

    driver.merge_point(|ctx, sym| {
        let cond = ctx.const_int(1);
        ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[cond], 2, &[sym[0], sym[1]]);
        TraceAction::CloseLoop
    });

    assert!(driver.back_edge_keyed(91, 91, &mut state, &(), || {}));
    assert_eq!(state.before_sync_calls, 3);
    assert_eq!(state.after_sync_calls, 1);
    assert!(!state.saw_info_on_after);
}

#[test]
fn declarative_driver_auto_syncs_virtualizable_heap_state() {
    let descriptor = AutoVirtualizableDriver::descriptor(
        &[Type::Int, Type::Int],
        &[Type::Ref, Type::Int, Type::Int],
    )
    .expect("descriptor should build");
    let mut driver = JitDriver::<AutoVirtualizableState>::with_descriptor(2, descriptor);
    let mut info = VirtualizableInfo::new(0);
    info.add_field("stackpos", Type::Int, 8);
    info.add_array_field("stack", Type::Int, 16);
    driver.set_virtualizable_info(info);

    let mut stack = vec![41_i64];
    let mut frame = AutoVirtualizableFrame {
        token: 123,
        stackpos: 7,
        stack_ptr: stack.as_mut_ptr().cast(),
    };
    let mut state = AutoVirtualizableState {
        frame: (&mut frame as *mut AutoVirtualizableFrame).cast::<u8>() as usize,
        stackpos: -1,
        top: -1,
    };

    assert!(!driver.back_edge_keyed(111, 111, &mut state, &(), || {}));
    assert_eq!(state.stackpos, 7);
    assert_eq!(state.top, 41);
    assert!(!driver.back_edge_keyed(111, 111, &mut state, &(), || {}));
    assert!(driver.is_tracing());

    driver.merge_point(|ctx, sym| {
        let one = ctx.const_int(1);
        sym[1] = ctx.record_op(OpCode::IntAdd, &[sym[1], one]);
        sym[2] = ctx.record_op(OpCode::IntAdd, &[sym[2], one]);
        let cond = ctx.const_int(1);
        ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[cond], 3, &[sym[0], sym[1], sym[2]]);
        TraceAction::CloseLoop
    });

    state.stackpos = -10;
    state.top = -10;
    assert!(driver.back_edge_keyed(111, 111, &mut state, &(), || {}));
    assert_eq!(state.stackpos, 8);
    assert_eq!(state.top, 42);
    assert_eq!(frame.stackpos, 8);
    assert_eq!(stack[0], 42);
    assert_eq!(frame.token, 0);
}

#[test]
fn declarative_driver_blackhole_fallback_jump_restores_and_syncs_virtualizable_state() {
    let descriptor = AutoVirtualizableDriver::descriptor(
        &[Type::Int, Type::Int],
        &[Type::Ref, Type::Int, Type::Int],
    )
    .expect("descriptor should build");
    let mut driver = JitDriver::<AutoVirtualizableState>::with_descriptor(2, descriptor);
    let mut info = VirtualizableInfo::new(0);
    info.add_field("stackpos", Type::Int, 8);
    info.add_array_field("stack", Type::Int, 16);
    driver.set_virtualizable_info(info);

    let mut stack = vec![41_i64];
    let mut frame = AutoVirtualizableFrame {
        token: 456,
        stackpos: 7,
        stack_ptr: stack.as_mut_ptr().cast(),
    };
    let mut state = AutoVirtualizableState {
        frame: (&mut frame as *mut AutoVirtualizableFrame).cast::<u8>() as usize,
        stackpos: -1,
        top: -1,
    };

    assert!(!driver.back_edge_keyed(222, 222, &mut state, &(), || {}));
    assert!(!driver.back_edge_keyed(222, 222, &mut state, &(), || {}));
    driver.merge_point(|ctx, sym| {
        ctx.record_guard_with_fail_args(
            OpCode::GuardFalse,
            &[sym[1]],
            3,
            &[sym[0], sym[1], sym[2]],
        );
        let one = ctx.const_int(1);
        sym[1] = ctx.record_op(OpCode::IntAdd, &[sym[1], one]);
        sym[2] = ctx.record_op(OpCode::IntAdd, &[sym[2], one]);
        TraceAction::CloseLoop
    });

    state.stackpos = -10;
    state.top = -10;
    match driver.run_compiled_with_blackhole_fallback_keyed(222, &mut state, || {}) {
        DriverRunOutcome::Jump { .. } => {}
        other => panic!("expected Jump outcome, got {other:?}"),
    }

    assert_eq!(state.stackpos, 8);
    assert_eq!(state.top, 42);
    assert_eq!(frame.stackpos, 8);
    assert_eq!(stack[0], 42);
    assert_eq!(frame.token, 0);
}

#[test]
fn declarative_driver_blackhole_guard_failure_restores_and_syncs_virtualizable_state() {
    let descriptor = AutoVirtualizableDriver::descriptor(
        &[Type::Int, Type::Int],
        &[Type::Ref, Type::Int, Type::Int],
    )
    .expect("descriptor should build");
    let mut driver = JitDriver::<AutoVirtualizableState>::with_descriptor(2, descriptor);
    let mut info = VirtualizableInfo::new(0);
    info.add_field("stackpos", Type::Int, 8);
    info.add_array_field("stack", Type::Int, 16);
    driver.set_virtualizable_info(info);

    let mut stack = vec![41_i64];
    let mut frame = AutoVirtualizableFrame {
        token: 789,
        stackpos: 7,
        stack_ptr: stack.as_mut_ptr().cast(),
    };
    let mut state = AutoVirtualizableState {
        frame: (&mut frame as *mut AutoVirtualizableFrame).cast::<u8>() as usize,
        stackpos: -1,
        top: -1,
    };

    assert!(!driver.back_edge_keyed(333, 333, &mut state, &(), || {}));
    assert!(!driver.back_edge_keyed(333, 333, &mut state, &(), || {}));
    driver.merge_point(|ctx, sym| {
        ctx.record_guard_with_fail_args(
            OpCode::GuardFalse,
            &[sym[1]],
            3,
            &[sym[0], sym[1], sym[2]],
        );
        let one = ctx.const_int(1);
        sym[1] = ctx.record_op(OpCode::IntAdd, &[sym[1], one]);
        ctx.record_guard_with_fail_args(
            OpCode::GuardFalse,
            &[sym[1]],
            3,
            &[sym[0], sym[1], sym[2]],
        );
        TraceAction::CloseLoop
    });

    state.stackpos = -10;
    state.top = -10;
    match driver.run_compiled_with_blackhole_fallback_keyed(333, &mut state, || {}) {
        DriverRunOutcome::GuardFailure { restored, .. } => assert!(restored),
        other => panic!("expected GuardFailure outcome, got {other:?}"),
    }

    assert_eq!(state.stackpos, 8);
    assert_eq!(state.top, 41);
    assert_eq!(frame.stackpos, 8);
    assert_eq!(stack[0], 41);
    assert_eq!(frame.token, 0);
}

#[test]
fn declarative_driver_guard_failure_restores_from_reconstructed_resume_frame() {
    let descriptor =
        ResumeMappedDriver::descriptor(&[Type::Int, Type::Int], &[Type::Ref, Type::Int, Type::Int])
            .expect("descriptor should build");
    let mut driver = JitDriver::<ResumeMappedState>::with_descriptor(2, descriptor);
    let frame_ptr = 0x4444usize;
    let mut state = ResumeMappedState {
        frame: frame_ptr,
        stackpos: 0,
        top: 5,
    };

    assert!(!driver.back_edge_keyed(444, 444, &mut state, &(), || {}));
    assert!(!driver.back_edge_keyed(444, 444, &mut state, &(), || {}));
    driver.merge_point(|ctx, sym| {
        ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[sym[1]], 1, &[sym[1]]);
        let one = ctx.const_int(1);
        sym[1] = ctx.record_op(OpCode::IntAdd, &[sym[1], one]);
        ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[sym[1]], 1, &[sym[1]]);
        TraceAction::CloseLoop
    });

    let mut resume = ResumeDataBuilder::new();
    resume.push_frame(444);
    resume.set_slot_constant(0, frame_ptr as i64);
    resume.map_slot(1, 0);
    resume.set_slot_constant(2, 99);
    driver
        .meta_interp_mut()
        .attach_resume_data(444, 1, resume.build());

    state.frame = 0;
    state.stackpos = 1;
    state.top = -10;
    match driver.run_compiled_with_blackhole_fallback_keyed(444, &mut state, || {}) {
        DriverRunOutcome::GuardFailure {
            restored,
            via_blackhole,
            ..
        } => {
            assert!(restored);
            assert!(via_blackhole);
        }
        other => panic!("expected GuardFailure outcome, got {other:?}"),
    }

    assert_eq!(state.frame, frame_ptr);
    assert_eq!(state.stackpos, 2);
    assert_eq!(state.top, 99);
}

#[test]
fn declarative_driver_guard_failure_materializes_virtual_ref_from_resume_state() {
    let descriptor =
        VirtualResumeDriver::descriptor(&[Type::Int, Type::Int], &[Type::Ref, Type::Int])
            .expect("descriptor should build");
    let mut driver = JitDriver::<VirtualResumeState>::with_descriptor(2, descriptor);
    let mut state = VirtualResumeState {
        obj: 0,
        flag: 0,
        materialized_ref: 0xfeedusize,
        materialize_calls: 0,
    };

    assert!(!driver.back_edge_keyed(555, 555, &mut state, &(), || {}));
    assert!(!driver.back_edge_keyed(555, 555, &mut state, &(), || {}));
    driver.merge_point(|ctx, sym| {
        ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[sym[1]], 1, &[sym[1]]);
        let one = ctx.const_int(1);
        sym[1] = ctx.record_op(OpCode::IntAdd, &[sym[1], one]);
        ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[sym[1]], 1, &[sym[1]]);
        TraceAction::CloseLoop
    });

    let mut resume = ResumeDataBuilder::new();
    resume.push_frame(555);
    let virtual_index = resume.add_virtual_struct(
        0,
        7,
        vec![(3, majit_meta::resume::ResumeValueSource::Constant(55))],
    );
    resume.set_slot_virtual(0, virtual_index);
    resume.map_slot(1, 0);
    driver
        .meta_interp_mut()
        .attach_resume_data(555, 1, resume.build());

    state.obj = 0;
    state.flag = 1;
    match driver.run_compiled_with_blackhole_fallback_keyed(555, &mut state, || {}) {
        DriverRunOutcome::GuardFailure {
            restored,
            via_blackhole,
            ..
        } => {
            assert!(restored);
            assert!(via_blackhole);
        }
        other => panic!("expected GuardFailure outcome, got {other:?}"),
    }

    assert_eq!(state.obj, 0xfeedusize);
    assert_eq!(state.flag, 2);
    assert_eq!(state.materialize_calls, 1);
}

#[test]
fn jit_state_restore_guard_failure_materializes_nested_virtual_refs_in_dependency_order() {
    let mut state = NestedVirtualResumeState {
        outer: 0,
        inner: 0,
        alias: 0,
        outer_ref: 0xbeefusize,
        inner_ref: 0xabbausize,
        materialize_order: Vec::new(),
    };
    let meta = TestMeta { header_pc: 556 };
    let mut resume = ResumeDataBuilder::new();
    resume.push_frame(556);
    let inner = resume.add_virtual_struct(
        0,
        10,
        vec![(0, majit_meta::resume::ResumeValueSource::Constant(77))],
    );
    let outer = resume.add_virtual_obj(
        0,
        20,
        vec![
            (0, majit_meta::resume::ResumeValueSource::Virtual(inner)),
            (1, majit_meta::resume::ResumeValueSource::Constant(99)),
        ],
    );
    resume.set_slot_virtual(0, outer);
    resume.set_slot_virtual(1, inner);
    resume.set_slot_virtual(2, inner);
    let reconstructed_state = resume.build().reconstruct_state(&[]);

    assert!(state.restore_guard_failure(
        &meta,
        &[],
        Some(&reconstructed_state),
        &reconstructed_state.virtuals,
        &[],
        &majit_meta::blackhole::ExceptionState::default(),
    ));

    assert_eq!(state.outer, state.outer_ref);
    assert_eq!(state.inner, state.inner_ref);
    assert_eq!(state.alias, state.inner_ref);
    assert_eq!(state.materialize_order, vec![0, 1]);
}

#[test]
fn jit_state_restore_guard_failure_replays_pending_writes_with_virtual_target_and_value() {
    let mut parent_cell = Box::new(PendingRefCell { child: 0 });
    let mut child_cell = Box::new(PendingRefCell { child: 0 });
    let parent_ref = (&mut *parent_cell as *mut PendingRefCell) as usize;
    let child_ref = (&mut *child_cell as *mut PendingRefCell) as usize;
    let mut state = PendingVirtualWriteState {
        parent: 0,
        child: 0,
        parent_ref,
        child_ref,
        materialize_order: Vec::new(),
    };
    let meta = TestMeta { header_pc: 557 };
    let mut resume = ResumeDataBuilder::new();
    resume.push_frame(557);
    let parent = resume.add_virtual_struct(0, 30, vec![]);
    let child = resume.add_virtual_struct(0, 31, vec![]);
    resume.set_slot_virtual(0, parent);
    resume.set_slot_virtual(1, child);
    resume.add_pending_field_write(
        21,
        majit_meta::resume::ResumeValueSource::Virtual(parent),
        majit_meta::resume::ResumeValueSource::Virtual(child),
    );
    let reconstructed_state = resume.build().reconstruct_state(&[]);

    assert!(state.restore_guard_failure(
        &meta,
        &[],
        Some(&reconstructed_state),
        &reconstructed_state.virtuals,
        &reconstructed_state.pending_fields,
        &majit_meta::blackhole::ExceptionState::default(),
    ));

    assert_eq!(state.parent, parent_ref);
    assert_eq!(state.child, child_ref);
    assert_eq!(parent_cell.child, child_ref);
    assert_eq!(state.materialize_order, vec![0, 1]);
}

#[test]
fn declarative_driver_guard_failure_replays_pending_field_writes() {
    let descriptor =
        PendingWriteDriver::descriptor(&[Type::Int, Type::Int], &[Type::Ref, Type::Int])
            .expect("descriptor should build");
    let mut driver = JitDriver::<PendingWriteState>::with_descriptor(2, descriptor);
    let mut cell = PendingWriteCell { field: 11 };
    let mut state = PendingWriteState {
        obj: (&mut cell as *mut PendingWriteCell) as usize,
        flag: 0,
    };

    assert!(!driver.back_edge_keyed(666, 666, &mut state, &(), || {}));
    assert!(!driver.back_edge_keyed(666, 666, &mut state, &(), || {}));
    driver.merge_point(|ctx, sym| {
        ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[sym[1]], 1, &[sym[1]]);
        let one = ctx.const_int(1);
        sym[1] = ctx.record_op(OpCode::IntAdd, &[sym[1], one]);
        ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[sym[1]], 1, &[sym[1]]);
        TraceAction::CloseLoop
    });

    let mut resume = ResumeDataBuilder::new();
    resume.push_frame(666);
    resume.set_slot_constant(0, state.obj as i64);
    resume.map_slot(1, 0);
    resume.add_pending_field_write(
        9,
        majit_meta::resume::ResumeValueSource::Constant(state.obj as i64),
        majit_meta::resume::ResumeValueSource::Constant(77),
    );
    driver
        .meta_interp_mut()
        .attach_resume_data(666, 1, resume.build());

    state.flag = 1;
    match driver.run_compiled_with_blackhole_fallback_keyed(666, &mut state, || {}) {
        DriverRunOutcome::GuardFailure {
            restored,
            via_blackhole,
            ..
        } => {
            assert!(restored);
            assert!(via_blackhole);
        }
        other => panic!("expected GuardFailure outcome, got {other:?}"),
    }

    assert_eq!(state.flag, 2);
    assert_eq!(cell.field, 77);
}

#[test]
fn declarative_driver_guard_failure_replays_pending_array_writes_via_layout_hook() {
    let descriptor =
        PendingArrayWriteDriver::descriptor(&[Type::Int, Type::Int], &[Type::Ref, Type::Int])
            .expect("descriptor should build");
    let mut driver = JitDriver::<PendingArrayWriteState>::with_descriptor(2, descriptor);
    let mut array = vec![10_i64, 20_i64, 30_i64];
    let mut state = PendingArrayWriteState {
        array: array.as_mut_ptr() as usize,
        flag: 0,
    };

    assert!(!driver.back_edge_keyed(888, 888, &mut state, &(), || {}));
    assert!(!driver.back_edge_keyed(888, 888, &mut state, &(), || {}));
    driver.merge_point(|ctx, sym| {
        ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[sym[1]], 1, &[sym[1]]);
        let one = ctx.const_int(1);
        sym[1] = ctx.record_op(OpCode::IntAdd, &[sym[1], one]);
        ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[sym[1]], 1, &[sym[1]]);
        TraceAction::CloseLoop
    });

    let mut resume = ResumeDataBuilder::new();
    resume.push_frame(888);
    resume.set_slot_constant(0, state.array as i64);
    resume.map_slot(1, 0);
    resume.add_pending_arrayitem_write(
        12,
        majit_meta::resume::ResumeValueSource::Constant(state.array as i64),
        1,
        majit_meta::resume::ResumeValueSource::Constant(88),
    );
    driver
        .meta_interp_mut()
        .attach_resume_data(888, 1, resume.build());

    state.flag = 1;
    match driver.run_compiled_with_blackhole_fallback_keyed(888, &mut state, || {}) {
        DriverRunOutcome::GuardFailure {
            restored,
            via_blackhole,
            ..
        } => {
            assert!(restored);
            assert!(via_blackhole);
        }
        other => panic!("expected GuardFailure outcome, got {other:?}"),
    }

    assert_eq!(state.flag, 2);
    assert_eq!(array, vec![10, 88, 30]);
}

#[test]
fn declarative_driver_guard_failure_can_restore_multi_frame_resume_state() {
    let descriptor = MultiFrameDriver::descriptor(&[Type::Int, Type::Int], &[Type::Ref, Type::Int])
        .expect("descriptor should build");
    let mut driver = JitDriver::<MultiFrameResumeState>::with_descriptor(2, descriptor);
    let frame_ptr = 0xabcusize;
    let mut state = MultiFrameResumeState {
        frame: frame_ptr,
        flag: 0,
        restored_pcs: Vec::new(),
    };

    assert!(!driver.back_edge_keyed(777, 777, &mut state, &(), || {}));
    assert!(!driver.back_edge_keyed(777, 777, &mut state, &(), || {}));
    driver.merge_point(|ctx, sym| {
        ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[sym[1]], 1, &[sym[1]]);
        let one = ctx.const_int(1);
        sym[1] = ctx.record_op(OpCode::IntAdd, &[sym[1], one]);
        ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[sym[1]], 1, &[sym[1]]);
        TraceAction::CloseLoop
    });

    let mut resume = ResumeDataBuilder::new();
    resume.push_frame(100);
    resume.set_slot_constant(0, frame_ptr as i64);
    resume.set_slot_constant(1, 1);
    resume.push_frame(200);
    resume.set_slot_constant(0, frame_ptr as i64);
    resume.map_slot(1, 0);
    driver
        .meta_interp_mut()
        .attach_resume_data(777, 1, resume.build());

    state.frame = 0;
    state.flag = 1;
    match driver.run_compiled_with_blackhole_fallback_keyed(777, &mut state, || {}) {
        DriverRunOutcome::GuardFailure {
            restored,
            via_blackhole,
            ..
        } => {
            assert!(restored);
            assert!(via_blackhole);
        }
        other => panic!("expected GuardFailure outcome, got {other:?}"),
    }

    assert_eq!(state.frame, frame_ptr);
    assert_eq!(state.flag, 2);
    assert_eq!(state.restored_pcs, vec![100, 200]);
}

#[test]
fn declarative_driver_guard_failure_can_restore_multi_frame_state_via_generic_frame_hooks() {
    let descriptor =
        GenericMultiFrameDriver::descriptor(&[Type::Int, Type::Int], &[Type::Ref, Type::Int])
            .expect("descriptor should build");
    let mut driver = JitDriver::<GenericMultiFrameResumeState>::with_descriptor(2, descriptor);
    let materialized_ref = 0xfeedusize;
    let mut state = GenericMultiFrameResumeState {
        frame: 0,
        flag: 0,
        restored_pcs: Vec::new(),
        restored_frames: Vec::new(),
        materialized_ref,
        materialize_calls: 0,
    };

    assert!(!driver.back_edge_keyed(778, 778, &mut state, &(), || {}));
    assert!(!driver.back_edge_keyed(778, 778, &mut state, &(), || {}));
    driver.merge_point(|ctx, sym| {
        ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[sym[1]], 1, &[sym[1]]);
        let one = ctx.const_int(1);
        sym[1] = ctx.record_op(OpCode::IntAdd, &[sym[1], one]);
        ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[sym[1]], 1, &[sym[1]]);
        TraceAction::CloseLoop
    });

    let mut resume = ResumeDataBuilder::new();
    resume.push_frame(300);
    let virtual_index = resume.add_virtual_struct(0, 55, vec![]);
    resume.set_slot_virtual(0, virtual_index);
    resume.set_slot_constant(1, 1);
    resume.push_frame(400);
    resume.set_slot_virtual(0, virtual_index);
    resume.map_slot(1, 0);
    driver
        .meta_interp_mut()
        .attach_resume_data(778, 1, resume.build());

    state.flag = 1;
    match driver.run_compiled_with_blackhole_fallback_keyed(778, &mut state, || {}) {
        DriverRunOutcome::GuardFailure {
            restored,
            via_blackhole,
            ..
        } => {
            assert!(restored);
            assert!(via_blackhole);
        }
        other => panic!("expected GuardFailure outcome, got {other:?}"),
    }

    assert_eq!(state.frame, materialized_ref);
    assert_eq!(state.flag, 2);
    assert_eq!(state.restored_pcs, vec![300, 400]);
    assert_eq!(
        state.restored_frames,
        vec![(materialized_ref, 1), (materialized_ref, 2)]
    );
    assert_eq!(state.materialize_calls, 1);
}

#[test]
fn declarative_driver_generic_multi_frame_restore_reuses_virtual_cache_for_pending_writes() {
    let descriptor =
        GenericMultiFrameDriver::descriptor(&[Type::Int, Type::Int], &[Type::Ref, Type::Int])
            .expect("descriptor should build");
    let mut driver = JitDriver::<GenericMultiFrameResumeState>::with_descriptor(2, descriptor);
    let mut cell = PendingWriteCell { field: 0 };
    let cell_ptr = (&mut cell as *mut PendingWriteCell) as usize;
    let mut state = GenericMultiFrameResumeState {
        frame: 0,
        flag: 0,
        restored_pcs: Vec::new(),
        restored_frames: Vec::new(),
        materialized_ref: cell_ptr,
        materialize_calls: 0,
    };

    assert!(!driver.back_edge_keyed(779, 779, &mut state, &(), || {}));
    assert!(!driver.back_edge_keyed(779, 779, &mut state, &(), || {}));
    driver.merge_point(|ctx, sym| {
        ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[sym[1]], 1, &[sym[1]]);
        let one = ctx.const_int(1);
        sym[1] = ctx.record_op(OpCode::IntAdd, &[sym[1], one]);
        ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[sym[1]], 1, &[sym[1]]);
        TraceAction::CloseLoop
    });

    let mut resume = ResumeDataBuilder::new();
    resume.push_frame(500);
    let virtual_index = resume.add_virtual_struct(0, 56, vec![]);
    resume.set_slot_virtual(0, virtual_index);
    resume.set_slot_constant(1, 1);
    resume.push_frame(600);
    resume.set_slot_virtual(0, virtual_index);
    resume.map_slot(1, 0);
    resume.add_pending_field_write(
        31,
        majit_meta::resume::ResumeValueSource::Virtual(virtual_index),
        majit_meta::resume::ResumeValueSource::Constant(77),
    );
    driver
        .meta_interp_mut()
        .attach_resume_data(779, 1, resume.build());

    state.flag = 1;
    match driver.run_compiled_with_blackhole_fallback_keyed(779, &mut state, || {}) {
        DriverRunOutcome::GuardFailure {
            restored,
            via_blackhole,
            ..
        } => {
            assert!(restored);
            assert!(via_blackhole);
        }
        other => panic!("expected GuardFailure outcome, got {other:?}"),
    }

    assert_eq!(state.materialize_calls, 1);
    assert_eq!(cell.field, 77);
}

#[test]
fn declarative_driver_guard_failure_uses_resume_layout_slot_types_for_generic_restore() {
    let descriptor =
        LayoutTypedFrameDriver::descriptor(&[Type::Int, Type::Int], &[Type::Ref, Type::Int])
            .expect("descriptor should build");
    let mut driver = JitDriver::<LayoutTypedFrameRestoreState>::with_descriptor(2, descriptor);
    let frame_ptr = 0xfaceusize;
    let mut state = LayoutTypedFrameRestoreState {
        frame: frame_ptr,
        flag: 1,
        fallback_restore_calls: 0,
        layout_restore_calls: 0,
        restored_frames: Vec::new(),
    };

    assert!(!driver.back_edge_keyed(780, 780, &mut state, &(), || {}));
    assert!(!driver.back_edge_keyed(780, 780, &mut state, &(), || {}));
    driver.merge_point(|ctx, sym| {
        ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[sym[1]], 1, &[sym[1]]);
        let one = ctx.const_int(1);
        sym[1] = ctx.record_op(OpCode::IntAdd, &[sym[1], one]);
        ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[sym[1]], 1, &[sym[1]]);
        TraceAction::CloseLoop
    });

    let mut resume = ResumeDataBuilder::new();
    resume.push_frame(780);
    resume.set_slot_constant(0, frame_ptr as i64);
    resume.map_slot(1, 0);
    driver
        .meta_interp_mut()
        .attach_resume_data(780, 1, resume.build());

    state.frame = 0;
    state.flag = 1;
    match driver.run_compiled_with_blackhole_fallback_keyed(780, &mut state, || {}) {
        DriverRunOutcome::GuardFailure {
            restored,
            via_blackhole,
            ..
        } => {
            assert!(restored);
            assert!(via_blackhole);
        }
        other => panic!("expected GuardFailure outcome, got {other:?}"),
    }

    assert_eq!(state.frame, frame_ptr);
    assert_eq!(state.flag, 2);
    assert_eq!(state.layout_restore_calls, 1);
    assert_eq!(state.fallback_restore_calls, 0);
    assert_eq!(state.restored_frames, vec![(frame_ptr, 2)]);
}
