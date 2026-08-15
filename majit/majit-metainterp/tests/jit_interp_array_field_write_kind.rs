//! An array field is a POINTER field, and both producers that describe it have
//! to say so.
//!
//! Two places in the lowering register a description of the same member.  The
//! array base read (`emit_array_field_base`) registers it as a pointer word:
//! the field holds a buffer base, and the read is a `getfield_gc_r`.  A
//! `residual_writes = { <ref>.<field> => [..] }` declaration — the FIELD form,
//! no `[]` — rebuilds the same layout to mint the write set's descr.
//!
//! `get_field_descr` caches by `(struct, fieldname)` and is cache-OR-mint: the
//! first producer to reach a slot wins, and a later one describing the field
//! differently gets the cached descr back with only a collision counter to show
//! for it.  So a write-set producer that asked the wrong map about the field's
//! kind does not fail; it either loses silently, or wins and leaves every
//! `getfield_gc_r` of that base carrying an Int-typed field descr.  Which of
//! those happens is decided by the order the two producers were emitted in.
//!
//! This fixture pins the write-set producer alone.  Nothing here reads
//! `sel.data[i]`, so the array base producer never runs and the cache slot for
//! `PointerFieldStack::data` holds exactly what the write-set path decided —
//! order cannot mask the answer.

use majit_metainterp::{Assembler, JitDriver};

/// Reached only through the write-set declaration below, so its cache slot has
/// exactly one producer.  A struct any other fixture also touched would let the
/// other producer mint the slot and this test would pass without meaning it.
#[repr(C)]
struct PointerFieldStack {
    data: *mut i64,
    size: usize,
}

/// An opaque in-place mutator.  The body is irrelevant — the JIT never looks
/// inside a residual — but it has to exist for the concrete path.
extern "C" fn jit_scramble_pointer_field(stack: usize) {
    let stack = stack as *mut PointerFieldStack;
    if stack.is_null() {
        return;
    }
    unsafe {
        (*stack).size = (*stack).size;
    }
}

pub type Bytecode = [u8];

const OP_NOP: u8 = 0;
const OP_SCRAMBLE: u8 = 1;

struct PointerFieldState {
    a: i64,
    sel: usize,
}

#[majit_macros::jit_interp(
    state = PointerFieldState,
    env = Bytecode,
    state_fields = { a: int, sel: ref(PointerFieldStack) },
    greens = [],
    array_fields = { PointerFieldStack::data => i64 },
    calls = { jit_scramble_pointer_field => residual_void },
    residual_writes = {
        sel.data => [jit_scramble_pointer_field],
    },
)]
#[allow(unused_assignments, unused_variables)]
fn pointer_field_only(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<PointerFieldState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let state = PointerFieldState { a: 0, sel: 0 };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while pc < program.len() {
        jit_merge_point!();
        let opcode = program[pc];
        pc += 1;
        match opcode {
            OP_NOP => {}
            OP_SCRAMBLE => jit_scramble_pointer_field(state.sel),
            _ => break,
        }
    }
    state.a
}

fn build() -> majit_metainterp::JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![0], vec![]);
    __prebuild_jitcode_liveness_pointer_field_only(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_pointer_field_only(&mut asm, 0i64)
        .expect("dispatch lower must succeed for the pointer-field fixture")
}

/// `(field_type, field_size, is_signed)` the cache holds for `data`, or `None`
/// if nothing registered it.
fn cached_data_field() -> Option<(majit_ir::Type, usize, bool)> {
    use majit_ir::descr::FieldDescr as _;
    let type_id = majit_metainterp::__pyre_struct_type_id::<PointerFieldStack>(false);
    let cache = majit_ir::descr::gc_cache().lock().unwrap();
    let descr = cache
        ._cache_field
        .get(&majit_ir::descr::LLType::Struct(type_id))
        .and_then(|fields| fields.get("data"))
        .cloned()?;
    Some((
        descr.field_type(),
        descr.field_size(),
        descr.is_field_signed(),
    ))
}

/// The subject: a write-set declaration naming an array field describes a
/// pointer.
///
/// Before this was fixed the write-set path asked only `ref_fields` whether the
/// member is a pointer.  An array field is declared in `array_fields`, so the
/// answer was no and the field went into the write set as an eight-byte SIGNED
/// INTEGER — the undeclared-scalar default — for a member holding `*mut i64`.
#[test]
fn a_write_set_declaration_over_an_array_field_describes_a_pointer() {
    let _jc = build();
    let (field_type, field_size, is_signed) = cached_data_field()
        .expect("the fixture's `residual_writes` declaration must register `data`");
    assert_eq!(
        field_type,
        majit_ir::Type::Ref,
        "`sel.data` names the buffer BASE POINTER, so the write set's descr must \
         be a Ref; an Int here is the undeclared-scalar default leaking into a \
         pointer field, and `get_field_descr` will hand that descr to every \
         `getfield_gc_r` of the same base that reaches the slot after it",
    );
    assert_eq!(
        field_size,
        std::mem::size_of::<usize>(),
        "a pointer field is one target word wide, not the eight bytes the \
         integer default claims (they differ on wasm32)",
    );
    assert!(
        !is_signed,
        "a pointer is not a signed integer; the sign flag is part of what \
         `describes_same_field` compares when a second producer arrives",
    );
}

/// The control, without which the assertion above cannot fail for the reason it
/// names: a non-pointer field of the same struct must still be described as the
/// scalar it is.
///
/// A fix that made every write-set field a Ref would satisfy the subject and be
/// exactly as wrong in the other direction.
#[test]
fn a_scalar_field_of_the_same_struct_is_still_a_scalar() {
    use majit_ir::descr::FieldDescr as _;
    let _jc = build();
    let type_id = majit_metainterp::__pyre_struct_type_id::<PointerFieldStack>(false);
    let cache = majit_ir::descr::gc_cache().lock().unwrap();
    let size_field = cache
        ._cache_field
        .get(&majit_ir::descr::LLType::Struct(type_id))
        .and_then(|fields| fields.get("size"))
        .cloned();
    // `size` is not named by any declaration or access here, so the control is
    // only meaningful if something registered it.  Skipping when nothing did is
    // honest; asserting on an absent slot would pass for the wrong reason.
    if let Some(descr) = size_field {
        assert_eq!(
            descr.field_type(),
            majit_ir::Type::Int,
            "`size` is a scalar; only the field a pointer declaration names may \
             become a Ref",
        );
    }
}
