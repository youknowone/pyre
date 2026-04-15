//! Smoke tests for `#[jit_struct]` — descr.py:105-127 / :218-239 auto-discovery.

use majit_ir::descr::{GcCache, LLType};
use majit_ir::value::Type;
use majit_macros::jit_struct;

#[jit_struct]
struct Node {
    value: i64,
    next: Option<Box<Node>>,
}

#[jit_struct]
struct Stack {
    head: Option<Box<Node>>,
    size: usize,
}

#[test]
fn field_names_are_declaration_order() {
    assert_eq!(Node::__MAJIT_FIELD_NAMES, &["value", "next"]);
    assert_eq!(Stack::__MAJIT_FIELD_NAMES, &["head", "size"]);
}

#[test]
fn type_id_is_stable_within_process() {
    let a = Node::__majit_type_id();
    let b = Node::__majit_type_id();
    assert_eq!(a, b);
    assert_ne!(Node::__majit_type_id(), Stack::__majit_type_id());
}

#[test]
fn register_descrs_populates_gc_cache() {
    let mut gc = GcCache::new();
    let node_size = Node::__majit_register_descrs(&mut gc);
    let size_sd = node_size.as_size_descr().expect("SizeDescr");
    assert_eq!(size_sd.size(), std::mem::size_of::<Node>());

    // Field cache populated with the two named fields.
    let key = LLType::struct_key(Node::__majit_type_id());
    let field_cache = gc._cache_field.get(&key).expect("field cache entry");
    assert_eq!(field_cache.len(), 2);
    assert!(field_cache.contains_key("value"));
    assert!(field_cache.contains_key("next"));

    let value_fd = field_cache.get("value").unwrap().as_field_descr().unwrap();
    assert_eq!(value_fd.offset(), std::mem::offset_of!(Node, value));
    assert_eq!(value_fd.field_size(), std::mem::size_of::<i64>());
    assert_eq!(value_fd.field_type(), Type::Int);
    assert_eq!(value_fd.index_in_parent(), 0);

    let next_fd = field_cache.get("next").unwrap().as_field_descr().unwrap();
    assert_eq!(next_fd.offset(), std::mem::offset_of!(Node, next));
    assert_eq!(next_fd.field_type(), Type::Ref);
    assert_eq!(next_fd.index_in_parent(), 1);
}

#[test]
fn register_descrs_is_idempotent() {
    let mut gc = GcCache::new();
    let a = Node::__majit_register_descrs(&mut gc);
    let b = Node::__majit_register_descrs(&mut gc);
    // Second call must return the cached SizeDescr, not a new one.
    assert!(std::sync::Arc::ptr_eq(&a, &b));
    let key = LLType::struct_key(Node::__majit_type_id());
    assert_eq!(gc._cache_field.get(&key).unwrap().len(), 2);
}

#[test]
fn parent_descr_backref_wired() {
    let mut gc = GcCache::new();
    let node_size = Node::__majit_register_descrs(&mut gc);
    let key = LLType::struct_key(Node::__majit_type_id());
    let value_fd = gc
        ._cache_field
        .get(&key)
        .unwrap()
        .get("value")
        .unwrap()
        .as_field_descr()
        .unwrap();
    // descr.py:238: fielddescr.parent_descr = get_size_descr(...)
    let parent = value_fd.get_parent_descr().expect("parent_descr wired");
    assert!(std::sync::Arc::ptr_eq(&parent, &node_size));
}

// ─────────────────────────────────────────────────────────────────────
// rpaheui shape parity: `LinkedList`/`Stack`/`Queue` / `Port` structures
// from `rpaheui/aheui/storage/linkedlist.py` recoded as `#[jit_struct]`.
// These tests document the end-state of the generic-storage migration:
// once the tracer consumes descrs through GcCache lookup, the existing
// `linked_list_*` trait methods on `JitCodeSym` become redundant.
// ─────────────────────────────────────────────────────────────────────

/// rpaheui/aheui/storage/linkedlist.py:4-11 (`class Node`).
#[jit_struct]
struct AheuiNode {
    value: i64,
    next: Option<Box<AheuiNode>>,
}

/// rpaheui/aheui/storage/linkedlist.py:67-91 (`class Stack(LinkedList)`).
#[jit_struct]
struct AheuiStack {
    head: Option<Box<AheuiNode>>,
    size: usize,
}

/// rpaheui/aheui/storage/linkedlist.py:94-122 (`class Queue(LinkedList)`).
#[jit_struct]
struct AheuiQueue {
    head: Option<Box<AheuiNode>>,
    tail: Option<Box<AheuiNode>>,
    size: usize,
}

/// rpaheui/aheui/storage/linkedlist.py:125-148 (`class Port(LinkedList)`).
#[jit_struct]
struct AheuiPort {
    head: Option<Box<AheuiNode>>,
    size: usize,
    last_push: i64,
}

#[test]
fn aheui_shapes_register_descrs() {
    let mut gc = GcCache::new();
    let node = AheuiNode::__majit_register_descrs(&mut gc);
    let stack = AheuiStack::__majit_register_descrs(&mut gc);
    let queue = AheuiQueue::__majit_register_descrs(&mut gc);
    let port = AheuiPort::__majit_register_descrs(&mut gc);

    // Each shape gets a distinct SizeDescr.
    for (a, b) in [
        (&node, &stack),
        (&node, &queue),
        (&node, &port),
        (&stack, &queue),
        (&stack, &port),
        (&queue, &port),
    ] {
        assert!(!std::sync::Arc::ptr_eq(a, b));
    }

    assert_eq!(AheuiNode::__MAJIT_FIELD_NAMES, &["value", "next"]);
    assert_eq!(AheuiStack::__MAJIT_FIELD_NAMES, &["head", "size"]);
    assert_eq!(AheuiQueue::__MAJIT_FIELD_NAMES, &["head", "tail", "size"]);
    assert_eq!(
        AheuiPort::__MAJIT_FIELD_NAMES,
        &["head", "size", "last_push"]
    );
}

#[test]
fn aheui_node_field_types_match_rpaheui() {
    let mut gc = GcCache::new();
    let _ = AheuiNode::__majit_register_descrs(&mut gc);
    let key = LLType::struct_key(AheuiNode::__majit_type_id());
    let fields = gc._cache_field.get(&key).unwrap();

    // `value: i64` → Int (rpaheui bigint lowered to i64 for the integer trace).
    let value_fd = fields.get("value").unwrap().as_field_descr().unwrap();
    assert_eq!(value_fd.field_type(), Type::Int);

    // `next: Option<Box<Node>>` → Ref (the Node* in RPython).
    let next_fd = fields.get("next").unwrap().as_field_descr().unwrap();
    assert_eq!(next_fd.field_type(), Type::Ref);
}

#[test]
fn aheui_queue_tail_descr_distinct_from_head() {
    let mut gc = GcCache::new();
    let _ = AheuiQueue::__majit_register_descrs(&mut gc);
    let key = LLType::struct_key(AheuiQueue::__majit_type_id());
    let fields = gc._cache_field.get(&key).unwrap();
    let head = fields.get("head").unwrap();
    let tail = fields.get("tail").unwrap();
    assert!(!std::sync::Arc::ptr_eq(head, tail));
    let head_fd = head.as_field_descr().unwrap();
    let tail_fd = tail.as_field_descr().unwrap();
    assert_ne!(head_fd.offset(), tail_fd.offset());
    assert_eq!(head_fd.index_in_parent(), 0);
    assert_eq!(tail_fd.index_in_parent(), 1);
}
