use majit_ir::{GreenKey, Type, VarKind};
use majit_macros::jit_driver;

#[jit_driver(greens = [pc, code], reds = [frame, stack], virtualizable = frame)]
struct DriverWithVirtualizable;

#[jit_driver(greens = [pc], reds = [frame])]
struct SimpleDriver;

#[test]
fn jit_driver_generates_runtime_descriptor_with_virtualizable_metadata() {
    let descriptor =
        DriverWithVirtualizable::descriptor(&[Type::Int, Type::Ref], &[Type::Ref, Type::Int])
            .expect("descriptor should build");

    assert_eq!(descriptor.num_greens(), 2);
    assert_eq!(descriptor.num_reds(), 2);
    assert_eq!(descriptor.vars.len(), 4);
    assert_eq!(descriptor.vars[0].name, "pc");
    assert_eq!(descriptor.vars[0].tp, Type::Int);
    assert_eq!(descriptor.vars[0].kind, VarKind::Green);
    assert_eq!(descriptor.vars[1].name, "code");
    assert_eq!(descriptor.vars[1].tp, Type::Ref);
    assert_eq!(descriptor.vars[2].name, "frame");
    assert_eq!(descriptor.vars[2].tp, Type::Ref);
    assert_eq!(descriptor.vars[2].kind, VarKind::Red);
    assert_eq!(descriptor.vars[3].name, "stack");
    assert_eq!(descriptor.vars[3].tp, Type::Int);
    assert_eq!(descriptor.virtualizable.as_deref(), Some("frame"));
    assert_eq!(
        descriptor.virtualizable().map(|var| var.name.as_str()),
        Some("frame")
    );
}

#[test]
fn jit_driver_descriptor_rejects_wrong_type_counts() {
    let green_err =
        SimpleDriver::descriptor(&[], &[Type::Int]).expect_err("green count mismatch must fail");
    assert_eq!(green_err, "wrong number of green variable types");

    let red_err =
        SimpleDriver::descriptor(&[Type::Int], &[]).expect_err("red count mismatch must fail");
    assert_eq!(red_err, "wrong number of red variable types");
}

#[test]
fn jit_driver_descriptor_rejects_nonref_virtualizable_red() {
    let err = DriverWithVirtualizable::descriptor(&[Type::Int, Type::Ref], &[Type::Int, Type::Int])
        .expect_err("virtualizable must be a ref");
    assert_eq!(err, "virtualizable red must have Ref type");
}

#[test]
fn jit_driver_generates_checked_green_key_builder() {
    let key = DriverWithVirtualizable::green_key(&[10, 20]).expect("green key should build");
    assert_eq!(key, GreenKey::new(vec![10, 20]));

    let err = DriverWithVirtualizable::green_key(&[10]).expect_err("length mismatch must fail");
    assert_eq!(err, "wrong number of green key values");
}
