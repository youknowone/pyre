use majit_macros::{dont_look_inside, elidable, jit_driver};

#[jit_driver(greens = [pc, code], reds = [frame])]
struct MyDriver;

#[test]
fn test_driver_greens() {
    assert_eq!(MyDriver::GREENS, &["pc", "code"]);
}

#[test]
fn test_driver_reds() {
    assert_eq!(MyDriver::REDS, &["frame"]);
}

#[test]
fn test_driver_num_greens() {
    assert_eq!(MyDriver::NUM_GREENS, 2);
}

#[test]
fn test_driver_num_reds() {
    assert_eq!(MyDriver::NUM_REDS, 1);
}

#[test]
fn test_driver_num_vars() {
    assert_eq!(MyDriver::NUM_VARS, 3);
}

#[jit_driver(greens = [pc], reds = [frame, stack])]
struct SingleGreenDriver;

#[test]
fn test_single_green_driver() {
    assert_eq!(SingleGreenDriver::GREENS, &["pc"]);
    assert_eq!(SingleGreenDriver::REDS, &["frame", "stack"]);
    assert_eq!(SingleGreenDriver::NUM_GREENS, 1);
    assert_eq!(SingleGreenDriver::NUM_REDS, 2);
    assert_eq!(SingleGreenDriver::NUM_VARS, 3);
}

#[elidable]
fn compute(x: i64) -> i64 {
    x * x + 1
}

#[test]
fn test_elidable_function() {
    assert_eq!(compute(5), 26);
    assert_eq!(compute(0), 1);
    assert_eq!(compute(-3), 10);
}

#[dont_look_inside]
fn opaque_call(x: i64, y: i64) -> i64 {
    x + y
}

#[test]
fn test_opaque_function() {
    assert_eq!(opaque_call(2, 3), 5);
    assert_eq!(opaque_call(-1, 1), 0);
}
