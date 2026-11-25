//! `rpython/rlib/objectmodel.py`.

/// `objectmodel.py` `keepalive_until_here(*values)`: keeps `x` alive up
/// to this point. The translator lowers the call to the `keepalive` llop.
#[inline(never)]
pub fn keepalive_until_here<T>(x: T) {
    std::hint::black_box(x);
}
