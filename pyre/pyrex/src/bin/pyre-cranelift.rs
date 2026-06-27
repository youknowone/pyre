// Opt-in (`--features mimalloc`); see `pyre-dynasm.rs`. OFF by default.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    pyrex::main_entry("pyre-cranelift");
}
