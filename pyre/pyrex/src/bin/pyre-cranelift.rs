// Enabled by default; see `pyre-dynasm.rs`.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    pyrex::main_entry("pyre-cranelift");
}
