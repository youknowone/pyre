// Enabled by default; allocator-sensitive diagnostics can disable default
// features and select only `dynasm`. See pyrex/Cargo.toml.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    pyrex::main_entry("pyre-dynasm");
}
