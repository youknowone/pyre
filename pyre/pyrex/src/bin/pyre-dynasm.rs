// Opt-in (`--features mimalloc`): mimalloc roughly halves bignum-heavy
// workloads by replacing the platform default heap on the per-rbigint-op limb
// Vec allocation. OFF by default.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    pyrex::main_entry("pyre-dynasm");
}
