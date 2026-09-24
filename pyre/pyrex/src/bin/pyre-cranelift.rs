#[global_allocator]
static GLOBAL: pyrex::ProcessAllocator = pyrex::ProcessAllocator::new();

fn main() {
    pyrex::memory_ceiling::install();
    pyrex::main_entry("pyre-cranelift");
}
