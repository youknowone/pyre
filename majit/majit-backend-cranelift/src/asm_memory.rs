use std::io;
use std::sync::Arc;

use cranelift_jit::{BranchProtection, JITMemoryKind, JITMemoryProvider};
use cranelift_module::{ModuleError, ModuleResult};
use majit_backend::{AsmMemoryBlock, AsmMemoryManager};

struct Allocation {
    block: AsmMemoryBlock,
    kind: JITMemoryKind,
    finalized: bool,
}

struct Shared {
    arena: Arc<AsmMemoryManager>,
    allocations: parking_lot::Mutex<Vec<Allocation>>,
}

/// Handle retained by the backend after the provider itself moves into
/// `JITModule`. It transfers finalized executable ranges to the owning
/// `CompiledLoopToken`, matching `BaseAssembler.get_asmmemmgr_blocks`.
pub(crate) struct CraneliftArenaHandle {
    shared: Arc<Shared>,
}

impl CraneliftArenaHandle {
    pub(crate) fn take_executable(&self, ptr: *const u8) -> Option<AsmMemoryBlock> {
        let mut allocations = self.shared.allocations.lock();
        let position = allocations.iter().position(|allocation| {
            matches!(allocation.kind, JITMemoryKind::Executable)
                && allocation.finalized
                && allocation.block.contains(ptr)
        })?;
        Some(allocations.remove(position).block)
    }
}

/// Cranelift's `ArenaMemoryProvider` proves that one provider may suballocate
/// a retained mapping. This provider delegates those allocations to pyre's
/// RPython-compatible free list and leaves lifetime ownership with CLTs.
pub(crate) struct CraneliftArenaMemoryProvider {
    shared: Arc<Shared>,
}

impl CraneliftArenaMemoryProvider {
    pub(crate) fn new(arena: Arc<AsmMemoryManager>) -> (Self, CraneliftArenaHandle) {
        let shared = Arc::new(Shared {
            arena,
            allocations: parking_lot::Mutex::new(Vec::new()),
        });
        (
            Self {
                shared: Arc::clone(&shared),
            },
            CraneliftArenaHandle { shared },
        )
    }
}

impl JITMemoryProvider for CraneliftArenaMemoryProvider {
    fn allocate(&mut self, size: usize, align: u64, kind: JITMemoryKind) -> io::Result<*mut u8> {
        let align = usize::try_from(align)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "JIT alignment overflow"))?;
        if !align.is_power_of_two() || align > region::page::size() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "JIT alignment must be a power of two no larger than a page",
            ));
        }
        let used = if matches!(kind, JITMemoryKind::Executable) {
            size
        } else {
            0
        };
        let block = self.shared.arena.allocate(size, used)?;
        let ptr = block.ptr() as *mut u8;
        self.shared.allocations.lock().push(Allocation {
            block,
            kind,
            finalized: false,
        });
        Ok(ptr)
    }

    unsafe fn free_memory(&mut self) {
        self.shared.allocations.lock().clear();
    }

    fn finalize(&mut self, branch_protection: BranchProtection) -> ModuleResult<()> {
        if branch_protection != BranchProtection::None {
            return Err(ModuleError::Backend(
                io::Error::new(
                    io::ErrorKind::Unsupported,
                    "branch-protected Cranelift arena mappings are not configured",
                )
                .into(),
            ));
        }
        let mut allocations = self.shared.allocations.lock();
        for allocation in allocations.iter_mut().filter(|entry| !entry.finalized) {
            let result = match allocation.kind {
                JITMemoryKind::Executable => allocation.block.make_executable(),
                JITMemoryKind::ReadOnly => allocation.block.make_read_only(),
                JITMemoryKind::Writable => Ok(()),
            };
            result.map_err(|error| ModuleError::Backend(error.into()))?;
            allocation.finalized = true;
        }
        Ok(())
    }
}
