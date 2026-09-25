//! Per-loop wasm resources owned by `CompiledLoopToken.asmmemmgr_blocks`.
//!
//! `llmodel.py` `AbstractLLCPU.free_loop_and_bridges` frees the loop by
//! dropping `asmmemmgr_blocks` and `asmmemmgr_gcreftracers`. A target another
//! token still jumps to stays alive through `JitCellToken.record_jump_to`
//! (`history.py`), so this token is not freed while that caller exists.

use std::any::Any;

use majit_backend::JitCellToken;

/// Compile-time maps, table slots, label rows, fail indices, and bridge
/// cells of one emission. Pushed into `asmmemmgr_blocks`; `Drop` runs when
/// `free_loop_and_bridges` clears that list.
#[derive(Default)]
pub struct LoopAsmResources {
    pub gcmaps: Vec<Box<[usize]>>,
    /// Real `__indirect_function_table` pair bases. `0` is not a host slot.
    pub table_slots: Vec<u32>,
    pub label_ids: Vec<usize>,
    pub label_handle: u32,
    /// `JitCellToken.number` that published `label_ids`. A later compile that
    /// overwrites the same descr updates `LabelTarget.owner_token`, and this
    /// drop leaves that row alone.
    pub label_owner: u64,
    /// One [`crate::failguard::FailDescrCell`] per guard exit. The address is
    /// what the exit stores in `jf_descr` (`get_latest_descr`). The cell
    /// outlives every module that can still leave through that exit because
    /// this block is `asmmemmgr_blocks`.
    pub fail_cells: Vec<Box<crate::failguard::FailDescrCell>>,
    /// CALL_ASSEMBLER indirect cell. Its address is `JitCellToken._ll_function_addr`.
    pub ca_entry: Option<Box<crate::failguard::WasmCaDispatchEntry>>,
    pub bridge_cells: Vec<Box<[u32]>>,
    /// `[descr_cell, gcmap]` pairs the exit loads. The address is baked
    /// into the module; the allocation does not move.
    pub exit_table: Option<Box<[usize]>>,
}

impl LoopAsmResources {
    pub fn park_gcmap(&mut self, map: Box<[usize]>) -> usize {
        let ptr = map.as_ptr() as usize;
        self.gcmaps.push(map);
        ptr
    }

    /// Stable address baked into `jf_descr`. The `Box` does not move for
    /// the rest of this block's life.
    pub fn alloc_fail_cell(
        &mut self,
        descr: std::sync::Arc<crate::failguard::WasmFailDescr>,
    ) -> usize {
        let cell = Box::new(crate::failguard::FailDescrCell::new(descr));
        let ptr = &*cell as *const crate::failguard::FailDescrCell as usize;
        self.fail_cells.push(cell);
        ptr
    }

    /// `count` guard exits, two `usize` words each. Returns the guest
    /// address baked into the module, or 0 when there is nothing to name.
    pub fn alloc_exit_table(&mut self, count: usize) -> usize {
        if count == 0 {
            return 0;
        }
        let table = vec![0usize; count * 2].into_boxed_slice();
        let ptr = table.as_ptr() as usize;
        self.exit_table = Some(table);
        ptr
    }

    pub fn write_exit_slot(&mut self, index: usize, descr_cell: usize, gcmap: usize) {
        let Some(table) = self.exit_table.as_mut() else {
            return;
        };
        let base = index * 2;
        if base + 1 < table.len() {
            table[base] = descr_cell;
            table[base + 1] = gcmap;
        }
    }
}

impl Drop for LoopAsmResources {
    fn drop(&mut self) {
        for slot in self.table_slots.drain(..) {
            if slot != 0 {
                #[cfg(target_arch = "wasm32")]
                crate::glue::free(slot);
                #[cfg(not(target_arch = "wasm32"))]
                let _ = slot;
            }
        }
        if self.label_owner != 0 {
            let mut reg = crate::failguard::LABEL_TARGETS.lock();
            if let Some(labels) = reg.as_mut() {
                for id in self.label_ids.drain(..) {
                    let still_ours = labels
                        .get(&id)
                        .is_some_and(|target| target.owner_token == self.label_owner);
                    if still_ours {
                        labels.remove(&id);
                    }
                }
            }
        }
    }
}

/// `codegen` publishes a home gcmap while building a module. `sink` is the
/// `LoopAsmResources` the caller will push into `asmmemmgr_blocks`. A null
/// sink is a direct `build_wasm_module` test: the map is leaked the same way
/// `allocate_gcmap` leaks via `Box::into_raw`.
pub fn park_gcmap_raw(sink: usize, map: Box<[usize]>) -> usize {
    if sink == 0 {
        return Box::into_raw(map) as *mut usize as usize;
    }
    // SAFETY: `sink` is `&mut LoopAsmResources` held by the compile that
    // called `build_wasm_module`, and that borrow lasts until the build
    // returns.
    let resources = unsafe { &mut *(sink as *mut LoopAsmResources) };
    resources.park_gcmap(map)
}

pub fn push_resources(token: &JitCellToken, resources: LoopAsmResources) {
    let Some(clt) = token.compiled_loop_token() else {
        return;
    };
    let block: Box<dyn Any + Send> = Box::new(resources);
    clt.asmmemmgr_blocks.lock().push(block);
}
