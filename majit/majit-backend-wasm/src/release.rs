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
    pub fail_indices: Vec<u32>,
    pub bridge_cells: Vec<Box<[u32]>>,
}

impl LoopAsmResources {
    pub fn park_gcmap(&mut self, map: Box<[usize]>) -> usize {
        let ptr = map.as_ptr() as usize;
        self.gcmaps.push(map);
        ptr
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
        let mut reg = crate::failguard::FAIL_DESCR_REGISTRY.lock();
        if let Some(vec) = reg.as_mut() {
            for index in self.fail_indices.drain(..) {
                if index < crate::failguard::FINISH_EXIT_INDEX_COUNT {
                    continue;
                }
                if let Some(slot) = vec.get_mut(index as usize) {
                    *slot = crate::failguard::FailDescrSlot::Reserved;
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
