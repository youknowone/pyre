//! RPython `rpython/translator/sandbox/interact.py`.

use crate::translator::sandbox::sandlib::SimpleIOSandboxedProc;

pub fn interact(args: Vec<String>) -> Result<(), String> {
    SimpleIOSandboxedProc::new(args).proc.interact()
}
