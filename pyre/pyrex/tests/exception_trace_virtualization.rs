//! Keeps a caught exception virtual across the loop and its handler bridge.
//!
//! `PyFrame.handle_operation_error` records a traceback on the live
//! `OperationError`, and RPython's `OptVirtualize.optimize_NEW_WITH_VTABLE`
//! keeps both objects virtual until they escape.  A caught exception in this
//! loop never escapes, so neither the loop nor its handler bridge may retain
//! an allocation, residual call, or force operation after optimization.

#![cfg(feature = "dynasm")]

use std::process::Command;

const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

const SOURCE: &str = r#"
def main():
    total = 0
    i = 0
    while i < 5000:
        try:
            if i % 7 == 0:
                raise ValueError("v")
            total += 1
        except ValueError:
            total += 2
        i += 1
    print(total)

main()
"#;

fn optimized_sections(stderr: &str) -> Vec<(&str, Vec<&str>)> {
    let mut sections = Vec::new();
    let mut current: Option<(&str, Vec<&str>)> = None;

    for line in stderr.lines() {
        if line.starts_with("--- ") {
            if let Some(section) = current.take() {
                sections.push(section);
            }
            current = match line {
                "--- peeled trace (assembled) ---" | "--- bridge trace (after opt) ---" => {
                    Some((line, Vec::new()))
                }
                _ => None,
            };
        } else if let Some((_, lines)) = current.as_mut() {
            lines.push(line);
        }
    }
    if let Some(section) = current {
        sections.push(section);
    }
    sections
}

fn opcode(line: &str) -> Option<&str> {
    let operation = line
        .trim()
        .split_once(" = ")
        .map_or_else(|| line.trim(), |(_, operation)| operation);
    operation.split_once('(').map(|(opcode, _)| opcode)
}

#[test]
fn caught_exception_and_traceback_stay_virtual() {
    let output = Command::new(PYRE)
        .args(["-c", SOURCE])
        .env("MAJIT_LOG", "1")
        .env("MAJIT_STATS", "1")
        .output()
        .expect("spawn pyre-dynasm");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "exception probe exited {:?}\nstdout:\n{stdout}\nstderr tail:\n{}",
        output.status,
        stderr.lines().rev().take(40).collect::<Vec<_>>().join("\n"),
    );
    assert_eq!(stdout.trim(), "5715");
    assert!(stderr.contains("loops_compiled=1"), "{stderr}");
    assert!(stderr.contains("bridges_compiled=1"), "{stderr}");
    assert!(stderr.contains("loops_aborted=0"), "{stderr}");

    let sections = optimized_sections(&stderr);
    assert_eq!(
        sections
            .iter()
            .filter(|(header, _)| *header == "--- peeled trace (assembled) ---")
            .count(),
        1,
        "expected one optimized loop section; headers were {:?}",
        sections
            .iter()
            .map(|(header, _)| header)
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        sections
            .iter()
            .filter(|(header, _)| *header == "--- bridge trace (after opt) ---")
            .count(),
        1,
        "expected one optimized exception bridge; headers were {:?}",
        sections
            .iter()
            .map(|(header, _)| header)
            .collect::<Vec<_>>(),
    );

    let forbidden = sections
        .iter()
        .flat_map(|(header, lines)| {
            lines.iter().filter_map(move |line| {
                let opcode = opcode(line)?;
                let is_allocation = matches!(
                    opcode,
                    "New"
                        | "NewWithVtable"
                        | "NewArray"
                        | "NewArrayClear"
                        | "NewStr"
                        | "NewUnicode"
                );
                (is_allocation || opcode.starts_with("Call") || opcode == "ForceToken")
                    .then_some((*header, *line))
            })
        })
        .collect::<Vec<_>>();
    assert!(
        forbidden.is_empty(),
        "caught exception escaped virtualization:\n{}",
        forbidden
            .iter()
            .map(|(header, line)| format!("{header}\n{line}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );

    let redundant_ec_loads = sections
        .iter()
        .flat_map(|(header, lines)| {
            lines.iter().filter_map(move |line| {
                line.contains("PyFrame.execution_context")
                    .then_some((*header, *line))
            })
        })
        .collect::<Vec<_>>();
    assert!(
        redundant_ec_loads.is_empty(),
        "portal trace reloaded the ExecutionContext instead of using its second red input:\n{}",
        redundant_ec_loads
            .iter()
            .map(|(header, line)| format!("{header}\n{line}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
}
