use std::process::Command;

#[test]
fn log_probe() {
    if std::env::var_os("MAJIT_LOG_TEST_CHILD").is_none() {
        return;
    }
    let ops = majit_backend_dynasm::majit_ops_log_enabled();
    assert_eq!(ops, majit_backend_dynasm::majit_ops_log_enabled());
    println!(
        "ops={ops},full={}",
        majit_backend_dynasm::majit_log_enabled()
    );
}

#[test]
fn emission_logging_is_independent_and_announces_its_version_once() {
    for (ops, full) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "log_probe", "--nocapture"])
            .env("MAJIT_LOG_TEST_CHILD", "1")
            .env_remove("MAJIT_LOG_OPS")
            .env_remove("MAJIT_LOG");
        if ops {
            command.env("MAJIT_LOG_OPS", "1");
        }
        if full {
            command.env("MAJIT_LOG", "1");
        }
        let output = command.output().unwrap();
        assert!(output.status.success());
        assert!(
            String::from_utf8(output.stdout)
                .unwrap()
                .contains(&format!("ops={},full={full}", ops || full))
        );
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert_eq!(
            stderr.matches("[dynasm] op-log-version=1").count(),
            usize::from(ops)
        );
    }
}
