//! RPython `rpython/translator/revdb/pplog.py`.

pub fn post_process_line(line: &str) -> Option<String> {
    let line_no_newline = line.trim_end_matches('\n');
    if let Some(rest) = line_no_newline.strip_prefix("revdb.c:") {
        let mut parts = rest.splitn(2, ": ");
        let line_number = parts.next().unwrap_or_default();
        let tail = parts.next().unwrap_or_default();
        if line_number.chars().all(|c| c.is_ascii_digit())
            && !tail.is_empty()
            && tail.chars().all(|c| c.is_ascii_hexdigit())
        {
            return Some(format!(
                "revdb.c:after pplog.py: {}\n",
                "#".repeat(tail.len())
            ));
        }
    }
    if starts_removed_line(line) {
        return None;
    }
    Some(line.to_string())
}

pub fn post_process(input: &str) -> String {
    input.lines().fold(String::new(), |mut out, line| {
        let mut owned = line.to_string();
        owned.push('\n');
        if let Some(processed) = post_process_line(&owned) {
            out.push_str(&processed);
        }
        out
    })
}

fn starts_removed_line(line: &str) -> bool {
    if line.starts_with('[') {
        return true;
    }
    if line.starts_with("PID ") && line.contains(" starting, log file disabled") {
        return true;
    }
    let Some((head, _)) = line.split_once(": obj 92233720368") else {
        return false;
    };
    let Some((stem, lineno)) = head.rsplit_once(':') else {
        return false;
    };
    stem.ends_with(".c") && lineno.chars().all(|c| c.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_process_hides_revdb_tail() {
        assert_eq!(
            post_process_line("revdb.c:123: 0a1b\n"),
            Some("revdb.c:after pplog.py: ####\n".to_string())
        );
    }

    #[test]
    fn post_process_removes_noise_lines() {
        assert_eq!(post_process_line("[noise]\n"), None);
        assert_eq!(
            post_process_line("PID 123 starting, log file disabled\n"),
            None
        );
        assert_eq!(post_process_line("foo.c:17: obj 92233720368\n"), None);
    }
}
