//! Port of `rpython/config/parse.py`.

use std::collections::HashMap;

use crate::config::config::OptionValue;

pub fn parse_info(text: &str) -> HashMap<String, OptionValue> {
    let text = text.trim_start();
    let mut result = HashMap::new();
    if index_or_end(&(text.to_string() + ":"), ':') > index_or_end(&(text.to_string() + "="), '=') {
        let mut current: HashMap<usize, String> = HashMap::from([(0, String::new())]);
        let mut indentation_prefix: Option<String> = None;
        for raw_line in text.lines() {
            let line = raw_line.trim_end();
            if line.is_empty() {
                continue;
            }
            let realline = line.trim_start();
            let indent = line.len() - realline.len();

            if let Some(prefix) = indentation_prefix.take() {
                assert!(
                    indent > current.keys().copied().max().unwrap_or(0),
                    "missing indent?"
                );
                current.insert(indent, prefix);
            } else {
                let to_delete: Vec<usize> =
                    current.keys().copied().filter(|n| *n > indent).collect();
                for n in to_delete {
                    current.remove(&n);
                }
            }

            let prefix = current
                .get(&indent)
                .unwrap_or_else(|| panic!("bad dedent at indentation {indent}"))
                .clone();
            if realline.starts_with('[') && realline.ends_with(']') {
                indentation_prefix =
                    Some(format!("{}{}.", prefix, &realline[1..realline.len() - 1]));
            } else {
                let i = realline
                    .find(" = ")
                    .unwrap_or_else(|| panic!("missing ` = ` in config line {realline:?}"));
                let key = format!("{}{}", prefix, &realline[..i]);
                let value = parse_literal(&realline[i + 3..]);
                result.insert(key, value);
            }
        }
    } else {
        for line in text.lines() {
            let i = line
                .find(':')
                .unwrap_or_else(|| panic!("missing `:` in config line {line:?}"));
            let key = line[..i].trim().to_string();
            let value = line[i + 1..].trim();
            result.insert(key, parse_old_literal(value));
        }
    }
    result
}

fn index_or_end(text: &str, needle: char) -> usize {
    text.find(needle).unwrap_or(text.len())
}

fn parse_old_literal(value: &str) -> OptionValue {
    if let Ok(i) = value.parse::<i64>() {
        return OptionValue::Int(i);
    }
    match value {
        "True" => OptionValue::Bool(true),
        "False" => OptionValue::Bool(false),
        "None" => OptionValue::None,
        _ => OptionValue::Str(value.to_string()),
    }
}

fn parse_literal(value: &str) -> OptionValue {
    match value {
        "True" => OptionValue::Bool(true),
        "False" => OptionValue::Bool(false),
        "None" => OptionValue::None,
        _ => {
            if let Ok(i) = value.parse::<i64>() {
                return OptionValue::Int(i);
            }
            if let Some(s) = parse_quoted_string(value) {
                return OptionValue::Str(s);
            }
            OptionValue::Str(value.to_string())
        }
    }
}

fn parse_quoted_string(value: &str) -> Option<String> {
    let bytes = value.as_bytes();
    if bytes.len() < 2 {
        return None;
    }
    let quote = bytes[0];
    if quote != b'\'' && quote != b'"' {
        return None;
    }
    if bytes[bytes.len() - 1] != quote {
        return None;
    }
    let inner = &value[1..value.len() - 1];
    let mut out = String::new();
    let mut chars = inner.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        let Some(escaped) = chars.next() else {
            out.push('\\');
            break;
        };
        match escaped {
            '\\' => out.push('\\'),
            '\'' => out.push('\''),
            '"' => out.push('"'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            other => {
                out.push('\\');
                out.push(other);
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_new_format() {
        let parsed = parse_info("[foo]\n    bar = True\n");
        assert_eq!(parsed.len(), 1);
        assert!(matches!(
            parsed.get("foo.bar"),
            Some(OptionValue::Bool(true))
        ));

        let parsed = parse_info(
            "[objspace]\n    x = 'hello'\n[translation]\n    bar = 42\n    [egg]\n        something = None\n    foo = True\n",
        );
        assert_eq!(parsed.len(), 4);
        assert!(matches!(
            parsed.get("translation.foo"),
            Some(OptionValue::Bool(true))
        ));
        assert!(matches!(
            parsed.get("translation.bar"),
            Some(OptionValue::Int(42))
        ));
        assert!(matches!(
            parsed.get("translation.egg.something"),
            Some(OptionValue::None)
        ));
        assert!(matches!(
            parsed.get("objspace.x"),
            Some(OptionValue::Str(s)) if s == "hello"
        ));

        let parsed = parse_info("simple = 43\n");
        assert_eq!(parsed.len(), 1);
        assert!(matches!(parsed.get("simple"), Some(OptionValue::Int(43))));
    }

    #[test]
    fn parse_old_format() {
        let parsed = parse_info(
            "                          objspace.allworkingmodules: True\n                    objspace.disable_call_speedhacks: False\n                                 objspace.extmodules: None\n                        objspace.std.prebuiltintfrom: -5\n",
        );
        assert_eq!(parsed.len(), 4);
        assert!(matches!(
            parsed.get("objspace.allworkingmodules"),
            Some(OptionValue::Bool(true))
        ));
        assert!(matches!(
            parsed.get("objspace.disable_call_speedhacks"),
            Some(OptionValue::Bool(false))
        ));
        assert!(matches!(
            parsed.get("objspace.extmodules"),
            Some(OptionValue::None)
        ));
        assert!(matches!(
            parsed.get("objspace.std.prebuiltintfrom"),
            Some(OptionValue::Int(-5))
        ));
    }
}
