//! Compile-time `SyntaxWarning` for an unrecognized escape sequence.
//!
//! `warn_invalid_escape_sequence()` in `Parser/string_parser.c` reports the
//! escape while the literal is being decoded, and
//! `_PyTokenizer_warn_invalid_escape_sequence()` in
//! `Parser/tokenizer/helpers.c` does the same for the literal halves of an
//! f-string.  The compiler pyre embeds hands back a finished code object with
//! no diagnostic channel for either, so the literals are walked here over
//! their own parse, before compilation.
//!
//! A filter that turns the warning into an error yields a `SyntaxError`
//! carrying the literal's position, and its message drops the "Such sequences
//! will not work in the future." sentence the warning carries.

use crate::PyError;
use crate::compile::{ast, parser};
use ast::visitor::Visitor;
use ruff_text_size::TextRange;
use rustpython_wtf8::Wtf8;

/// 1-indexed line number at a byte offset.
fn line_number_at(source: &str, offset: usize) -> usize {
    source[..offset.min(source.len())]
        .bytes()
        .filter(|&b| b == b'\n')
        .count()
        + 1
}

/// 1-indexed (line, column) at a byte offset.  The column counts characters,
/// not bytes, because that is what a `SyntaxError.offset` reports.
fn line_offset_at(source: &str, offset: usize) -> (usize, usize) {
    let offset = offset.min(source.len());
    let prefix = &source[..offset];
    let lineno = prefix.bytes().filter(|&b| b == b'\n').count() + 1;
    let line_start = prefix.rfind('\n').map_or(0, |index| index + 1);
    let column = source[line_start..offset].chars().count() + 1;
    (lineno, column)
}

/// The byte bounds of a quoted literal's content, with the prefix letters and
/// the quote delimiters removed.  `range` covers the whole literal token.
fn content_bounds(source: &str, range: TextRange) -> Option<(usize, usize)> {
    let start = range.start().to_usize();
    let end = range.end().to_usize();
    if start >= end || end > source.len() {
        return None;
    }
    let bytes = &source.as_bytes()[start..end];
    let quote_index = bytes.iter().position(|&c| c == b'\'' || c == b'"')?;
    let quote = bytes[quote_index];
    let quote_len = if bytes.get(quote_index + 1) == Some(&quote)
        && bytes.get(quote_index + 2) == Some(&quote)
    {
        3
    } else {
        1
    };
    let content_start = start + quote_index + quote_len;
    let content_end = end.checked_sub(quote_len)?;
    (content_start <= content_end).then_some((content_start, content_end))
}

/// An escape the decoder does not recognize, named by the text that follows
/// its backslash so the report can quote the whole sequence.
#[derive(Debug, PartialEq, Eq)]
struct InvalidEscape {
    /// One character for an unrecognized escape, the whole digit run for an
    /// octal one.
    sequence: String,
    /// An octal escape above `\377` is out of range rather than unrecognized,
    /// and says so.
    octal: bool,
}

/// The first invalid escape in `source[start..end]`, with the backslash's own
/// byte offset.
///
/// Only the first is reported per literal:
/// `_PyUnicode_DecodeUnicodeEscapeInternal2` keeps a single
/// `first_invalid_escape_char`.  In a bytes literal `\u`, `\U` and `\N` are
/// invalid, the literal being byte-oriented.
fn first_invalid_escape(
    source: &str,
    start: usize,
    end: usize,
    is_bytes: bool,
) -> Option<(InvalidEscape, usize)> {
    let character = |ch: char| InvalidEscape {
        sequence: ch.to_string(),
        octal: false,
    };
    let mut chars = source[start..end].char_indices().peekable();
    while let Some((index, ch)) = chars.next() {
        if ch != '\\' {
            continue;
        }
        let Some((_, next)) = chars.next() else {
            break;
        };
        match next {
            '\\' | '\'' | '"' | 'a' | 'b' | 'f' | 'n' | 'r' | 't' | 'v' | '\n' => {}
            '\r' => {
                if matches!(chars.peek(), Some(&(_, '\n'))) {
                    chars.next();
                }
            }
            '0'..='7' => {
                let mut digits = String::from(next);
                for _ in 0..2 {
                    match chars.peek() {
                        Some(&(_, digit @ '0'..='7')) => {
                            digits.push(digit);
                            chars.next();
                        }
                        _ => break,
                    }
                }
                // Three digits led by 4 or above exceed `\377`, the largest
                // value one byte holds; two or fewer cannot.
                if digits.len() == 3 && digits.as_bytes()[0] >= b'4' {
                    return Some((
                        InvalidEscape {
                            sequence: digits,
                            octal: true,
                        },
                        start + index,
                    ));
                }
            }
            'x' | 'u' | 'U' if !(is_bytes && next != 'x') => {
                let digits = match next {
                    'x' => 2,
                    'u' => 4,
                    _ => 8,
                };
                for _ in 0..digits {
                    if chars.peek().is_some_and(|&(_, c)| c.is_ascii_hexdigit()) {
                        chars.next();
                    } else {
                        break;
                    }
                }
            }
            'N' if !is_bytes => {
                if matches!(chars.peek(), Some(&(_, '{'))) {
                    chars.next();
                    for (_, c) in chars.by_ref() {
                        if c == '}' {
                            break;
                        }
                    }
                }
            }
            _ => return Some((character(next), start + index)),
        }
    }
    None
}

fn describe(escape: &InvalidEscape) -> &'static str {
    if escape.octal {
        "an invalid octal escape sequence"
    } else {
        "an invalid escape sequence"
    }
}

fn warning_message(escape: &InvalidEscape) -> String {
    let sequence = &escape.sequence;
    let kind = describe(escape);
    format!(
        "\"\\{sequence}\" is {kind}. \
         Such sequences will not work in the future. \
         Did you mean \"\\\\{sequence}\"? A raw string is also an option."
    )
}

/// The same report without the "will not work in the future" sentence, which
/// a raised `SyntaxError` drops.
fn error_message(escape: &InvalidEscape) -> String {
    let sequence = &escape.sequence;
    let kind = describe(escape);
    format!(
        "\"\\{sequence}\" is {kind}. \
         Did you mean \"\\\\{sequence}\"? A raw string is also an option."
    )
}

/// Turn a filter-escalated `SyntaxWarning` into the `SyntaxError` that names
/// the offending literal.  Anything else the warning machinery raised is the
/// caller's to propagate unchanged.
fn escalate(
    err: PyError,
    source: &str,
    filename: &str,
    offset: usize,
    escape: &InvalidEscape,
) -> PyError {
    if err.exc_object.is_null() {
        return err;
    }
    let Some(category) = crate::builtins::lookup_exc_class("SyntaxWarning") else {
        return err;
    };
    if !matches!(
        crate::baseobjspace::isinstance(err.exc_object, category),
        Ok(true)
    ) {
        return err;
    }
    let (lineno, column) = line_offset_at(source, offset);
    let text = source
        .split('\n')
        .nth(lineno.saturating_sub(1))
        .map(|line| format!("{}\n", line.trim_end_matches('\r')));
    PyError::syntax_error_located(
        error_message(escape),
        Wtf8::new(filename),
        lineno as i64,
        column as i64,
        lineno as i64,
        // The range covers the backslash and the sequence it introduced.
        (column + 1 + escape.sequence.chars().count()) as i64,
        text.as_deref(),
    )
}

/// `PyErr_WarnExplicitObject(PyExc_SyntaxWarning, ...)` for one invalid escape.
fn warn_invalid_escape_sequence(
    source: &str,
    filename: &str,
    escape: &InvalidEscape,
    offset: usize,
) -> Result<(), PyError> {
    // The filters and the once-registry are installed with the `_warnings`
    // module. A compile that runs before that point has nothing to match
    // against, and reporting the escape unfiltered would put a line on stderr
    // for every literal the bootstrap itself compiles.
    if !crate::module::_warnings::state_is_readable() {
        return Ok(());
    }
    let Some(category) = crate::builtins::lookup_exc_class("SyntaxWarning") else {
        return Ok(());
    };
    let lineno = line_number_at(source, offset) as i64;

    // `w_str_new` for the message can collect, which would relocate the
    // category and then the filename, so each is reloaded from its slot at the
    // call rather than held in a Rust local across the next allocation.
    let _roots = pyre_object::gc_roots::push_roots();
    let category_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(category);
    let message_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(pyre_object::w_str_new(&warning_message(escape)));
    let filename_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(pyre_object::w_str_new(filename));

    crate::module::_warnings::do_warn_explicit(
        pyre_object::gc_roots::shadow_stack_get(category_slot),
        pyre_object::gc_roots::shadow_stack_get(message_slot),
        pyre_object::gc_roots::shadow_stack_get(filename_slot),
        lineno,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
    )
    .map_err(|err| escalate(err, source, filename, offset, escape))
}

struct EscapeWarningVisitor<'a> {
    source: &'a str,
    filename: &'a str,
    error: Option<PyError>,
}

impl<'a> EscapeWarningVisitor<'a> {
    fn record(&mut self, result: Result<(), PyError>) {
        if self.error.is_none()
            && let Err(err) = result
        {
            self.error = Some(err);
        }
    }

    /// A quoted literal, `range` covering its prefix and delimiters too.
    fn check_quoted_literal(&mut self, range: TextRange, is_bytes: bool) {
        if let Some((start, end)) = content_bounds(self.source, range)
            && let Some((escape, offset)) = first_invalid_escape(self.source, start, end, is_bytes)
        {
            let result = warn_invalid_escape_sequence(self.source, self.filename, &escape, offset);
            self.record(result);
        }
    }

    /// One literal run of an f-string, `range` covering content only.
    ///
    /// `_PyTokenizer_warn_invalid_escape_sequence` also sees `\{` and `\}` in
    /// an `FSTRING_MIDDLE` / `FSTRING_END` token.  The parser splits the run
    /// before the interpolation delimiter, leaving the backslash at the end of
    /// the run and the brace just past it, so that pair is checked separately.
    /// An even number of trailing backslashes escape each other and warn on
    /// none.
    fn check_fstring_literal(&mut self, range: TextRange) {
        let start = range.start().to_usize();
        let end = range.end().to_usize();
        if start >= end || end > self.source.len() {
            return;
        }
        if let Some((escape, offset)) = first_invalid_escape(self.source, start, end, false) {
            let result = warn_invalid_escape_sequence(self.source, self.filename, &escape, offset);
            self.record(result);
            return;
        }
        let trailing_backslashes = self.source.as_bytes()[start..end]
            .iter()
            .rev()
            .take_while(|&&b| b == b'\\')
            .count();
        if trailing_backslashes % 2 == 1
            && let Some(&after) = self.source.as_bytes().get(end)
            && (after == b'{' || after == b'}')
        {
            let brace = InvalidEscape {
                sequence: (after as char).to_string(),
                octal: false,
            };
            let result = warn_invalid_escape_sequence(self.source, self.filename, &brace, end - 1);
            self.record(result);
        }
    }

    fn visit_interpolated_elements(&mut self, elements: &'a ast::InterpolatedStringElements) {
        for element in elements {
            if self.error.is_some() {
                return;
            }
            match element {
                ast::InterpolatedStringElement::Literal(literal) => {
                    self.check_fstring_literal(literal.range);
                }
                ast::InterpolatedStringElement::Interpolation(interpolation) => {
                    self.visit_expr(&interpolation.expression);
                    if let Some(spec) = &interpolation.format_spec {
                        self.visit_interpolated_elements(&spec.elements);
                    }
                }
            }
        }
    }
}

impl<'a> Visitor<'a> for EscapeWarningVisitor<'a> {
    fn visit_expr(&mut self, expr: &'a ast::Expr) {
        if self.error.is_some() {
            return;
        }
        match expr {
            ast::Expr::StringLiteral(string) => {
                for part in string.value.as_slice() {
                    if !matches!(
                        part.flags.prefix(),
                        ast::str_prefix::StringLiteralPrefix::Raw { .. }
                    ) {
                        self.check_quoted_literal(part.range, false);
                    }
                }
            }
            ast::Expr::BytesLiteral(bytes) => {
                for part in bytes.value.as_slice() {
                    if !matches!(
                        part.flags.prefix(),
                        ast::str_prefix::ByteStringPrefix::Raw { .. }
                    ) {
                        self.check_quoted_literal(part.range, true);
                    }
                }
            }
            ast::Expr::FString(fstring) => {
                for part in fstring.value.as_slice() {
                    match part {
                        ast::FStringPart::Literal(literal) => {
                            if !matches!(
                                literal.flags.prefix(),
                                ast::str_prefix::StringLiteralPrefix::Raw { .. }
                            ) {
                                self.check_quoted_literal(literal.range, false);
                            }
                        }
                        ast::FStringPart::FString(fstring) => {
                            if !matches!(
                                fstring.flags.prefix(),
                                ast::str_prefix::FStringPrefix::Raw { .. }
                            ) {
                                self.visit_interpolated_elements(&fstring.elements);
                            }
                        }
                    }
                }
            }
            _ => ast::visitor::walk_expr(self, expr),
        }
    }
}

/// Walk every string literal in `source` and warn for the invalid escapes.
///
/// A source that does not parse is still walked, over whatever the parser
/// recovered: the tokenizer reports an escape it has already passed even when
/// the parse fails later, so `'\\e' $` warns once and then raises.
pub fn emit_escape_warnings(source: &str, filename: &str) -> Result<(), PyError> {
    // No escape without a backslash, so the reparse is skipped for the vast
    // majority of modules rather than paid on every compile.
    if !source.contains('\\') {
        return Ok(());
    }
    let parsed = parser::parse_unchecked(source, parser::Mode::Module.into());
    let mut visitor = EscapeWarningVisitor {
        source,
        filename,
        error: None,
    };
    match parsed.syntax() {
        ast::Mod::Module(module) => {
            for stmt in &module.body {
                visitor.visit_stmt(stmt);
            }
        }
        ast::Mod::Expression(expression) => visitor.visit_expr(&expression.body),
    }
    visitor.error.map_or(Ok(()), Err)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(source: &str, is_bytes: bool) -> Option<(InvalidEscape, usize)> {
        let len = u32::try_from(source.len()).unwrap();
        let (start, end) = content_bounds(source, TextRange::new(0.into(), len.into())).unwrap();
        first_invalid_escape(source, start, end, is_bytes)
    }

    fn character(ch: char, offset: usize) -> Option<(InvalidEscape, usize)> {
        Some((
            InvalidEscape {
                sequence: ch.to_string(),
                octal: false,
            },
            offset,
        ))
    }

    #[test]
    fn only_the_first_invalid_escape_is_reported() {
        // The offset names the backslash, which is what a `SyntaxError.offset`
        // of 1 for a literal starting a line requires.
        assert_eq!(scan(r"'\d and \q'", false), character('d', 1));
    }

    #[test]
    fn recognized_escapes_are_not_reported() {
        assert_eq!(scan(r"'\n\t\\\x41\u0041\N{BULLET}\101'", false), None);
    }

    #[test]
    fn bytes_reject_the_character_escapes() {
        assert_eq!(scan(r"b'\u0041'", true), character('u', 2));
        assert_eq!(scan(r"b'\u0041'", false), None);
    }

    #[test]
    fn an_octal_escape_above_377_is_out_of_range() {
        assert_eq!(
            scan(r"'\407'", false),
            Some((
                InvalidEscape {
                    sequence: "407".to_owned(),
                    octal: true,
                },
                1
            ))
        );
        // `\377` is the largest byte, and two digits cannot exceed it.
        assert_eq!(scan(r"'\377'", false), None);
        assert_eq!(scan(r"'\77'", false), None);
    }

    #[test]
    fn the_octal_report_quotes_the_whole_run() {
        let escape = InvalidEscape {
            sequence: "407".to_owned(),
            octal: true,
        };
        assert_eq!(
            warning_message(&escape),
            r#""\407" is an invalid octal escape sequence. Such sequences will not work in the future. Did you mean "\\407"? A raw string is also an option."#
        );
        assert_eq!(
            error_message(&escape),
            r#""\407" is an invalid octal escape sequence. Did you mean "\\407"? A raw string is also an option."#
        );
    }

    #[test]
    fn a_triple_quoted_literal_reports_the_second_line() {
        let source = "'''\n\\z'''";
        assert_eq!(line_number_at(source, 4), 2);
        assert_eq!(line_offset_at(source, 4), (2, 1));
    }

    #[test]
    fn content_bounds_skips_the_prefix_and_one_delimiter() {
        // The inner `'` runs belong to the content: only the opening `"` and
        // its closing partner are delimiters.
        let source = "\"'''''invalid\\ escape\"";
        let len = u32::try_from(source.len()).unwrap();
        let (start, end) = content_bounds(source, TextRange::new(0.into(), len.into())).unwrap();
        assert_eq!(&source[start..end], "'''''invalid\\ escape");
        assert_eq!(scan(source, false), character(' ', 13));
    }
}
