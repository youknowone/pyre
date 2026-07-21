//! `_tokenize` — CPython 3.14 tokenizer iterator, adapted from RustPython's
//! `crates/stdlib/src/_tokenize.rs` at the workspace-pinned revision.
//!
//! The PyPy snapshot vendored by pyre predates `_tokenize.TokenizerIter`.
//! Python 3.14's `tokenize.py` requires it, so the common RustPython lexer is
//! the upstream owner for this compatibility layer.  The iterator keeps the
//! same source-reading and token-emission phases as that implementation.

use pyre_object::*;
use rustpython_compiler::{
    ast::{
        PySourceType,
        token::{Token, TokenKind},
    },
    parser::{LexicalErrorType, ParseError, ParseErrorType, parse_unchecked_source},
};

const TOKEN_ENDMARKER: u8 = 0;
const TOKEN_DEDENT: u8 = 6;
const TOKEN_OP: u8 = 55;
const TOKEN_COMMENT: u8 = 65;
const TOKEN_NL: u8 = 66;

#[derive(Clone, Debug)]
struct SourceLines {
    starts: Vec<usize>,
}

impl SourceLines {
    fn new(source: &str) -> Self {
        let mut starts = vec![0];
        for (i, byte) in source.bytes().enumerate() {
            if byte == b'\n' {
                starts.push(i + 1);
            }
        }
        Self { starts }
    }

    /// Ruff's `LineIndex::line_column` uses UTF-32 columns, which are the
    /// character offsets exposed by CPython's tokenizer.
    fn line_col(&self, source: &str, byte_offset: usize) -> (usize, usize) {
        let offset = byte_offset.min(source.len());
        let line_index = self.starts.partition_point(|&start| start <= offset) - 1;
        let start = self.starts[line_index];
        let mut column = source[start..offset].chars().count();
        if line_index == 0 && source.starts_with('\u{feff}') && column > 0 {
            column -= 1;
        }
        (line_index + 1, column)
    }

    fn full_line<'a>(&self, source: &'a str, byte_offset: usize) -> &'a str {
        if source.is_empty() {
            return "";
        }
        let offset = byte_offset.min(source.len().saturating_sub(1));
        let line_index = self.starts.partition_point(|&start| start <= offset) - 1;
        let start = self.starts[line_index];
        let end = source[start..]
            .find('\n')
            .map(|i| start + i + 1)
            .unwrap_or(source.len());
        &source[start..end]
    }
}

#[derive(Clone, Copy, Debug)]
enum TokenizerPhase {
    Reading,
    Yielding,
    Done,
}

#[crate::pyre_class("_tokenize.TokenizerIter")]
pub struct W_TokenizerIter {
    readline: PyObjectRef,
    extra_tokens: bool,
    encoding: Option<String>,
    phase: TokenizerPhase,
    source: String,
    tokens: Vec<Token>,
    errors: Vec<ParseError>,
    index: usize,
    indent_depth: usize,
    lines: SourceLines,
    need_implicit_nl: bool,
    pending_fstring_parts: Vec<(u8, String, usize, usize, usize, usize)>,
    pending_empty_fstring_middle: Option<(u8, usize, usize, String)>,
}

/// Sweep-time cleanup for a GC-reclaimed `W_TokenizerIter`: run its Drop glue
/// so the owned Rust heap (source `String`, token / error `Vec`s, the line
/// table) is freed instead of leaked. The only managed child, `readline`, is a
/// raw `PyObjectRef` (Copy), and `Token` / `ParseError` carry no GC pointers,
/// so this touches no GC memory and is safe to run at sweep time.
///
/// # Safety
/// `obj` must point to a valid, GC-dead `W_TokenizerIter`.
pub unsafe fn w_tokenizer_iter_dealloc(obj: PyObjectRef) {
    unsafe { std::ptr::drop_in_place(obj as *mut W_TokenizerIter) }
}

fn read_line(self_obj: PyObjectRef) -> Result<String, crate::PyError> {
    let (readline, encoding) = {
        let this = W_TokenizerIter::from_obj(self_obj)
            .ok_or_else(|| crate::PyError::type_error("invalid tokenizer iterator"))?;
        (this.readline, this.encoding.clone())
    };
    let _roots = gc_roots::push_roots();
    gc_roots::pin_root(self_obj);
    gc_roots::pin_root(readline);
    let raw = match crate::builtins::call_and_check(
        gc_roots::shadow_stack_get(gc_roots::shadow_stack_len() - 1),
        &[],
    ) {
        Ok(value) => value,
        Err(err) if err.kind == crate::PyErrorKind::StopIteration => return Ok(String::new()),
        Err(err) => return Err(err),
    };
    match encoding {
        Some(encoding) => unsafe {
            if !is_bytes(raw) {
                return Err(crate::PyError::type_error(
                    "readline() returned a non-bytes object",
                ));
            }
            // Decoding may enter the codec registry.  Do not retain a slice
            // borrowed from a movable bytes object across that call.
            let bytes = pyre_object::bytesobject::bytes_like_data(raw).to_vec();
            let decoded = crate::typedef::decode_bytes_to_wtf8(&bytes, &encoding, "strict")?;
            Ok(decoded.to_string_lossy().into_owned())
        },
        None => unsafe {
            if !is_str(raw) {
                return Err(crate::PyError::type_error(
                    "readline() returned a non-string object",
                ));
            }
            Ok(w_str_get_wtf8(raw).to_string_lossy().into_owned())
        },
    }
}

fn prepare_tokens(self_obj: PyObjectRef) {
    let this = W_TokenizerIter::from_obj(self_obj).expect("TokenizerIter payload");
    let parsed = parse_unchecked_source(&this.source, PySourceType::Python);
    this.tokens = parsed.tokens().iter().copied().collect();
    this.errors = parsed.errors().to_vec();
    this.lines = SourceLines::new(&this.source);
    this.need_implicit_nl = !this.source.ends_with('\n');
    this.index = 0;
    this.indent_depth = 0;
    this.pending_fstring_parts.clear();
    this.pending_empty_fstring_middle = None;
    this.phase = TokenizerPhase::Yielding;
}

fn tokenizer_next(self_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let _roots = gc_roots::push_roots();
    gc_roots::pin_root(self_obj);
    let slot = gc_roots::shadow_stack_len() - 1;
    loop {
        let self_obj = gc_roots::shadow_stack_get(slot);
        match W_TokenizerIter::from_obj(self_obj)
            .expect("TokenizerIter payload")
            .phase
        {
            TokenizerPhase::Reading => {
                let line = read_line(self_obj)?;
                let self_obj = gc_roots::shadow_stack_get(slot);
                if line.is_empty() {
                    prepare_tokens(self_obj);
                } else {
                    W_TokenizerIter::from_obj(self_obj)
                        .expect("TokenizerIter payload")
                        .source
                        .push_str(&line);
                }
            }
            TokenizerPhase::Yielding => return emit_next_token(self_obj),
            TokenizerPhase::Done => return Err(crate::PyError::stop_iteration()),
        }
    }
}

#[crate::pyre_methods]
impl W_TokenizerIter {
    #[staticmethod]
    fn __new__(_cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
        // The descriptor ABI includes the requested class at positional[0].
        let positional = positional.get(1..).unwrap_or(&[]);
        crate::builtins::kwarg_reject_unknown(
            kwargs,
            &["encoding", "extra_tokens"],
            "tokenizeriter",
        )?;
        if positional.len() != 1 {
            return Err(crate::PyError::type_error(format!(
                "tokenizeriter() takes exactly 1 positional argument ({} given)",
                positional.len()
            )));
        }
        let readline = positional[0];
        if !crate::baseobjspace::callable_w(readline) {
            return Err(crate::PyError::type_error("source must be callable"));
        }
        let w_extra = crate::builtins::kwarg_get(kwargs, "extra_tokens").ok_or_else(|| {
            crate::PyError::type_error(
                "tokenizeriter() missing required argument 'extra_tokens' (pos 2)",
            )
        })?;
        let extra_tokens = crate::baseobjspace::is_true(w_extra)?;
        let encoding = match crate::builtins::kwarg_get(kwargs, "encoding") {
            Some(value) => unsafe {
                if !is_str(value) {
                    return Err(crate::PyError::type_error(format!(
                        "tokenizeriter() argument 'encoding' must be str, not {}",
                        type_name_of(value)
                    )));
                }
                Some(w_str_get_wtf8(value).to_string_lossy().into_owned())
            },
            None => None,
        };
        let _ = type_object();
        Ok(W_TokenizerIter::allocate_stable(W_TokenizerIter {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            readline,
            extra_tokens,
            encoding,
            phase: TokenizerPhase::Reading,
            source: String::new(),
            tokens: Vec::new(),
            errors: Vec::new(),
            index: 0,
            indent_depth: 0,
            lines: SourceLines { starts: vec![0] },
            need_implicit_nl: false,
            pending_fstring_parts: Vec::new(),
            pending_empty_fstring_middle: None,
        }))
    }

    fn __iter__(&self, args: &[PyObjectRef]) -> PyObjectRef {
        args[0]
    }

    fn __next__(&mut self) -> Result<PyObjectRef, crate::PyError> {
        tokenizer_next(self as *mut W_TokenizerIter as PyObjectRef)
    }
}

fn emit_next_token(self_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let this = W_TokenizerIter::from_obj(self_obj).expect("TokenizerIter payload");

    if let Some((kind, line, col, line_str)) = this.pending_empty_fstring_middle.take() {
        return Ok(make_token_tuple(
            kind,
            "",
            line,
            col as isize,
            line,
            col as isize,
            &line_str,
        ));
    }
    if let Some((kind, text, sl, sc, el, ec)) = this.pending_fstring_parts.pop() {
        let line = this
            .lines
            .full_line(&this.source, line_start_offset(&this.source, sl));
        return Ok(make_token_tuple(
            kind,
            &text,
            sl,
            sc as isize,
            el,
            ec as isize,
            line,
        ));
    }

    while this.index < this.tokens.len() {
        let token = this.tokens[this.index];
        this.index += 1;
        let kind = token.kind();
        let range = token.as_tuple().1;
        let start = u32::from(range.start()) as usize;
        let end = u32::from(range.end()) as usize;

        if kind == TokenKind::Indent {
            this.indent_depth += 1;
            if this.indent_depth >= 100 {
                let (line, column) = this.lines.line_col(&this.source, start);
                let text = this.lines.full_line(&this.source, start).to_owned();
                return Err(positioned_syntax_error(
                    "IndentationError",
                    "too many levels of indentation",
                    line,
                    column + 1,
                    Some(&text),
                ));
            }
        } else if kind == TokenKind::Dedent {
            this.indent_depth = this.indent_depth.saturating_sub(1);
        }

        // Ruff deliberately recovers from malformed numeric literals by
        // splitting them into adjacent NUMBER/NAME fragments (`1_` -> `1`,
        // `_`; `0b12` -> `0b1`, `2`).  CPython's tokenizer treats an
        // identifier or another number immediately following a number as a
        // lexical error, while whitespace-separated `1 2` remains tokenizable.
        if is_number_kind(kind) {
            if let Some(next) = this.tokens.get(this.index) {
                let next_range = next.as_tuple().1;
                let next_start = u32::from(next_range.start()) as usize;
                let next_end = u32::from(next_range.end()) as usize;
                let next_text = if next_end <= this.source.len() {
                    &this.source[next_start..next_end]
                } else {
                    ""
                };
                if next_start == end
                    && (is_number_kind(next.kind())
                        || (next.kind() == TokenKind::Name
                            && (!this.extra_tokens
                                || next_text.starts_with('_')
                                || next_text.eq_ignore_ascii_case("e"))))
                {
                    let (line, column) = this.lines.line_col(&this.source, next_end);
                    return Err(raise_syntax_error("invalid decimal literal", line, column));
                }
            }
        }

        for error in &this.errors {
            let es = u32::from(error.location.start()) as usize;
            let ee = u32::from(error.location.end()) as usize;
            if ranges_touch(es, ee, start, end) {
                if matches!(
                    error.error,
                    ParseErrorType::Lexical(LexicalErrorType::IndentationError)
                ) {
                    // Ruff and CPython expand tab indentation differently;
                    // retain RustPython's false-positive suppression here.
                    if !this.source.contains('\t') {
                        return Err(raise_indentation_error(error, &this.source, &this.lines));
                    }
                } else if matches!(error.error, ParseErrorType::Lexical(_))
                    && !tolerated_extra_numeric_error(
                        this.extra_tokens,
                        error,
                        &this.tokens,
                        &this.source,
                    )
                {
                    return Err(raise_lexical_error(error, &this.source, &this.lines));
                }
            }
        }
        if kind == TokenKind::EndOfFile {
            continue;
        }
        if !this.extra_tokens && matches!(kind, TokenKind::Comment | TokenKind::NonLogicalNewline) {
            continue;
        }

        let raw_type = if this.extra_tokens
            && kind == TokenKind::Unknown
            && end <= this.source.len()
            && this.source[start..end]
                .bytes()
                .all(|byte| byte.is_ascii_digit())
        {
            2
        } else {
            token_kind_value(kind)
        };
        let token_type = if this.extra_tokens && raw_type > TOKEN_DEDENT && raw_type < TOKEN_OP {
            TOKEN_OP
        } else {
            raw_type
        };

        let (token_str, start_line, start_col, end_line, end_col, line_str) =
            if kind == TokenKind::Dedent {
                let last_line = this.source.lines().count();
                let default_pos = if this.extra_tokens {
                    (last_line + 1, 0)
                } else {
                    (last_line, 0)
                };
                let (pos, line) = next_non_dedent_info(
                    &this.tokens,
                    this.index,
                    &this.source,
                    &this.lines,
                    default_pos,
                );
                ("", pos.0, pos.1, pos.0, pos.1, line)
            } else {
                let (sl, sc) = this.lines.line_col(&this.source, start);
                let implicit_newline = start >= this.source.len();
                let in_source = end <= this.source.len();
                let (text, el, ec) = if kind == TokenKind::Newline {
                    if this.extra_tokens {
                        if implicit_newline {
                            ("", sl, sc + 1)
                        } else {
                            let text = if this.source[start..end].starts_with('\r') {
                                "\r\n"
                            } else {
                                "\n"
                            };
                            (text, sl, sc + text.len())
                        }
                    } else {
                        ("", sl, sc)
                    }
                } else if kind == TokenKind::NonLogicalNewline {
                    let text = if in_source {
                        &this.source[start..end]
                    } else {
                        ""
                    };
                    (text, sl, sc + text.chars().count())
                } else {
                    let (el, ec) = this.lines.line_col(&this.source, end);
                    let text = if in_source {
                        &this.source[start..end]
                    } else {
                        ""
                    };
                    (text, el, ec)
                };
                (
                    text,
                    sl,
                    sc,
                    el,
                    ec,
                    this.lines.full_line(&this.source, start),
                )
            };

        if matches!(kind, TokenKind::FStringMiddle | TokenKind::TStringMiddle)
            && (token_str.contains("{{") || token_str.contains("}}"))
        {
            let mut parts =
                split_fstring_middle(token_str, token_type, start_line, start_col).into_iter();
            let (kind, text, sl, sc, el, ec) = parts.next().expect("non-empty f-string split");
            for part in parts.collect::<Vec<_>>().into_iter().rev() {
                this.pending_fstring_parts.push(part);
            }
            return Ok(make_token_tuple(
                kind,
                &text,
                sl,
                sc as isize,
                el,
                ec as isize,
                line_str,
            ));
        }

        if kind == TokenKind::Rbrace
            && this
                .tokens
                .get(this.index)
                .is_some_and(|t| t.kind() == TokenKind::Rbrace)
        {
            let middle_type = find_fstring_middle_type(&this.tokens, this.index);
            this.pending_empty_fstring_middle =
                Some((middle_type, end_line, end_col, line_str.to_owned()));
        }

        // CPython's parser-facing tokenizer suppresses indentation spelling
        // and reports its synthetic column as -1.  `extra_tokens=True` is
        // the public `tokenize` mode and retains the actual whitespace.
        if kind == TokenKind::Indent && !this.extra_tokens {
            return Ok(make_token_tuple(
                token_type, "", end_line, -1, end_line, -1, line_str,
            ));
        }
        if kind == TokenKind::Dedent && !this.extra_tokens {
            return Ok(make_token_tuple(
                token_type, "", start_line, -1, start_line, -1, line_str,
            ));
        }

        return Ok(make_token_tuple(
            token_type,
            token_str,
            start_line,
            start_col as isize,
            end_line,
            end_col as isize,
            line_str,
        ));
    }

    if this.extra_tokens && std::mem::take(&mut this.need_implicit_nl) {
        if let Some(last) = this
            .tokens
            .iter()
            .rev()
            .find(|token| token.kind() != TokenKind::EndOfFile)
            .filter(|token| token.kind() == TokenKind::Comment)
        {
            let range = last.as_tuple().1;
            let start = u32::from(range.start()) as usize;
            let end = u32::from(range.end()) as usize;
            let (line, col) = this.lines.line_col(&this.source, end);
            return Ok(make_token_tuple(
                TOKEN_NL,
                "",
                line,
                col as isize,
                line,
                col as isize + 1,
                this.lines.full_line(&this.source, start),
            ));
        }
        let line_start = this.source.rfind('\n').map_or(0, |index| index + 1);
        let tail = &this.source[line_start..];
        if !tail.is_empty() && tail.bytes().all(|byte| matches!(byte, b' ' | b'\t' | 0x0c)) {
            let (line, col) = this.lines.line_col(&this.source, this.source.len());
            return Ok(make_token_tuple(
                TOKEN_NL,
                "",
                line,
                col as isize,
                line,
                col as isize + 1,
                tail,
            ));
        }
    }

    for error in &this.errors {
        if !matches!(error.error, ParseErrorType::Lexical(_)) {
            continue;
        }
        if matches!(
            error.error,
            ParseErrorType::Lexical(LexicalErrorType::IndentationError)
        ) {
            if !this.source.contains('\t') {
                return Err(raise_indentation_error(error, &this.source, &this.lines));
            }
        } else if !tolerated_extra_numeric_error(
            this.extra_tokens,
            error,
            &this.tokens,
            &this.source,
        ) {
            return Err(raise_lexical_error(error, &this.source, &this.lines));
        }
    }

    let (mismatched_close, unclosed_or_too_deep) = bracket_problem(&this.tokens);
    if unclosed_or_too_deep || (mismatched_close && !this.extra_tokens) {
        return Err(raise_syntax_error(
            "EOF in multi-line statement",
            this.source.lines().count() + 1,
            0,
        ));
    }

    let last_line = this.source.lines().count();
    let (line, col, line_str) = if this.extra_tokens {
        (last_line + 1, 0, "")
    } else {
        (
            last_line,
            -1,
            this.lines
                .full_line(&this.source, this.source.len().saturating_sub(1)),
        )
    };
    this.phase = TokenizerPhase::Done;
    Ok(make_token_tuple(
        TOKEN_ENDMARKER,
        "",
        line,
        col,
        line,
        col,
        line_str,
    ))
}

fn ranges_touch(
    error_start: usize,
    error_end: usize,
    token_start: usize,
    token_end: usize,
) -> bool {
    error_start <= token_end && token_start <= error_end.max(error_start)
}

const fn is_number_kind(kind: TokenKind) -> bool {
    matches!(kind, TokenKind::Int | TokenKind::Float | TokenKind::Complex)
}

fn tolerated_extra_numeric_error(
    extra_tokens: bool,
    error: &ParseError,
    tokens: &[Token],
    source: &str,
) -> bool {
    if !extra_tokens {
        return false;
    }
    let error_start = u32::from(error.location.start()) as usize;
    tokens.iter().any(|token| {
        if token.kind() != TokenKind::Unknown {
            return false;
        }
        let range = token.as_tuple().1;
        let start = u32::from(range.start()) as usize;
        let end = u32::from(range.end()) as usize;
        start <= error_start
            && error_start <= end
            && end <= source.len()
            && source[start..end].bytes().all(|byte| byte.is_ascii_digit())
    })
}

/// `(mismatched closing delimiter, unclosed or excessive nesting)`.
fn bracket_problem(tokens: &[Token]) -> (bool, bool) {
    let mut stack = Vec::new();
    for token in tokens {
        match token.kind() {
            TokenKind::Lpar | TokenKind::Lsqb | TokenKind::Lbrace => {
                stack.push(token.kind());
                if stack.len() > 200 {
                    return (false, true);
                }
            }
            TokenKind::Rpar | TokenKind::Rsqb | TokenKind::Rbrace => {
                let expected = match token.kind() {
                    TokenKind::Rpar => TokenKind::Lpar,
                    TokenKind::Rsqb => TokenKind::Lsqb,
                    TokenKind::Rbrace => TokenKind::Lbrace,
                    _ => unreachable!(),
                };
                if stack.last().copied() != Some(expected) {
                    return (true, !stack.is_empty());
                }
                stack.pop();
            }
            _ => {}
        }
    }
    (false, !stack.is_empty())
}

fn line_start_offset(source: &str, one_indexed_line: usize) -> usize {
    source
        .split_inclusive('\n')
        .take(one_indexed_line.saturating_sub(1))
        .map(str::len)
        .sum()
}

fn next_non_dedent_info<'a>(
    tokens: &[Token],
    index: usize,
    source: &'a str,
    lines: &SourceLines,
    default_pos: (usize, usize),
) -> ((usize, usize), &'a str) {
    for token in &tokens[index..] {
        match token.kind() {
            TokenKind::Dedent => continue,
            TokenKind::EndOfFile => return (default_pos, ""),
            _ => {
                let start = u32::from(token.as_tuple().1.start()) as usize;
                return (
                    lines.line_col(source, start),
                    lines.full_line(source, start),
                );
            }
        }
    }
    (default_pos, "")
}

fn raise_indentation_error(
    error: &ParseError,
    source: &str,
    lines: &SourceLines,
) -> crate::PyError {
    let start = u32::from(error.location.start()) as usize;
    let (line, _) = lines.line_col(source, start);
    let text = lines
        .full_line(source, start)
        .trim_end_matches(['\n', '\r'])
        .to_owned();
    let message = error.error.to_string();
    positioned_syntax_error(
        "IndentationError",
        &message,
        line,
        text.len() + 1,
        Some(&text),
    )
}

fn raise_lexical_error(error: &ParseError, source: &str, lines: &SourceLines) -> crate::PyError {
    let start = u32::from(error.location.start()) as usize;
    let (line, column) = lines.line_col(source, start);
    raise_syntax_error(&error.error.to_string(), line, column + 1)
}

fn raise_syntax_error(message: &str, line: usize, offset: usize) -> crate::PyError {
    positioned_syntax_error("SyntaxError", message, line, offset, None)
}

fn positioned_syntax_error(
    class_name: &str,
    message: &str,
    line: usize,
    offset: usize,
    text: Option<&str>,
) -> crate::PyError {
    let Some(class) = crate::builtins::lookup_exc_class(class_name) else {
        return crate::PyError::syntax_error(message);
    };
    let _roots = gc_roots::push_roots();
    gc_roots::pin_root(class);
    gc_roots::pin_root(w_str_new(&message));
    let base = gc_roots::shadow_stack_len() - 2;
    let exc = match crate::builtins::call_and_check(
        gc_roots::shadow_stack_get(base),
        &[gc_roots::shadow_stack_get(base + 1)],
    ) {
        Ok(exc) => exc,
        Err(_) => return crate::PyError::syntax_error(message),
    };
    gc_roots::pin_root(exc);
    let exc_slot = gc_roots::shadow_stack_len() - 1;
    for (name, value) in [("lineno", line as i64), ("offset", offset as i64)] {
        let value = w_int_new(value);
        let _ = crate::baseobjspace::setattr_str(gc_roots::shadow_stack_get(exc_slot), name, value);
    }
    for (name, value) in [("msg", message), ("filename", "<string>")] {
        let value = w_str_new(value);
        let _ = crate::baseobjspace::setattr_str(gc_roots::shadow_stack_get(exc_slot), name, value);
    }
    let text_value = text.map(w_str_new).unwrap_or_else(w_none);
    let _ =
        crate::baseobjspace::setattr_str(gc_roots::shadow_stack_get(exc_slot), "text", text_value);
    unsafe { crate::PyError::from_exc_object(gc_roots::shadow_stack_get(exc_slot)) }
}

fn find_fstring_middle_type(tokens: &[Token], index: usize) -> u8 {
    let mut depth = 0i32;
    for token in tokens[..index].iter().rev() {
        match token.kind() {
            TokenKind::FStringEnd | TokenKind::TStringEnd => depth += 1,
            TokenKind::FStringStart => {
                if depth == 0 {
                    return 60;
                }
                depth -= 1;
            }
            TokenKind::TStringStart => {
                if depth == 0 {
                    return 63;
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    60
}

fn split_fstring_middle(
    raw: &str,
    token_type: u8,
    start_line: usize,
    start_col: usize,
) -> Vec<(u8, String, usize, usize, usize, usize)> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let (mut line, mut col) = (start_line, start_col);
    let (mut part_line, mut part_col) = (line, col);
    let mut chars = raw.chars().peekable();
    let end_pos = |text: &str, mut line: usize, mut col: usize| {
        for ch in text.chars() {
            if ch == '\n' {
                line += 1;
                col = 0;
            } else {
                col += ch.len_utf8();
            }
        }
        (line, col)
    };
    while let Some(ch) = chars.next() {
        if ch == '{' && chars.peek() == Some(&'{') {
            chars.next();
            current.push('{');
            col += 2;
        } else if ch == '}' && chars.peek() == Some(&'}') {
            chars.next();
            if !current.is_empty() {
                let (el, ec) = end_pos(&current, part_line, part_col);
                parts.push((
                    token_type,
                    std::mem::take(&mut current),
                    part_line,
                    part_col,
                    el,
                    ec,
                ));
            }
            parts.push((token_type, "}".to_owned(), line, col, line, col + 1));
            col += 2;
            part_line = line;
            part_col = col;
        } else {
            if current.is_empty() {
                part_line = line;
                part_col = col;
            }
            current.push(ch);
            if ch == '\n' {
                line += 1;
                col = 0;
            } else {
                col += ch.len_utf8();
            }
        }
    }
    if !current.is_empty() {
        let (el, ec) = end_pos(&current, part_line, part_col);
        parts.push((token_type, current, part_line, part_col, el, ec));
    }
    parts
}

#[allow(clippy::too_many_arguments)]
fn make_token_tuple(
    token_type: u8,
    text: &str,
    start_line: usize,
    start_col: isize,
    end_line: usize,
    end_col: isize,
    line: &str,
) -> PyObjectRef {
    // Callers usually pass slices borrowed from the iterator's `source`.
    // Copy them before the first collecting allocation can move that owner.
    let text = text.to_owned();
    let line = line.to_owned();
    let _roots = gc_roots::push_roots();
    let base = gc_roots::shadow_stack_len();
    gc_roots::pin_root(w_int_new(token_type as i64));
    gc_roots::pin_root(w_str_new(&text));
    gc_roots::pin_root(w_int_new(start_line as i64));
    gc_roots::pin_root(w_int_new(start_col as i64));
    gc_roots::pin_root(w_int_new(end_line as i64));
    gc_roots::pin_root(w_int_new(end_col as i64));
    gc_roots::pin_root(w_str_new(&line));
    let start = w_tuple_new(vec![
        gc_roots::shadow_stack_get(base + 2),
        gc_roots::shadow_stack_get(base + 3),
    ]);
    gc_roots::pin_root(start);
    let end = w_tuple_new(vec![
        gc_roots::shadow_stack_get(base + 4),
        gc_roots::shadow_stack_get(base + 5),
    ]);
    gc_roots::pin_root(end);
    w_tuple_new(vec![
        gc_roots::shadow_stack_get(base),
        gc_roots::shadow_stack_get(base + 1),
        gc_roots::shadow_stack_get(base + 7),
        gc_roots::shadow_stack_get(base + 8),
        gc_roots::shadow_stack_get(base + 6),
    ])
}

const fn token_kind_value(kind: TokenKind) -> u8 {
    match kind {
        TokenKind::EndOfFile => 0,
        TokenKind::Name
        | TokenKind::For
        | TokenKind::In
        | TokenKind::Pass
        | TokenKind::Class
        | TokenKind::And
        | TokenKind::Is
        | TokenKind::Raise
        | TokenKind::True
        | TokenKind::False
        | TokenKind::Assert
        | TokenKind::Try
        | TokenKind::While
        | TokenKind::Yield
        | TokenKind::Lambda
        | TokenKind::None
        | TokenKind::Not
        | TokenKind::Or
        | TokenKind::Break
        | TokenKind::Continue
        | TokenKind::Global
        | TokenKind::Nonlocal
        | TokenKind::Return
        | TokenKind::Except
        | TokenKind::Import
        | TokenKind::Case
        | TokenKind::Match
        | TokenKind::Type
        | TokenKind::Await
        | TokenKind::With
        | TokenKind::Del
        | TokenKind::Finally
        | TokenKind::From
        | TokenKind::Def
        | TokenKind::If
        | TokenKind::Else
        | TokenKind::Elif
        | TokenKind::As
        | TokenKind::Async => 1,
        TokenKind::Int | TokenKind::Complex | TokenKind::Float => 2,
        TokenKind::String => 3,
        TokenKind::Newline => 4,
        TokenKind::Indent => 5,
        TokenKind::Dedent => 6,
        TokenKind::Lpar => 7,
        TokenKind::Rpar => 8,
        TokenKind::Lsqb => 9,
        TokenKind::Rsqb => 10,
        TokenKind::Colon => 11,
        TokenKind::Comma => 12,
        TokenKind::Semi => 13,
        TokenKind::Plus => 14,
        TokenKind::Minus => 15,
        TokenKind::Star => 16,
        TokenKind::Slash => 17,
        TokenKind::Vbar => 18,
        TokenKind::Amper => 19,
        TokenKind::Less => 20,
        TokenKind::Greater => 21,
        TokenKind::Equal => 22,
        TokenKind::Dot => 23,
        TokenKind::Percent => 24,
        TokenKind::Lbrace => 25,
        TokenKind::Rbrace => 26,
        TokenKind::EqEqual => 27,
        TokenKind::NotEqual => 28,
        TokenKind::LessEqual => 29,
        TokenKind::GreaterEqual => 30,
        TokenKind::Tilde => 31,
        TokenKind::CircumFlex => 32,
        TokenKind::LeftShift => 33,
        TokenKind::RightShift => 34,
        TokenKind::DoubleStar => 35,
        TokenKind::PlusEqual => 36,
        TokenKind::MinusEqual => 37,
        TokenKind::StarEqual => 38,
        TokenKind::SlashEqual => 39,
        TokenKind::PercentEqual => 40,
        TokenKind::AmperEqual => 41,
        TokenKind::VbarEqual => 42,
        TokenKind::CircumflexEqual => 43,
        TokenKind::LeftShiftEqual => 44,
        TokenKind::RightShiftEqual => 45,
        TokenKind::DoubleStarEqual => 46,
        TokenKind::DoubleSlash => 47,
        TokenKind::DoubleSlashEqual => 48,
        TokenKind::At => 49,
        TokenKind::AtEqual => 50,
        TokenKind::Rarrow => 51,
        TokenKind::Ellipsis => 52,
        TokenKind::ColonEqual => 53,
        TokenKind::Exclamation => 54,
        TokenKind::FStringStart => 59,
        TokenKind::FStringMiddle => 60,
        TokenKind::FStringEnd => 61,
        TokenKind::TStringStart => 62,
        TokenKind::TStringMiddle => 63,
        TokenKind::TStringEnd => 64,
        TokenKind::Comment => TOKEN_COMMENT,
        TokenKind::NonLogicalNewline => TOKEN_NL,
        TokenKind::IpyEscapeCommand | TokenKind::Question | TokenKind::Unknown => 67,
        TokenKind::Lazy => u8::MAX,
    }
}

crate::py_module! {
    "_tokenize",
    interpleveldefs: {
        "TokenizerIter" => type_object(),
    },
}
