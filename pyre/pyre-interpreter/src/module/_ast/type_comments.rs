//! `# type:` comments for a `type_comments=True` parse.
//!
//! The tree has no node for a comment, so the only place a type comment
//! survives a parse is the token list handed back beside it. `lexer.c` decides
//! what a comment is by matching `"# type: "` against its text, with each space
//! in that pattern standing for any run of spaces and tabs, and splits what
//! matches into `TYPE_IGNORE` — the word `ignore` followed by the end of the
//! comment or by an ASCII non-alphanumeric byte — and `TYPE_COMMENT`, which is
//! everything else.
//!
//! Where one attaches is the grammar's business, and the grammar takes a
//! `TYPE_COMMENT` in five places: after an assignment's value, after the colon
//! of a `for`, a `with` or a `def`, and after a parameter's comma. Each of
//! those is the span between two nodes whose ranges the tree already carries,
//! so attaching here is a scan of the comment list against those spans rather
//! than a second parse.

use ruff_text_size::Ranged;
use rustpython_compiler::ast::{
    self,
    token::{Token, TokenKind},
};

/// The literal `lexer.c` matches a comment against. A space stands for a run
/// of spaces and tabs, so `#type:int` matches as well as `# type: int`.
const PREFIX: &[u8] = b"# type: ";

/// One classified comment: where it starts, the line it is on, and the text
/// the tree publishes — the type for a `TYPE_COMMENT`, the tag for a
/// `TYPE_IGNORE`.
pub struct TypeComment {
    start: u32,
    /// Where `text` begins.  `lexer.c` starts the token past the prefix, so
    /// this is the position a diagnostic about the token reports.
    text_start: u32,
    /// Just past the last byte before the comment that is not a space, a tab
    /// or a form feed.  A rule that takes the comment as its very next token
    /// is satisfied exactly when the node it follows ends at or after this.
    code_end: u32,
    pub lineno: u32,
    pub text: String,
}

impl TypeComment {
    /// One read back off an `_ast` object rather than out of a source, which
    /// has no offset to carry.
    pub fn new(lineno: u32, text: String) -> Self {
        Self {
            start: 0,
            text_start: 0,
            code_end: 0,
            lineno,
            text,
        }
    }
}

/// Every `# type:` comment a parse found, split by what the lexer would call
/// it. The comments are consumed as they are attached; the grammar gives each
/// one at most one home, and one it never reaches is a `TYPE_COMMENT` token
/// standing where no rule accepts it, which is a parse failure -- see
/// [`TypeComments::misplaced`].
#[derive(Default)]
pub struct TypeComments {
    comments: Vec<TypeComment>,
    attached: Vec<bool>,
    pub ignores: Vec<TypeComment>,
    line_starts: Vec<u32>,
}

/// The offset just past `PREFIX` within `text`, or `None` when the comment is
/// an ordinary one.
///
/// The trailing space of the pattern matches an empty run, so a comment that
/// stops at the colon still matches and publishes an empty type.
fn after_prefix(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut at = 0usize;
    for want in PREFIX.iter().copied() {
        if want == b' ' {
            while matches!(bytes.get(at), Some(b' ' | b'\t')) {
                at += 1;
            }
        } else if bytes.get(at) == Some(&want) {
            at += 1;
        } else {
            return None;
        }
    }
    Some(at)
}

/// Whether what follows the prefix is the word `ignore` standing on its own:
/// the comment ends there, or the next byte is ASCII and not alphanumeric.
///
/// `_` is not alphanumeric, so `# type: ignore_x` is an ignore whose tag is
/// `_x`, while `# type: ignorex` is an ordinary type comment.
fn is_ignore(rest: &[u8]) -> bool {
    rest.starts_with(b"ignore")
        && !matches!(rest.get(6), Some(&byte) if byte >= 128 || byte.is_ascii_alphanumeric())
}

/// Split a parse's comment tokens into the two kinds.
pub fn collect(tokens: &[Token], source: &str) -> TypeComments {
    let mut out = TypeComments {
        line_starts: std::iter::once(0)
            .chain(
                source
                    .bytes()
                    .enumerate()
                    .filter(|(_, byte)| *byte == b'\n')
                    .map(|(offset, _)| offset as u32 + 1),
            )
            .collect(),
        ..TypeComments::default()
    };
    for token in tokens {
        if token.kind() != TokenKind::Comment {
            continue;
        }
        let start = u32::from(token.range().start()) as usize;
        let end = u32::from(token.range().end()) as usize;
        let text = &source[start..end];
        let Some(at) = after_prefix(text) else {
            continue;
        };
        let lineno = out
            .line_starts
            .partition_point(|first| *first as usize <= start) as u32;
        // The three bytes skipped are ASCII, so the last byte that is not one
        // of them ends whatever character it belongs to.
        let code_end = source.as_bytes()[..start]
            .iter()
            .rposition(|byte| !matches!(byte, b' ' | b'\t' | 0x0c))
            .map_or(0, |index| index + 1) as u32;
        if is_ignore(&text.as_bytes()[at..]) {
            // The tag runs to the end of the comment, and takes the line break
            // with it when nothing but whitespace precedes the comment on its
            // line: `lexer.c` consumes the newline for an ignore that stands
            // alone, so `# type: ignore` on its own line reports `"\n"`.
            let mut tag = text[at + "ignore".len()..].to_owned();
            let line_start = out.line_starts[lineno as usize - 1] as usize;
            let alone = source[line_start..start]
                .bytes()
                .all(|byte| matches!(byte, b' ' | b'\t' | 0x0c));
            // The tokenizer supplies the line break a source without one is
            // missing, so an ignore that ends the file reports it too.
            if alone {
                tag.push('\n');
            }
            out.ignores.push(TypeComment {
                start: start as u32,
                text_start: (start + at) as u32,
                code_end,
                lineno,
                text: tag,
            });
        } else {
            out.comments.push(TypeComment {
                start: start as u32,
                text_start: (start + at) as u32,
                code_end,
                lineno,
                text: text[at..].to_owned(),
            });
            out.attached.push(false);
        }
    }
    out
}

impl TypeComments {
    /// The first unattached comment starting inside `[from, to)`, consumed.
    fn take(&mut self, from: u32, to: u32) -> Option<Box<str>> {
        let index = self
            .comments
            .iter()
            .enumerate()
            .find(|(index, comment)| {
                !self.attached[*index] && comment.start >= from && comment.start < to
            })
            .map(|(index, _)| index)?;
        self.attached[index] = true;
        Some(self.comments[index].text.as_str().into())
    }

    /// The unattached comment standing directly after `offset`, consumed.
    ///
    /// An assignment's rule reads its `TYPE_COMMENT` straight after the value,
    /// so the comment has to be the next token: in `x = 1; y = 2  # type: int`
    /// it belongs to the second assignment and in `x = 1;  # type: int` to
    /// neither.  Only spaces and tabs may stand between, which a line break
    /// is not -- a comment on the following line is nobody's.
    fn take_adjacent(&mut self, offset: u32) -> Option<Box<str>> {
        let index = self
            .comments
            .iter()
            .enumerate()
            .find(|(index, comment)| {
                !self.attached[*index] && comment.start >= offset && comment.code_end <= offset
            })
            .map(|(index, _)| index)?;
        self.attached[index] = true;
        Some(self.comments[index].text.as_str().into())
    }

    /// The `SyntaxError` a comment the grammar had no place for owes, if any.
    ///
    /// A `TYPE_COMMENT` is a token, so a rule that does not accept one leaves
    /// the parser looking at a token it cannot shift and the parse fails
    /// there: `x += 1  # type: int` and `x: int = 1  # type: int` are both
    /// refused, while the five accepted positions are not. The failure is the
    /// parser's ordinary one, so the message is `invalid syntax` and the span
    /// is the token -- the type text, not the `#`.
    ///
    /// Answers the first such comment in source order, which is the one the
    /// parser would have reached first.
    ///
    /// `file_input` says whether the tokenizer appends the line break a source
    /// without one is missing: file input gets one, so its reported line always
    /// ends in `\n`, while `eval` and `single` input report the line as it
    /// stands.
    pub fn misplaced(&self, source: &str, file_input: bool) -> Option<crate::PyError> {
        let comment = self
            .comments
            .iter()
            .zip(self.attached.iter())
            .find(|(_, attached)| !**attached)
            .map(|(comment, _)| comment)?;
        let line_start = self.line_starts[comment.lineno as usize - 1] as usize;
        let column = crate::builtins::syntax_error_character_offset(
            source,
            comment.lineno as usize,
            comment.text_start as usize - line_start + 1,
        ) as i64;
        let mut line = source[line_start..]
            .split_inclusive('\n')
            .next()
            .unwrap_or_default()
            .to_owned();
        if file_input && !line.ends_with('\n') {
            line.push('\n');
        }
        Some(crate::PyError::syntax_error_located(
            "invalid syntax",
            // `compile()` overwrites this with the name it was given.
            rustpython_wtf8::Wtf8::new(""),
            comment.lineno as i64,
            column,
            comment.lineno as i64,
            column + comment.text.chars().count() as i64,
            Some(&line),
        ))
    }

    /// Attach every comment the grammar has a place for, and answer the
    /// module's `TypeIgnore` list.
    pub fn attach(&mut self, module: &mut ast::Mod) {
        if let ast::Mod::Module(module) = module {
            self.stmts(&mut module.body);
        }
    }

    fn stmts(&mut self, body: &mut [ast::Stmt]) {
        for stmt in body {
            self.stmt(stmt);
        }
    }

    fn stmt(&mut self, stmt: &mut ast::Stmt) {
        match stmt {
            ast::Stmt::Assign(node) => {
                node.runtime_type_comment = self.take_adjacent(node.range.end().into());
            }
            ast::Stmt::For(node) => {
                let from = node.iter.range().end().into();
                if let Some(to) = node.body.first().map(|s| s.range().start().into()) {
                    node.runtime_type_comment = self.take(from, to);
                }
                self.stmts(&mut node.body);
                self.stmts(&mut node.orelse);
            }
            ast::Stmt::With(node) => {
                let from = node
                    .items
                    .last()
                    .map(|item| item.range().end().into())
                    .unwrap_or_else(|| node.range.start().into());
                if let Some(to) = node.body.first().map(|s| s.range().start().into()) {
                    node.runtime_type_comment = self.take(from, to);
                }
                self.stmts(&mut node.body);
            }
            ast::Stmt::FunctionDef(node) => {
                // A parameter's comment sits inside the parentheses and the
                // function's sits past them, so the two spans cannot collide.
                self.parameters(&mut node.parameters);
                let from = node
                    .returns
                    .as_ref()
                    .map(|returns| returns.range().end().into())
                    .unwrap_or_else(|| node.parameters.range().end().into());
                if let Some(to) = node.body.first().map(|s| s.range().start().into()) {
                    node.runtime_type_comment = self.take(from, to);
                }
                self.stmts(&mut node.body);
            }
            ast::Stmt::ClassDef(node) => self.stmts(&mut node.body),
            ast::Stmt::While(node) => {
                self.stmts(&mut node.body);
                self.stmts(&mut node.orelse);
            }
            ast::Stmt::If(node) => {
                self.stmts(&mut node.body);
                for clause in &mut node.elif_else_clauses {
                    self.stmts(&mut clause.body);
                }
            }
            ast::Stmt::Try(node) => {
                self.stmts(&mut node.body);
                for handler in &mut node.handlers {
                    let ast::ExceptHandler::ExceptHandler(handler) = handler;
                    self.stmts(&mut handler.body);
                }
                self.stmts(&mut node.orelse);
                self.stmts(&mut node.finalbody);
            }
            ast::Stmt::Match(node) => {
                for case in &mut node.cases {
                    self.stmts(&mut case.body);
                }
            }
            _ => {}
        }
    }

    /// A parameter's comment follows its comma, so its span runs to the start
    /// of the next parameter, or to the closing parenthesis for the last one.
    fn parameters(&mut self, parameters: &mut ast::Parameters) {
        let close = parameters.range().end().into();
        let mut spans: Vec<(u32, &mut Option<Box<str>>)> = Vec::new();
        for with_default in parameters
            .posonlyargs
            .iter_mut()
            .chain(parameters.args.iter_mut())
        {
            spans.push((
                with_default.range().end().into(),
                &mut with_default.parameter.runtime_type_comment,
            ));
        }
        if let Some(vararg) = parameters.vararg.as_mut() {
            spans.push((
                vararg.range().end().into(),
                &mut vararg.runtime_type_comment,
            ));
        }
        for with_default in parameters.kwonlyargs.iter_mut() {
            spans.push((
                with_default.range().end().into(),
                &mut with_default.parameter.runtime_type_comment,
            ));
        }
        if let Some(kwarg) = parameters.kwarg.as_mut() {
            spans.push((kwarg.range().end().into(), &mut kwarg.runtime_type_comment));
        }
        spans.sort_by_key(|(end, _)| *end);
        let ends: Vec<u32> = spans.iter().map(|(end, _)| *end).collect();
        for (index, (end, slot)) in spans.into_iter().enumerate() {
            let to = ends.get(index + 1).copied().unwrap_or(close);
            *slot = self.take(end, to);
        }
    }
}
