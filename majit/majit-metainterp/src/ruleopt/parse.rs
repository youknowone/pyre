//! Parser for the integer optimization rule DSL.
//!
//! RPython counterpart: `rpython/jit/metainterp/ruleopt/parse.py`.
//!
//! PyPy uses `rply` to build this grammar.  The Rust port keeps the same AST
//! surface and type-checking rules, but uses a small hand-written lexer and
//! Pratt parser because the DSL grammar is fixed and local to `real.rules`.

use std::collections::HashMap;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct SourcePos {
    pub lineno: usize,
    pub colno: usize,
}

pub trait BaseAst {
    fn sourcepos(&self) -> Option<SourcePos>;
    fn endsourcepos(&self) -> Option<SourcePos>;
}

pub trait Visitor {
    fn default_visit(&mut self) {}
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct File {
    pub rules: Vec<Rule>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rule {
    pub name: String,
    pub pattern: Pattern,
    pub cantproof: bool,
    pub elements: Vec<Element>,
    pub target: Pattern,
    pub sourcepos: Option<SourcePos>,
    pub endsourcepos: Option<SourcePos>,
}

impl Rule {
    pub fn newpattern(&self, pattern: Pattern) -> Self {
        Self {
            name: self.name.clone(),
            pattern,
            cantproof: self.cantproof,
            elements: self.elements.clone(),
            target: self.target.clone(),
            sourcepos: self.sourcepos,
            endsourcepos: self.endsourcepos,
        }
    }
}

impl BaseAst for Rule {
    fn sourcepos(&self) -> Option<SourcePos> {
        self.sourcepos
    }

    fn endsourcepos(&self) -> Option<SourcePos> {
        self.endsourcepos
    }
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}: {}", self.name, self.pattern)?;
        for element in &self.elements {
            writeln!(f, "    {element}")?;
        }
        write!(f, "    => {}", self.target)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum Pattern {
    PatternVar(PatternVar),
    PatternConst(PatternConst),
    PatternOp(PatternOp),
}

impl Pattern {
    pub fn typ(&self) -> RuleType {
        match self {
            Pattern::PatternVar(v) => v.typ.unwrap_or_else(|| inferred_name_type(&v.name)),
            Pattern::PatternConst(_) => RuleType::Int,
            Pattern::PatternOp(_) => RuleType::IntBound,
        }
    }

    pub fn sourcepos(&self) -> Option<SourcePos> {
        match self {
            Pattern::PatternVar(v) => v.sourcepos,
            Pattern::PatternConst(v) => v.sourcepos,
            Pattern::PatternOp(v) => v.sourcepos,
        }
    }

    pub fn matches_constant(&self) -> bool {
        match self {
            Pattern::PatternVar(v) => v.matches_constant(),
            Pattern::PatternConst(v) => v.matches_constant(),
            Pattern::PatternOp(v) => v.matches_constant(),
        }
    }

    pub fn sort_key(&self) -> String {
        match self {
            Pattern::PatternConst(v) => format!("0:{}", v.const_value),
            Pattern::PatternOp(v) => {
                let mut arg_keys = v.args.iter().map(Pattern::sort_key).collect::<Vec<_>>();
                arg_keys.sort();
                format!("1:{}({})", v.opname, arg_keys.join(","))
            }
            Pattern::PatternVar(v) => format!("2:{}", v.name),
        }
    }

    pub fn newargs(&self, args: Vec<Pattern>) -> Self {
        match self {
            Pattern::PatternOp(v) => Pattern::PatternOp(v.newargs(args)),
            _ => self.clone(),
        }
    }
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Pattern::PatternVar(v) => write!(f, "{}", v.name),
            Pattern::PatternConst(v) => write!(f, "{}", v.const_value),
            Pattern::PatternOp(v) => {
                write!(f, "{}(", v.opname)?;
                for (i, arg) in v.args.iter().enumerate() {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PatternVar {
    pub name: String,
    pub typ: Option<RuleType>,
    pub sourcepos: Option<SourcePos>,
}

impl PatternVar {
    pub fn matches_constant(&self) -> bool {
        self.name.starts_with('C')
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PatternConst {
    pub const_value: String,
    pub typ: RuleType,
    pub sourcepos: Option<SourcePos>,
}

impl PatternConst {
    pub fn matches_constant(&self) -> bool {
        true
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PatternOp {
    pub opname: String,
    pub args: Vec<Pattern>,
    pub sourcepos: Option<SourcePos>,
}

impl PatternOp {
    pub fn newargs(&self, args: Vec<Pattern>) -> Self {
        Self {
            opname: self.opname.clone(),
            args,
            sourcepos: self.sourcepos,
        }
    }

    pub fn matches_constant(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Element {
    Compute(Compute),
    Check(Check),
}

impl fmt::Display for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Element::Compute(c) => write!(f, "compute {} = {}", c.name, c.expr),
            Element::Check(c) => write!(f, "check {}", c.expr),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Compute {
    pub name: String,
    pub expr: Expression,
    pub sourcepos: Option<SourcePos>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Check {
    pub expr: Expression,
    pub sourcepos: Option<SourcePos>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Expression {
    Name(Name),
    Number(Number),
    Add(Box<Add>),
    Sub(Box<Sub>),
    Mul(Box<Mul>),
    Div(Box<Div>),
    LShift(Box<LShift>),
    URShift(Box<URShift>),
    ARShift(Box<ARShift>),
    OpAnd(Box<OpAnd>),
    OpOr(Box<OpOr>),
    OpXor(Box<OpXor>),
    Eq(Box<Eq>),
    Ge(Box<Ge>),
    Gt(Box<Gt>),
    Le(Box<Le>),
    Lt(Box<Lt>),
    Ne(Box<Ne>),
    ShortcutAnd(Box<ShortcutAnd>),
    ShortcutOr(Box<ShortcutOr>),
    Invert(Box<Invert>),
    Attribute(Attribute),
    MethodCall(Box<MethodCall>),
    FuncCall(FuncCall),
}

impl Expression {
    fn sourcepos(&self) -> Option<SourcePos> {
        match self {
            Expression::Name(v) => v.sourcepos,
            Expression::Number(v) => v.sourcepos,
            Expression::Add(v) => v.sourcepos,
            Expression::Sub(v) => v.sourcepos,
            Expression::Mul(v) => v.sourcepos,
            Expression::Div(v) => v.sourcepos,
            Expression::LShift(v) => v.sourcepos,
            Expression::URShift(v) => v.sourcepos,
            Expression::ARShift(v) => v.sourcepos,
            Expression::OpAnd(v) => v.sourcepos,
            Expression::OpOr(v) => v.sourcepos,
            Expression::OpXor(v) => v.sourcepos,
            Expression::Eq(v) => v.sourcepos,
            Expression::Ge(v) => v.sourcepos,
            Expression::Gt(v) => v.sourcepos,
            Expression::Le(v) => v.sourcepos,
            Expression::Lt(v) => v.sourcepos,
            Expression::Ne(v) => v.sourcepos,
            Expression::ShortcutAnd(v) => v.sourcepos,
            Expression::ShortcutOr(v) => v.sourcepos,
            Expression::Invert(v) => v.sourcepos,
            Expression::Attribute(v) => v.sourcepos,
            Expression::MethodCall(v) => v.sourcepos,
            Expression::FuncCall(v) => v.sourcepos,
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Name(v) => write!(f, "{}", v.name),
            Expression::Number(v) => write!(f, "{}", v.value),
            Expression::Add(v) => display_binop(f, &v.left, "+", &v.right),
            Expression::Sub(v) => display_binop(f, &v.left, "-", &v.right),
            Expression::Mul(v) => display_binop(f, &v.left, "*", &v.right),
            Expression::Div(v) => display_binop(f, &v.left, "//", &v.right),
            Expression::LShift(v) => display_binop(f, &v.left, "<<", &v.right),
            Expression::URShift(v) => display_binop(f, &v.left, ">>u", &v.right),
            Expression::ARShift(v) => display_binop(f, &v.left, ">>", &v.right),
            Expression::OpAnd(v) => display_binop(f, &v.left, "&", &v.right),
            Expression::OpOr(v) => display_binop(f, &v.left, "|", &v.right),
            Expression::OpXor(v) => display_binop(f, &v.left, "^", &v.right),
            Expression::Eq(v) => display_binop(f, &v.left, "==", &v.right),
            Expression::Ge(v) => display_binop(f, &v.left, ">=", &v.right),
            Expression::Gt(v) => display_binop(f, &v.left, ">", &v.right),
            Expression::Le(v) => display_binop(f, &v.left, "<=", &v.right),
            Expression::Lt(v) => display_binop(f, &v.left, "<", &v.right),
            Expression::Ne(v) => display_binop(f, &v.left, "!=", &v.right),
            Expression::ShortcutAnd(v) => display_binop(f, &v.left, "and", &v.right),
            Expression::ShortcutOr(v) => display_binop(f, &v.left, "or", &v.right),
            Expression::Invert(v) => write!(f, "~{}", v.left),
            Expression::Attribute(v) => write!(f, "{}.{}", v.varname, v.attrname),
            Expression::MethodCall(v) => {
                write!(f, "{}.{}(", v.value, v.methname)?;
                display_args(f, &v.args)?;
                write!(f, ")")
            }
            Expression::FuncCall(v) => {
                write!(f, "{}(", v.funcname)?;
                display_args(f, &v.args)?;
                write!(f, ")")
            }
        }
    }
}

fn display_binop(
    f: &mut fmt::Formatter<'_>,
    left: &Expression,
    symbol: &str,
    right: &Expression,
) -> fmt::Result {
    write!(f, "{left} {symbol} {right}")
}

fn display_args(f: &mut fmt::Formatter<'_>, args: &[Expression]) -> fmt::Result {
    for (i, arg) in args.iter().enumerate() {
        if i != 0 {
            write!(f, ", ")?;
        }
        write!(f, "{arg}")?;
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Name {
    pub name: String,
    pub typ: Option<RuleType>,
    pub sourcepos: Option<SourcePos>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Number {
    pub value: i64,
    pub typ: RuleType,
    pub sourcepos: Option<SourcePos>,
}

pub trait BinOp {}

pub trait IntBinOp: BinOp {}

pub trait BoolBinOp: BinOp {}

pub trait UnaryOp {}

pub trait IntUnaryOp: UnaryOp {}

macro_rules! binop_struct {
    ($name:ident, $typ:expr, $opname:expr, $pysymbol:expr, $need_ruint:expr) => {
        #[derive(Clone, Debug, Eq, PartialEq)]
        pub struct $name {
            pub left: Expression,
            pub right: Expression,
            pub typ: RuleType,
            pub opname: Option<&'static str>,
            pub pysymbol: &'static str,
            pub need_ruint: bool,
            pub sourcepos: Option<SourcePos>,
        }

        impl $name {
            fn new(left: Expression, right: Expression, sourcepos: Option<SourcePos>) -> Self {
                Self {
                    left,
                    right,
                    typ: $typ,
                    opname: $opname,
                    pysymbol: $pysymbol,
                    need_ruint: $need_ruint,
                    sourcepos,
                }
            }
        }

        impl BinOp for $name {}
    };
}

binop_struct!(Add, RuleType::Int, Some("int_add"), "+", true);
binop_struct!(Sub, RuleType::Int, Some("int_sub"), "-", true);
binop_struct!(Mul, RuleType::Int, Some("int_mul"), "*", true);
binop_struct!(Div, RuleType::Int, None, "//", false);
binop_struct!(LShift, RuleType::Int, Some("int_lshift"), "<<", false);
binop_struct!(URShift, RuleType::Int, Some("uint_rshift"), ">>", true);
binop_struct!(ARShift, RuleType::Int, Some("int_rshift"), ">>", false);
binop_struct!(OpAnd, RuleType::Int, Some("int_and"), "&", false);
binop_struct!(OpOr, RuleType::Int, Some("int_or"), "|", false);
binop_struct!(OpXor, RuleType::Int, Some("int_xor"), "^", false);
binop_struct!(Eq, RuleType::Bool, Some("int_eq"), "==", false);
binop_struct!(Ge, RuleType::Bool, Some("int_ge"), ">=", false);
binop_struct!(Gt, RuleType::Bool, Some("int_gt"), ">", false);
binop_struct!(Le, RuleType::Bool, Some("int_le"), "<=", false);
binop_struct!(Lt, RuleType::Bool, Some("int_lt"), "<", false);
binop_struct!(Ne, RuleType::Bool, Some("int_ne"), "!=", false);
binop_struct!(ShortcutAnd, RuleType::Bool, None, "and", false);
binop_struct!(ShortcutOr, RuleType::Bool, None, "or", false);

impl IntBinOp for Add {}
impl IntBinOp for Sub {}
impl IntBinOp for Mul {}
impl IntBinOp for Div {}
impl IntBinOp for LShift {}
impl IntBinOp for URShift {}
impl IntBinOp for ARShift {}
impl IntBinOp for OpAnd {}
impl IntBinOp for OpOr {}
impl IntBinOp for OpXor {}

impl BoolBinOp for Eq {}
impl BoolBinOp for Ge {}
impl BoolBinOp for Gt {}
impl BoolBinOp for Le {}
impl BoolBinOp for Lt {}
impl BoolBinOp for Ne {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Invert {
    pub left: Expression,
    pub typ: RuleType,
    pub opname: &'static str,
    pub pysymbol: &'static str,
    pub sourcepos: Option<SourcePos>,
}

impl UnaryOp for Invert {}
impl IntUnaryOp for Invert {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Attribute {
    pub varname: String,
    pub attrname: String,
    pub typ: RuleType,
    pub sourcepos: Option<SourcePos>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MethodCall {
    pub value: Expression,
    pub methname: String,
    pub args: Vec<Expression>,
    pub typ: Option<RuleType>,
    pub sourcepos: Option<SourcePos>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FuncCall {
    pub funcname: String,
    pub args: Vec<Expression>,
    pub typ: Option<RuleType>,
    pub sourcepos: Option<SourcePos>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum RuleType {
    Int,
    Bool,
    IntBound,
}

impl RuleType {
    fn py_name(self) -> &'static str {
        match self {
            RuleType::Int => "int",
            RuleType::Bool => "bool",
            RuleType::IntBound => "IntBound",
        }
    }
}

impl fmt::Display for RuleType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.py_name())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParseError {
    pub msg: String,
    pub sourcepos: Option<SourcePos>,
}

impl ParseError {
    fn new(msg: impl Into<String>, sourcepos: Option<SourcePos>) -> Self {
        Self {
            msg: msg.into(),
            sourcepos,
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}

impl std::error::Error for ParseError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TypeCheckError {
    pub msg: String,
    pub sourcepos: Option<SourcePos>,
}

impl TypeCheckError {
    fn new(msg: impl Into<String>, sourcepos: Option<SourcePos>) -> Self {
        Self {
            msg: msg.into(),
            sourcepos,
        }
    }

    pub fn format(&self, sourcelines: Option<&str>) -> String {
        let mut res = vec![format!("Type error: {}", self.msg)];
        if let Some(pos) = self.sourcepos {
            res.push(format!("in line {}", pos.lineno));
            if let Some(source) = sourcelines
                && let Some(line) = source.lines().nth(pos.lineno.saturating_sub(1))
            {
                res.push(format!("    {line}"));
                res.push(format!("    {}^", " ".repeat(pos.colno.saturating_sub(1))));
            }
        }
        res.join("\n")
    }
}

impl fmt::Display for TypeCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}

impl std::error::Error for TypeCheckError {}

#[derive(Debug)]
pub enum RuleParseError {
    Parse(ParseError),
    TypeCheck(TypeCheckError),
}

impl fmt::Display for RuleParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuleParseError::Parse(e) => write!(f, "{e}"),
            RuleParseError::TypeCheck(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for RuleParseError {}

impl From<ParseError> for RuleParseError {
    fn from(value: ParseError) -> Self {
        RuleParseError::Parse(value)
    }
}

impl From<TypeCheckError> for RuleParseError {
    fn from(value: TypeCheckError) -> Self {
        RuleParseError::TypeCheck(value)
    }
}

pub fn parse(source: &str) -> Result<File, RuleParseError> {
    let mut rules = Vec::new();
    let mut current: Option<PartialRule> = None;
    for (idx, raw_line) in source.lines().enumerate() {
        let lineno = idx + 1;
        let Some((line, base_col)) = strip_rule_comment(raw_line) else {
            continue;
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !line.starts_with(' ') && !line.starts_with('\t') {
            if let Some(rule) = current.take() {
                rules.push(rule.finish()?);
            }
            current = Some(parse_rule_header(trimmed, lineno, base_col)?);
            continue;
        }
        let partial = current.as_mut().ok_or_else(|| {
            ParseError::new(
                "ruleopt parse.py: element before rule header",
                Some(SourcePos {
                    lineno,
                    colno: base_col,
                }),
            )
        })?;
        parse_rule_element(partial, line, lineno)?;
    }
    if let Some(rule) = current.take() {
        rules.push(rule.finish()?);
    }
    let mut file = File { rules };
    typecheck_file(&mut file)?;
    Ok(file)
}

fn strip_rule_comment(line: &str) -> Option<(&str, usize)> {
    let mut in_comment_at = None;
    for (idx, ch) in line.char_indices() {
        if ch == '#' {
            in_comment_at = Some(idx);
            break;
        }
    }
    let without_comment = &line[..in_comment_at.unwrap_or(line.len())];
    if without_comment.trim().is_empty() {
        return None;
    }
    let col = without_comment
        .find(|c: char| !c.is_whitespace())
        .map(|idx| idx + 1)
        .unwrap_or(1);
    Some((without_comment, col))
}

#[derive(Clone, Debug)]
struct PartialRule {
    name: String,
    pattern: Pattern,
    cantproof: bool,
    elements: Vec<Element>,
    target: Option<Pattern>,
    sourcepos: SourcePos,
    endsourcepos: Option<SourcePos>,
}

impl PartialRule {
    fn finish(self) -> Result<Rule, ParseError> {
        let target = self.target.ok_or_else(|| {
            ParseError::new(
                format!("rule {} has no target", self.name),
                Some(self.sourcepos),
            )
        })?;
        Ok(Rule {
            name: self.name,
            pattern: self.pattern,
            cantproof: self.cantproof,
            elements: self.elements,
            target,
            sourcepos: Some(self.sourcepos),
            endsourcepos: self.endsourcepos,
        })
    }
}

fn parse_rule_header(
    line: &str,
    lineno: usize,
    base_col: usize,
) -> Result<PartialRule, ParseError> {
    let Some((name, pattern_source)) = line.split_once(':') else {
        return Err(ParseError::new(
            "ruleopt parse.py: expected ':' in rule header",
            Some(SourcePos {
                lineno,
                colno: base_col,
            }),
        ));
    };
    let name = name.trim();
    if name.is_empty() {
        return Err(ParseError::new(
            "ruleopt parse.py: empty rule name",
            Some(SourcePos {
                lineno,
                colno: base_col,
            }),
        ));
    }
    let pattern_offset = line.find(pattern_source).unwrap_or(name.len() + 1);
    let pattern_col = base_col + pattern_offset + leading_ws(pattern_source);
    let mut parser = PatternParser::new(pattern_source.trim(), lineno, pattern_col)?;
    let pattern = parser.parse_pattern()?;
    parser.expect_eof()?;
    Ok(PartialRule {
        name: name.to_string(),
        pattern,
        cantproof: false,
        elements: Vec::new(),
        target: None,
        sourcepos: SourcePos {
            lineno,
            colno: base_col,
        },
        endsourcepos: None,
    })
}

fn parse_rule_element(rule: &mut PartialRule, line: &str, lineno: usize) -> Result<(), ParseError> {
    let trimmed_start = leading_ws(line);
    let content = &line[trimmed_start..];
    let col = trimmed_start + 1;
    let trimmed = content.trim();
    if trimmed == "SORRY_Z3" {
        rule.cantproof = true;
        return Ok(());
    }
    if let Some(target) = trimmed.strip_prefix("=>") {
        let arrow_idx = line.find("=>").unwrap_or(trimmed_start);
        let target_col = arrow_idx + 3 + leading_ws(target);
        let mut parser = PatternParser::new(target.trim(), lineno, target_col)?;
        let target = parser.parse_pattern()?;
        parser.expect_eof()?;
        rule.endsourcepos = target.sourcepos();
        rule.target = Some(target);
        return Ok(());
    }
    if let Some(expr) = trimmed.strip_prefix("check ") {
        let expr_col = line.find("check").unwrap_or(trimmed_start) + "check ".len() + 1;
        let mut parser = ExprParser::new(expr.trim(), lineno, expr_col)?;
        let expr = parser.parse_expression(0)?;
        parser.expect_eof()?;
        rule.elements.push(Element::Check(Check {
            expr,
            sourcepos: Some(SourcePos { lineno, colno: col }),
        }));
        return Ok(());
    }
    let Some((name, expr)) = trimmed.split_once('=') else {
        return Err(ParseError::new(
            "ruleopt parse.py: expected rule element",
            Some(SourcePos { lineno, colno: col }),
        ));
    };
    let name = name.trim();
    let expr_col = line.find('=').unwrap_or(trimmed_start) + 2 + leading_ws(expr);
    let mut parser = ExprParser::new(expr.trim(), lineno, expr_col)?;
    let expr = parser.parse_expression(0)?;
    parser.expect_eof()?;
    rule.elements.push(Element::Compute(Compute {
        name: name.to_string(),
        expr,
        sourcepos: Some(SourcePos { lineno, colno: col }),
    }));
    Ok(())
}

fn leading_ws(s: &str) -> usize {
    s.chars().take_while(|c| c.is_whitespace()).count()
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Token {
    kind: TokenKind,
    text: String,
    pos: SourcePos,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TokenKind {
    Number,
    Name,
    Check,
    And,
    Or,
    SorryZ3,
    LShift,
    URShift,
    ARShift,
    Arrow,
    LParen,
    RParen,
    Comma,
    EqualEqual,
    Ne,
    Equal,
    Colon,
    Dot,
    Ge,
    Gt,
    Le,
    Lt,
    Plus,
    Minus,
    Mul,
    Div,
    OpAnd,
    OpOr,
    OpXor,
    Invert,
    Eof,
}

fn lex(input: &str, lineno: usize, base_col: usize) -> Result<Vec<Token>, ParseError> {
    let chars: Vec<char> = input.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if ch == ' ' || ch == '\t' {
            i += 1;
            continue;
        }
        let pos = SourcePos {
            lineno,
            colno: base_col + i,
        };
        if ch.is_ascii_digit() || is_signed_number(&chars, i, out.last()) {
            let start = i;
            if ch == '+' || ch == '-' {
                i += 1;
            }
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            out.push(Token {
                kind: TokenKind::Number,
                text: chars[start..i].iter().collect(),
                pos,
            });
            continue;
        }
        if ch.is_ascii_alphabetic() || ch == '_' {
            let start = i;
            i += 1;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let text: String = chars[start..i].iter().collect();
            let kind = match text.as_str() {
                "check" => TokenKind::Check,
                "and" => TokenKind::And,
                "or" => TokenKind::Or,
                "SORRY_Z3" => TokenKind::SorryZ3,
                _ => TokenKind::Name,
            };
            out.push(Token { kind, text, pos });
            continue;
        }
        let (kind, len) = match ch {
            '<' if chars.get(i + 1) == Some(&'<') => (TokenKind::LShift, 2),
            '>' if chars.get(i + 1) == Some(&'>') && chars.get(i + 2) == Some(&'u') => {
                (TokenKind::URShift, 3)
            }
            '>' if chars.get(i + 1) == Some(&'>') => (TokenKind::ARShift, 2),
            '=' if chars.get(i + 1) == Some(&'>') => (TokenKind::Arrow, 2),
            '=' if chars.get(i + 1) == Some(&'=') => (TokenKind::EqualEqual, 2),
            '!' if chars.get(i + 1) == Some(&'=') => (TokenKind::Ne, 2),
            '>' if chars.get(i + 1) == Some(&'=') => (TokenKind::Ge, 2),
            '<' if chars.get(i + 1) == Some(&'=') => (TokenKind::Le, 2),
            '/' if chars.get(i + 1) == Some(&'/') => (TokenKind::Div, 2),
            '(' => (TokenKind::LParen, 1),
            ')' => (TokenKind::RParen, 1),
            ',' => (TokenKind::Comma, 1),
            '=' => (TokenKind::Equal, 1),
            ':' => (TokenKind::Colon, 1),
            '.' => (TokenKind::Dot, 1),
            '>' => (TokenKind::Gt, 1),
            '<' => (TokenKind::Lt, 1),
            '+' => (TokenKind::Plus, 1),
            '-' => (TokenKind::Minus, 1),
            '*' => (TokenKind::Mul, 1),
            '&' => (TokenKind::OpAnd, 1),
            '|' => (TokenKind::OpOr, 1),
            '^' => (TokenKind::OpXor, 1),
            '~' => (TokenKind::Invert, 1),
            _ => {
                return Err(ParseError::new(
                    format!("ruleopt parse.py: unexpected character {ch:?}"),
                    Some(pos),
                ));
            }
        };
        out.push(Token {
            kind,
            text: chars[i..i + len].iter().collect(),
            pos,
        });
        i += len;
    }
    out.push(Token {
        kind: TokenKind::Eof,
        text: String::new(),
        pos: SourcePos {
            lineno,
            colno: base_col + input.len(),
        },
    });
    Ok(out)
}

fn is_signed_number(chars: &[char], i: usize, prev: Option<&Token>) -> bool {
    let ch = chars[i];
    if ch != '+' && ch != '-' {
        return false;
    }
    if chars.get(i + 1).is_none_or(|next| !next.is_ascii_digit()) {
        return false;
    }
    match prev.map(|t| &t.kind) {
        None => true,
        Some(
            TokenKind::LParen
            | TokenKind::Comma
            | TokenKind::Equal
            | TokenKind::Colon
            | TokenKind::Arrow
            | TokenKind::Plus
            | TokenKind::Minus
            | TokenKind::Mul
            | TokenKind::Div
            | TokenKind::LShift
            | TokenKind::URShift
            | TokenKind::ARShift
            | TokenKind::And
            | TokenKind::Or
            | TokenKind::OpAnd
            | TokenKind::OpOr
            | TokenKind::OpXor
            | TokenKind::EqualEqual
            | TokenKind::Ne
            | TokenKind::Ge
            | TokenKind::Gt
            | TokenKind::Le
            | TokenKind::Lt
            | TokenKind::Invert,
        ) => true,
        _ => false,
    }
}

struct PatternParser {
    tokens: Vec<Token>,
    pos: usize,
}

impl PatternParser {
    fn new(input: &str, lineno: usize, base_col: usize) -> Result<Self, ParseError> {
        Ok(Self {
            tokens: lex(input, lineno, base_col)?,
            pos: 0,
        })
    }

    fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
        let token = self.advance().clone();
        match token.kind {
            TokenKind::Name => {
                if self.peek().kind == TokenKind::LParen {
                    self.advance();
                    let args = self.parse_patternargs()?;
                    self.expect(TokenKind::RParen)?;
                    Ok(Pattern::PatternOp(PatternOp {
                        opname: token.text,
                        args,
                        sourcepos: Some(token.pos),
                    }))
                } else if matches!(token.text.as_str(), "LONG_BIT" | "MININT" | "MAXINT") {
                    Ok(Pattern::PatternConst(PatternConst {
                        const_value: token.text,
                        typ: RuleType::Int,
                        sourcepos: Some(token.pos),
                    }))
                } else {
                    Ok(Pattern::PatternVar(PatternVar {
                        name: token.text,
                        typ: None,
                        sourcepos: Some(token.pos),
                    }))
                }
            }
            TokenKind::Number => Ok(Pattern::PatternConst(PatternConst {
                const_value: token.text,
                typ: RuleType::Int,
                sourcepos: Some(token.pos),
            })),
            _ => Err(ParseError::new(
                format!("ruleopt parse.py: expected pattern, got {:?}", token.kind),
                Some(token.pos),
            )),
        }
    }

    fn parse_patternargs(&mut self) -> Result<Vec<Pattern>, ParseError> {
        if self.peek().kind == TokenKind::RParen {
            return Ok(Vec::new());
        }
        let mut args = vec![self.parse_pattern()?];
        while self.peek().kind == TokenKind::Comma {
            self.advance();
            args.push(self.parse_pattern()?);
        }
        Ok(args)
    }

    fn expect(&mut self, kind: TokenKind) -> Result<Token, ParseError> {
        let token = self.advance().clone();
        if token.kind != kind {
            return Err(ParseError::new(
                format!(
                    "ruleopt parse.py: expected {:?}, got {:?}",
                    kind, token.kind
                ),
                Some(token.pos),
            ));
        }
        Ok(token)
    }

    fn expect_eof(&self) -> Result<(), ParseError> {
        let token = self.peek();
        if token.kind != TokenKind::Eof {
            return Err(ParseError::new(
                format!("ruleopt parse.py: trailing token {:?}", token.kind),
                Some(token.pos),
            ));
        }
        Ok(())
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn advance(&mut self) -> &Token {
        let idx = self.pos;
        self.pos += 1;
        &self.tokens[idx]
    }
}

struct ExprParser {
    tokens: Vec<Token>,
    pos: usize,
}

impl ExprParser {
    fn new(input: &str, lineno: usize, base_col: usize) -> Result<Self, ParseError> {
        Ok(Self {
            tokens: lex(input, lineno, base_col)?,
            pos: 0,
        })
    }

    fn parse_expression(&mut self, min_prec: u8) -> Result<Expression, ParseError> {
        let mut left = self.parse_prefix()?;
        loop {
            if self.peek().kind == TokenKind::Dot {
                if 11 < min_prec {
                    break;
                }
                self.advance();
                let name = self.expect(TokenKind::Name)?;
                let sourcepos = left.sourcepos();
                if self.peek().kind == TokenKind::LParen {
                    self.advance();
                    let args = self.parse_args()?;
                    self.expect(TokenKind::RParen)?;
                    left = Expression::MethodCall(Box::new(MethodCall {
                        value: left,
                        methname: name.text,
                        args,
                        typ: None,
                        sourcepos,
                    }));
                } else {
                    let Expression::Name(base_name) = left else {
                        return Err(ParseError::new(
                            "ruleopt parse.py: attribute base must be a name",
                            Some(name.pos),
                        ));
                    };
                    left = Expression::Attribute(Attribute {
                        varname: base_name.name,
                        attrname: name.text,
                        typ: RuleType::Int,
                        sourcepos,
                    });
                }
                continue;
            }
            let Some((prec, kind)) = infix_precedence(&self.peek().kind) else {
                break;
            };
            if prec < min_prec {
                break;
            }
            let sourcepos = left.sourcepos();
            self.advance();
            let right = self.parse_expression(prec + 1)?;
            left = make_binop(kind, left, right, sourcepos);
        }
        Ok(left)
    }

    fn parse_prefix(&mut self) -> Result<Expression, ParseError> {
        let token = self.advance().clone();
        match token.kind {
            TokenKind::Number => {
                let value = token.text.parse::<i64>().map_err(|e| {
                    ParseError::new(
                        format!("ruleopt parse.py: invalid number {:?}: {e}", token.text),
                        Some(token.pos),
                    )
                })?;
                Ok(Expression::Number(Number {
                    value,
                    typ: RuleType::Int,
                    sourcepos: Some(token.pos),
                }))
            }
            TokenKind::Name => {
                if self.peek().kind == TokenKind::LParen {
                    self.advance();
                    let args = self.parse_args()?;
                    self.expect(TokenKind::RParen)?;
                    Ok(Expression::FuncCall(FuncCall {
                        funcname: token.text,
                        args,
                        typ: None,
                        sourcepos: Some(token.pos),
                    }))
                } else {
                    Ok(Expression::Name(Name {
                        typ: Some(inferred_name_type(&token.text)),
                        name: token.text,
                        sourcepos: Some(token.pos),
                    }))
                }
            }
            TokenKind::LParen => {
                let expr = self.parse_expression(0)?;
                self.expect(TokenKind::RParen)?;
                Ok(expr)
            }
            TokenKind::Invert => {
                let left = self.parse_expression(10)?;
                Ok(Expression::Invert(Box::new(Invert {
                    left,
                    typ: RuleType::Int,
                    opname: "int_invert",
                    pysymbol: "~",
                    sourcepos: Some(token.pos),
                })))
            }
            _ => Err(ParseError::new(
                format!(
                    "ruleopt parse.py: expected expression, got {:?}",
                    token.kind
                ),
                Some(token.pos),
            )),
        }
    }

    fn parse_args(&mut self) -> Result<Vec<Expression>, ParseError> {
        if self.peek().kind == TokenKind::RParen {
            return Ok(Vec::new());
        }
        let mut args = vec![self.parse_expression(0)?];
        while self.peek().kind == TokenKind::Comma {
            self.advance();
            args.push(self.parse_expression(0)?);
        }
        Ok(args)
    }

    fn expect(&mut self, kind: TokenKind) -> Result<Token, ParseError> {
        let token = self.advance().clone();
        if token.kind != kind {
            return Err(ParseError::new(
                format!(
                    "ruleopt parse.py: expected {:?}, got {:?}",
                    kind, token.kind
                ),
                Some(token.pos),
            ));
        }
        Ok(token)
    }

    fn expect_eof(&self) -> Result<(), ParseError> {
        let token = self.peek();
        if token.kind != TokenKind::Eof {
            return Err(ParseError::new(
                format!("ruleopt parse.py: trailing token {:?}", token.kind),
                Some(token.pos),
            ));
        }
        Ok(())
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn advance(&mut self) -> &Token {
        let idx = self.pos;
        self.pos += 1;
        &self.tokens[idx]
    }
}

fn infix_precedence(kind: &TokenKind) -> Option<(u8, TokenKind)> {
    let prec = match kind {
        TokenKind::Or => 1,
        TokenKind::And => 2,
        TokenKind::EqualEqual
        | TokenKind::Ge
        | TokenKind::Gt
        | TokenKind::Le
        | TokenKind::Lt
        | TokenKind::Ne => 3,
        TokenKind::OpOr => 4,
        TokenKind::OpXor => 5,
        TokenKind::OpAnd => 6,
        TokenKind::LShift | TokenKind::ARShift | TokenKind::URShift => 7,
        TokenKind::Plus | TokenKind::Minus => 8,
        TokenKind::Mul | TokenKind::Div => 9,
        _ => return None,
    };
    Some((prec, kind.clone()))
}

fn make_binop(
    kind: TokenKind,
    left: Expression,
    right: Expression,
    sourcepos: Option<SourcePos>,
) -> Expression {
    match kind {
        TokenKind::Plus => Expression::Add(Box::new(Add::new(left, right, sourcepos))),
        TokenKind::Minus => Expression::Sub(Box::new(Sub::new(left, right, sourcepos))),
        TokenKind::Mul => Expression::Mul(Box::new(Mul::new(left, right, sourcepos))),
        TokenKind::Div => Expression::Div(Box::new(Div::new(left, right, sourcepos))),
        TokenKind::LShift => Expression::LShift(Box::new(LShift::new(left, right, sourcepos))),
        TokenKind::URShift => Expression::URShift(Box::new(URShift::new(left, right, sourcepos))),
        TokenKind::ARShift => Expression::ARShift(Box::new(ARShift::new(left, right, sourcepos))),
        TokenKind::And => {
            Expression::ShortcutAnd(Box::new(ShortcutAnd::new(left, right, sourcepos)))
        }
        TokenKind::Or => Expression::ShortcutOr(Box::new(ShortcutOr::new(left, right, sourcepos))),
        TokenKind::OpAnd => Expression::OpAnd(Box::new(OpAnd::new(left, right, sourcepos))),
        TokenKind::OpOr => Expression::OpOr(Box::new(OpOr::new(left, right, sourcepos))),
        TokenKind::OpXor => Expression::OpXor(Box::new(OpXor::new(left, right, sourcepos))),
        TokenKind::EqualEqual => Expression::Eq(Box::new(Eq::new(left, right, sourcepos))),
        TokenKind::Ge => Expression::Ge(Box::new(Ge::new(left, right, sourcepos))),
        TokenKind::Gt => Expression::Gt(Box::new(Gt::new(left, right, sourcepos))),
        TokenKind::Le => Expression::Le(Box::new(Le::new(left, right, sourcepos))),
        TokenKind::Lt => Expression::Lt(Box::new(Lt::new(left, right, sourcepos))),
        TokenKind::Ne => Expression::Ne(Box::new(Ne::new(left, right, sourcepos))),
        _ => unreachable!("non-infix token"),
    }
}

fn inferred_name_type(name: &str) -> RuleType {
    if name.starts_with('C') {
        RuleType::Int
    } else {
        RuleType::IntBound
    }
}

pub struct TypingVisitor {
    bindings: HashMap<String, RuleType>,
}

impl TypingVisitor {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    fn must_be_same_typ(
        &self,
        sourcepos: Option<SourcePos>,
        typ: RuleType,
        targettyp: RuleType,
    ) -> Result<(), TypeCheckError> {
        if targettyp != typ {
            return Err(TypeCheckError::new(
                format!("{typ} must have type {targettyp}, got {typ}"),
                sourcepos,
            ));
        }
        Ok(())
    }

    fn error<T>(
        &self,
        msg: impl Into<String>,
        sourcepos: Option<SourcePos>,
    ) -> Result<T, TypeCheckError> {
        Err(TypeCheckError::new(msg, sourcepos))
    }

    fn visit_rule(&mut self, rule: &mut Rule) -> Result<(), TypeCheckError> {
        self.bindings.clear();
        self.visit_pattern(&mut rule.pattern, true)?;
        for el in &mut rule.elements {
            self.visit_element(el)?;
        }
        self.visit_pattern(&mut rule.target, false)?;
        Ok(())
    }

    fn visit_pattern(
        &mut self,
        pattern: &mut Pattern,
        patterndefine: bool,
    ) -> Result<RuleType, TypeCheckError> {
        match pattern {
            Pattern::PatternVar(ast) => {
                if patterndefine {
                    let typ = inferred_name_type(&ast.name);
                    if let Some(previous) = self.bindings.get(&ast.name).copied() {
                        self.must_be_same_typ(ast.sourcepos, previous, typ)?;
                    } else {
                        self.bindings.insert(ast.name.clone(), typ);
                        ast.typ = Some(typ);
                    }
                    Ok(typ)
                } else {
                    let Some(typ) = self.bindings.get(&ast.name).copied() else {
                        return self.error(
                            format!("variable {} is not defined", py_repr_string(&ast.name)),
                            ast.sourcepos,
                        );
                    };
                    ast.typ = Some(typ);
                    Ok(typ)
                }
            }
            Pattern::PatternConst(ast) => Ok(ast.typ),
            Pattern::PatternOp(ast) => {
                for arg in &mut ast.args {
                    self.visit_pattern(arg, patterndefine)?;
                }
                Ok(RuleType::IntBound)
            }
        }
    }

    fn visit_element(&mut self, element: &mut Element) -> Result<(), TypeCheckError> {
        match element {
            Element::Compute(ast) => {
                if self.bindings.contains_key(&ast.name) {
                    return self.error(
                        format!("{} is already defined", py_repr_string(&ast.name)),
                        ast.sourcepos,
                    );
                }
                let typ = self.visit_expression(&mut ast.expr)?;
                self.bindings.insert(ast.name.clone(), typ);
                Ok(())
            }
            Element::Check(ast) => {
                let typ = self.visit_expression(&mut ast.expr)?;
                if typ != RuleType::Bool {
                    return self.error(
                        format!(
                            "expected check expression to return a bool, got {}",
                            typ.py_name()
                        ),
                        ast.expr.sourcepos(),
                    );
                }
                Ok(())
            }
        }
    }

    fn visit_expression(&mut self, ast: &mut Expression) -> Result<RuleType, TypeCheckError> {
        match ast {
            Expression::Name(ast) => {
                if ast.name == "LONG_BIT" {
                    ast.typ = Some(RuleType::Int);
                    return Ok(RuleType::Int);
                }
                let Some(typ) = self.bindings.get(&ast.name).copied() else {
                    return self.error(
                        format!("variable {} is not defined", py_repr_string(&ast.name)),
                        ast.sourcepos,
                    );
                };
                ast.typ = Some(typ);
                Ok(typ)
            }
            Expression::Number(ast) => Ok(ast.typ),
            Expression::Add(ast) => self.visit_int_binop(&mut ast.left, &mut ast.right),
            Expression::Sub(ast) => self.visit_int_binop(&mut ast.left, &mut ast.right),
            Expression::Mul(ast) => self.visit_int_binop(&mut ast.left, &mut ast.right),
            Expression::Div(ast) => self.visit_int_binop(&mut ast.left, &mut ast.right),
            Expression::LShift(ast) => self.visit_int_binop(&mut ast.left, &mut ast.right),
            Expression::URShift(ast) => self.visit_int_binop(&mut ast.left, &mut ast.right),
            Expression::ARShift(ast) => self.visit_int_binop(&mut ast.left, &mut ast.right),
            Expression::OpAnd(ast) => self.visit_int_binop(&mut ast.left, &mut ast.right),
            Expression::OpOr(ast) => self.visit_int_binop(&mut ast.left, &mut ast.right),
            Expression::OpXor(ast) => self.visit_int_binop(&mut ast.left, &mut ast.right),
            Expression::Eq(ast) => self.visit_bool_binop(&mut ast.left, &mut ast.right),
            Expression::Ge(ast) => self.visit_bool_binop(&mut ast.left, &mut ast.right),
            Expression::Gt(ast) => self.visit_bool_binop(&mut ast.left, &mut ast.right),
            Expression::Le(ast) => self.visit_bool_binop(&mut ast.left, &mut ast.right),
            Expression::Lt(ast) => self.visit_bool_binop(&mut ast.left, &mut ast.right),
            Expression::Ne(ast) => self.visit_bool_binop(&mut ast.left, &mut ast.right),
            Expression::ShortcutAnd(ast) => {
                self.visit_shortcut_binop(&mut ast.left, &mut ast.right)
            }
            Expression::ShortcutOr(ast) => self.visit_shortcut_binop(&mut ast.left, &mut ast.right),
            Expression::Invert(ast) => {
                let left_typ = self.visit_expression(&mut ast.left)?;
                self.must_be_same_typ(ast.left.sourcepos(), RuleType::Int, left_typ)?;
                Ok(RuleType::Int)
            }
            Expression::Attribute(ast) => {
                if !self.bindings.contains_key(&ast.varname) && ast.varname != "LONG_BIT" {
                    return self.error(
                        format!("variable {} is not defined", py_repr_string(&ast.varname)),
                        ast.sourcepos,
                    );
                }
                Ok(ast.typ)
            }
            Expression::MethodCall(ast) => {
                let Some((receiver, argtyps, restyp)) = intbound_methodtype(&ast.methname) else {
                    return self.error(format!("unknown method {:?}", ast.methname), ast.sourcepos);
                };
                let receiver_typ = self.visit_expression(&mut ast.value)?;
                self.must_be_same_typ(ast.value.sourcepos(), receiver, receiver_typ)?;
                for (arg, typ) in ast.args.iter_mut().zip(argtyps.iter().copied()) {
                    let hastyp = self.visit_expression(arg)?;
                    self.must_be_same_typ(arg.sourcepos(), hastyp, typ)?;
                }
                ast.typ = Some(restyp);
                Ok(restyp)
            }
            Expression::FuncCall(ast) => {
                let Some((argtyps, restyp)) = functype(&ast.funcname) else {
                    return self.error(
                        format!("unknown function {:?}", ast.funcname),
                        ast.sourcepos,
                    );
                };
                for (arg, typ) in ast.args.iter_mut().zip(argtyps.iter().copied()) {
                    let hastyp = self.visit_expression(arg)?;
                    self.must_be_same_typ(arg.sourcepos(), hastyp, typ)?;
                }
                ast.typ = Some(restyp);
                Ok(restyp)
            }
        }
    }

    fn visit_int_binop(
        &mut self,
        left: &mut Expression,
        right: &mut Expression,
    ) -> Result<RuleType, TypeCheckError> {
        let left_typ = self.visit_expression(left)?;
        self.must_be_same_typ(left.sourcepos(), RuleType::Int, left_typ)?;
        let right_typ = self.visit_expression(right)?;
        self.must_be_same_typ(right.sourcepos(), RuleType::Int, right_typ)?;
        Ok(RuleType::Int)
    }

    fn visit_bool_binop(
        &mut self,
        left: &mut Expression,
        right: &mut Expression,
    ) -> Result<RuleType, TypeCheckError> {
        let left_typ = self.visit_expression(left)?;
        self.must_be_same_typ(left.sourcepos(), RuleType::Int, left_typ)?;
        let right_typ = self.visit_expression(right)?;
        self.must_be_same_typ(right.sourcepos(), RuleType::Int, right_typ)?;
        Ok(RuleType::Bool)
    }

    fn visit_shortcut_binop(
        &mut self,
        left: &mut Expression,
        right: &mut Expression,
    ) -> Result<RuleType, TypeCheckError> {
        let left_typ = self.visit_expression(left)?;
        self.must_be_same_typ(left.sourcepos(), RuleType::Bool, left_typ)?;
        let right_typ = self.visit_expression(right)?;
        self.must_be_same_typ(right.sourcepos(), RuleType::Bool, right_typ)?;
        Ok(RuleType::Bool)
    }
}

impl Default for TypingVisitor {
    fn default() -> Self {
        Self::new()
    }
}

fn typecheck_file(file: &mut File) -> Result<(), TypeCheckError> {
    let mut visitor = TypingVisitor::new();
    for rule in &mut file.rules {
        visitor.visit_rule(rule)?;
    }
    Ok(())
}

fn intbound_methodtype(name: &str) -> Option<(RuleType, &'static [RuleType], RuleType)> {
    use RuleType::{Bool, Int, IntBound};
    match name {
        "known_eq_const" => Some((IntBound, &[Int], Bool)),
        "known_le_const" => Some((IntBound, &[Int], Bool)),
        "known_lt_const" => Some((IntBound, &[Int], Bool)),
        "known_ge_const" => Some((IntBound, &[Int], Bool)),
        "known_gt_const" => Some((IntBound, &[Int], Bool)),
        "known_ne" => Some((IntBound, &[IntBound], Bool)),
        "known_nonnegative" => Some((IntBound, &[], Bool)),
        "is_constant" => Some((IntBound, &[], Bool)),
        "is_bool" => Some((IntBound, &[], Bool)),
        "get_constant_int" => Some((IntBound, &[], Int)),
        "lshift_bound_cannot_overflow" => Some((IntBound, &[IntBound], Bool)),
        "and_bound" => Some((IntBound, &[IntBound], IntBound)),
        "or_bound" => Some((IntBound, &[IntBound], IntBound)),
        "sub_bound" => Some((IntBound, &[IntBound], IntBound)),
        "sub_bound_cannot_overflow" => Some((IntBound, &[IntBound], Bool)),
        "lshift_bound" => Some((IntBound, &[IntBound], IntBound)),
        "rshift_bound" => Some((IntBound, &[IntBound], IntBound)),
        "urshift_bound" => Some((IntBound, &[IntBound], IntBound)),
        "known_le" => Some((IntBound, &[IntBound], Bool)),
        "known_gt" => Some((IntBound, &[IntBound], Bool)),
        _ => None,
    }
}

fn functype(name: &str) -> Option<(&'static [RuleType], RuleType)> {
    use RuleType::Int;
    match name {
        "highest_bit" => Some((&[Int], Int)),
        "min" => Some((&[Int, Int], Int)),
        _ => None,
    }
}

fn py_repr_string(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_int_add_zero() {
        let s = "add_zero: int_add(x, 0)\n    => x\n";
        let ast = parse(s).unwrap();
        assert_eq!(
            ast.rules[0].to_string(),
            "add_zero: int_add(x, 0)\n    => x"
        );
    }

    #[test]
    fn test_parse_source_positions() {
        let s = "add_zero: int_add(x, 0)\n    => x\n";
        let ast = parse(s).unwrap();
        let rule = &ast.rules[0];
        assert_eq!(rule.sourcepos.unwrap().lineno, 1);
        assert_eq!(rule.target.sourcepos().unwrap().lineno, 2);
        assert_eq!(rule.target.sourcepos().unwrap().colno, 8);
    }

    #[test]
    fn test_parse_function_many_args() {
        let s = "n: op(C)\n    C1 = min(min(C, C), min(C, C))\n    => C\n";
        parse(s).unwrap();
    }

    #[test]
    fn test_sorry() {
        let s = "eq_different_knownbits: int_eq(x, y)\n    SORRY_Z3\n    => 0\n";
        let ast = parse(s).unwrap();
        assert!(ast.rules[0].cantproof);
    }

    #[test]
    fn test_parse_lshift_rshift() {
        let s = "int_lshift_int_rshift_consts: int_lshift(int_rshift(x, C1), C1)\n    C = (-1 >> C1) << C1\n    => int_and(x, C)\n";
        parse(s).unwrap();
    }

    #[test]
    fn test_parse_all() {
        parse(crate::ruleopt::REAL_RULES).unwrap();
    }

    #[test]
    fn test_undefined_name() {
        let s = "n: op(C)\n    => x\n";
        let err = parse(s).unwrap_err();
        let RuleParseError::TypeCheck(err) = err else {
            panic!("expected type error");
        };
        assert_eq!(err.to_string(), "variable 'x' is not defined");
        assert_eq!(
            err.format(Some(s)),
            "Type error: variable 'x' is not defined\nin line 2\n        => x\n           ^"
        );
    }

    #[test]
    fn test_doubly_defined_name() {
        let s = "n: op(C)\n    C = C + 1\n    => C\n";
        let err = parse(s).unwrap_err();
        let RuleParseError::TypeCheck(err) = err else {
            panic!("expected type error");
        };
        assert_eq!(err.to_string(), "'C' is already defined");
    }

    #[test]
    fn test_check_not_bool() {
        let s = "n: op(C)\n    check C\n    => C\n";
        let err = parse(s).unwrap_err();
        let RuleParseError::TypeCheck(err) = err else {
            panic!("expected type error");
        };
        assert_eq!(
            err.to_string(),
            "expected check expression to return a bool, got int"
        );
        assert_eq!(
            err.format(Some(s)),
            "Type error: expected check expression to return a bool, got int\nin line 2\n        check C\n              ^"
        );
    }
}
