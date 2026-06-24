//! Proof support for the integer optimization rule DSL.
//!
//! RPython counterpart: `rpython/jit/metainterp/ruleopt/proof.py`.
//!
//! PyPy proves rules with Z3 before generating `autogenintrules.py`. This
//! module carries the same public proof surface and the pure helper logic; the
//! solver backend is intentionally explicit so callers cannot mistake
//! parse-only validation for a successful proof.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::ruleopt::parse::{self, File, Pattern, Rule};

pub const LONG_BIT: u32 = 64;
pub const MAXINT: i64 = i64::MAX;
pub const MININT: i64 = i64::MIN;

pub const TRUEBV: &str = "BitVecVal(1, LONG_BIT)";
pub const FALSEBV: &str = "BitVecVal(0, LONG_BIT)";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProofProblem {
    CouldNotProve(CouldNotProve),
    RuleCannotApply(RuleCannotApply),
    SolverUnavailable { rule: String },
}

impl ProofProblem {
    pub fn format(&self) -> String {
        match self {
            ProofProblem::CouldNotProve(problem) => problem.format(),
            ProofProblem::RuleCannotApply(problem) => problem.format(),
            ProofProblem::SolverUnavailable { rule } => {
                format!("Z3 proof backend is not available for rule '{rule}'")
            }
        }
    }
}

impl fmt::Display for ProofProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format())
    }
}

impl std::error::Error for ProofProblem {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CouldNotProve {
    pub rule: Rule,
    pub cond: String,
    pub model: Option<String>,
    pub lhs: String,
    pub rhs: String,
}

impl CouldNotProve {
    pub fn format(&self) -> String {
        let mut res = vec![format!(
            "Could not prove correctness of rule '{}'",
            self.rule.name
        )];
        if let Some(sourcepos) = self.rule.sourcepos {
            res.push(format!("in line {}", sourcepos.lineno));
        }
        if let Some(model) = &self.model {
            res.push("counterexample given by Z3:".to_string());
            res.push(model.clone());
        }
        res.push(format!(
            "operation {} with Z3 formula {}",
            self.rule.pattern, self.lhs
        ));
        res.push("BUT".to_string());
        res.push(format!(
            "target expression: {} with Z3 formula {}",
            self.rule.target, self.rhs
        ));
        res.join("\n")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuleCannotApply {
    pub rule: Rule,
    pub cond: String,
    pub names: Vec<String>,
}

impl RuleCannotApply {
    pub fn format(&self) -> String {
        let mut res = vec![format!("Rule '{}' cannot ever apply", self.rule.name)];
        if let Some(sourcepos) = self.rule.sourcepos {
            res.push(format!("in line {}", sourcepos.lineno));
        }
        res.push(format!(
            "Z3 did not manage to find values for variables {} such that the following condition becomes True:",
            self.names.join(", ")
        ));
        res.push(self.cond.clone());
        res.join("\n")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Z3Formula {
    pub expr: String,
    pub valid: String,
}

impl Z3Formula {
    fn new(expr: impl Into<String>, valid: impl Into<String>) -> Self {
        Self {
            expr: expr.into(),
            valid: valid.into(),
        }
    }
}

pub fn z3_cond(z3expr: impl AsRef<str>) -> String {
    format!("If({}, {TRUEBV}, {FALSEBV})", z3expr.as_ref())
}

pub fn z3_bool_expression(opname: &str, arg0: &str, arg1: Option<&str>) -> Z3Formula {
    let arg1 = arg1.unwrap_or("");
    let expr = match opname {
        "int_eq" => format!("{arg0} == {arg1}"),
        "int_ne" => format!("{arg0} != {arg1}"),
        "int_lt" => format!("{arg0} < {arg1}"),
        "int_le" => format!("{arg0} <= {arg1}"),
        "int_gt" => format!("{arg0} > {arg1}"),
        "int_ge" => format!("{arg0} >= {arg1}"),
        "uint_lt" => format!("ULT({arg0}, {arg1})"),
        "uint_le" => format!("ULE({arg0}, {arg1})"),
        "uint_gt" => format!("UGT({arg0}, {arg1})"),
        "uint_ge" => format!("UGE({arg0}, {arg1})"),
        "int_is_true" => format!("{arg0} != {FALSEBV}"),
        "int_is_zero" => format!("{arg0} == {FALSEBV}"),
        _ => panic!("unknown bool ruleopt operation {opname}"),
    };
    Z3Formula::new(expr, "True")
}

pub fn z3_expression(opname: &str, arg0: &str, arg1: Option<&str>) -> Z3Formula {
    let arg1 = arg1.unwrap_or("");
    match opname {
        "int_add" => Z3Formula::new(format!("{arg0} + {arg1}"), "True"),
        "int_sub" => Z3Formula::new(format!("{arg0} - {arg1}"), "True"),
        "int_mul" => Z3Formula::new(format!("{arg0} * {arg1}"), "True"),
        "int_and" => Z3Formula::new(format!("{arg0} & {arg1}"), "True"),
        "int_or" => Z3Formula::new(format!("{arg0} | {arg1}"), "True"),
        "int_xor" => Z3Formula::new(format!("{arg0} ^ {arg1}"), "True"),
        "int_lshift" => Z3Formula::new(
            format!("{arg0} << {arg1}"),
            format!("And({arg1} >= 0, {arg1} < LONG_BIT)"),
        ),
        "int_rshift" => Z3Formula::new(
            format!("{arg0} >> {arg1}"),
            format!("And({arg1} >= 0, {arg1} < LONG_BIT)"),
        ),
        "uint_rshift" => Z3Formula::new(
            format!("LShR({arg0}, {arg1})"),
            format!("And({arg1} >= 0, {arg1} < LONG_BIT)"),
        ),
        "uint_mul_high" => Z3Formula::new(
            format!(
                "Extract(LONG_BIT * 2 - 1, LONG_BIT, ZeroExt(LONG_BIT, {arg0}) * ZeroExt(LONG_BIT, {arg1}))"
            ),
            "True",
        ),
        "int_neg" => Z3Formula::new(format!("-{arg0}"), "True"),
        "int_invert" => Z3Formula::new(format!("~{arg0}"), "True"),
        "int_force_ge_zero" => Z3Formula::new(format!("If({arg0} < 0, 0, {arg0})"), "True"),
        _ => {
            let bool_expr = z3_bool_expression(opname, arg0, Some(arg1));
            Z3Formula::new(z3_cond(bool_expr.expr), bool_expr.valid)
        }
    }
}

pub fn z3_and<I, S>(args: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let args = args
        .into_iter()
        .filter_map(|arg| {
            let arg = arg.as_ref().to_string();
            (arg != "True").then_some(arg)
        })
        .collect::<Vec<_>>();
    match args.len() {
        0 => "True".to_string(),
        1 => args[0].clone(),
        _ => format!("And({})", args.join(", ")),
    }
}

pub fn z3_implies(a: &str, b: &str) -> String {
    if a == "True" {
        b.to_string()
    } else {
        format!("Implies({a}, {b})")
    }
}

pub fn popcount64(w: u64) -> u32 {
    w.count_ones()
}

pub fn highest_bit(x: u64) -> i32 {
    if x == 0 {
        return -1;
    }
    63 - x.leading_zeros() as i32
}

pub fn z3_highest_bit(x: &str) -> String {
    format!("z3_highest_bit({x})")
}

pub fn z3_min(a: &str, b: &str) -> String {
    format!("If({a} <= {b}, {a}, {b})")
}

#[derive(Clone, Debug, Default)]
pub struct Prover {
    pub name_to_z3: HashMap<String, String>,
    pub name_to_intbound: HashMap<String, String>,
    pub glue_conditions_added: HashSet<String>,
    pub glue_conditions: Vec<String>,
}

impl Prover {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn prove(&mut self, _cond: &str) -> bool {
        false
    }

    pub fn check_rule(&mut self, rule: &Rule) -> Result<(), ProofProblem> {
        if rule.cantproof {
            return Ok(());
        }
        Err(ProofProblem::SolverUnavailable {
            rule: rule.name.clone(),
        })
    }

    fn _convert_var(&mut self, name: &str) -> String {
        if let Some(existing) = self.name_to_z3.get(name) {
            return existing.clone();
        }
        let res = format!("BitVec({name:?}, LONG_BIT)");
        self.name_to_z3.insert(name.to_string(), res.clone());
        self.name_to_intbound
            .insert(name.to_string(), format!("IntBound({name})"));
        res
    }

    fn _convert_intbound(&mut self, name: &str) -> String {
        if !self.glue_conditions_added.contains(name) {
            self.glue_conditions
                .push(format!("{}.z3_formula()", self.name_to_intbound[name]));
            self.glue_conditions_added.insert(name.to_string());
        }
        self.name_to_intbound[name].clone()
    }

    fn _convert_attr(&mut self, varname: &str, attrname: &str) -> String {
        let bound = self._convert_intbound(varname);
        match attrname {
            "ones" => format!("{bound}.tvalue"),
            "zeros" => format!("~({bound}.tvalue | {bound}.tmask)"),
            _ => format!("{bound}.{attrname}"),
        }
    }
}

pub fn prove_source(source: &str, _force: bool) -> Result<File, ProofProblem> {
    let ast = parse::parse(source).map_err(|err| {
        ProofProblem::CouldNotProve(CouldNotProve {
            rule: Rule {
                name: "<parse>".to_string(),
                pattern: Pattern::PatternConst(parse::PatternConst {
                    const_value: "0".to_string(),
                    typ: parse::RuleType::Int,
                    sourcepos: None,
                }),
                cantproof: false,
                elements: Vec::new(),
                target: Pattern::PatternConst(parse::PatternConst {
                    const_value: "0".to_string(),
                    typ: parse::RuleType::Int,
                    sourcepos: None,
                }),
                sourcepos: None,
                endsourcepos: None,
            },
            cond: err.to_string(),
            model: None,
            lhs: "parse".to_string(),
            rhs: "parse".to_string(),
        })
    })?;
    for rule in &ast.rules {
        let mut prover = Prover::new();
        prover.check_rule(rule)?;
    }
    Ok(ast)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highest_bit() {
        for i in 0..64 {
            assert_eq!(highest_bit(1u64 << i), i);
        }
        assert_eq!(highest_bit(0), -1);
    }

    #[test]
    fn test_z3_expression() {
        let expr = z3_expression("int_and", "x", Some("y"));
        assert_eq!(expr.expr, "x & y");
        assert_eq!(expr.valid, "True");
        let expr = z3_expression("uint_rshift", "x", Some("C"));
        assert_eq!(expr.expr, "LShR(x, C)");
        assert_eq!(expr.valid, "And(C >= 0, C < LONG_BIT)");
    }

    #[test]
    fn test_z3_and() {
        assert_eq!(z3_and(["True"]), "True");
        assert_eq!(z3_and(["True", "x > 0"]), "x > 0");
        assert_eq!(z3_and(["x > 0", "y > 0"]), "And(x > 0, y > 0)");
    }

    #[test]
    fn test_sorry_rule_is_accepted() {
        let source = "eq_different_knownbits: int_eq(x, y)\n    SORRY_Z3\n    => 0\n";
        let ast = prove_source(source, false).unwrap();
        assert!(ast.rules[0].cantproof);
    }

    #[test]
    fn test_non_sorry_rule_reports_solver_gap() {
        let source = "bug: int_and(x, y)\n    => 1\n";
        let err = prove_source(source, false).unwrap_err();
        assert_eq!(
            err.format(),
            "Z3 proof backend is not available for rule 'bug'"
        );
    }
}
