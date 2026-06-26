//! RPython parity module for `rpython/jit/metainterp/graphpage.py`.
//!
//! The upstream module is a debugging/viewer helper: it turns optimized
//! `ResOperation` procedures into Graphviz DOT and opens `dotviewer`.
//! Pyre does not embed `dotviewer`, so this port preserves the module and
//! symbol surface while returning DOT source for callers/tests to inspect.

use majit_ir::box_ref::BoxRef;
use majit_ir::{DescrRef, Op, OpCode, Type, Value, descr_identity};

const BOX_COLOR: (u8, u8, u8) = (128, 0, 96);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkInfo {
    pub label: String,
    pub color: (u8, u8, u8),
}

#[derive(Default)]
pub struct ResOpMemo {
    names: Vec<(usize, String)>,
}

impl ResOpMemo {
    pub fn get(&self, box_: &BoxRef) -> Option<&str> {
        let key = box_.as_ptr() as usize;
        self.names
            .iter()
            .find_map(|(k, v)| (*k == key).then_some(v.as_str()))
    }

    pub fn set(&mut self, box_: &BoxRef, value: String) {
        let key = box_.as_ptr() as usize;
        if let Some((_, old)) = self.names.iter_mut().find(|(k, _)| *k == key) {
            *old = value;
        } else {
            self.names.push((key, value));
        }
    }
}

pub trait ResOpProcedure {
    fn get_operations(&self) -> &[Op];

    fn get_display_text(&self, _memo: &mut ResOpMemo) -> Option<String> {
        None
    }
}

impl ResOpProcedure for Vec<Op> {
    fn get_operations(&self) -> &[Op] {
        self.as_slice()
    }
}

impl ResOpProcedure for [Op] {
    fn get_operations(&self) -> &[Op] {
        self
    }
}

pub struct SubGraph {
    failargs: Vec<BoxRef>,
    subinputargs: Vec<BoxRef>,
    suboperations: Vec<Op>,
}

impl SubGraph {
    pub fn new(failargs: Vec<BoxRef>, subinputargs: Vec<BoxRef>, suboperations: Vec<Op>) -> Self {
        Self {
            failargs,
            subinputargs,
            suboperations,
        }
    }
}

impl ResOpProcedure for SubGraph {
    fn get_operations(&self) -> &[Op] {
        &self.suboperations
    }

    fn get_display_text(&self, memo: &mut ResOpMemo) -> Option<String> {
        for (failarg, inputarg) in self.failargs.iter().zip(self.subinputargs.iter()) {
            if let Some(name) = memo.get(failarg).map(str::to_string) {
                memo.set(inputarg, name);
            }
        }
        None
    }
}

pub fn display_procedures(
    procedures: &[&dyn ResOpProcedure],
    errmsg: Option<&str>,
    highlight_procedures: &[u8],
) -> String {
    let graphs = procedures
        .iter()
        .enumerate()
        .map(|(i, procedure)| (*procedure, *highlight_procedures.get(i).unwrap_or(&0)))
        .collect::<Vec<_>>();
    let mut graphpage = ResOpGraphPage::default();
    graphpage.compute(&graphs, errmsg);
    graphpage.source
}

pub fn is_interesting_guard(_op: &Op) -> bool {
    // PyPy checks for `op.getdescr()._debug_suboperations`. Pyre does not
    // currently attach that viewer-only bridge payload to fail descrs.
    false
}

pub fn getdescr(op: &Op) -> Option<DescrRef> {
    op.getdescr()
}

#[derive(Default)]
pub struct ResOpGraphPage {
    pub source: String,
    pub links: Vec<(String, LinkInfo)>,
}

impl ResOpGraphPage {
    pub fn compute(&mut self, graphs: &[(&dyn ResOpProcedure, u8)], errmsg: Option<&str>) {
        let mut resopgen = ResOpGen::new(None);
        for (graph, highlight) in graphs {
            resopgen.add_graph(*graph, *highlight);
        }
        if let Some(errmsg) = errmsg {
            resopgen.set_errmsg(errmsg.to_string());
        }
        self.source = resopgen.getsource();
        self.links = resopgen.getlinks();
    }
}

struct GraphEntry<'a> {
    procedure: &'a dyn ResOpProcedure,
    highlight: u8,
}

pub struct ResOpGen<'a> {
    graphs: Vec<GraphEntry<'a>>,
    block_starters: Vec<Vec<usize>>,
    all_operations: Vec<(usize, (usize, usize))>,
    errmsg: Option<String>,
    target_tokens: Vec<(usize, (usize, usize))>,
    metainterp_sd: Option<()>,
    memo: ResOpMemo,
    pendingedges: Vec<(String, String, EdgeAttrs)>,
    dotgen: Option<DotGen>,
}

impl<'a> ResOpGen<'a> {
    pub const CLUSTERING: bool = true;
    pub const BOX_COLOR: (u8, u8, u8) = BOX_COLOR;

    pub fn new(metainterp_sd: Option<()>) -> Self {
        Self {
            graphs: Vec::new(),
            block_starters: Vec::new(),
            all_operations: Vec::new(),
            errmsg: None,
            target_tokens: Vec::new(),
            metainterp_sd,
            memo: ResOpMemo::default(),
            pendingedges: Vec::new(),
            dotgen: None,
        }
    }

    pub fn op_name(&self, graphindex: usize, opindex: usize) -> String {
        format!("g{graphindex}op{opindex}")
    }

    pub fn mark_starter(&mut self, graphindex: usize, opindex: usize) {
        let starters = &mut self.block_starters[graphindex];
        if !starters.contains(&opindex) {
            starters.push(opindex);
        }
    }

    pub fn add_graph(&mut self, graph: &'a dyn ResOpProcedure, highlight: u8) {
        let graphindex = self.graphs.len();
        self.graphs.push(GraphEntry {
            procedure: graph,
            highlight,
        });
        for (i, op) in graph.get_operations().iter().enumerate() {
            self.all_operations
                .push((op as *const Op as usize, (graphindex, i)));
        }
    }

    pub fn find_starters(&mut self) {
        self.block_starters = (0..self.graphs.len()).map(|_| vec![0]).collect();
        for graphindex in 0..self.graphs.len() {
            let mut mergepointblock = None;
            let operations = self.graphs[graphindex].procedure.get_operations();
            for (i, op) in operations.iter().enumerate() {
                if is_interesting_guard(op) {
                    self.mark_starter(graphindex, i + 1);
                }
                if op.opcode == OpCode::DebugMergePoint {
                    if mergepointblock.is_none() {
                        mergepointblock = Some(i);
                    }
                } else if op.opcode == OpCode::Label {
                    self.mark_starter(graphindex, i);
                    if let Some(descr) = getdescr(op) {
                        self.target_tokens
                            .push((descr_identity(&descr), (graphindex, i)));
                    }
                    mergepointblock = Some(i);
                } else if let Some(block) = mergepointblock.take() {
                    self.mark_starter(graphindex, block);
                }
            }
        }
        for starters in &mut self.block_starters {
            starters.sort_unstable();
        }
    }

    pub fn set_errmsg(&mut self, errmsg: String) {
        self.errmsg = Some(errmsg);
    }

    pub fn getsource(&mut self) -> String {
        self.find_starters();
        self.pendingedges.clear();
        self.dotgen = Some(DotGen::new("resop"));
        self.emit(r#"clusterrank="local""#);
        self.generrmsg();
        for graphindex in 0..self.graphs.len() {
            self.gengraph(graphindex);
        }
        for (from, to, attrs) in std::mem::take(&mut self.pendingedges) {
            self.dotgen
                .as_mut()
                .expect("dotgen initialized")
                .emit_edge(&from, &to, attrs);
        }
        self.dotgen.take().expect("dotgen initialized").generate()
    }

    fn emit(&mut self, line: impl Into<String>) {
        self.dotgen.as_mut().expect("dotgen initialized").emit(line);
    }

    fn generrmsg(&mut self) {
        let Some(errmsg) = self.errmsg.clone() else {
            return;
        };
        self.dotgen
            .as_mut()
            .expect("dotgen initialized")
            .emit_node("errmsg", NodeAttrs::new("box", &errmsg).fillcolor("red"));
        if !self.graphs.is_empty() && !self.block_starters[0].is_empty() {
            let opindex = *self.block_starters[0].iter().max().unwrap();
            let blockname = self.op_name(0, opindex);
            self.pendingedges
                .push((blockname, "errmsg".to_string(), EdgeAttrs::default()));
        }
    }

    pub fn getgraphname(&self, graphindex: usize) -> String {
        format!("graph{graphindex}")
    }

    fn gengraph(&mut self, graphindex: usize) {
        let graphname = self.getgraphname(graphindex);
        if Self::CLUSTERING {
            self.emit(format!("subgraph cluster{graphindex} {{"));
        }
        let label = self.graphs[graphindex]
            .procedure
            .get_display_text(&mut self.memo);
        if let Some(label) = label {
            let fillcolor = match self.graphs[graphindex].highlight {
                1 => "#f084c2",
                2 => "#808080",
                _ => "#84f0c2",
            };
            self.dotgen.as_mut().expect("dotgen initialized").emit_node(
                &graphname,
                NodeAttrs::new("octagon", &label).fillcolor(fillcolor),
            );
            self.pendingedges
                .push((graphname, self.op_name(graphindex, 0), EdgeAttrs::default()));
        }
        let starters = self.block_starters[graphindex].clone();
        for opindex in starters {
            self.genblock(graphindex, opindex);
        }
        if Self::CLUSTERING {
            self.emit("}");
        }
    }

    fn genedge(&mut self, from: (usize, usize), to: (usize, usize), attrs: EdgeAttrs) {
        self.pendingedges.push((
            self.op_name(from.0, from.1),
            self.op_name(to.0, to.1),
            attrs,
        ));
    }

    fn genblock(&mut self, graphindex: usize, opstartindex: usize) {
        let operations = self.graphs[graphindex].procedure.get_operations();
        if opstartindex >= operations.len() {
            return;
        }
        let blockname = self.op_name(graphindex, opstartindex);
        let block_starters = self.block_starters[graphindex].clone();
        let mut lines = Vec::new();
        let mut opindex = opstartindex;
        loop {
            let op = &operations[opindex];
            lines.push(self.repr_op(op));
            opindex += 1;
            if opindex >= operations.len() {
                break;
            }
            if block_starters.contains(&opindex) {
                self.genedge(
                    (graphindex, opstartindex),
                    (graphindex, opindex),
                    EdgeAttrs::default(),
                );
                break;
            }
        }
        let op = &operations[opindex.saturating_sub(1)];
        if op.opcode == OpCode::Jump {
            if let Some(descr) = getdescr(op) {
                let key = descr_identity(&descr);
                if let Some((_, target)) = self.target_tokens.iter().find(|(k, _)| *k == key) {
                    self.genedge(
                        (graphindex, opstartindex),
                        *target,
                        EdgeAttrs::default().weight("0"),
                    );
                }
            }
        }
        lines.push(String::new());
        let label = lines.join(r"\l");
        self.dotgen
            .as_mut()
            .expect("dotgen initialized")
            .emit_node(&blockname, NodeAttrs::new("box", &label));
    }

    fn repr_op(&self, op: &Op) -> String {
        if op.opcode == OpCode::DebugMergePoint && self.metainterp_sd.is_some() {
            // PyPy asks warmstate for a printable location here. Pyre does not
            // expose the equivalent metainterp_sd viewer hook yet, so keep the
            // operation's regular display until that hook exists.
        }
        format!("{op}")
    }

    pub fn getlinks(&mut self) -> Vec<(String, LinkInfo)> {
        let mut links = Vec::new();
        for graphindex in 0..self.graphs.len() {
            let operations = self.graphs[graphindex].procedure.get_operations();
            for op in operations {
                for arg in op.getarglist() {
                    self.add_link_for_box(&mut links, &arg);
                }
                if op.result_type() != Type::Void && !op.pos.get().is_none() {
                    let result = BoxRef::from_opref(op.pos.get());
                    self.add_link_for_box(&mut links, &result);
                }
            }
        }
        links
    }

    fn add_link_for_box(&mut self, links: &mut Vec<(String, LinkInfo)>, box_: &BoxRef) {
        let short = self.repr_short(box_);
        if short.len() > 1
            && matches!(short.as_bytes()[0], b'i' | b'r' | b'f')
            && short[1..].chars().all(|c| c.is_ascii_digit())
            && !links.iter().any(|(key, _)| key == &short)
        {
            links.push((
                short.clone(),
                LinkInfo {
                    label: self.repr_box(box_),
                    color: BOX_COLOR,
                },
            ));
        }
    }

    fn repr_short(&mut self, box_: &BoxRef) -> String {
        if let Some(name) = self.memo.get(box_) {
            return name.to_string();
        }
        let name = match box_.const_value() {
            Some(value) => format_value(value),
            None if box_.is_none() => "_".to_string(),
            None => {
                let prefix = match box_.type_() {
                    Type::Int => "i",
                    Type::Ref => "r",
                    Type::Float => "f",
                    Type::Void => "v",
                };
                match box_.position() {
                    Some(pos) => format!("{prefix}{pos}"),
                    None => "_".to_string(),
                }
            }
        };
        self.memo.set(box_, name.clone());
        name
    }

    fn repr_box(&mut self, box_: &BoxRef) -> String {
        match box_.const_value() {
            Some(value) => format_value(value),
            None => self.repr_short(box_),
        }
    }
}

#[derive(Clone, Default)]
struct EdgeAttrs {
    label: String,
    style: String,
    color: String,
    dir: String,
    weight: String,
}

impl EdgeAttrs {
    fn weight(mut self, weight: &str) -> Self {
        self.weight = weight.to_string();
        self
    }
}

#[derive(Clone)]
struct NodeAttrs {
    shape: String,
    label: String,
    color: String,
    fillcolor: String,
    style: String,
    width: String,
}

impl NodeAttrs {
    fn new(shape: &str, label: &str) -> Self {
        Self {
            shape: shape.to_string(),
            label: label.to_string(),
            color: "black".to_string(),
            fillcolor: "white".to_string(),
            style: "filled".to_string(),
            width: "0.75".to_string(),
        }
    }

    fn fillcolor(mut self, fillcolor: &str) -> Self {
        self.fillcolor = fillcolor.to_string();
        self
    }
}

struct DotGen {
    graphname: String,
    lines: Vec<String>,
}

impl DotGen {
    fn new(graphname: &str) -> Self {
        let graphname = safename(graphname);
        let mut dotgen = Self {
            graphname,
            lines: Vec::new(),
        };
        dotgen.emit(format!("digraph {} {{", dotgen.graphname));
        dotgen
    }

    fn emit(&mut self, line: impl Into<String>) {
        self.lines.push(line.into());
    }

    fn emit_edge(&mut self, name1: &str, name2: &str, attrs: EdgeAttrs) {
        let attrs = EdgeAttrs {
            label: attrs.label,
            style: if attrs.style.is_empty() {
                "dashed".to_string()
            } else {
                attrs.style
            },
            color: if attrs.color.is_empty() {
                "black".to_string()
            } else {
                attrs.color
            },
            dir: if attrs.dir.is_empty() {
                "forward".to_string()
            } else {
                attrs.dir
            },
            weight: if attrs.weight.is_empty() {
                "5".to_string()
            } else {
                attrs.weight
            },
        };
        self.emit(format!(
            r#"edge [label="{}", style="{}", color="{}", dir="{}", weight="{}"];"#,
            escape_attr(&attrs.label),
            escape_attr(&attrs.style),
            escape_attr(&attrs.color),
            escape_attr(&attrs.dir),
            escape_attr(&attrs.weight)
        ));
        self.emit(format!("{} -> {}", safename(name1), safename(name2)));
    }

    fn emit_node(&mut self, name: &str, attrs: NodeAttrs) {
        self.emit(format!(
            r#"{} [shape="{}", label="{}", color="{}", fillcolor="{}", style="{}", width="{}"];"#,
            safename(name),
            escape_attr(&attrs.shape),
            escape_attr(&attrs.label),
            escape_attr(&attrs.color),
            escape_attr(&attrs.fillcolor),
            escape_attr(&attrs.style),
            escape_attr(&attrs.width)
        ));
    }

    fn generate(mut self) -> String {
        self.emit("}");
        self.lines.join("\n")
    }
}

fn escape_attr(value: &str) -> String {
    value
        .replace('\\', r"\\")
        .replace('"', "\\\"")
        .replace('\n', r"\n")
}

fn format_value(value: Value) -> String {
    match value {
        Value::Int(v) => v.to_string(),
        Value::Float(v) => v.to_string(),
        Value::Ref(v) => format!("ptr({:#x})", v.0),
        Value::Void => "void".to_string(),
    }
}

fn safename(name: &str) -> String {
    let mut result = String::from("_");
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            result.push(ch);
        } else if ch == '_' {
            result.push_str("__");
        } else {
            result.push_str(&format!("_{:02X}", ch as u32));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::test_support::rooted_resop_operand;
    use majit_ir::{Op, OpCode, OpRef, Type, box_ref::BoxRef, operand::Operand, value::InputArg};

    #[test]
    fn display_procedures_returns_dot_source() {
        let i0 = InputArg::new_int_rc(0);
        let i1 = InputArg::new_int_rc(1);
        let op = Op::new(
            OpCode::IntAdd,
            &[
                Operand::from_bound_inputarg(&i0),
                Operand::from_bound_inputarg(&i1),
            ],
        );
        op.pos.set(OpRef::int_op(2));
        let jump = Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 2)]);
        let procedure = vec![op, jump];

        let source = display_procedures(&[&procedure], None, &[]);

        assert!(source.contains("digraph _resop {"));
        assert!(source.contains("IntAdd"));
        assert!(source.contains("Jump"));
    }

    #[test]
    fn graphpage_exposes_box_links() {
        let i0 = InputArg::new_int_rc(0);
        let i1 = InputArg::new_int_rc(1);
        let op = Op::new(
            OpCode::IntAdd,
            &[
                Operand::from_bound_inputarg(&i0),
                Operand::from_bound_inputarg(&i1),
            ],
        );
        op.pos.set(OpRef::int_op(1));
        let procedure = vec![op];
        let mut page = ResOpGraphPage::default();

        page.compute(&[(&procedure as &dyn ResOpProcedure, 0)], None);

        assert!(page.links.iter().any(|(name, _)| name == "i0"));
        assert!(page.links.iter().any(|(name, _)| name == "i1"));
    }

    #[test]
    fn subgraph_copies_parent_failarg_names() {
        let failarg = BoxRef::new_inputarg(Type::Int, 7);
        let subinputarg = BoxRef::new_inputarg(Type::Int, 0);
        let subgraph = SubGraph::new(vec![failarg.clone()], vec![subinputarg.clone()], vec![]);
        let mut memo = ResOpMemo::default();
        memo.set(&failarg, "i7".to_string());

        subgraph.get_display_text(&mut memo);

        assert_eq!(memo.get(&subinputarg), Some("i7"));
    }
}
