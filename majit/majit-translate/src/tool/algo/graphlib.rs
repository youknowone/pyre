//! Port of `rpython/tool/algo/graphlib.py`.
//!
//! Utilities to manipulate graphs (vertices and edges, not control flow
//! graphs). Convention (`graphlib.py:4-7`):
//!
//!   * `vertices` is a set of vertices (or a dict with vertices as keys);
//!   * `edges` is a dict mapping a vertex to the list of edges with that
//!     vertex as their source.
//!
//! `Edge` objects are shared: upstream files the *same* `Edge` instance
//! into the `edges` dict, into the lists returned by `all_cycles`, and so
//! on. `Rc<Edge>` is that shared Python object, so edge identity is
//! preserved across every container here.
//!
//! Two upstream functions are intentionally omitted:
//!   * `break_cycles` — upstream disables it with `py.test.skip(...)`
//!     ("not used any more", `graphlib.py:205-209`); the live edge-cutting
//!     entry point is [`break_cycles_v`].
//!   * `show_graph` — a graphviz/GUI debug helper (`graphlib.py`) with
//!     no translation-time consumer.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use std::rc::Rc;

/// `class Edge` at `graphlib.py`.
///
/// Upstream stores only `source`/`target`, but callers tack extra
/// attributes onto the instance — e.g. `innerloop` sets `edge.link`
/// (`innerloop.py:53`). Rust cannot attach attributes dynamically, so
/// that idiom is modelled with a generic payload `P` (defaulting to
/// `()`). `source` and `target` — the only fields graphlib itself
/// reads — keep their upstream spelling.
#[derive(Clone)]
pub struct Edge<V, P = ()> {
    pub source: V,
    pub target: V,
    pub payload: P,
}

/// `Edge.__repr__` at `graphlib.py`: `'%r -> %r' % (source, target)`.
/// `%r` is repr, so this lands on `Debug`; the payload is not part of the
/// upstream repr.
impl<V: fmt::Debug, P> fmt::Debug for Edge<V, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} -> {:?}", self.source, self.target)
    }
}

impl<V> Edge<V, ()> {
    /// `Edge(source, target)` with no extra payload (`graphlib.py`).
    pub fn new(source: V, target: V) -> Self {
        Edge {
            source,
            target,
            payload: (),
        }
    }
}

impl<V, P> Edge<V, P> {
    /// `Edge(source, target)` carrying the payload a caller would tack
    /// on as an extra attribute (e.g. `edge.link = link`).
    pub fn with_payload(source: V, target: V, payload: P) -> Self {
        Edge {
            source,
            target,
            payload,
        }
    }
}

/// Membership/iteration view over a vertex collection.
///
/// Upstream states `vertices` "is a set of vertices (or a dict with
/// vertices as keys)" (`graphlib.py:4-5`). The algorithms test `v in
/// vertices` and iterate `for v in vertices`; this trait is that
/// `__contains__` + iteration, implemented for a set (`HashSet<V>`), a
/// dict-shaped `HashMap<V, _>`, and a plain list (`Vec<V>` / `[V]`) — the
/// tests pass a list, e.g. `depth_first_search('A', list('ABCEFG'), edges)`
/// (`test_graphlib.py:19`), so callers can pass any of the three.
pub trait VertexSet<V> {
    /// `v in self`.
    fn contains_vertex(&self, v: &V) -> bool;
    /// A snapshot of the vertices, for `for v in self` / `self.copy()`.
    fn vertex_snapshot(&self) -> Vec<V>;
}

impl<V: Eq + Hash + Clone> VertexSet<V> for HashSet<V> {
    fn contains_vertex(&self, v: &V) -> bool {
        self.contains(v)
    }
    fn vertex_snapshot(&self) -> Vec<V> {
        self.iter().cloned().collect()
    }
}

impl<V: Eq + Hash + Clone, T> VertexSet<V> for HashMap<V, T> {
    fn contains_vertex(&self, v: &V) -> bool {
        self.contains_key(v)
    }
    fn vertex_snapshot(&self) -> Vec<V> {
        self.keys().cloned().collect()
    }
}

impl<V: Eq + Hash + Clone> VertexSet<V> for [V] {
    fn contains_vertex(&self, v: &V) -> bool {
        self.contains(v)
    }
    fn vertex_snapshot(&self) -> Vec<V> {
        self.to_vec()
    }
}

impl<V: Eq + Hash + Clone> VertexSet<V> for Vec<V> {
    fn contains_vertex(&self, v: &V) -> bool {
        self.contains(v)
    }
    fn vertex_snapshot(&self) -> Vec<V> {
        self.clone()
    }
}

/// The official `edges` dict shape: a vertex mapped to the shared edges
/// with that vertex as their source.
pub type EdgeDict<V, P> = HashMap<V, Vec<Rc<Edge<V, P>>>>;

/// `make_edge_dict(edge_list)` at `graphlib.py`.
///
/// Puts a list of edges into the official dict format: every edge is
/// filed under its `source`, and every `target` is guaranteed to be a
/// key (with a possibly-empty list), mirroring upstream's
/// `edges.setdefault(edge.source, []).append(edge)` /
/// `edges.setdefault(edge.target, [])`. The `Rc<Edge>` handed in is the
/// same object that lands in the dict.
pub fn make_edge_dict<V, P>(edge_list: Vec<Rc<Edge<V, P>>>) -> EdgeDict<V, P>
where
    V: Eq + Hash + Clone,
{
    let mut edges: EdgeDict<V, P> = HashMap::new();
    for edge in edge_list {
        let source = edge.source.clone();
        let target = edge.target.clone();
        edges.entry(source).or_default().push(edge);
        edges.entry(target).or_default();
    }
    edges
}

/// `copy_edges(edges)` at `graphlib.py`: a copy whose lists are
/// fresh (`value[:]`) but whose `Edge` objects are shared.
pub fn copy_edges<V, P>(edges: &EdgeDict<V, P>) -> EdgeDict<V, P>
where
    V: Eq + Hash + Clone,
{
    let mut result: EdgeDict<V, P> = HashMap::new();
    for (key, value) in edges {
        result.insert(key.clone(), value.clone());
    }
    result
}

/// `('start', v)` / `('stop', v)` events from [`depth_first_search`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DfsEvent {
    Start,
    Stop,
}

/// `depth_first_search(root, vertices, edges)` at `graphlib.py`.
///
/// Returns the start/stop event stream of an iterative DFS from `root`,
/// only descending into targets that are in `vertices`.
pub fn depth_first_search<V, P, S>(
    root: &V,
    vertices: &S,
    edges: &EdgeDict<V, P>,
) -> Vec<(DfsEvent, V)>
where
    V: Eq + Hash + Clone,
    S: VertexSet<V>,
{
    let mut seen: HashSet<V> = HashSet::new();
    seen.insert(root.clone());
    let mut result: Vec<(DfsEvent, V)> = Vec::new();
    // each frame is `(vertex, next edge index)`, the persistent position
    // of upstream's `iter(edges[root])`.
    let mut stack: Vec<(V, usize)> = Vec::new();
    let mut root = root.clone();
    loop {
        result.push((DfsEvent::Start, root.clone()));
        stack.push((root.clone(), 0));
        loop {
            let last = stack.len() - 1;
            let (vertex, idx) = {
                let top = &stack[last];
                (top.0.clone(), top.1)
            };
            // `edges[root]` raises KeyError when absent (`:35`).
            let out = edges.get(&vertex).unwrap_or_else(|| {
                panic!("graphlib.py:35 depth_first_search: edges[root] KeyError")
            });
            if idx >= out.len() {
                // StopIteration (`:40-44`).
                stack.pop();
                result.push((DfsEvent::Stop, vertex));
                if stack.is_empty() {
                    return result;
                }
            } else {
                // one `next(iterator)` (`:39`).
                let w = out[idx].target.clone();
                stack[last].1 = idx + 1;
                if vertices.contains_vertex(&w) && !seen.contains(&w) {
                    seen.insert(w.clone());
                    root = w;
                    break;
                }
            }
        }
    }
}

/// `vertices_reachable_from(root, vertices, edges)` at `graphlib.py`.
pub fn vertices_reachable_from<V, P, S>(root: &V, vertices: &S, edges: &EdgeDict<V, P>) -> Vec<V>
where
    V: Eq + Hash + Clone,
    S: VertexSet<V>,
{
    depth_first_search(root, vertices, edges)
        .into_iter()
        .filter(|(event, _)| *event == DfsEvent::Start)
        .map(|(_, v)| v)
        .collect()
}

/// `strong_components(vertices, edges)` at `graphlib.py`.
///
/// Enumerates the strongly connected components of the graph; each is a
/// set of vertices mutually reachable along the edges.
pub fn strong_components<V, P, S>(vertices: &S, edges: &EdgeDict<V, P>) -> Vec<Vec<V>>
where
    V: Eq + Hash + Clone,
    S: VertexSet<V>,
{
    let mut component_root: HashMap<V, V> = HashMap::new();
    let mut discovery_time: HashMap<V, usize> = HashMap::new();
    let mut remaining: HashMap<V, ()> = vertices
        .vertex_snapshot()
        .into_iter()
        .map(|v| (v, ()))
        .collect();
    let mut stack: Vec<V> = Vec::new();
    let mut result: Vec<Vec<V>> = Vec::new();

    for root in vertices.vertex_snapshot() {
        if !remaining.contains_key(&root) {
            continue;
        }
        for (event, v) in depth_first_search(&root, &remaining, edges) {
            match event {
                DfsEvent::Start => {
                    remaining.remove(&v);
                    let t = discovery_time.len();
                    discovery_time.insert(v.clone(), t);
                    component_root.insert(v.clone(), v.clone());
                    stack.push(v);
                }
                DfsEvent::Stop => {
                    let mut vroot = v.clone();
                    for edge in &edges[&v] {
                        let w = &edge.target;
                        if let Some(wroot) = component_root.get(w)
                            && discovery_time[wroot] < discovery_time[&vroot]
                        {
                            vroot = wroot.clone();
                        }
                    }
                    if vroot == v {
                        let mut component: Vec<V> = Vec::new();
                        loop {
                            let w = stack.pop().expect("strong_components: stack underflow");
                            component_root.remove(&w);
                            let is_v = w == v;
                            component.push(w);
                            if is_v {
                                break;
                            }
                        }
                        result.push(component);
                    } else {
                        component_root.insert(v.clone(), vroot);
                    }
                }
            }
        }
    }
    result
}

/// One frame of the `visit` generator trampoline (see [`all_cycles`]).
enum Visit<V> {
    /// `visit(v)` whose first `next()` has not yet run.
    Pending(V),
    /// `visit(v)` for a fresh vertex, paused inside its `for` loop:
    /// `edge_idx` is the next edge to scan; `awaiting_child` is set while
    /// the generator sits at its `yield`, i.e. an edge it pushed is still
    /// on `edgestack` waiting for the yielded child to finish.
    Active {
        v: V,
        edge_idx: usize,
        awaiting_child: bool,
    },
}

/// `all_cycles(root, vertices, edges)` at `graphlib.py`.
///
/// Enumerates cycles, each returned as a list of edges. As upstream
/// notes, this may not give strictly all cycles when many cycles are
/// intermixed. Each returned edge is a clone of the shared `Rc<Edge>`
/// stored in `edges`, so edge identity (and any payload) is preserved.
///
/// `vertices` is any [`VertexSet`] — a `HashSet<V>` or a dict-shaped
/// `HashMap<V, _>` keyed by vertices — matching upstream's "set of
/// vertices (or a dict with vertices as keys)" (`graphlib.py:4-5`).
///
/// Upstream avoids CPython's recursion limit by driving the inner
/// `visit` generator through an explicit stack of generators
/// (`graphlib.py:118-124`); [`Visit`] reproduces that generator's paused
/// states and the loop below is the trampoline.
pub fn all_cycles<V, P, S>(
    root: &V,
    vertices: &S,
    edges: &EdgeDict<V, P>,
) -> Vec<Vec<Rc<Edge<V, P>>>>
where
    V: Eq + Hash + Clone,
    S: VertexSet<V>,
{
    // `stackpos[v]` is the depth of `v` on the current edge stack while
    // `v` is being visited, and `None` once `v` is finished
    // (`graphlib.py:102,107,113`). Absence of `v` means "not yet seen".
    let mut stackpos: HashMap<V, Option<usize>> = HashMap::new();
    let mut edgestack: Vec<Rc<Edge<V, P>>> = Vec::new();
    let mut result: Vec<Vec<Rc<Edge<V, P>>>> = Vec::new();

    // `pending = [visit(root)]` (`graphlib.py`).
    let mut pending: Vec<Visit<V>> = vec![Visit::Pending(root.clone())];
    while !pending.is_empty() {
        let last = pending.len() - 1;
        match &pending[last] {
            Visit::Pending(v) => {
                let v = v.clone();
                // `if v not in stackpos:` (`graphlib.py`).
                match stackpos.get(&v).copied() {
                    Some(pos) => {
                        // else-branch: a back-edge to a vertex still on
                        // the stack closes a cycle (`:114-116`); the
                        // generator yields nothing and returns.
                        if let Some(pos) = pos {
                            result.push(edgestack[pos..].to_vec());
                        }
                        pending.pop();
                    }
                    None => {
                        // `stackpos[v] = len(edgestack)` (`:107`); the
                        // generator now enters its `for` loop.
                        stackpos.insert(v.clone(), Some(edgestack.len()));
                        pending[last] = Visit::Active {
                            v,
                            edge_idx: 0,
                            awaiting_child: false,
                        };
                    }
                }
            }
            Visit::Active {
                v,
                edge_idx,
                awaiting_child,
            } => {
                let v = v.clone();
                let mut idx = *edge_idx;
                if *awaiting_child {
                    // resumed after the yielded child returned:
                    // `edgestack.pop()` (`:112`).
                    edgestack.pop();
                }
                // `for edge in edges[v]:` — `edges[v]` raises KeyError
                // when `v` is absent (`:108`).
                let out = edges
                    .get(&v)
                    .unwrap_or_else(|| panic!("graphlib.py:108 all_cycles: edges[v] KeyError"));
                let mut chosen: Option<Rc<Edge<V, P>>> = None;
                while idx < out.len() {
                    let edge = &out[idx];
                    idx += 1;
                    // `if edge.target in vertices:` (`:109`).
                    if vertices.contains_vertex(&edge.target) {
                        chosen = Some(edge.clone());
                        break;
                    }
                }
                match chosen {
                    Some(edge) => {
                        // `edgestack.append(edge)` then
                        // `yield visit(edge.target)` (`:110-111`).
                        let target = edge.target.clone();
                        edgestack.push(edge);
                        pending[last] = Visit::Active {
                            v,
                            edge_idx: idx,
                            awaiting_child: true,
                        };
                        pending.push(Visit::Pending(target));
                    }
                    None => {
                        // `for` loop exhausted: `stackpos[v] = None`
                        // (`:113`); the generator returns.
                        stackpos.insert(v, None);
                        pending.pop();
                    }
                }
            }
        }
    }
    result
}

/// `find_roots(vertices, edges)` at `graphlib.py`.
///
/// A minimal set of vertices from which all others are reachable.
pub fn find_roots<V, P, S>(vertices: &S, edges: &EdgeDict<V, P>) -> HashSet<V>
where
    V: Eq + Hash + Clone,
    S: VertexSet<V>,
{
    // maps all vertices to a representing vertex of their strongly
    // connected component (`:132-138`).
    let mut rep: HashMap<V, V> = HashMap::new();
    for mut component in strong_components(vertices, edges) {
        let random_vertex = component
            .pop()
            .expect("strong_components yields non-empty components");
        rep.insert(random_vertex.clone(), random_vertex.clone());
        for v in component {
            rep.insert(v, random_vertex.clone());
        }
    }

    let mut roots: HashSet<V> = rep.values().cloned().collect();
    for v in vertices.vertex_snapshot() {
        let v1 = rep[&v].clone();
        for edge in &edges[&v] {
            // `rep[edge.target]` may KeyError (target outside vertices);
            // and `roots.remove` is also under the same try/except, so a
            // repeated removal is a no-op (`:144-149`).
            if let Some(v2) = rep.get(&edge.target) {
                let v2 = v2.clone();
                if v1 != v2 {
                    roots.remove(&v2);
                }
            }
        }
    }
    roots
}

/// `compute_depths(roots, vertices, edges)` at `graphlib.py`.
///
/// The 'depth' of a vertex is its minimal distance from any root.
pub fn compute_depths<V, P, S>(
    roots: &HashSet<V>,
    vertices: &S,
    edges: &EdgeDict<V, P>,
) -> HashMap<V, usize>
where
    V: Eq + Hash + Clone,
    S: VertexSet<V>,
{
    let mut depths: HashMap<V, usize> = HashMap::new();
    let mut curdepth = 0usize;
    for v in roots {
        depths.insert(v.clone(), 0);
    }
    let mut pending: Vec<V> = roots.iter().cloned().collect();
    while !pending.is_empty() {
        curdepth += 1;
        let prev_generation = std::mem::take(&mut pending);
        for v in prev_generation {
            for edge in &edges[&v] {
                let v2 = &edge.target;
                if vertices.contains_vertex(v2) && !depths.contains_key(v2) {
                    depths.insert(v2.clone(), curdepth);
                    pending.push(v2.clone());
                }
            }
        }
    }
    depths
}

/// One frame of the `visit` trampoline inside [`is_acyclic`].
enum AcyclicVisit<V> {
    Pending(V),
    Active { v: V, idx: usize },
}

/// `is_acyclic(vertices, edges)` at `graphlib.py`.
pub fn is_acyclic<V, P, S>(vertices: &S, edges: &EdgeDict<V, P>) -> bool
where
    V: Eq + Hash + Clone,
    S: VertexSet<V>,
{
    let mut unvisited: HashMap<V, ()> = vertices
        .vertex_snapshot()
        .into_iter()
        .map(|v| (v, ()))
        .collect();
    // `while unvisited:` (`:189`).
    while let Some(root) = unvisited.keys().next().cloned() {
        unvisited.remove(&root);
        let mut visiting: HashSet<V> = HashSet::new();
        let mut pending: Vec<AcyclicVisit<V>> = vec![AcyclicVisit::Pending(root)];
        while !pending.is_empty() {
            let last = pending.len() - 1;
            match &pending[last] {
                AcyclicVisit::Pending(v) => {
                    let v = v.clone();
                    // `visiting[vertex] = True` (`:178`).
                    visiting.insert(v.clone());
                    pending[last] = AcyclicVisit::Active { v, idx: 0 };
                }
                AcyclicVisit::Active { v, idx } => {
                    let v = v.clone();
                    let mut i = *idx;
                    let out = &edges[&v];
                    let mut pushed = false;
                    while i < out.len() {
                        let w = out[i].target.clone();
                        i += 1;
                        // `if w in visiting: raise CycleFound` (`:181-182`).
                        if visiting.contains(&w) {
                            return false;
                        }
                        // `if w in unvisited:` (`:183-185`).
                        if unvisited.contains_key(&w) {
                            unvisited.remove(&w);
                            pending[last] = AcyclicVisit::Active {
                                v: v.clone(),
                                idx: i,
                            };
                            pending.push(AcyclicVisit::Pending(w));
                            pushed = true;
                            break;
                        }
                    }
                    if !pushed {
                        // `del visiting[vertex]` then the generator
                        // returns (`:186`).
                        visiting.remove(&v);
                        pending.pop();
                    }
                }
            }
        }
    }
    true
}

/// `compute_predecessors(vertices, edgedict)` at `graphlib.py`.
///
/// Upstream's `vertices` argument is unused, so it is dropped here.
pub fn compute_predecessors<V, P>(edgedict: &EdgeDict<V, P>) -> HashMap<V, HashSet<V>>
where
    V: Eq + Hash + Clone,
{
    let mut result: HashMap<V, HashSet<V>> = HashMap::new();
    for edges in edgedict.values() {
        for edge in edges {
            result
                .entry(edge.target.clone())
                .or_default()
                .insert(edge.source.clone());
        }
    }
    result
}

/// `remove_leaves(vertices, edgedict)` at `graphlib.py`.
///
/// Recursively removes leaves — vertices with no outgoing edges —
/// mutating both `vertices` and `edgedict`.
pub fn remove_leaves<V, P, T>(vertices: &mut HashMap<V, T>, edgedict: &mut EdgeDict<V, P>)
where
    V: Eq + Hash + Clone,
{
    let incoming = compute_predecessors(edgedict);
    remove_leaves_incoming(vertices, edgedict, &incoming, None);
}

/// `remove_leaves_incoming(vertices, edgedict, incoming, leaves=None)` at
/// `graphlib.py:275-310`.
///
/// `incoming` is the [`compute_predecessors`] result, reusable across
/// many leaf-removals on the same graph; `leaves`, when given, seeds the
/// removal from those nodes.
pub fn remove_leaves_incoming<V, P, T>(
    vertices: &mut HashMap<V, T>,
    edgedict: &mut EdgeDict<V, P>,
    incoming: &HashMap<V, HashSet<V>>,
    leaves: Option<HashSet<V>>,
) where
    V: Eq + Hash + Clone,
{
    let mut leaves = match leaves {
        Some(leaves) => leaves,
        None => {
            let leaves: HashSet<V> = edgedict
                .iter()
                .filter(|(_, edges)| edges.is_empty())
                .map(|(source, _)| source.clone())
                .collect();
            for leave in &leaves {
                edgedict.remove(leave);
                vertices.remove(leave);
            }
            leaves
        }
    };
    loop {
        if leaves.is_empty() {
            break;
        }
        let mut new_leaves: HashSet<V> = HashSet::new();
        let mut to_update: HashSet<V> = HashSet::new();
        for leave in &leaves {
            if let Some(preds) = incoming.get(leave) {
                for p in preds {
                    to_update.insert(p.clone());
                }
            }
        }
        for vertex in &to_update {
            let Some(edges) = edgedict.get_mut(vertex) else {
                continue;
            };
            // `del edges[i]` for every edge whose target is a leaf
            // (`:297-303`).
            edges.retain(|edge| !leaves.contains(&edge.target));
            if edges.is_empty() {
                new_leaves.insert(vertex.clone());
            }
        }
        leaves = new_leaves;
        for leave in &leaves {
            edgedict.remove(leave);
            vertices.remove(leave);
        }
    }
}

/// `break_cycles_v(vertices, edges)` at `graphlib.py`.
///
/// Enumerates a reasonably minimal set of *vertices* to remove to make
/// the graph acyclic. Each cycle is broken at the vertex furthest from a
/// root (largest depth), so the stack check lands as late as possible.
pub fn break_cycles_v<V, P, S>(vertices: &S, edges: &EdgeDict<V, P>) -> Vec<V>
where
    V: Eq + Hash + Clone,
    S: VertexSet<V>,
{
    let mut edges = copy_edges(edges); // we mutate it
    let incoming = compute_predecessors(&edges);
    let vertices_count = vertices.vertex_snapshot().len();

    let mut v_depths: HashMap<V, usize> = HashMap::new();
    let mut first_time = true;
    let mut roots_finished: HashSet<V> = HashSet::new();
    let mut result: Vec<V> = Vec::new();
    let mut progress = true;
    while progress {
        // first pass roots come from `vertices`; later passes from the
        // shrinking `v_depths` (`:346`).
        let roots: HashSet<V> = if first_time {
            find_roots(vertices, &edges)
        } else {
            find_roots(&v_depths, &edges)
        };
        if first_time {
            v_depths = compute_depths(&roots, vertices, &edges);
            assert_eq!(
                v_depths.len(),
                vertices_count,
                "break_cycles_v: roots must cover all vertices"
            );
            // leaves never contribute to cycles; drop them so the
            // `all_cycles` calls below don't walk into them (`:352-355`).
            remove_leaves_incoming(&mut v_depths, &mut edges, &incoming, None);
            first_time = false;
        }
        progress = false;
        for root in &roots {
            if roots_finished.contains(root) || !v_depths.contains_key(root) {
                continue;
            }
            let cycles = all_cycles(root, &v_depths, &edges);
            if cycles.is_empty() {
                roots_finished.insert(root.clone());
                continue;
            }
            // depth of a cycle = how far it reaches from any root
            // (`:368-371`).
            #[expect(
                clippy::type_complexity,
                reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
            )]
            let mut allcycles: Vec<(usize, Vec<Rc<Edge<V, P>>>)> = Vec::new();
            for cycle in cycles {
                let cycledepth = cycle
                    .iter()
                    .map(|edge| v_depths[&edge.source])
                    .max()
                    .expect("a cycle has at least one edge");
                allcycles.push((cycledepth, cycle));
            }
            // upstream `allcycles.sort()` tuple-sorts `(depth, cycle)`; on
            // a depth tie it falls through to comparing the cycles, i.e.
            // Python-2 `Edge` object-id order (non-deterministic). Stable
            // sort on depth alone keeps the `all_cycles` discovery order.
            allcycles.sort_by_key(|a| a.0);
            let mut removed: HashSet<V> = HashSet::new();
            for (_, cycle) in &allcycles {
                // `v_depths[edge.source]` KeyErrors once a vertex on this
                // cycle was already removed → the cycle is already broken
                // (`:376-380`).
                let mut choices: Vec<(usize, V)> = Vec::with_capacity(cycle.len());
                let mut already_broken = false;
                for edge in cycle {
                    match v_depths.get(&edge.source) {
                        Some(&d) => choices.push((d, edge.source.clone())),
                        None => {
                            already_broken = true;
                            break;
                        }
                    }
                }
                if already_broken {
                    continue;
                }
                // break this cycle by removing the furthest vertex
                // (`:382-388`). Upstream `max(choices)` tuple-maxes over
                // `(depth, vertex)`; depth dominates, and a depth tie falls
                // through to Python-2 `vertex` object-id order
                // (non-deterministic — and the vertices, e.g. flow-graph
                // `Block`s from `insert_ll_stackcheck`, carry no ordering).
                // Maxing by depth alone keeps the same furthest-vertex choice
                // without imposing `Ord` on `V`; `max_by_key` returns the last
                // such vertex in cycle-edge order, which is deterministic.
                let (_max_depth, max_vertex) = choices
                    .into_iter()
                    .max_by_key(|(depth, _)| *depth)
                    .expect("non-empty choices");
                v_depths.remove(&max_vertex);
                edges.remove(&max_vertex);
                result.push(max_vertex.clone());
                removed.insert(max_vertex);
                progress = true;
            }
            // early exit when done (`:390-392`).
            if is_acyclic(&v_depths, &edges) {
                return result;
            }
            // remove leaves, starting from the nodes we just removed
            // (`:393-395`).
            remove_leaves_incoming(&mut v_depths, &mut edges, &incoming, Some(removed));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(source: &'static str, target: &'static str) -> Rc<Edge<&'static str>> {
        Rc::new(Edge::new(source, target))
    }

    fn ep(source: u32, target: u32, payload: &'static str) -> Rc<Edge<u32, &'static str>> {
        Rc::new(Edge::with_payload(source, target, payload))
    }

    fn en(source: u32, target: u32) -> Rc<Edge<u32>> {
        Rc::new(Edge::new(source, target))
    }

    fn cycle_targets<P>(cycle: &[Rc<Edge<&'static str, P>>]) -> Vec<(&'static str, &'static str)> {
        cycle.iter().map(|e| (e.source, e.target)).collect()
    }

    /// The `TestSimple.edges` graph (`test_graphlib.py`).
    fn simple_edges() -> EdgeDict<&'static str, ()> {
        let mut edges: EdgeDict<&'static str, ()> = HashMap::new();
        edges.insert("A", vec![e("A", "B"), e("A", "C")]);
        edges.insert("B", vec![e("B", "D"), e("B", "E")]);
        edges.insert("C", vec![e("C", "F")]);
        edges.insert("D", vec![e("D", "D")]);
        edges.insert("E", vec![e("E", "A"), e("E", "C")]);
        edges.insert("F", vec![]);
        edges.insert("G", vec![]);
        edges
    }

    // ---- existing edge-level unit tests, on the shared Rc edges ----

    #[test]
    fn make_edge_dict_files_sources_and_seeds_targets() {
        let edges = make_edge_dict(vec![en(1, 2), en(2, 3)]);
        assert_eq!(edges[&1].len(), 1);
        assert_eq!(edges[&1][0].target, 2);
        assert_eq!(edges[&2].len(), 1);
        // target-only vertex is present with an empty list.
        assert!(edges[&3].is_empty());
    }

    #[test]
    fn edge_repr_is_source_arrow_target() {
        assert_eq!(format!("{:?}", Edge::new(1u32, 2u32)), "1 -> 2");
    }

    #[test]
    fn self_loop_is_a_one_edge_cycle() {
        let edges = make_edge_dict(vec![en(1, 1)]);
        let vertices: HashSet<u32> = [1].into_iter().collect();
        let cycles = all_cycles(&1, &vertices, &edges);
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0].len(), 1);
        assert_eq!((cycles[0][0].source, cycles[0][0].target), (1, 1));
    }

    #[test]
    fn target_outside_vertices_is_not_traversed() {
        // 1 -> 2 -> 1, but 2 is excluded from `vertices`: no cycle.
        let edges = make_edge_dict(vec![en(1, 2), en(2, 1)]);
        let vertices: HashSet<u32> = [1].into_iter().collect();
        assert!(all_cycles(&1, &vertices, &edges).is_empty());
    }

    #[test]
    fn payload_is_preserved_through_cycles() {
        let edges = make_edge_dict(vec![ep(1, 2, "a"), ep(2, 1, "b")]);
        let vertices: HashSet<u32> = [1, 2].into_iter().collect();
        let cycles = all_cycles(&1, &vertices, &edges);
        assert_eq!(cycles.len(), 1);
        let payloads: Vec<&str> = cycles[0].iter().map(|e| e.payload).collect();
        assert_eq!(payloads, vec!["a", "b"]);
    }

    #[test]
    #[should_panic(expected = "edges[v] KeyError")]
    fn missing_root_in_edges_panics_like_keyerror() {
        let edges: EdgeDict<u32, ()> = HashMap::new();
        let vertices: HashSet<u32> = [1].into_iter().collect();
        all_cycles(&1, &vertices, &edges);
    }

    // ---- TestSimple ports (test_graphlib.py) ----

    #[test]
    fn test_depth_first_search() {
        // 'D' missing from the list of vertices (`test_graphlib.py:17-31`).
        let edges = simple_edges();
        let vertices: HashSet<&str> = ["A", "B", "C", "E", "F", "G"].into_iter().collect();
        let lst = depth_first_search(&"A", &vertices, &edges);
        let expected = vec![
            (DfsEvent::Start, "A"),
            (DfsEvent::Start, "B"),
            (DfsEvent::Start, "E"),
            (DfsEvent::Start, "C"),
            (DfsEvent::Start, "F"),
            (DfsEvent::Stop, "F"),
            (DfsEvent::Stop, "C"),
            (DfsEvent::Stop, "E"),
            (DfsEvent::Stop, "B"),
            (DfsEvent::Stop, "A"),
        ];
        assert_eq!(lst, expected);
    }

    #[test]
    fn test_strong_components() {
        // `test_graphlib.py:33-43`.
        let edges = simple_edges();
        let result = strong_components(&edges, &edges);
        let mut names: Vec<String> = result
            .iter()
            .map(|comp| {
                let mut chars: Vec<&str> = comp.clone();
                chars.sort();
                chars.concat()
            })
            .collect();
        names.sort();
        assert_eq!(names, vec!["ABE", "C", "D", "F", "G"]);
    }

    #[test]
    fn test_all_cycles() {
        // `test_graphlib.py:45-56`.
        let edges = simple_edges();
        let cycles = all_cycles(&"A", &edges, &edges);
        let mut got: Vec<Vec<(&str, &str)>> = cycles.iter().map(|c| cycle_targets(c)).collect();
        got.sort();
        let mut expected = vec![vec![("A", "B"), ("B", "E"), ("E", "A")], vec![("D", "D")]];
        expected.sort();
        assert_eq!(got, expected);
    }

    #[test]
    fn test_find_roots() {
        // `test_graphlib.py:82-91`.
        let edges = simple_edges();
        let mut roots: Vec<&str> = find_roots(&edges, &edges).into_iter().collect();
        roots.sort();
        let joined = roots.concat();
        assert!(["AG", "BG", "EG"].contains(&joined.as_str()));

        // adding R -> B makes R the only root reaching the A/B/E cycle.
        let mut edges = simple_edges();
        edges.insert("R", vec![e("R", "B")]);
        let mut roots: Vec<&str> = find_roots(&edges, &edges).into_iter().collect();
        roots.sort();
        assert_eq!(roots.concat(), "GR");
    }

    #[test]
    fn test_break_cycles_v() {
        // `test_graphlib.py:69-80`.
        let mut edges = simple_edges();
        edges.insert("R", vec![e("R", "B")]);
        let mut result = break_cycles_v(&edges, &edges);
        result.sort();
        // 'A' (furthest from root 'R' on the A/B/E cycle) and 'D' (the
        // self-loop). 'BD' / 'DE' would also break the cycle, but 'A' is
        // the furthest vertex so it is picked.
        assert_eq!(result.concat(), "AD");
    }

    #[test]
    fn test_remove_leaves() {
        // `test_graphlib.py:93-101`.
        let mut edges = simple_edges();
        let mut vertices: HashMap<&str, ()> = edges.keys().map(|k| (*k, ())).collect();
        remove_leaves(&mut vertices, &mut edges);
        assert!(!edges.contains_key("F"));
        assert!(!edges.contains_key("C"));
        assert_eq!(edges["A"].len(), 1);
        assert_eq!(edges["A"][0].target, "B");
        assert_eq!(edges["E"].len(), 1);
        assert_eq!(edges["E"][0].target, "A");
    }

    #[test]
    fn is_acyclic_detects_cycle_and_tree() {
        // the A/B/E cycle makes the simple graph cyclic.
        let edges = simple_edges();
        assert!(!is_acyclic(&edges, &edges));

        // a plain chain 1 -> 2 -> 3 is acyclic.
        let chain = make_edge_dict(vec![en(1, 2), en(2, 3)]);
        let vertices: HashSet<u32> = [1, 2, 3].into_iter().collect();
        assert!(is_acyclic(&vertices, &chain));
    }

    #[test]
    fn depth_first_search_accepts_a_list() {
        // upstream passes a plain list as `vertices` (`test_graphlib.py:19`:
        // `depth_first_search('A', list('ABCEFG'), edges)`); the list view
        // must agree with the set view.
        let edges = simple_edges();
        let vertices: Vec<&str> = vec!["A", "B", "C", "E", "F", "G"];
        let from_list = depth_first_search(&"A", &vertices, &edges);
        let set: HashSet<&str> = vertices.iter().copied().collect();
        assert_eq!(from_list, depth_first_search(&"A", &set, &edges));
        assert_eq!(from_list.first(), Some(&(DfsEvent::Start, "A")));
        assert_eq!(from_list.last(), Some(&(DfsEvent::Stop, "A")));
    }

    #[test]
    fn break_cycles_v_accepts_unordered_vertices() {
        // `insert_ll_stackcheck` calls `break_cycles_v(edgedict, edgedict)`
        // (`transform.py:231`) with flow-graph `Block`s as vertices, which
        // are identity-keyed and carry no ordering. This vertex type has
        // `Eq + Hash` but deliberately no `Ord`, guarding against the
        // `V: Ord` bound creeping back.
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        struct Vid(u32);
        // 0 -> 1 -> 2 -> 1: a 1/2 cycle reachable from root 0.
        let edges = make_edge_dict(vec![
            Rc::new(Edge::new(Vid(0), Vid(1))),
            Rc::new(Edge::new(Vid(1), Vid(2))),
            Rc::new(Edge::new(Vid(2), Vid(1))),
        ]);
        // 2 is the furthest vertex on the cycle, so it is removed.
        assert_eq!(break_cycles_v(&edges, &edges), vec![Vid(2)]);
    }
}
