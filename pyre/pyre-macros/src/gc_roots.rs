//! `#[gc_roots]` — `shadowcolor.py postprocess_graph` over a Rust function
//! body.
//!
//! Upstream writes the shadow-stack bracket twice. `framework.py` /
//! `shadowstack.py` emit the naive form -- one `gc_push_roots` /
//! `gc_pop_roots` pair immediately around every operation that can reach the
//! collector -- and `shadowcolor.py` then rewrites that form into the one that
//! ships: `allocate_registers` colours the roots so slots are reused,
//! `expand_push_roots` / `expand_pop_roots` turn the bulk pair into per-root
//! `gc_save_root` / `gc_restore_root`, `move_pushes_earlier` hoists saves out
//! of loops, and `add_enter_leave_roots_frame` reduces the whole graph to a
//! single frame entered as late as possible.
//!
//! Both halves are needed, and only the second one is an optimiser. Writing
//! the shipped form by hand means doing a register allocator's job in one's
//! head at every call site; writing the naive form by hand is mechanical.
//! This attribute is what makes that split available here: a function body
//! states one [`with_roots!`] bracket per collecting operation, and the
//! attribute performs the `shadowcolor.py` half.
//!
//! [`with_roots!`]: pyre_object::with_roots
//!
//! ```ignore
//! #[gc_roots]
//! fn f(mut obj: PyObjectRef, mut value: PyObjectRef) -> Result<(), PyError> {
//!     with_roots!(obj, value => switch_strategy(obj));
//!     let key = with_roots!(obj, value => object_key_for(obj)?);
//!     ...
//! }
//! ```
//!
//! Two brackets, two frames, four slot pushes and two pops in the naive
//! expansion; one frame and two slots after this pass.
//!
//! # What this pass does not do
//!
//! `move_pushes_earlier` is not implemented. It is the one upstream pass whose
//! benefit is a measurement rather than a structural property, and the saves
//! it hoists are still correct where they stand.
//!
//! Upstream also skips the walker over slots a frame is not currently using,
//! via `make_bitmask`. This pass has no equivalent, so a slot keeps naming its
//! bracket's root until the next bracket overwrites the colour or the frame
//! pops. That retains an object the function has finished with; it does not
//! read a stale one, because a slot the walker can reach is a slot the walker
//! forwards.

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use std::collections::BTreeMap;
use syn::parse::{Parse, ParseStream};
use syn::visit_mut::{self, VisitMut};
use syn::{Expr, Ident, ItemFn, Stmt, Token};

/// One `with_roots!(a, b => body)` invocation, as the attribute sees it.
///
/// The attribute runs before `macro_rules!` expansion, so the invocation is
/// still literal tokens here and can be rewritten rather than expanded. That
/// ordering is what lets the naive form stay correct on its own: a function
/// without the attribute expands `with_roots!` to its own per-operation
/// bracket, exactly as an untransformed graph would run.
struct WithRoots {
    locals: Vec<Ident>,
    body: Expr,
}

impl Parse for WithRoots {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut locals = Vec::new();
        loop {
            locals.push(input.parse::<Ident>()?);
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
                if input.peek(Token![=>]) {
                    break; // trailing comma before the arrow
                }
                continue;
            }
            break;
        }
        input.parse::<Token![=>]>()?;
        Ok(Self {
            locals,
            body: input.parse()?,
        })
    }
}

/// Does this macro path name `with_roots`, however it was brought into scope?
///
/// Matched on the last segment: the call sites spell it `with_roots!`,
/// `crate::with_roots!` and `pyre_object::with_roots!`, and `#[macro_export]`
/// puts it at the crate root under all three.
fn is_with_roots(mac: &syn::Macro) -> bool {
    mac.path
        .segments
        .last()
        .is_some_and(|seg| seg.ident == "with_roots")
}

/// `allocate_registers`, at the granularity this pass can prove.
///
/// Upstream runs a real register allocator over the flow graph, so two roots
/// share a colour when their live ranges are disjoint. A colour here is a
/// scratch slot rather than a live range: it is written immediately before a
/// bracket's body and read back immediately after, and nothing between two
/// brackets reads it. So the only rule that has to hold is that the roots
/// *within one bracket* get distinct colours, and naming them is enough to
/// guarantee it.
///
/// The cost of the weaker rule is frame width -- one slot per distinct root
/// name in the function rather than per simultaneously-live root. The saving
/// this pass is actually after is the frame count, which goes to one either
/// way.
fn allocate_colors(brackets: &[WithRoots]) -> BTreeMap<String, usize> {
    let mut colors = BTreeMap::new();
    let mut next = 0usize;
    for bracket in brackets {
        for local in &bracket.locals {
            colors.entry(local.to_string()).or_insert_with(|| {
                let color = next;
                next += 1;
                color
            });
        }
    }
    colors
}

/// Collects every rewritable `with_roots!` in a body, in source order.
///
/// A bracket inside a closure or a nested item is not collected, so it keeps
/// the naive expansion. The frame is a local of the enclosing function, and a
/// closure that captured it would extend the bracket past the frame's own
/// scope -- upstream has the same boundary for free, because a nested graph is
/// a separate graph with its own frame.
struct Collect {
    brackets: Vec<WithRoots>,
    error: Option<syn::Error>,
}

impl VisitMut for Collect {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if let Expr::Closure(_) = expr {
            return;
        }
        if let Expr::Macro(mac) = expr
            && is_with_roots(&mac.mac)
        {
            match mac.mac.parse_body::<WithRoots>() {
                Ok(parsed) => {
                    // The body can hold further brackets; visit it, not the
                    // unparsed token stream.
                    let mut parsed = parsed;
                    self.visit_expr_mut(&mut parsed.body);
                    self.brackets.push(parsed);
                }
                Err(err) => {
                    self.error.get_or_insert(err);
                }
            }
            return;
        }
        visit_mut::visit_expr_mut(self, expr);
    }

    /// A bracket written as a statement -- `with_roots!(obj, value => f(obj));`
    /// -- parses as `Stmt::Macro`, never reaching `visit_expr_mut`. That is
    /// the position most brackets are in, because most collecting operations
    /// are performed for their effect.
    fn visit_stmt_mut(&mut self, stmt: &mut Stmt) {
        if let Stmt::Macro(mac) = stmt
            && is_with_roots(&mac.mac)
        {
            match mac.mac.parse_body::<WithRoots>() {
                Ok(mut parsed) => {
                    self.visit_expr_mut(&mut parsed.body);
                    self.brackets.push(parsed);
                }
                Err(err) => {
                    self.error.get_or_insert(err);
                }
            }
            return;
        }
        visit_mut::visit_stmt_mut(self, stmt);
    }

    fn visit_item_mut(&mut self, _item: &mut syn::Item) {}
}

/// Rewrites each collected bracket into its `gc_save_root` /
/// `gc_restore_root` series against the function's frame.
struct Rewrite<'a> {
    colors: &'a BTreeMap<String, usize>,
    frame: &'a Ident,
    base: &'a Ident,
    error: Option<syn::Error>,
}

impl VisitMut for Rewrite<'_> {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if let Expr::Closure(_) = expr {
            return;
        }
        let Expr::Macro(mac) = expr else {
            visit_mut::visit_expr_mut(self, expr);
            return;
        };
        if !is_with_roots(&mac.mac) {
            visit_mut::visit_expr_mut(self, expr);
            return;
        }
        let mut parsed = match mac.mac.parse_body::<WithRoots>() {
            Ok(parsed) => parsed,
            Err(err) => {
                self.error.get_or_insert(err);
                return;
            }
        };
        self.visit_expr_mut(&mut parsed.body);
        *expr = self.expand(&parsed);
    }

    /// The [`Collect`] twin: rewrite a statement-position bracket in place,
    /// keeping its trailing semicolon so the block's value is unchanged.
    fn visit_stmt_mut(&mut self, stmt: &mut Stmt) {
        let Stmt::Macro(mac) = stmt else {
            visit_mut::visit_stmt_mut(self, stmt);
            return;
        };
        if !is_with_roots(&mac.mac) {
            visit_mut::visit_stmt_mut(self, stmt);
            return;
        }
        let semi = mac.semi_token;
        let mut parsed = match mac.mac.parse_body::<WithRoots>() {
            Ok(parsed) => parsed,
            Err(err) => {
                self.error.get_or_insert(err);
                return;
            }
        };
        self.visit_expr_mut(&mut parsed.body);
        *stmt = Stmt::Expr(self.expand(&parsed), semi);
    }

    fn visit_item_mut(&mut self, _item: &mut syn::Item) {}
}

impl Rewrite<'_> {
    fn expand(&self, bracket: &WithRoots) -> Expr {
        let (frame, base) = (self.frame, self.base);
        let slots: Vec<TokenStream> = bracket
            .locals
            .iter()
            .map(|local| {
                // `Collect` and `Rewrite` traverse identically, so every
                // rewritten root was coloured; naming the invariant beats an
                // index panic if a future traversal drifts apart.
                let color = *self
                    .colors
                    .get(&local.to_string())
                    .expect("a rewritten root was not coloured: the collect and rewrite traversals disagree");
                quote!(#base + #color)
            })
            .collect();
        let locals = &bracket.locals;
        let body = &bracket.body;
        // `save_run` publishes every value before resolving forwarding for
        // any, which is `expand_one_push_roots`'s series plus the safepoint
        // rule `pin_roots` states. The restores are `expand_one_pop_roots`:
        // `gc_restore_root`'s whole body is `setvar(v_value, newvalue)`, so
        // after the bracket the local *is* the live word.
        syn::parse_quote!({
            #frame.save_run(&[#((#slots, #locals)),*]);
            let __roots_result = #body;
            #(#locals = #frame.get(#slots);)*
            __roots_result
        })
    }
}

/// `add_enter_leave_roots_frame`, at the granularity this pass can prove.
///
/// Upstream puts `gc_enter_roots_frame` as late as possible -- just before the
/// first `gc_save_root` any path reaches -- and the leave as early as possible.
/// A frame at the top of the body would instead charge every call that never
/// reaches a bracket, and a bracket usually guards a slow path: the two pilot
/// functions bracket only their strategy-switch and reentrant-probe arms, so
/// the ordinary store would pay a frame it never uses.
///
/// The block that gets the frame is the innermost one still containing every
/// bracket, found by descending while a single branch holds all of them.
/// Loops are not entered: a frame inside a loop body is re-entered per
/// iteration, which is the direction `move_pushes_earlier` exists to undo.
/// Closures are not entered either, matching [`Collect`].
mod place {
    use super::{Stmt, is_with_roots};
    use syn::{Block, Expr};

    pub fn in_block(block: &Block) -> usize {
        block.stmts.iter().map(in_stmt).sum()
    }

    pub fn in_stmt(stmt: &Stmt) -> usize {
        match stmt {
            Stmt::Macro(mac) if is_with_roots(&mac.mac) => 1,
            Stmt::Expr(expr, _) => in_expr(expr),
            Stmt::Local(local) => local
                .init
                .as_ref()
                .map(|init| {
                    in_expr(&init.expr) + init.diverge.as_ref().map_or(0, |(_, e)| in_expr(e))
                })
                .unwrap_or(0),
            // An item is a separate graph with its own frame.
            Stmt::Item(_) => 0,
            Stmt::Macro(_) => 0,
        }
    }

    /// Counts brackets in `expr`, by the same traversal [`super::Collect`]
    /// uses -- the two must agree, or the frame could be installed somewhere
    /// that does not dominate every rewritten bracket.
    pub fn in_expr(expr: &Expr) -> usize {
        struct Count(usize);
        impl<'ast> syn::visit::Visit<'ast> for Count {
            fn visit_expr(&mut self, expr: &'ast Expr) {
                if let Expr::Closure(_) = expr {
                    return;
                }
                if let Expr::Macro(mac) = expr
                    && is_with_roots(&mac.mac)
                {
                    self.0 += 1;
                    if let Ok(parsed) = mac.mac.parse_body::<super::WithRoots>() {
                        self.visit_expr(&parsed.body);
                    }
                    return;
                }
                syn::visit::visit_expr(self, expr);
            }
            fn visit_stmt(&mut self, stmt: &'ast Stmt) {
                if let Stmt::Macro(mac) = stmt
                    && is_with_roots(&mac.mac)
                {
                    self.0 += 1;
                    if let Ok(parsed) = mac.mac.parse_body::<super::WithRoots>() {
                        self.visit_expr(&parsed.body);
                    }
                    return;
                }
                syn::visit::visit_stmt(self, stmt);
            }
            fn visit_item(&mut self, _item: &'ast syn::Item) {}
        }
        let mut count = Count(0);
        syn::visit::Visit::visit_expr(&mut count, expr);
        count.0
    }

    /// The blocks a statement branches into, for a descent that must not
    /// change how often the frame is entered.
    fn branches(expr: &mut Expr) -> Vec<&mut Block> {
        match expr {
            Expr::Block(e) => vec![&mut e.block],
            Expr::Unsafe(e) => vec![&mut e.block],
            Expr::If(e) => {
                let mut out = vec![&mut e.then_branch];
                if let Some((_, alt)) = &mut e.else_branch
                    && let Expr::Block(alt) = &mut **alt
                {
                    out.push(&mut alt.block);
                }
                out
            }
            Expr::Match(e) => e
                .arms
                .iter_mut()
                .filter_map(|arm| match &mut *arm.body {
                    Expr::Block(b) => Some(&mut b.block),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Insert `prologue` at the latest point that still dominates every
    /// bracket, and report where it landed for the caller to check.
    pub fn install(block: &mut Block, prologue: &[Stmt]) {
        let total = in_block(block);
        debug_assert!(total > 0);
        let counts: Vec<usize> = block.stmts.iter().map(in_stmt).collect();
        let first = counts.iter().position(|&n| n > 0).expect("a bracket");
        if counts[first] == total {
            let descend = match &mut block.stmts[first] {
                Stmt::Expr(expr, _) => Some(expr),
                Stmt::Local(local) => local.init.as_mut().map(|init| &mut *init.expr),
                _ => None,
            };
            if let Some(expr) = descend
                && let Some(inner) = branches(expr)
                    .into_iter()
                    .find(|inner| in_block(inner) == total)
            {
                install(inner, prologue);
                return;
            }
        }
        block.stmts.splice(first..first, prologue.iter().cloned());
    }
}

pub fn expand(mut func: ItemFn) -> syn::Result<TokenStream> {
    let mut collect = Collect {
        brackets: Vec::new(),
        error: None,
    };
    // `visit_block_mut` rather than `visit_item_fn_mut`: the latter would be
    // stopped by this visitor's own `visit_item_mut` guard.
    collect.visit_block_mut(&mut func.block);
    if let Some(err) = collect.error {
        return Err(err);
    }
    if collect.brackets.is_empty() {
        return Err(syn::Error::new(
            func.sig.ident.span(),
            "#[gc_roots] found no `with_roots!` bracket to transform. \
             An attribute that transforms nothing still emits a frame, so it \
             costs a push and a pop to say nothing; remove it.",
        ));
    }

    let colors = allocate_colors(&collect.brackets);
    let numcolors = colors.len();
    let frame = format_ident!("__pyre_gc_frame", span = Span::call_site());
    let base = format_ident!("__pyre_gc_base", span = Span::call_site());

    // `add_enter_leave_roots_frame` before the expansion, so the search counts
    // `with_roots!` invocations rather than the blocks they become. The leave
    // needs no placement: the frame is a scope guard, so it pops at the end of
    // whichever block the enter landed in -- the earliest point that still
    // covers every restore.
    let prologue: Vec<Stmt> = syn::parse_quote! {
        let #frame = ::pyre_object::gc_roots::enter_roots_frame(#numcolors);
        let #base = ::pyre_object::gc_roots::RootScope::base(&#frame);
    };
    place::install(&mut func.block, &prologue);

    let mut rewrite = Rewrite {
        colors: &colors,
        frame: &frame,
        base: &base,
        error: None,
    };
    rewrite.visit_block_mut(&mut func.block);
    if let Some(err) = rewrite.error {
        return Err(err);
    }

    Ok(quote!(#func))
}
