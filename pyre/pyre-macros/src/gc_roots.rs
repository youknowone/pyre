//! `#[gc_roots]` — `shadowcolor.py postprocess_graph` over a Rust function
//! body.
//!
//! Upstream writes the shadow-stack bracket twice. `framework.py` /
//! `shadowstack.py` emit the naive form -- one `gc_push_roots` /
//! `gc_pop_roots` pair immediately around every operation that can reach the
//! collector -- and `shadowcolor.py` then rewrites that form into the one that
//! ships: `allocate_registers` colours the roots so slots are reused,
//! `expand_push_roots` / `expand_pop_roots` turn the bulk pair into per-root
//! `gc_save_root` / `gc_restore_root`, `move_pushes_earlier` hoists saves
//! earlier, and `add_enter_leave_roots_frame` reduces the whole graph to a
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
//! expansion; one frame, two slots and one save after this pass.
//!
//! `majit-translate`'s `memory::gctransform::shadowcolor` is the strict port of
//! the same upstream module, over the flowspace graphs the translator builds.
//! This one answers for the interpreter, whose graphs rustc owns, so it works
//! on the token tree instead and is an approximation at the two points named
//! below.
//!
//! # Where this pass is weaker than upstream
//!
//! Colours are per root *name*, not per live range, because a colour here is a
//! scratch slot rather than a range: nothing between two brackets reads it, so
//! the only rule that has to hold is that the roots within one bracket differ.
//! The cost is frame width, not frame count.
//!
//! `move_pushes_earlier` hoists only to the frame prologue, and only for roots
//! [`hoistable`] can prove constant. Upstream hoists any save to any earlier
//! point its colouring admits.
//!
//! Upstream also skips the walker over slots a frame is not currently using,
//! via `make_bitmask`. This pass has no equivalent, so a slot keeps naming its
//! root until the next bracket overwrites the colour or the frame pops. That
//! retains an object the function has finished with; it does not read a stale
//! one, because a slot the walker can reach is a slot the walker forwards.

use proc_macro2::{Span, TokenStream, TokenTree};
use quote::{format_ident, quote};
use std::collections::{BTreeMap, BTreeSet};
use syn::parse::{Parse, ParseStream};
use syn::visit_mut::{self, VisitMut};
use syn::{Block, Expr, FnArg, Ident, ItemFn, Pat, Stmt, Token};

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

// ── The one traversal ───────────────────────────────────────────────
//
// Every pass below walks the body looking for the same thing, and they have to
// agree about what they find: a bracket the colouring misses but the rewrite
// reaches has no slot to address, and a bracket the frame placement misses can
// end up outside the frame that serves it. So the traversal is written once
// and the passes differ only in what they do with a bracket.

/// What a pass does with each bracket the walk reaches.
trait Handler {
    /// Called once per bracket, after the bracket's own body has been walked.
    /// `Some` replaces the invocation, `None` leaves it in place.
    fn bracket(&mut self, bracket: WithRoots) -> Option<Expr>;
    fn error(&mut self, err: syn::Error);
}

/// Walks a body and reports every rewritable bracket to a [`Handler`].
///
/// A bracket inside a closure or a nested item is not reported, so it keeps
/// the naive expansion. The frame is a local of the enclosing function, and a
/// closure that captured it would extend the bracket past the frame's own
/// scope -- upstream has the same boundary for free, because a nested graph is
/// a separate graph with its own frame.
struct Walk<H>(H);

impl<H: Handler> Walk<H> {
    /// Parse an invocation and hand it to the handler, walking the bracket's
    /// own body first so a nested bracket is reported before its parent.
    fn take(&mut self, mac: &syn::Macro) -> Option<Expr> {
        match mac.parse_body::<WithRoots>() {
            Ok(mut bracket) => {
                self.visit_expr_mut(&mut bracket.body);
                self.0.bracket(bracket)
            }
            Err(err) => {
                self.0.error(err);
                None
            }
        }
    }
}

impl<H: Handler> VisitMut for Walk<H> {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if let Expr::Closure(_) = expr {
            return;
        }
        if let Expr::Macro(mac) = expr
            && is_with_roots(&mac.mac)
        {
            let mac = mac.mac.clone();
            if let Some(replacement) = self.take(&mac) {
                *expr = replacement;
            }
            return;
        }
        visit_mut::visit_expr_mut(self, expr);
    }

    /// A bracket written as a statement -- `with_roots!(obj, value => f(obj));`
    /// -- parses as `Stmt::Macro`, never reaching `visit_expr_mut`. That is the
    /// position most brackets are in, because most collecting operations are
    /// performed for their effect.
    fn visit_stmt_mut(&mut self, stmt: &mut Stmt) {
        if let Stmt::Macro(mac) = stmt
            && is_with_roots(&mac.mac)
        {
            let (inner, semi) = (mac.mac.clone(), mac.semi_token);
            if let Some(replacement) = self.take(&inner) {
                // Keeping the semicolon leaves the block's value unchanged.
                *stmt = Stmt::Expr(replacement, semi);
            }
            return;
        }
        visit_mut::visit_stmt_mut(self, stmt);
    }

    fn visit_item_mut(&mut self, _item: &mut syn::Item) {}
}

/// Records every bracket, changing nothing.
#[derive(Default)]
struct Collect {
    brackets: Vec<WithRoots>,
    error: Option<syn::Error>,
}

impl Handler for Collect {
    fn bracket(&mut self, bracket: WithRoots) -> Option<Expr> {
        self.brackets.push(bracket);
        None
    }
    fn error(&mut self, err: syn::Error) {
        self.error.get_or_insert(err);
    }
}

/// Counts brackets, changing nothing — the frame placement's measure.
#[derive(Default)]
struct Count(usize);

impl Handler for Count {
    fn bracket(&mut self, _bracket: WithRoots) -> Option<Expr> {
        self.0 += 1;
        None
    }
    fn error(&mut self, _err: syn::Error) {}
}

/// Collects the root names brackets address, changing nothing — what a frame
/// site needs in order to know which roots it can save on their behalf.
#[derive(Default)]
struct Names(BTreeSet<String>);

impl Handler for Names {
    fn bracket(&mut self, bracket: WithRoots) -> Option<Expr> {
        self.0.extend(bracket.locals.iter().map(Ident::to_string));
        None
    }
    fn error(&mut self, _err: syn::Error) {}
}

// ── allocate_registers ──────────────────────────────────────────────

/// `allocate_registers`, at the granularity this pass can prove.
///
/// Upstream runs a real register allocator over the flow graph, so two roots
/// share a colour when their live ranges are disjoint. A colour here is a
/// scratch slot: it is written before a bracket's body and read back after,
/// and nothing between two brackets reads it. So the only rule that has to
/// hold is that the roots *within one bracket* get distinct colours, and
/// naming them is enough to guarantee it.
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

// ── move_pushes_earlier ─────────────────────────────────────────────

/// `move_pushes_earlier`: which roots can be saved once, in the prologue,
/// instead of at every bracket.
///
/// Upstream's example is a loop -- the store hoisted out and the load left
/// inside -- and that is the shape this answers for. A root saved in the
/// prologue needs no save at any bracket: the slot is the authoritative word,
/// so each bracket reloads the local from it instead, which is what the
/// hand-written brackets do when they read `shadow_stack_get(slot)` at every
/// use rather than re-pinning per iteration.
///
/// Hoisting is sound for a root whose binding cannot change between the
/// prologue and the bracket. A name qualifies when it is a parameter, or a
/// `let` with a plain name bound exactly once, and the body never rebinds it:
/// no assignment, no `&mut` borrow, no second binding of the name anywhere --
/// a `for` pattern, a match arm and a closure parameter all rebind per entry.
/// The restores this pass emits are assignments too, but they write the slot's
/// own word, which is the invariant rather than a violation of it, so they are
/// not counted.
///
/// A `let` qualifying does not prove it is in scope where the prologue lands,
/// and this pass does not check: a `let` after that point, or inside a loop
/// the prologue sits ahead of, fails to compile rather than hoisting a name
/// that names nothing.
///
/// A macro body is opaque tokens at this point, so it is scanned for the same
/// three shapes rather than parsed. The scan over-reports -- `a == b` is not
/// an assignment, and neither is a `=` inside a nested macro that never
/// expands -- and over-reporting only declines a hoist.
fn hoistable(func: &ItemFn) -> BTreeSet<String> {
    let mut params = BTreeSet::new();
    for arg in &func.sig.inputs {
        if let FnArg::Typed(typed) = arg
            && let Pat::Ident(ident) = &*typed.pat
        {
            params.insert(ident.ident.to_string());
        }
    }
    #[derive(Default)]
    struct Rebound {
        /// Names an assignment, a `&mut` borrow or an opaque macro can change.
        assigned: BTreeSet<String>,
        /// Plain `let name = ...;` bindings, and how many were seen.
        let_bound: BTreeMap<String, usize>,
        /// Bound somewhere a value arrives per entry rather than once.
        other_bound: BTreeSet<String>,
    }
    impl Rebound {
        fn note_path(&mut self, expr: &Expr) {
            if let Expr::Path(path) = expr
                && let Some(ident) = path.path.get_ident()
            {
                self.assigned.insert(ident.to_string());
            }
        }
        /// `ident =` (but not `==`), `&mut ident`, and `let` bindings, over
        /// tokens no parser will reach.
        fn note_tokens(&mut self, tokens: &TokenStream) {
            let trees: Vec<TokenTree> = tokens.clone().into_iter().collect();
            for (i, tree) in trees.iter().enumerate() {
                match tree {
                    TokenTree::Group(group) => self.note_tokens(&group.stream()),
                    TokenTree::Ident(ident) => {
                        if let Some(TokenTree::Punct(punct)) = trees.get(i + 1)
                            && punct.as_char() == '='
                            && punct.spacing() == proc_macro2::Spacing::Alone
                        {
                            self.assigned.insert(ident.to_string());
                        }
                        // `&mut name`, and `let name`.
                        if (ident == "mut" || ident == "let")
                            && let Some(TokenTree::Ident(name)) = trees.get(i + 1)
                        {
                            self.assigned.insert(name.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    impl<'ast> syn::visit::Visit<'ast> for Rebound {
        fn visit_expr(&mut self, expr: &'ast Expr) {
            match expr {
                // The bracket's own restores are the invariant, not a rebind;
                // its body is still walked for everything else.
                Expr::Macro(mac) if is_with_roots(&mac.mac) => {
                    if let Ok(bracket) = mac.mac.parse_body::<WithRoots>() {
                        self.visit_expr(&bracket.body);
                    }
                    return;
                }
                Expr::Assign(assign) => self.note_path(&assign.left),
                Expr::Binary(binary) => {
                    use syn::BinOp::*;
                    if matches!(
                        binary.op,
                        AddAssign(_)
                            | SubAssign(_)
                            | MulAssign(_)
                            | DivAssign(_)
                            | RemAssign(_)
                            | BitXorAssign(_)
                            | BitAndAssign(_)
                            | BitOrAssign(_)
                            | ShlAssign(_)
                            | ShrAssign(_)
                    ) {
                        self.note_path(&binary.left);
                    }
                }
                Expr::Reference(reference) if reference.mutability.is_some() => {
                    self.note_path(&reference.expr)
                }
                Expr::Macro(mac) => self.note_tokens(&mac.mac.tokens),
                _ => {}
            }
            syn::visit::visit_expr(self, expr);
        }
        fn visit_stmt(&mut self, stmt: &'ast Stmt) {
            match stmt {
                Stmt::Macro(mac) => {
                    if is_with_roots(&mac.mac) {
                        if let Ok(bracket) = mac.mac.parse_body::<WithRoots>() {
                            self.visit_expr(&bracket.body);
                        }
                        return;
                    }
                    self.note_tokens(&mac.mac.tokens);
                }
                // A plain `let name = ...;` is the one binding form that
                // delivers a value once; every other one this visitor reaches
                // through `visit_pat_ident` delivers per entry.
                Stmt::Local(local) => {
                    if let Pat::Ident(ident) = &local.pat
                        && ident.subpat.is_none()
                    {
                        *self.let_bound.entry(ident.ident.to_string()).or_default() += 1;
                        if let Some(init) = &local.init {
                            self.visit_expr(&init.expr);
                            if let Some((_, diverge)) = &init.diverge {
                                self.visit_expr(diverge);
                            }
                        }
                        return;
                    }
                }
                _ => {}
            }
            syn::visit::visit_stmt(self, stmt);
        }
        fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
            self.other_bound.insert(pat.ident.to_string());
            syn::visit::visit_pat_ident(self, pat);
        }
        /// A nested item is a separate graph; its names are not these names.
        fn visit_item(&mut self, _item: &'ast syn::Item) {}
    }

    let mut rebound = Rebound::default();
    for stmt in &func.block.stmts {
        syn::visit::Visit::visit_stmt(&mut rebound, stmt);
    }
    let mut names = params;
    names.extend(
        rebound
            .let_bound
            .iter()
            .filter(|(_, count)| **count == 1)
            .map(|(name, _)| name.clone()),
    );
    names.retain(|name| !rebound.assigned.contains(name) && !rebound.other_bound.contains(name));
    names
}

// ── expand_push_roots / expand_pop_roots ────────────────────────────

/// Rewrites each bracket into its `gc_save_root` / `gc_restore_root` series
/// against the function's frame.
pub struct Rewrite<'a> {
    colors: &'a BTreeMap<String, usize>,
    /// The roots this site saved on arrival; owned, because the set differs
    /// per site and the sites are built one at a time.
    hoisted: BTreeSet<String>,
    frame: &'a Ident,
    base: &'a Ident,
    error: Option<syn::Error>,
}

impl Rewrite<'_> {
    fn slot(&self, local: &Ident) -> TokenStream {
        let base = self.base;
        // The colouring walked the same traversal, so every rewritten root has
        // a colour; naming the invariant beats an index panic if the two ever
        // drift apart.
        let color = *self.colors.get(&local.to_string()).expect(
            "a rewritten root was not coloured: the collect and rewrite traversals disagree",
        );
        quote!(#base + #color)
    }
}

impl Handler for Rewrite<'_> {
    fn bracket(&mut self, bracket: WithRoots) -> Option<Expr> {
        let frame = self.frame;
        let mut saves = Vec::new();
        let mut reloads = Vec::new();
        let mut restores = Vec::new();
        for local in &bracket.locals {
            let slot = self.slot(local);
            if self.hoisted.contains(&local.to_string()) {
                // Saved once in the prologue. The slot is the live word and
                // the local may not be, so the body reads the slot.
                reloads.push(quote!(#local = #frame.get(#slot);));
            } else {
                saves.push(quote!((#slot, #local)));
            }
            restores.push(quote!(#local = #frame.get(#slot);));
        }
        // `expand_one_push_roots`'s series, plus the safepoint rule
        // `pin_roots` states: publish the whole set before querying any of it.
        let save = (!saves.is_empty()).then(|| quote!(#frame.save_run(&[#(#saves),*]);));
        let body = &bracket.body;
        // `expand_one_pop_roots`: `gc_restore_root`'s whole body is
        // `setvar(v_value, newvalue)`, so after the bracket the local *is* the
        // live word.
        // Parenthesised: a bare block in a condition -- `if let Some(c) =
        // with_roots!(..) && ..` -- would be read as the `if`'s own body.  The
        // `macro_rules!` form is immune because a macro call is an expression
        // before it expands; this output is tokens rustc parses fresh.
        Some(syn::parse_quote!(({
            #save
            #(#reloads)*
            let __roots_result = #body;
            #(#restores)*
            __roots_result
        })))
    }

    fn error(&mut self, err: syn::Error) {
        self.error.get_or_insert(err);
    }
}

// ── add_enter_leave_roots_frame ─────────────────────────────────────

/// Where the frame is entered, and what it saves on arrival.
///
/// Upstream puts `gc_enter_roots_frame` as late as possible -- just before the
/// first `gc_save_root` any path reaches -- and the leave as early as
/// possible. A frame at the top of the body would instead charge every call
/// that never reaches a bracket, and a bracket usually guards a slow path.
///
/// Guard arms that leave the function take their own frame, so a chain of them
/// charges only the arm that runs. That is where the hand-written brackets put
/// `push_roots()`: inside the arm, not ahead of the chain that selects it.
/// Whatever is left over shares one frame, placed before the first statement
/// that needs it, and descending while a single branch holds all of it. Loops
/// are not entered -- a frame inside a loop body is re-entered per iteration,
/// which is the direction `move_pushes_earlier` exists to undo -- and neither
/// are closures, matching [`Walk`].
///
/// The leave needs no placement: the frame is a scope guard, so it pops at the
/// end of whichever block the enter landed in.
///
/// Placement and rewriting are one pass because they are one decision.
/// `move_pushes_earlier`'s hoist is what a site saves on arrival, and a root
/// can only be hoisted where it is already in scope -- which differs per site,
/// so the brackets a site serves have to be expanded against that site's own
/// answer.  A single global answer would have a bracket reload a colour its
/// site never saved.
mod place {
    use super::{Block, Count, Expr, Names, Rewrite, Stmt, Walk};
    use std::collections::BTreeSet;
    use syn::visit_mut::VisitMut;

    /// Brackets in a statement, by the traversal every other pass uses.
    fn in_stmt(stmt: &Stmt) -> usize {
        let mut walk = Walk(Count::default());
        walk.visit_stmt_mut(&mut stmt.clone());
        walk.0.0
    }

    pub fn in_block(block: &Block) -> usize {
        block.stmts.iter().map(in_stmt).sum()
    }

    /// The root names the brackets in these statements address.
    fn names_in(stmts: &[Stmt]) -> BTreeSet<String> {
        let mut walk = Walk(Names::default());
        for stmt in stmts {
            walk.visit_stmt_mut(&mut stmt.clone());
        }
        walk.0.0
    }

    /// The name a plain `let name = ...;` introduces, which is the only
    /// binding form [`super::hoistable`] admits.
    fn binds(stmt: &Stmt) -> Option<String> {
        let Stmt::Local(local) = stmt else {
            return None;
        };
        match &local.pat {
            syn::Pat::Ident(ident) if ident.subpat.is_none() => Some(ident.ident.to_string()),
            _ => None,
        }
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

    /// Does this block leave the enclosing function or loop at its end?
    ///
    /// A guard arm that does is exclusive with everything after it, so a frame
    /// inside it is entered exactly on the paths that reach a bracket.
    fn diverges(block: &Block) -> bool {
        matches!(
            block.stmts.last(),
            Some(Stmt::Expr(
                Expr::Return(_) | Expr::Break(_) | Expr::Continue(_),
                _
            ))
        )
    }

    /// The arms of a guard statement that hold brackets, when every one of
    /// them diverges. `None` keeps the statement on the shared frame.
    fn guard_arms(stmt: &mut Stmt) -> Option<Vec<&mut Block>> {
        let Stmt::Expr(expr, _) = stmt else {
            return None;
        };
        if !matches!(expr, Expr::If(_) | Expr::Match(_)) {
            return None;
        }
        let mut arms = branches(expr);
        arms.retain(|arm| in_block(arm) > 0);
        if arms.is_empty() || !arms.iter().all(|arm| diverges(arm)) {
            return None;
        }
        Some(arms)
    }

    /// What every site shares: how to build a prologue, and how to expand a
    /// bracket once the site's hoist set is known.
    pub struct Sites<'a> {
        pub hoistable: &'a BTreeSet<String>,
        pub prologue: &'a dyn Fn(&BTreeSet<String>) -> Vec<Stmt>,
        pub rewrite: &'a dyn Fn(&BTreeSet<String>) -> Rewrite<'a>,
    }

    /// Place the frames in `block` and expand the brackets each one serves.
    ///
    /// `in_scope` is the names an enclosing block has already bound at the
    /// point this one starts; a root is hoisted only if it is in scope where
    /// its site lands, so a `let` further down cannot be saved ahead of
    /// itself.
    pub fn install(
        block: &mut Block,
        in_scope: &BTreeSet<String>,
        sites: &Sites<'_>,
    ) -> Option<syn::Error> {
        let mut scope = in_scope.clone();
        let mut scope_at: Vec<BTreeSet<String>> = Vec::with_capacity(block.stmts.len());
        for stmt in &block.stmts {
            scope_at.push(scope.clone());
            if let Some(name) = binds(stmt) {
                scope.insert(name);
            }
        }

        let mut error = None;
        let mut handled = vec![false; block.stmts.len()];
        for index in 0..block.stmts.len() {
            let outside = in_stmt(&block.stmts[index]);
            if outside == 0 {
                continue;
            }
            let here = scope_at[index].clone();
            let Some(arms) = guard_arms(&mut block.stmts[index]) else {
                continue;
            };
            // A bracket in the condition or the scrutinee is outside every
            // arm, and would lose the frame the arms take with them.
            if arms.iter().map(|arm| in_block(arm)).sum::<usize>() != outside {
                continue;
            }
            for arm in arms {
                error = error.or(install(arm, &here, sites));
            }
            handled[index] = true;
        }

        let counts: Vec<usize> = block
            .stmts
            .iter()
            .enumerate()
            .map(|(index, stmt)| if handled[index] { 0 } else { in_stmt(stmt) })
            .collect();
        let remaining: usize = counts.iter().sum();
        let Some(first) = counts.iter().position(|&n| n > 0) else {
            return error;
        };
        if counts[first] == remaining {
            let descend = match &mut block.stmts[first] {
                Stmt::Expr(expr, _) => Some(expr),
                Stmt::Local(local) => local.init.as_mut().map(|init| &mut *init.expr),
                _ => None,
            };
            let here = scope_at[first].clone();
            if let Some(expr) = descend
                && let Some(inner) = branches(expr)
                    .into_iter()
                    .find(|inner| in_block(inner) == remaining)
            {
                return error.or(install(inner, &here, sites));
            }
        }

        let hoist: BTreeSet<String> = names_in(&block.stmts[first..])
            .into_iter()
            .filter(|name| sites.hoistable.contains(name) && scope_at[first].contains(name))
            .collect();
        let prologue = (sites.prologue)(&hoist);
        let shift = prologue.len();
        block.stmts.splice(first..first, prologue);

        // From past the prologue, whose own statements hold no bracket. The
        // splice shifted every original statement at or after `first` by
        // `shift`, so that is what maps a position back into `handled`.
        let mut rewrite = Walk((sites.rewrite)(&hoist));
        for index in (first + shift)..block.stmts.len() {
            if handled[index - shift] {
                continue; // an arm rewrote this statement against its own site
            }
            rewrite.visit_stmt_mut(&mut block.stmts[index]);
        }
        error.or(rewrite.0.error)
    }
}

// ── postprocess_graph ───────────────────────────────────────────────

pub fn expand(mut func: ItemFn) -> syn::Result<TokenStream> {
    let mut collect = Walk(Collect::default());
    // `visit_block_mut` rather than `visit_item_fn_mut`: the latter would be
    // stopped by the walk's own `visit_item_mut` guard.
    collect.visit_block_mut(&mut func.block.clone());
    if let Some(err) = collect.0.error {
        return Err(err);
    }
    let brackets = collect.0.brackets;
    if brackets.is_empty() {
        return Err(syn::Error::new(
            func.sig.ident.span(),
            "#[gc_roots] found no `with_roots!` bracket to transform. \
             An attribute that transforms nothing still emits a frame, so it \
             costs a push and a pop to say nothing; remove it.",
        ));
    }

    let colors = allocate_colors(&brackets);
    let numcolors = colors.len();
    let named: BTreeSet<String> = brackets
        .iter()
        .flat_map(|bracket| bracket.locals.iter().map(Ident::to_string))
        .collect();
    let mut hoistable = hoistable(&func);
    hoistable.retain(|name| named.contains(name));

    let frame = format_ident!("__pyre_gc_frame", span = Span::call_site());
    let base = format_ident!("__pyre_gc_base", span = Span::call_site());

    // `move_pushes_earlier`'s output: one `gc_save_root` series on arrival at
    // the frame, standing in for the one every bracket would otherwise run.
    let prologue = |hoist: &BTreeSet<String>| -> Vec<Stmt> {
        let saves: Vec<TokenStream> = hoist
            .iter()
            .map(|name| {
                let local = format_ident!("{}", name, span = Span::call_site());
                let color = colors[name];
                quote!((#base + #color, #local))
            })
            .collect();
        let save = (!saves.is_empty()).then(|| quote!(#frame.save_run(&[#(#saves),*]);));
        syn::parse_quote! {
            let #frame = ::pyre_object::gc_roots::enter_roots_frame(#numcolors);
            let #base = ::pyre_object::gc_roots::RootScope::base(&#frame);
            #save
        }
    };
    let rewrite = |hoist: &BTreeSet<String>| Rewrite {
        colors: &colors,
        hoisted: hoist.clone(),
        frame: &frame,
        base: &base,
        error: None,
    };

    let mut in_scope = BTreeSet::new();
    for arg in &func.sig.inputs {
        if let FnArg::Typed(typed) = arg
            && let Pat::Ident(ident) = &*typed.pat
        {
            in_scope.insert(ident.ident.to_string());
        }
    }
    let sites = place::Sites {
        hoistable: &hoistable,
        prologue: &prologue,
        rewrite: &rewrite,
    };
    if let Some(err) = place::install(&mut func.block, &in_scope, &sites) {
        return Err(err);
    }

    Ok(quote!(#func))
}
