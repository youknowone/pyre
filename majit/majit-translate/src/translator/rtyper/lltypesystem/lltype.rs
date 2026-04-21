//! RPython `rpython/rtyper/lltypesystem/lltype.py`.
//!
//! This subset carries the function-pointer surface consumed by
//! `translator/simplify.py:get_graph`.

use std::hash::{Hash, Hasher};

use crate::flowspace::model::{ConcretetypePlaceholder, GraphKey, GraphRef, Hlvalue};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DelayedPointer;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FuncType {
    pub args: Vec<ConcretetypePlaceholder>,
    pub result: ConcretetypePlaceholder,
}

#[derive(Clone, Debug)]
pub struct _func {
    pub TYPE: FuncType,
    pub _name: String,
    pub graph: Option<usize>,
    pub _callable: Option<String>,
}

impl PartialEq for _func {
    fn eq(&self, other: &Self) -> bool {
        self.TYPE == other.TYPE
            && self._name == other._name
            && self._callable == other._callable
            && self.graph == other.graph
    }
}

impl Eq for _func {}

impl Hash for _func {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.TYPE.hash(state);
        self._name.hash(state);
        self._callable.hash(state);
        match &self.graph {
            Some(graph) => {
                true.hash(state);
                graph.hash(state);
            }
            None => false.hash(state),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct _ptr {
    pub _TYPE: FuncType,
    pub _obj0: Result<_func, DelayedPointer>,
}

impl _ptr {
    pub fn _obj(&self) -> Result<&_func, DelayedPointer> {
        self._obj0.as_ref().map_err(|_| DelayedPointer)
    }
}

pub fn functionptr(
    TYPE: FuncType,
    name: &str,
    graph: Option<usize>,
    _callable: Option<String>,
) -> _ptr {
    _ptr {
        _TYPE: TYPE.clone(),
        _obj0: Ok(_func {
            TYPE,
            _name: name.to_string(),
            graph,
            _callable,
        }),
    }
}

pub fn _getconcretetype(v: &Hlvalue) -> ConcretetypePlaceholder {
    match v {
        Hlvalue::Variable(v) => v.concretetype.unwrap_or(()),
        Hlvalue::Constant(c) => c.concretetype.unwrap_or(()),
    }
}

pub fn getfunctionptr(
    graph: &GraphRef,
    getconcretetype: fn(&Hlvalue) -> ConcretetypePlaceholder,
) -> _ptr {
    let graph_b = graph.borrow();
    let llinputs = graph_b.getargs().iter().map(getconcretetype).collect();
    let lloutput = getconcretetype(&graph_b.getreturnvar());
    let ft = FuncType {
        args: llinputs,
        result: lloutput,
    };
    let name = graph_b.name.clone();
    let callable = graph_b.func.as_ref().map(|func| func.name.clone());
    drop(graph_b);
    functionptr(ft, &name, Some(GraphKey::of(graph).as_usize()), callable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{Block, FunctionGraph};
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn functionptr_keeps_graph_on_funcobj() {
        let start = Rc::new(RefCell::new(Block::new(vec![])));
        let graph = Rc::new(RefCell::new(FunctionGraph::new("f", start)));
        let ptr = getfunctionptr(&graph, _getconcretetype);
        let funcobj = ptr._obj().unwrap();
        assert_eq!(funcobj.graph, Some(GraphKey::of(&graph).as_usize()));
    }

    #[test]
    fn getfunctionptr_calls_getconcretetype_for_args_and_result() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static CALLS: AtomicUsize = AtomicUsize::new(0);

        fn counting_getconcretetype(v: &Hlvalue) -> ConcretetypePlaceholder {
            let _ = v;
            CALLS.fetch_add(1, Ordering::Relaxed);
        }

        let start = Rc::new(RefCell::new(Block::new(vec![
            Hlvalue::Variable(crate::flowspace::model::Variable::new()),
            Hlvalue::Variable(crate::flowspace::model::Variable::new()),
        ])));
        let graph = Rc::new(RefCell::new(FunctionGraph::new("f", start)));
        CALLS.store(0, Ordering::Relaxed);

        let _ = getfunctionptr(&graph, counting_getconcretetype);

        assert_eq!(CALLS.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn delayed_pointer_raises_on_obj_access() {
        let ptr = _ptr {
            _TYPE: FuncType {
                args: vec![],
                result: (),
            },
            _obj0: Err(DelayedPointer),
        };
        assert_eq!(ptr._obj(), Err(DelayedPointer));
    }
}
