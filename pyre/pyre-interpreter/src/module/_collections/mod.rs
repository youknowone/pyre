//! _collections module — PyPy: pypy/module/_collections/
//!
//! Provides the C-accelerated deque/defaultdict/OrderedDict types.  Our
//! stubs are backed by lists/dicts, which is correct semantically but
//! not performant.

crate::pyre_module_init!(interp_collections);
