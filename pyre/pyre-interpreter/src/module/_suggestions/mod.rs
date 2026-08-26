//! _suggestions module — `Modules/_suggestions.c`.
//!
//! Exposes the interpreter's misspelling search to Python.  `traceback.py`
//! reaches it from `_compute_suggestion_error` and `_find_keyword_typos`,
//! and falls back to `difflib.get_close_matches` when the import fails.  The
//! fallback ranks candidates by a different metric, so the reported
//! suggestion only matches while this module is importable.

use pyre_object::*;

fn generate_suggestions(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (Some(candidates), Some(item)) = (args.first().copied(), args.get(1).copied()) else {
        return Err(crate::PyError::type_error(
            "_generate_suggestions expected 2 arguments",
        ));
    };
    // The `unicode` converter runs while the arguments are parsed, so a
    // non-`str` name is reported ahead of the list test in the body.
    if !unsafe { is_str(item) } {
        return Err(crate::PyError::type_error(format!(
            "_generate_suggestions() argument 2 must be str, not {}",
            crate::error::type_name_of(item)
        )));
    }
    if !unsafe { is_exact_type(candidates, &LIST_TYPE) } {
        return Err(crate::PyError::type_error("candidates must be a list"));
    }
    let items = unsafe { w_list_items_copy_as_vec(candidates) };
    let mut names = Vec::with_capacity(items.len());
    for element in items {
        if !unsafe { is_str(element) } {
            return Err(crate::PyError::type_error(
                "all elements in 'candidates' must be strings",
            ));
        }
        // A name carrying a lone surrogate has no `&str` view, and the search
        // compares `char`s.  Drop it from the candidate set rather than
        // reading it as UTF-8.
        if let Some(name) = unsafe { w_str_get_value_opt(element) } {
            names.push(name.to_string());
        }
    }
    let Some(wrong_name) = (unsafe { w_str_get_value_opt(item) }) else {
        return Ok(w_none());
    };
    Ok(match crate::error::best_suggestion(&names, wrong_name) {
        Some(name) => w_str_new(&name),
        None => w_none(),
    })
}

crate::py_module! {
    "_suggestions",
    functions: {
        "_generate_suggestions" / 2 = generate_suggestions,
    },
}
