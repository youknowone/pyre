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
    // `PyList_CheckExact`, so a list subclass is refused with everything else
    // that is not a list.
    if !unsafe { is_exact_type(candidates, &LIST_TYPE) } {
        return Err(crate::PyError::type_error("candidates must be a list"));
    }
    let items = unsafe { w_list_items_copy_as_vec(candidates) };
    for element in &items {
        if !unsafe { is_str(*element) } {
            return Err(crate::PyError::type_error(
                "all elements in 'candidates' must be strings",
            ));
        }
    }
    // `_Py_CalculateSuggestions` gives up on a set this large before it reads
    // any name at all, so a candidate that has no UTF-8 encoding is never
    // reached and never reported here either.
    if items.len() >= crate::error::MAX_SUGGESTION_CANDIDATES {
        return Ok(w_none());
    }
    // `PyUnicode_AsUTF8AndSize` is what the search reads each name through,
    // and a lone surrogate has no UTF-8 encoding: the failure is the one the
    // strict encoder reports, not a name quietly left out of the ranking.
    // The wrong name is read first, which is the one order in which a
    // candidate equal to it cannot be the name that reports the failure.
    let wrong_name = crate::baseobjspace::str_utf8_w(item)?.to_string();
    let mut names = Vec::with_capacity(items.len());
    for element in items {
        names.push(crate::baseobjspace::str_utf8_w(element)?.to_string());
    }
    Ok(match crate::error::best_suggestion(&names, &wrong_name) {
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
