//! pyexpat module — minimal stub.
//!
//! The real C extension wraps the Expat non-validating XML parser.  pyre
//! ports only the module surface (`error`/`ExpatError`, `ParserCreate`,
//! the `errors` / `model` constant namespaces, `version_info`) so that
//! `xml.parsers.expat` and the `xml.*` / `xmlrpc` / `plistlib` wrappers
//! import.  Actual parsing raises `NotImplementedError`.

use pyre_object::*;

/// Handler attributes a fresh parser exposes (settable to a callback or
/// `None`).  `xmlparser` instances carry a `__dict__`, so assignment
/// persists and `xml.sax` / `xml.dom` can wire their callbacks.
const HANDLER_NAMES: &[&str] = &[
    "StartElementHandler",
    "EndElementHandler",
    "ProcessingInstructionHandler",
    "CharacterDataHandler",
    "UnparsedEntityDeclHandler",
    "NotationDeclHandler",
    "StartNamespaceDeclHandler",
    "EndNamespaceDeclHandler",
    "CommentHandler",
    "StartCdataSectionHandler",
    "EndCdataSectionHandler",
    "DefaultHandler",
    "DefaultHandlerExpand",
    "NotStandaloneHandler",
    "ExternalEntityRefHandler",
    "StartDoctypeDeclHandler",
    "EndDoctypeDeclHandler",
    "EntityDeclHandler",
    "XmlDeclHandler",
    "ElementDeclHandler",
    "AttlistDeclHandler",
    "SkippedEntityHandler",
];

mod xmlparser_class {
    use super::*;

    crate::py_class! {
        "xmlparser",
        methods: {
            fn Parse(
                self_obj: PyObjectRef,
                data: PyObjectRef,
                #[default(w_bool_from(false))] isfinal: PyObjectRef,
            ) -> Result<PyObjectRef, crate::PyError> {
                let _ = (self_obj, data, isfinal);
                Err(crate::PyError::not_implemented("pyexpat parsing is unavailable"))
            }
            fn ParseFile(
                self_obj: PyObjectRef,
                file: PyObjectRef,
            ) -> Result<PyObjectRef, crate::PyError> {
                let _ = (self_obj, file);
                Err(crate::PyError::not_implemented("pyexpat parsing is unavailable"))
            }
            fn SetBase(self_obj: PyObjectRef, base: PyObjectRef) -> PyObjectRef {
                let _ = (self_obj, base);
                w_none()
            }
            fn GetBase(self_obj: PyObjectRef) -> PyObjectRef {
                let _ = self_obj;
                w_none()
            }
            fn GetInputContext(self_obj: PyObjectRef) -> PyObjectRef {
                let _ = self_obj;
                w_none()
            }
            fn SetParamEntityParsing(self_obj: PyObjectRef, flag: PyObjectRef) -> PyObjectRef {
                let _ = (self_obj, flag);
                w_int_new(0)
            }
            fn UseForeignDTD(
                self_obj: PyObjectRef,
                #[default(w_bool_from(true))] flag: PyObjectRef,
            ) -> PyObjectRef {
                let _ = (self_obj, flag);
                w_none()
            }
            fn ExternalEntityParserCreate(
                self_obj: PyObjectRef,
                context: PyObjectRef,
                #[default(w_none())] encoding: PyObjectRef,
            ) -> PyObjectRef {
                let _ = (context, encoding);
                self_obj
            }
        }
    }
}

fn init_parser_slots(parser: PyObjectRef) {
    for h in HANDLER_NAMES {
        let _ = crate::baseobjspace::setattr_str(parser, h, w_none());
    }
    let set_int = |name: &str, v: i64| {
        let _ = crate::baseobjspace::setattr_str(parser, name, w_int_new(v));
    };
    let set_bool = |name: &str, v: bool| {
        let _ = crate::baseobjspace::setattr_str(parser, name, w_bool_from(v));
    };
    set_bool("buffer_text", false);
    set_int("buffer_size", 8192);
    set_int("buffer_used", 0);
    set_bool("ordered_attributes", false);
    set_bool("specified_attributes", false);
    set_int("ErrorCode", 0);
    set_int("ErrorLineNumber", 0);
    set_int("ErrorColumnNumber", 0);
    set_int("ErrorByteIndex", 0);
    set_int("CurrentLineNumber", 0);
    set_int("CurrentColumnNumber", 0);
    set_int("CurrentByteIndex", 0);
    let _ = crate::baseobjspace::setattr_str(parser, "intern", w_dict_new());
}

/// `ParserCreate(encoding=None, namespace_separator=None, intern=None)`.
fn parser_create(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let parser = w_instance_new(xmlparser_class::type_object());
    init_parser_slots(parser);
    Ok(parser)
}

/// `ErrorString(code)` — map an error code to its message via the `errors`
/// table.  Returns `None` for an unknown code (matching the C behaviour).
fn error_string(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let code = match args.first().copied() {
        Some(o) if unsafe { is_int(o) } => unsafe { w_int_get_value(o) },
        _ => return Ok(w_none()),
    };
    Ok(ERROR_TABLE
        .iter()
        .find(|(_, c)| *c == code)
        .map(|(msg, _)| w_str_new(msg))
        .unwrap_or_else(w_none))
}

/// `(XML_ERROR_NAME message, code)` from Expat's `XML_Error` enum.
const ERROR_TABLE: &[(&str, i64)] = &[
    ("out of memory", 1),
    ("syntax error", 2),
    ("no element found", 3),
    ("not well-formed (invalid token)", 4),
    ("unclosed token", 5),
    ("partial character", 6),
    ("mismatched tag", 7),
    ("duplicate attribute", 8),
    ("junk after document element", 9),
    ("illegal parameter entity reference", 10),
    ("undefined entity", 11),
    ("recursive entity reference", 12),
    ("asynchronous entity", 13),
    ("reference to invalid character number", 14),
    ("reference to binary entity", 15),
    ("reference to external entity in attribute", 16),
    ("XML or text declaration not at start of entity", 17),
    ("unknown encoding", 18),
    ("encoding specified in XML declaration is incorrect", 19),
    ("unclosed CDATA section", 20),
    ("error in processing external entity reference", 21),
    ("document is not standalone", 22),
    ("unexpected parser state - please send a bug report", 23),
    ("entity declared in parameter entity", 24),
    ("requested feature requires XML_DTD support in Expat", 25),
    ("cannot change setting once parsing has begun", 26),
    ("unbound prefix", 27),
    ("must not undeclare prefix", 28),
    ("incomplete markup in parameter entity", 29),
    ("XML declaration not well-formed", 30),
    ("text declaration not well-formed", 31),
    ("illegal character(s) in public id", 32),
    ("parser suspended", 33),
    ("parser not suspended", 34),
    ("parsing aborted", 35),
    ("parsing finished", 36),
    ("cannot suspend in external parameter entity", 37),
];

/// `(constant name, value)` from Expat's content-model enums.
const MODEL_CONSTANTS: &[(&str, i64)] = &[
    ("XML_CQUANT_NONE", 0),
    ("XML_CQUANT_OPT", 1),
    ("XML_CQUANT_REP", 2),
    ("XML_CQUANT_PLUS", 3),
    ("XML_CTYPE_EMPTY", 1),
    ("XML_CTYPE_ANY", 2),
    ("XML_CTYPE_MIXED", 3),
    ("XML_CTYPE_NAME", 4),
    ("XML_CTYPE_CHOICE", 5),
    ("XML_CTYPE_SEQ", 6),
];

/// `XML_ERROR_NAME -> message` pairs, in `XML_Error` enum order so each
/// name's index+1 is its code.
const ERROR_NAMES: &[&str] = &[
    "XML_ERROR_NONE",
    "XML_ERROR_NO_MEMORY",
    "XML_ERROR_SYNTAX",
    "XML_ERROR_NO_ELEMENTS",
    "XML_ERROR_INVALID_TOKEN",
    "XML_ERROR_UNCLOSED_TOKEN",
    "XML_ERROR_PARTIAL_CHAR",
    "XML_ERROR_TAG_MISMATCH",
    "XML_ERROR_DUPLICATE_ATTRIBUTE",
    "XML_ERROR_JUNK_AFTER_DOC_ELEMENT",
    "XML_ERROR_PARAM_ENTITY_REF",
    "XML_ERROR_UNDEFINED_ENTITY",
    "XML_ERROR_RECURSIVE_ENTITY_REF",
    "XML_ERROR_ASYNC_ENTITY",
    "XML_ERROR_BAD_CHAR_REF",
    "XML_ERROR_BINARY_ENTITY_REF",
    "XML_ERROR_ATTRIBUTE_EXTERNAL_ENTITY_REF",
    "XML_ERROR_MISPLACED_XML_PI",
    "XML_ERROR_UNKNOWN_ENCODING",
    "XML_ERROR_INCORRECT_ENCODING",
    "XML_ERROR_UNCLOSED_CDATA_SECTION",
    "XML_ERROR_EXTERNAL_ENTITY_HANDLING",
    "XML_ERROR_NOT_STANDALONE",
    "XML_ERROR_UNEXPECTED_STATE",
    "XML_ERROR_ENTITY_DECLARED_IN_PE",
    "XML_ERROR_FEATURE_REQUIRES_XML_DTD",
    "XML_ERROR_CANT_CHANGE_FEATURE_ONCE_PARSING",
    "XML_ERROR_UNBOUND_PREFIX",
    "XML_ERROR_UNDECLARING_PREFIX",
    "XML_ERROR_INCOMPLETE_PE",
    "XML_ERROR_XML_DECL",
    "XML_ERROR_TEXT_DECL",
    "XML_ERROR_PUBLICID",
    "XML_ERROR_SUSPENDED",
    "XML_ERROR_NOT_SUSPENDED",
    "XML_ERROR_ABORTED",
    "XML_ERROR_FINISHED",
    "XML_ERROR_SUSPEND_PE",
];

/// Build a `hasdict` namespace object used for the `model` / `errors`
/// submodules; constants are written as instance attributes.
fn make_namespace(name: &'static str) -> PyObjectRef {
    let tp = crate::typedef::make_builtin_type(name, |_| {});
    unsafe { typeobject::w_type_set_hasdict(tp, true) };
    let obj = w_instance_new(tp);
    let _ = crate::baseobjspace::setattr_str(obj, "__name__", w_str_new(name));
    obj
}

crate::py_module! {
    "pyexpat",
    interpleveldefs: {
        "EXPAT_VERSION"   => w_str_new("expat_2.6.4"),
        "native_encoding" => w_str_new("UTF-8"),
        "XMLParserType"   => xmlparser_class::type_object(),
        "version_info"    => w_tuple_new(vec![w_int_new(2), w_int_new(6), w_int_new(4)]),
    },
    exceptions: {
        "error" => crate::builtins::lookup_exc_class("Exception")
            .expect("Exception must be installed before pyexpat init"),
    },
    functions: {
        "ParserCreate" / * = parser_create,
        "ErrorString"  / 1 = error_string,
    },
    extra_init: |ns| {
        // `ExpatError` is an alias of `error` (pyexpat exposes both).
        if let Some(err) = crate::runtime_ops::dict_storage_get(ns, "error") {
            crate::dict_storage_store(ns, "ExpatError", err);
        }

        // model — content-model integer constants.
        let model = make_namespace("pyexpat.model");
        for (name, value) in MODEL_CONSTANTS {
            let _ = crate::baseobjspace::setattr_str(model, name, w_int_new(*value));
        }
        crate::dict_storage_store(ns, "model", model);

        // errors — XML_ERROR_* message strings plus the `codes`
        // (message -> code) and `messages` (code -> message) maps.
        let errors = make_namespace("pyexpat.errors");
        let codes = w_dict_new();
        let messages = w_dict_new();
        for (idx, name) in ERROR_NAMES.iter().enumerate() {
            // ERROR_NAMES[0] is XML_ERROR_NONE (no message); codes start at 1.
            if idx == 0 {
                continue;
            }
            let (msg, code) = ERROR_TABLE[idx - 1];
            let w_msg = w_str_new(msg);
            let _ = crate::baseobjspace::setattr_str(errors, name, w_msg);
            unsafe {
                w_dict_setitem_str(codes, msg, w_int_new(code));
                w_dict_store(messages, w_int_new(code), w_msg);
            }
        }
        let _ = crate::baseobjspace::setattr_str(errors, "codes", codes);
        let _ = crate::baseobjspace::setattr_str(errors, "messages", messages);
        crate::dict_storage_store(ns, "errors", errors);

        // features — list of (name, value) capability tuples.
        let features = w_list_new(vec![
            w_tuple_new(vec![w_str_new("sizeof(XML_Char)"), w_int_new(1)]),
            w_tuple_new(vec![w_str_new("sizeof(XML_LChar)"), w_int_new(1)]),
            w_tuple_new(vec![w_str_new("XML_DTD"), w_int_new(0)]),
            w_tuple_new(vec![w_str_new("XML_CONTEXT_BYTES"), w_int_new(1024)]),
            w_tuple_new(vec![w_str_new("XML_NS"), w_int_new(0)]),
        ]);
        crate::dict_storage_store(ns, "features", features);
    },
}
