//! pyexpat module — Rust mini-parser adaptation.
//!
//! The real C extension wraps Expat. pyre deliberately keeps this as a small
//! Rust parser instead of embedding libexpat, with enough behavior for plistlib
//! and the stdlib XML wrappers that only use simple handler callbacks. Known
//! limits: callbacks are delayed until final input, input is UTF-8 only,
//! namespaces are not processed, and external entities are not loaded.

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

#[derive(Clone, Copy)]
struct XmlPos {
    index: usize,
    line: usize,
    col: usize,
}

struct MiniXmlParser<'a> {
    parser: PyObjectRef,
    input: &'a str,
    pos: XmlPos,
    stack: Vec<String>,
    root_closed: bool,
    char_buffer: String,
}

impl<'a> MiniXmlParser<'a> {
    fn new(parser: PyObjectRef, input: &'a str) -> Self {
        Self {
            parser,
            input,
            pos: XmlPos {
                index: 0,
                line: 1,
                col: 0,
            },
            stack: Vec::new(),
            root_closed: false,
            char_buffer: String::new(),
        }
    }

    fn parse(mut self, isfinal: bool) -> Result<(), crate::PyError> {
        self.skip_ws();
        if self.starts_with("<?xml") {
            self.parse_xml_decl()?;
        }
        loop {
            if self.eof() {
                break;
            }
            if self.root_closed {
                self.skip_ws();
                if self.eof() {
                    break;
                }
                if self.starts_with("<!--") {
                    self.parse_comment()?;
                    continue;
                }
                if self.starts_with("<?") {
                    self.parse_pi()?;
                    continue;
                }
                return self.fail("junk after document element");
            }
            if self.starts_with("<!--") {
                self.parse_comment()?;
            } else if self.starts_with("<?") {
                self.parse_pi()?;
            } else if self.starts_with("<!DOCTYPE") {
                self.parse_doctype()?;
            } else if self.starts_with("<![CDATA[") {
                self.parse_cdata()?;
            } else if self.starts_with("</") {
                self.parse_end_element()?;
            } else if self.starts_with("<") {
                self.parse_start_element()?;
            } else {
                self.parse_chardata()?;
            }
        }
        if isfinal && !self.stack.is_empty() {
            return self.fail("unclosed token");
        }
        self.flush_character_buffer()?;
        self.update_position_slots();
        Ok(())
    }

    fn eof(&self) -> bool {
        self.pos.index >= self.input.len()
    }

    fn rest(&self) -> &'a str {
        &self.input[self.pos.index..]
    }

    fn starts_with(&self, s: &str) -> bool {
        self.rest().starts_with(s)
    }

    fn peek_char(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn bump_char(&mut self) -> Option<char> {
        let ch = self.peek_char()?;
        self.pos.index += ch.len_utf8();
        if ch == '\n' {
            self.pos.line += 1;
            self.pos.col = 0;
        } else {
            self.pos.col += 1;
        }
        Some(ch)
    }

    fn consume(&mut self, s: &str) -> bool {
        if !self.starts_with(s) {
            return false;
        }
        for _ in s.chars() {
            self.bump_char();
        }
        true
    }

    fn expect(&mut self, s: &str) -> Result<(), crate::PyError> {
        if self.consume(s) {
            Ok(())
        } else {
            self.fail(&format!("expected {s}"))
        }
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek_char(), Some(' ' | '\t' | '\r' | '\n')) {
            self.bump_char();
        }
    }

    fn read_name(&mut self) -> Result<String, crate::PyError> {
        let mut name = String::new();
        while let Some(ch) = self.peek_char() {
            if ch.is_alphanumeric() || matches!(ch, '_' | '-' | ':' | '.') {
                name.push(ch);
                self.bump_char();
            } else {
                break;
            }
        }
        if name.is_empty() {
            self.fail("expected name")
        } else {
            Ok(name)
        }
    }

    fn read_quoted(&mut self) -> Result<String, crate::PyError> {
        let quote = match self.bump_char() {
            Some(q @ ('"' | '\'')) => q,
            _ => return self.fail("expected quoted string"),
        };
        let start = self.pos.index;
        while let Some(ch) = self.peek_char() {
            if ch == quote {
                let raw = &self.input[start..self.pos.index];
                self.bump_char();
                return expand_entities(raw).map_err(|m| self.make_error(&m));
            }
            self.bump_char();
        }
        self.fail("unterminated quoted string")
    }

    fn parse_xml_decl(&mut self) -> Result<(), crate::PyError> {
        self.expect("<?xml")?;
        let mut version = String::new();
        let mut encoding = w_none();
        let mut standalone = w_int_new(-1);
        loop {
            self.skip_ws();
            if self.consume("?>") {
                break;
            }
            let key = self.read_name()?;
            self.skip_ws();
            self.expect("=")?;
            self.skip_ws();
            let value = self.read_quoted()?;
            match key.as_str() {
                "version" => version = value,
                "encoding" => encoding = w_str_new(&value),
                "standalone" => {
                    standalone = w_int_new(if value == "yes" { 1 } else { 0 });
                }
                _ => {}
            }
        }
        self.call_handler(
            "XmlDeclHandler",
            &[w_str_new(&version), encoding, standalone],
        )
    }

    fn parse_comment(&mut self) -> Result<(), crate::PyError> {
        self.expect("<!--")?;
        let start = self.pos.index;
        let Some(rel) = self.rest().find("-->") else {
            return self.fail("unclosed token");
        };
        let text = self.input[start..start + rel].to_string();
        for _ in 0..rel + 3 {
            self.bump_char();
        }
        self.call_handler("CommentHandler", &[w_str_new(&text)])
    }

    fn parse_pi(&mut self) -> Result<(), crate::PyError> {
        self.expect("<?")?;
        let target = self.read_name()?;
        self.skip_ws();
        let start = self.pos.index;
        let Some(rel) = self.rest().find("?>") else {
            return self.fail("unclosed token");
        };
        let data = self.input[start..start + rel].trim().to_string();
        for _ in 0..rel + 2 {
            self.bump_char();
        }
        self.call_handler(
            "ProcessingInstructionHandler",
            &[w_str_new(&target), w_str_new(&data)],
        )
    }

    fn parse_cdata(&mut self) -> Result<(), crate::PyError> {
        self.expect("<![CDATA[")?;
        let start = self.pos.index;
        let Some(rel) = self.rest().find("]]>") else {
            return self.fail("unclosed CDATA section");
        };
        let text = self.input[start..start + rel].to_string();
        for _ in 0..rel + 3 {
            self.bump_char();
        }
        self.emit_character_data(&text)
    }

    fn parse_doctype(&mut self) -> Result<(), crate::PyError> {
        self.expect("<!DOCTYPE")?;
        self.skip_ws();
        let name = self.read_name()?;
        let mut sysid = w_none();
        let mut pubid = w_none();
        let mut has_internal_subset = false;
        self.skip_ws();
        if self.starts_with("PUBLIC") {
            self.expect("PUBLIC")?;
            self.skip_ws();
            pubid = w_str_new(&self.read_quoted()?);
            self.skip_ws();
            sysid = w_str_new(&self.read_quoted()?);
        } else if self.starts_with("SYSTEM") {
            self.expect("SYSTEM")?;
            self.skip_ws();
            sysid = w_str_new(&self.read_quoted()?);
        }
        self.skip_ws();
        if self.consume("[") {
            has_internal_subset = true;
            self.call_handler(
                "StartDoctypeDeclHandler",
                &[w_str_new(&name), sysid, pubid, w_int_new(1)],
            )?;
            self.parse_internal_subset()?;
        } else {
            self.call_handler(
                "StartDoctypeDeclHandler",
                &[w_str_new(&name), sysid, pubid, w_int_new(0)],
            )?;
        }
        self.skip_ws();
        self.expect(">")?;
        let _ = has_internal_subset;
        self.call_handler("EndDoctypeDeclHandler", &[])
    }

    fn parse_internal_subset(&mut self) -> Result<(), crate::PyError> {
        loop {
            self.skip_ws();
            if self.consume("]") {
                return Ok(());
            }
            if self.starts_with("<!ENTITY") {
                self.parse_entity_decl()?;
            } else if self.starts_with("<!--") {
                self.parse_comment()?;
            } else if self.starts_with("<?") {
                self.parse_pi()?;
            } else {
                self.skip_declaration()?;
            }
        }
    }

    fn parse_entity_decl(&mut self) -> Result<(), crate::PyError> {
        self.expect("<!ENTITY")?;
        self.skip_ws();
        let is_param = if self.consume("%") {
            self.skip_ws();
            1
        } else {
            0
        };
        let name = self.read_name()?;
        self.skip_ws();
        let mut value = w_none();
        let mut base = w_none();
        let mut sysid = w_none();
        let mut pubid = w_none();
        let mut notation = w_none();
        if matches!(self.peek_char(), Some('"' | '\'')) {
            value = w_str_new(&self.read_quoted()?);
        } else if self.starts_with("PUBLIC") {
            self.expect("PUBLIC")?;
            self.skip_ws();
            pubid = w_str_new(&self.read_quoted()?);
            self.skip_ws();
            sysid = w_str_new(&self.read_quoted()?);
        } else if self.starts_with("SYSTEM") {
            self.expect("SYSTEM")?;
            self.skip_ws();
            sysid = w_str_new(&self.read_quoted()?);
        }
        self.skip_ws();
        if self.starts_with("NDATA") {
            self.expect("NDATA")?;
            self.skip_ws();
            notation = w_str_new(&self.read_name()?);
        }
        self.skip_until_gt()?;
        let _ = &mut base;
        self.call_handler(
            "EntityDeclHandler",
            &[
                w_str_new(&name),
                w_int_new(is_param),
                value,
                base,
                sysid,
                pubid,
                notation,
            ],
        )
    }

    fn skip_declaration(&mut self) -> Result<(), crate::PyError> {
        if self.starts_with("<!") {
            self.skip_until_gt()
        } else {
            self.fail("syntax error")
        }
    }

    fn skip_until_gt(&mut self) -> Result<(), crate::PyError> {
        let mut quote: Option<char> = None;
        while let Some(ch) = self.bump_char() {
            if let Some(q) = quote {
                if ch == q {
                    quote = None;
                }
            } else if ch == '"' || ch == '\'' {
                quote = Some(ch);
            } else if ch == '>' {
                return Ok(());
            }
        }
        self.fail("unclosed token")
    }

    fn parse_start_element(&mut self) -> Result<(), crate::PyError> {
        self.expect("<")?;
        let name = self.read_name()?;
        let mut attrs: Vec<(String, String)> = Vec::new();
        loop {
            self.skip_ws();
            if self.consume("/>") {
                let w_attrs = self.convert_attributes(&attrs);
                self.call_handler("StartElementHandler", &[w_str_new(&name), w_attrs])?;
                self.call_handler("EndElementHandler", &[w_str_new(&name)])?;
                if self.stack.is_empty() {
                    self.root_closed = true;
                }
                return Ok(());
            }
            if self.consume(">") {
                self.stack.push(name.clone());
                let w_attrs = self.convert_attributes(&attrs);
                self.call_handler("StartElementHandler", &[w_str_new(&name), w_attrs])?;
                return Ok(());
            }
            let attr_name = self.read_name()?;
            if attrs.iter().any(|(existing, _)| existing == &attr_name) {
                return self.fail("duplicate attribute");
            }
            self.skip_ws();
            self.expect("=")?;
            self.skip_ws();
            let attr_value = self.read_quoted()?;
            attrs.push((attr_name, attr_value));
        }
    }

    fn parse_end_element(&mut self) -> Result<(), crate::PyError> {
        self.expect("</")?;
        let name = self.read_name()?;
        self.skip_ws();
        self.expect(">")?;
        match self.stack.pop() {
            Some(open) if open == name => {
                self.call_handler("EndElementHandler", &[w_str_new(&name)])?;
                if self.stack.is_empty() {
                    self.root_closed = true;
                }
                Ok(())
            }
            _ => self.fail("mismatched tag"),
        }
    }

    fn parse_chardata(&mut self) -> Result<(), crate::PyError> {
        let start = self.pos.index;
        while let Some(ch) = self.peek_char() {
            if ch == '<' {
                break;
            }
            self.bump_char();
        }
        let raw = &self.input[start..self.pos.index];
        if raw.is_empty() {
            return Ok(());
        }
        let text = expand_entities(raw).map_err(|m| self.make_error(&m))?;
        if self.stack.is_empty() && text.trim().is_empty() {
            return Ok(());
        }
        if text.is_empty() {
            Ok(())
        } else {
            self.emit_character_data(&text)
        }
    }

    fn convert_attributes(&self, attrs: &[(String, String)]) -> PyObjectRef {
        let ordered = crate::baseobjspace::getattr_str(self.parser, "ordered_attributes")
            .map(is_true_obj)
            .unwrap_or(false);
        if ordered {
            let mut items = Vec::with_capacity(attrs.len() * 2);
            for (name, value) in attrs {
                items.push(w_str_new(name));
                items.push(w_str_new(value));
            }
            w_list_new(items)
        } else {
            let w_attrs = w_dict_new();
            for (name, value) in attrs {
                unsafe { w_dict_setitem_str(w_attrs, name, w_str_new(value)) };
            }
            w_attrs
        }
    }

    fn emit_character_data(&mut self, text: &str) -> Result<(), crate::PyError> {
        let buffering = crate::baseobjspace::getattr_str(self.parser, "buffer_text")
            .map(is_true_obj)
            .unwrap_or(false);
        if buffering {
            self.char_buffer.push_str(text);
            crate::baseobjspace::setdictvalue(
                self.parser,
                "buffer_used",
                w_int_new(self.char_buffer.len() as i64),
            );
            Ok(())
        } else {
            self.call_handler_raw("CharacterDataHandler", &[w_str_new(text)])
        }
    }

    fn flush_character_buffer(&mut self) -> Result<(), crate::PyError> {
        if self.char_buffer.is_empty() {
            return Ok(());
        }
        let text = std::mem::take(&mut self.char_buffer);
        crate::baseobjspace::setdictvalue(self.parser, "buffer_used", w_int_new(0));
        self.call_handler_raw("CharacterDataHandler", &[w_str_new(&text)])
    }

    fn call_handler(&mut self, name: &str, args: &[PyObjectRef]) -> Result<(), crate::PyError> {
        if name != "CharacterDataHandler" {
            self.flush_character_buffer()?;
        }
        self.call_handler_raw(name, args)
    }

    fn call_handler_raw(&self, name: &str, args: &[PyObjectRef]) -> Result<(), crate::PyError> {
        let Ok(handler) = crate::baseobjspace::getattr_str(self.parser, name) else {
            return Ok(());
        };
        if handler.is_null() || unsafe { is_none(handler) } {
            return Ok(());
        }
        crate::call::call_function_impl_result(handler, args)?;
        Ok(())
    }

    fn fail<T>(&self, msg: &str) -> Result<T, crate::PyError> {
        Err(self.make_error(msg))
    }

    fn make_error(&self, msg: &str) -> crate::PyError {
        let code = error_code_for_message(msg);
        crate::baseobjspace::setdictvalue(
            self.parser,
            "ErrorLineNumber",
            w_int_new(self.pos.line as i64),
        );
        crate::baseobjspace::setdictvalue(
            self.parser,
            "ErrorColumnNumber",
            w_int_new(self.pos.col as i64),
        );
        crate::baseobjspace::setdictvalue(
            self.parser,
            "ErrorByteIndex",
            w_int_new(self.pos.index as i64),
        );
        crate::baseobjspace::setdictvalue(self.parser, "ErrorCode", w_int_new(code));
        pyexpat_error(
            format!("{msg}: line {}, column {}", self.pos.line, self.pos.col),
            code,
            self.pos.line as i64,
            self.pos.col as i64,
        )
    }

    fn update_position_slots(&self) {
        crate::baseobjspace::setdictvalue(
            self.parser,
            "CurrentLineNumber",
            w_int_new(self.pos.line as i64),
        );
        crate::baseobjspace::setdictvalue(
            self.parser,
            "CurrentColumnNumber",
            w_int_new(self.pos.col as i64),
        );
        crate::baseobjspace::setdictvalue(
            self.parser,
            "CurrentByteIndex",
            w_int_new(self.pos.index as i64),
        );
    }
}

fn expand_entities(raw: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut rest = raw;
    while let Some(pos) = rest.find('&') {
        out.push_str(&rest[..pos]);
        let after = &rest[pos + 1..];
        let Some(end) = after.find(';') else {
            return Err("undefined entity".to_string());
        };
        let ent = &after[..end];
        match ent {
            "amp" => out.push('&'),
            "lt" => out.push('<'),
            "gt" => out.push('>'),
            "quot" => out.push('"'),
            "apos" => out.push('\''),
            _ if ent.starts_with("#x") || ent.starts_with("#X") => {
                let code = u32::from_str_radix(&ent[2..], 16)
                    .map_err(|_| "reference to invalid character number".to_string())?;
                let ch = char::from_u32(code)
                    .ok_or_else(|| "reference to invalid character number".to_string())?;
                out.push(ch);
            }
            _ if ent.starts_with('#') => {
                let code = ent[1..]
                    .parse::<u32>()
                    .map_err(|_| "reference to invalid character number".to_string())?;
                let ch = char::from_u32(code)
                    .ok_or_else(|| "reference to invalid character number".to_string())?;
                out.push(ch);
            }
            _ => return Err("undefined entity".to_string()),
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

fn object_to_xml_string(obj: PyObjectRef) -> Result<String, crate::PyError> {
    unsafe {
        if is_str(obj) {
            Ok(w_str_get_value(obj).to_string())
        } else if pyre_object::bytesobject::is_bytes_like(obj) {
            String::from_utf8(pyre_object::bytesobject::bytes_like_data(obj).to_vec())
                .map_err(|_| crate::PyError::value_error("pyexpat only supports UTF-8 input"))
        } else {
            Err(crate::PyError::type_error(
                "Parse() argument must be str or bytes",
            ))
        }
    }
}

fn is_true_obj(obj: PyObjectRef) -> bool {
    unsafe {
        if obj.is_null() || is_none(obj) {
            false
        } else if is_bool(obj) {
            w_bool_get_value(obj)
        } else if is_int(obj) {
            w_int_get_value(obj) != 0
        } else {
            true
        }
    }
}

fn parser_pending(parser: PyObjectRef) -> String {
    match crate::baseobjspace::getattr_str(parser, "_pyre_pending_xml") {
        Ok(obj) if unsafe { is_str(obj) } => unsafe { w_str_get_value(obj) }.to_string(),
        _ => String::new(),
    }
}

fn set_parser_pending(parser: PyObjectRef, pending: &str) {
    crate::baseobjspace::setdictvalue(parser, "_pyre_pending_xml", w_str_new(pending));
}

fn parse_impl(
    parser: PyObjectRef,
    data: PyObjectRef,
    isfinal: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let mut input = parser_pending(parser);
    input.push_str(&object_to_xml_string(data)?);
    if !is_true_obj(isfinal) {
        set_parser_pending(parser, &input);
        return Ok(w_int_new(1));
    }
    set_parser_pending(parser, "");
    MiniXmlParser::new(parser, &input).parse(true)?;
    Ok(w_int_new(1))
}

fn pyexpat_error(msg: String, code: i64, lineno: i64, offset: i64) -> crate::PyError {
    let mut err = crate::PyError::value_error(msg.clone());
    if let Some(cls) = crate::builtins::lookup_exc_class("pyexpat.error") {
        let args = [cls, w_str_new(&msg)];
        if let Ok(exc) = crate::builtins::exc_exception_new(&args) {
            crate::baseobjspace::setdictvalue(exc, "code", w_int_new(code));
            crate::baseobjspace::setdictvalue(exc, "lineno", w_int_new(lineno));
            crate::baseobjspace::setdictvalue(exc, "offset", w_int_new(offset));
            err.exc_object = exc;
        }
    }
    err
}

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
                parse_impl(self_obj, data, isfinal)
            }
            fn ParseFile(
                self_obj: PyObjectRef,
                file: PyObjectRef,
            ) -> Result<PyObjectRef, crate::PyError> {
                let read = crate::baseobjspace::getattr_str(file, "read")?;
                let mut result = w_int_new(1);
                loop {
                    let data = crate::call::call_function_impl_result(read, &[w_int_new(2048)])?;
                    let chunk = object_to_xml_string(data)?;
                    let eof = chunk.is_empty();
                    result = parse_impl(self_obj, w_str_new(&chunk), w_bool_from(eof))?;
                    if eof {
                        return Ok(result);
                    }
                }
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
        crate::baseobjspace::setdictvalue(parser, h, w_none());
    }
    let set_int = |name: &str, v: i64| {
        crate::baseobjspace::setdictvalue(parser, name, w_int_new(v));
    };
    let set_bool = |name: &str, v: bool| {
        crate::baseobjspace::setdictvalue(parser, name, w_bool_from(v));
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
    crate::baseobjspace::setdictvalue(parser, "intern", w_dict_new());
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

fn error_code_for_message(msg: &str) -> i64 {
    ERROR_TABLE
        .iter()
        .find(|(known, _)| *known == msg)
        .map(|(_, code)| *code)
        .unwrap_or(2)
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
    crate::baseobjspace::setdictvalue(obj, "__name__", w_str_new(name));
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
            crate::baseobjspace::setdictvalue(model, name, w_int_new(*value));
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
            crate::baseobjspace::setdictvalue(errors, name, w_msg);
            unsafe {
                w_dict_setitem_str(codes, msg, w_int_new(code));
                w_dict_store(messages, w_int_new(code), w_msg);
            }
        }
        crate::baseobjspace::setdictvalue(errors, "codes", codes);
        crate::baseobjspace::setdictvalue(errors, "messages", messages);
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
