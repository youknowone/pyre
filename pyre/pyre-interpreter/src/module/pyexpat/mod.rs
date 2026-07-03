//! pyexpat module — Rust mini-parser adaptation.
//!
//! The real C extension wraps Expat. pyre deliberately keeps this as a small
//! Rust parser instead of embedding libexpat, with enough behavior for plistlib
//! and the stdlib XML wrappers that only use simple handler callbacks. Known
//! limits: the parser is intentionally incomplete and does not load external
//! resources itself.

use pyre_object::*;
use std::collections::HashMap;

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
    isfinal: bool,
    suppress_until: usize,
    suppress_current: bool,
    pos: XmlPos,
    stack: Vec<String>,
    ns_stack: Vec<Vec<String>>,
    namespaces: Vec<(String, String)>,
    internal_entities: HashMap<String, String>,
    external_entities: HashMap<String, (String, Option<String>)>,
    root_closed: bool,
    char_buffer: String,
}

impl<'a> MiniXmlParser<'a> {
    fn new(parser: PyObjectRef, input: &'a str, isfinal: bool, suppress_until: usize) -> Self {
        Self {
            parser,
            input,
            isfinal,
            suppress_until,
            suppress_current: false,
            pos: XmlPos {
                index: 0,
                line: 1,
                col: 0,
            },
            stack: Vec::new(),
            ns_stack: Vec::new(),
            namespaces: vec![(
                "xml".to_string(),
                "http://www.w3.org/XML/1998/namespace".to_string(),
            )],
            internal_entities: HashMap::new(),
            external_entities: HashMap::new(),
            root_closed: false,
            char_buffer: String::new(),
        }
    }

    fn parse(mut self) -> Result<usize, crate::PyError> {
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
        if self.isfinal && self.pos.index == 0 && !self.input.is_empty() {
            return self.fail("unclosed token");
        }
        if self.isfinal && !self.stack.is_empty() {
            return self.fail("unclosed token");
        }
        self.flush_character_buffer()?;
        self.update_position_slots();
        Ok(self.pos.index)
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
                return self
                    .expand_entities(raw, false)
                    .map_err(|m| self.make_error(&m));
            }
            self.bump_char();
        }
        self.fail("unterminated quoted string")
    }

    fn parse_xml_decl(&mut self) -> Result<(), crate::PyError> {
        let event_pos = self.pos;
        self.expect("<?xml")?;
        let mut version = String::new();
        let mut encoding = w_none();
        let mut standalone = w_int_new(-1);
        loop {
            self.skip_ws();
            if self.consume("?>") {
                break;
            }
            let key = self
                .read_name()
                .map_err(|_| self.make_error("XML declaration not well-formed"))?;
            self.skip_ws();
            self.expect("=")
                .map_err(|_| self.make_error("XML declaration not well-formed"))?;
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
        self.set_event_position(event_pos);
        if let Some(enc) = unsafe {
            if is_none(encoding) {
                None
            } else {
                Some(w_str_get_value(encoding).to_string())
            }
        } {
            validate_decl_encoding(&enc).map_err(|m| self.make_error(m))?;
        }
        self.call_handler(
            "XmlDeclHandler",
            &[w_str_new(&version), encoding, standalone],
        )?;
        if unsafe { is_int(standalone) && w_int_get_value(standalone) == 0 } {
            crate::baseobjspace::setdictvalue(
                self.parser,
                "_pyre_not_standalone_pending",
                w_bool_from(true),
            );
        }
        Ok(())
    }

    fn parse_comment(&mut self) -> Result<(), crate::PyError> {
        let event_pos = self.pos;
        self.expect("<!--")?;
        let start = self.pos.index;
        let Some(rel) = self.rest().find("-->") else {
            return if self.isfinal {
                self.fail("unclosed token")
            } else {
                Ok(())
            };
        };
        let text = self.input[start..start + rel].to_string();
        for _ in 0..rel + 3 {
            self.bump_char();
        }
        self.set_event_position(event_pos);
        self.call_handler("CommentHandler", &[w_str_new(&text)])
    }

    fn parse_pi(&mut self) -> Result<(), crate::PyError> {
        let event_pos = self.pos;
        self.expect("<?")?;
        let target = self.read_name()?;
        self.skip_ws();
        let start = self.pos.index;
        let Some(rel) = self.rest().find("?>") else {
            return if self.isfinal {
                self.fail("unclosed token")
            } else {
                Ok(())
            };
        };
        let data = self.input[start..start + rel].trim().to_string();
        for _ in 0..rel + 2 {
            self.bump_char();
        }
        self.set_event_position(event_pos);
        self.call_handler(
            "ProcessingInstructionHandler",
            &[w_str_new(&target), w_str_new(&data)],
        )
    }

    fn parse_cdata(&mut self) -> Result<(), crate::PyError> {
        let event_pos = self.pos;
        self.expect("<![CDATA[")?;
        let start = self.pos.index;
        let Some(rel) = self.rest().find("]]>") else {
            return if self.isfinal {
                self.fail("unclosed CDATA section")
            } else {
                Ok(())
            };
        };
        let text = self.input[start..start + rel].to_string();
        self.set_event_position(event_pos);
        self.flush_character_buffer()?;
        self.call_handler_raw("StartCdataSectionHandler", &[])?;
        for _ in 0..rel + 3 {
            self.bump_char();
        }
        self.emit_character_data(&text)?;
        self.flush_character_buffer()?;
        self.call_handler_raw("EndCdataSectionHandler", &[])
    }

    fn parse_doctype(&mut self) -> Result<(), crate::PyError> {
        let event_pos = self.pos;
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
        if crate::baseobjspace::getattr_str(self.parser, "_pyre_not_standalone_pending")
            .map(is_true_obj)
            .unwrap_or(false)
        {
            self.call_not_standalone()?;
            crate::baseobjspace::setdictvalue(
                self.parser,
                "_pyre_not_standalone_pending",
                w_bool_from(false),
            );
        }
        if self.consume("[") {
            has_internal_subset = true;
            self.set_event_position(event_pos);
            self.call_handler(
                "StartDoctypeDeclHandler",
                &[w_str_new(&name), sysid, pubid, w_int_new(1)],
            )?;
            self.parse_internal_subset()?;
        } else {
            self.set_event_position(event_pos);
            self.call_handler(
                "StartDoctypeDeclHandler",
                &[w_str_new(&name), sysid, pubid, w_int_new(0)],
            )?;
        }
        self.skip_ws();
        self.expect(">")?;
        self.maybe_external_dtd(pubid, sysid)?;
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
            } else if self.starts_with("%") {
                self.skip_parameter_entity_ref()?;
            } else if self.starts_with("<!ELEMENT") {
                self.parse_element_decl()?;
            } else if self.starts_with("<!ATTLIST") {
                self.parse_attlist_decl()?;
            } else if self.starts_with("<!NOTATION") {
                self.parse_notation_decl()?;
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
        let event_pos = self.pos;
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
            let v = self.read_quoted()?;
            self.internal_entities.insert(name.clone(), v.clone());
            value = w_str_new(&v);
        } else if self.starts_with("PUBLIC") {
            self.expect("PUBLIC")?;
            self.skip_ws();
            pubid = w_str_new(&self.read_quoted()?);
            self.skip_ws();
            let system = self.read_quoted()?;
            self.external_entities.insert(
                name.clone(),
                (
                    system.clone(),
                    Some(unsafe { w_str_get_value(pubid) }.to_string()),
                ),
            );
            sysid = w_str_new(&system);
        } else if self.starts_with("SYSTEM") {
            self.expect("SYSTEM")?;
            self.skip_ws();
            let system = self.read_quoted()?;
            self.external_entities
                .insert(name.clone(), (system.clone(), None));
            sysid = w_str_new(&system);
        }
        self.skip_ws();
        if self.starts_with("NDATA") {
            self.expect("NDATA")?;
            self.skip_ws();
            notation = w_str_new(&self.read_name()?);
        }
        self.skip_until_gt()?;
        let _ = &mut base;
        self.set_event_position(event_pos);
        if !unsafe { is_none(notation) } {
            self.call_handler(
                "UnparsedEntityDeclHandler",
                &[w_str_new(&name), base, sysid, pubid, notation],
            )?;
            return Ok(());
        }
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

    fn skip_parameter_entity_ref(&mut self) -> Result<(), crate::PyError> {
        self.expect("%")?;
        let _name = self.read_name()?;
        self.expect(";")?;
        self.call_not_standalone()
    }

    fn parse_element_decl(&mut self) -> Result<(), crate::PyError> {
        let event_pos = self.pos;
        self.expect("<!ELEMENT")?;
        self.skip_ws();
        let name = self.read_name()?;
        let mut depth = 0usize;
        while let Some(ch) = self.bump_char() {
            match ch {
                '(' => {
                    depth += 1;
                    if depth > 100_000 {
                        return Err(crate::PyError::runtime_error(
                            "maximum recursion depth exceeded",
                        ));
                    }
                }
                ')' => depth = depth.saturating_sub(1),
                '>' => break,
                _ => {}
            }
        }
        let model = w_tuple_new(vec![
            w_int_new(2),
            w_int_new(0),
            w_none(),
            w_tuple_new(vec![]),
        ]);
        self.set_event_position(event_pos);
        self.call_handler("ElementDeclHandler", &[w_str_new(&name), model])
    }

    fn parse_attlist_decl(&mut self) -> Result<(), crate::PyError> {
        let event_pos = self.pos;
        self.expect("<!ATTLIST")?;
        self.skip_ws();
        let elem = self.read_name()?;
        loop {
            self.skip_ws();
            if self.consume(">") {
                return Ok(());
            }
            let attr = self.read_name()?;
            self.skip_ws();
            let kind = self.read_name()?;
            self.skip_ws();
            let required = if self.consume("#REQUIRED") {
                1
            } else if self.consume("#IMPLIED") {
                0
            } else {
                if matches!(self.peek_char(), Some('"' | '\'')) {
                    let _ = self.read_quoted()?;
                } else {
                    let _ = self.read_name()?;
                }
                0
            };
            self.set_event_position(event_pos);
            self.call_handler(
                "AttlistDeclHandler",
                &[
                    w_str_new(&elem),
                    w_str_new(&attr),
                    w_str_new(&kind),
                    w_none(),
                    w_int_new(required),
                ],
            )?;
        }
    }

    fn parse_notation_decl(&mut self) -> Result<(), crate::PyError> {
        let event_pos = self.pos;
        self.expect("<!NOTATION")?;
        self.skip_ws();
        let name = self.read_name()?;
        self.skip_ws();
        let mut sysid = w_none();
        let mut pubid = w_none();
        if self.starts_with("PUBLIC") {
            self.expect("PUBLIC")?;
            self.skip_ws();
            pubid = w_str_new(&self.read_quoted()?);
            self.skip_ws();
            if matches!(self.peek_char(), Some('"' | '\'')) {
                sysid = w_str_new(&self.read_quoted()?);
            }
        } else if self.starts_with("SYSTEM") {
            self.expect("SYSTEM")?;
            self.skip_ws();
            sysid = w_str_new(&self.read_quoted()?);
        }
        self.skip_until_gt()?;
        self.set_event_position(event_pos);
        self.call_handler(
            "NotationDeclHandler",
            &[w_str_new(&name), w_none(), sysid, pubid],
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
        let event_pos = self.pos;
        self.expect("<")?;
        if self.eof() {
            return self.fail("unclosed token");
        }
        let raw_name = self.read_name()?;
        let mut attrs: Vec<(String, String)> = Vec::new();
        let mut ns_declared: Vec<String> = Vec::new();
        loop {
            self.skip_ws();
            if self.consume("/>") {
                self.apply_namespace_decls(&attrs, &mut ns_declared)?;
                let name = self.expand_name(&raw_name, false)?;
                let expanded_attrs = self.expand_attributes(&attrs)?;
                let w_attrs = self.convert_attributes(&expanded_attrs);
                self.set_event_position(event_pos);
                self.call_handler("StartElementHandler", &[self.intern_string(&name), w_attrs])?;
                self.set_event_position(self.pos);
                self.call_handler("EndElementHandler", &[self.intern_string(&name)])?;
                self.end_namespace_scope(ns_declared)?;
                if self.stack.is_empty() {
                    self.root_closed = true;
                }
                return Ok(());
            }
            if self.consume(">") {
                self.apply_namespace_decls(&attrs, &mut ns_declared)?;
                let name = self.expand_name(&raw_name, false)?;
                let expanded_attrs = self.expand_attributes(&attrs)?;
                self.stack.push(name.clone());
                self.ns_stack.push(ns_declared);
                let w_attrs = self.convert_attributes(&expanded_attrs);
                self.set_event_position(event_pos);
                self.call_handler("StartElementHandler", &[self.intern_string(&name), w_attrs])?;
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
        let event_pos = self.pos;
        self.expect("</")?;
        let raw_name = self.read_name()?;
        self.skip_ws();
        self.expect(">")?;
        let name = self.expand_name(&raw_name, false)?;
        match self.stack.pop() {
            Some(open) if open == name => {
                self.set_event_position(event_pos);
                self.call_handler("EndElementHandler", &[self.intern_string(&name)])?;
                if let Some(declared) = self.ns_stack.pop() {
                    self.end_namespace_scope(declared)?;
                }
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
            if ch == '\0' {
                return self.fail("unclosed token");
            }
            self.bump_char();
        }
        let raw = &self.input[start..self.pos.index];
        if raw.is_empty() {
            return Ok(());
        }
        self.set_event_position(XmlPos {
            index: start,
            line: self.pos.line,
            col: self.pos.col,
        });
        let text = self
            .expand_entities(raw, true)
            .map_err(|m| self.make_error(&m))?;
        if self.stack.is_empty() && text.trim().is_empty() {
            return Ok(());
        }
        if text.is_empty() {
            Ok(())
        } else {
            self.emit_character_data_split(&text)
        }
    }

    fn convert_attributes(&self, attrs: &[(String, String)]) -> PyObjectRef {
        let ordered = crate::baseobjspace::getattr_str(self.parser, "ordered_attributes")
            .map(is_true_obj)
            .unwrap_or(false);
        if ordered {
            let mut items = Vec::with_capacity(attrs.len() * 2);
            for (name, value) in attrs {
                items.push(self.intern_string(name));
                items.push(w_str_new(value));
            }
            w_list_new(items)
        } else {
            let w_attrs = w_dict_new();
            for (name, value) in attrs {
                unsafe { w_dict_store(w_attrs, self.intern_string(name), w_str_new(value)) };
            }
            w_attrs
        }
    }

    fn emit_character_data(&mut self, text: &str) -> Result<(), crate::PyError> {
        let buffering = crate::baseobjspace::getattr_str(self.parser, "buffer_text")
            .map(is_true_obj)
            .unwrap_or(false);
        if buffering {
            let size = get_parser_int(self.parser, "buffer_size", 8192).max(1) as usize;
            if self.char_buffer.len() + text.len() > size {
                self.flush_character_buffer()?;
            }
            if text.len() > size {
                return self.call_handler_raw("CharacterDataHandler", &[w_str_new(text)]);
            }
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

    fn emit_character_data_split(&mut self, text: &str) -> Result<(), crate::PyError> {
        let Some(first_non_ws) = text.find(|ch: char| !matches!(ch, ' ' | '\t' | '\r' | '\n'))
        else {
            return self.emit_character_data(text);
        };
        let (last_non_ws, last_len) = text
            .char_indices()
            .rev()
            .find(|(_, ch)| !matches!(ch, ' ' | '\t' | '\r' | '\n'))
            .map(|(idx, ch)| (idx, ch.len_utf8()))
            .unwrap();
        let content_end = last_non_ws + last_len;
        if first_non_ws > 0 {
            self.emit_character_data(&text[..first_non_ws])?;
        }
        self.emit_character_data(&text[first_non_ws..content_end])?;
        if content_end < text.len() {
            self.emit_character_data(&text[content_end..])?;
        }
        Ok(())
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
        if name != "CharacterDataHandler" && self.handler_is_set(name) {
            self.flush_character_buffer()?;
        }
        self.call_handler_raw(name, args)
    }

    fn call_handler_raw(&self, name: &str, args: &[PyObjectRef]) -> Result<(), crate::PyError> {
        if self.suppress_current {
            return Ok(());
        }
        let Ok(handler) = crate::baseobjspace::getattr_str(self.parser, name) else {
            return Ok(());
        };
        if handler.is_null() || unsafe { is_none(handler) } {
            return Ok(());
        }
        crate::call::call_function_impl_result(handler, args)?;
        Ok(())
    }

    fn call_not_standalone(&self) -> Result<(), crate::PyError> {
        if self.suppress_current {
            return Ok(());
        }
        let Ok(handler) = crate::baseobjspace::getattr_str(self.parser, "NotStandaloneHandler")
        else {
            return Ok(());
        };
        if handler.is_null() || unsafe { is_none(handler) } {
            return Ok(());
        }
        let ret = crate::call::call_function_impl_result(handler, &[])?;
        if unsafe { !(is_int(ret) || is_bool(ret)) } {
            return Err(crate::PyError::type_error(
                "NotStandaloneHandler must return an integer",
            ));
        }
        Ok(())
    }

    fn handler_is_set(&self, name: &str) -> bool {
        match crate::baseobjspace::getattr_str(self.parser, name) {
            Ok(handler) => !(handler.is_null() || unsafe { is_none(handler) }),
            Err(_) => false,
        }
    }

    fn set_event_position(&mut self, pos: XmlPos) {
        self.suppress_current = pos.index < self.suppress_until;
        crate::baseobjspace::setdictvalue(
            self.parser,
            "CurrentLineNumber",
            w_int_new(pos.line as i64),
        );
        crate::baseobjspace::setdictvalue(
            self.parser,
            "CurrentColumnNumber",
            w_int_new(pos.col as i64),
        );
        crate::baseobjspace::setdictvalue(
            self.parser,
            "CurrentByteIndex",
            w_int_new(pos.index as i64),
        );
    }

    fn namespace_separator(&self) -> Option<String> {
        match crate::baseobjspace::getattr_str(self.parser, "_pyre_namespace_separator") {
            Ok(obj) if unsafe { is_str(obj) } => Some(unsafe { w_str_get_value(obj) }.to_string()),
            _ => None,
        }
    }

    fn namespace_prefixes(&self) -> bool {
        crate::baseobjspace::getattr_str(self.parser, "namespace_prefixes")
            .map(is_true_obj)
            .unwrap_or(false)
    }

    fn apply_namespace_decls(
        &mut self,
        attrs: &[(String, String)],
        declared: &mut Vec<String>,
    ) -> Result<(), crate::PyError> {
        if self.namespace_separator().is_none() {
            return Ok(());
        }
        for (name, value) in attrs {
            let prefix = if name == "xmlns" {
                Some("")
            } else {
                name.strip_prefix("xmlns:")
            };
            if let Some(prefix) = prefix {
                let prefix = prefix.to_string();
                declared.push(prefix.clone());
                self.namespaces.push((prefix.clone(), value.clone()));
                let w_prefix = if prefix.is_empty() {
                    w_none()
                } else {
                    w_str_new(&prefix)
                };
                self.call_handler("StartNamespaceDeclHandler", &[w_prefix, w_str_new(value)])?;
            }
        }
        Ok(())
    }

    fn end_namespace_scope(&mut self, declared: Vec<String>) -> Result<(), crate::PyError> {
        for prefix in declared.into_iter().rev() {
            self.namespaces.pop();
            let w_prefix = if prefix.is_empty() {
                w_none()
            } else {
                w_str_new(&prefix)
            };
            self.call_handler("EndNamespaceDeclHandler", &[w_prefix])?;
        }
        Ok(())
    }

    fn namespace_uri(&self, prefix: &str) -> Option<&str> {
        self.namespaces
            .iter()
            .rev()
            .find(|(p, _)| p == prefix)
            .map(|(_, uri)| uri.as_str())
    }

    fn expand_name(&self, name: &str, is_attr: bool) -> Result<String, crate::PyError> {
        let Some(sep) = self.namespace_separator() else {
            return Ok(name.to_string());
        };
        if let Some((prefix, local)) = name.split_once(':') {
            let Some(uri) = self.namespace_uri(prefix) else {
                return self.fail("unbound prefix");
            };
            if self.namespace_prefixes() {
                Ok(format!("{uri}{sep}{local}{sep}{prefix}"))
            } else {
                Ok(format!("{uri}{sep}{local}"))
            }
        } else if !is_attr {
            if let Some(uri) = self.namespace_uri("") {
                Ok(format!("{uri}{sep}{name}"))
            } else {
                Ok(name.to_string())
            }
        } else {
            Ok(name.to_string())
        }
    }

    fn expand_attributes(
        &self,
        attrs: &[(String, String)],
    ) -> Result<Vec<(String, String)>, crate::PyError> {
        let mut out = Vec::new();
        for (name, value) in attrs {
            if self.namespace_separator().is_some()
                && (name == "xmlns" || name.starts_with("xmlns:"))
            {
                continue;
            }
            let expanded = self.expand_name(name, true)?;
            if out.iter().any(|(existing, _)| existing == &expanded) {
                return self.fail("duplicate attribute");
            }
            out.push((expanded, value.clone()));
        }
        Ok(out)
    }

    fn expand_entities(&mut self, raw: &str, content: bool) -> Result<String, String> {
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
                _ if self.internal_entities.contains_key(ent) => {
                    out.push_str(&self.internal_entities[ent].clone());
                }
                _ if content && self.external_entities.contains_key(ent) => {
                    let (sysid, pubid) = self.external_entities[ent].clone();
                    self.handle_external_entity(ent, &sysid, pubid.as_deref())
                        .map_err(|_| "error in processing external entity reference".to_string())?;
                }
                _ if content => {
                    self.call_handler("SkippedEntityHandler", &[w_str_new(ent), w_int_new(0)])
                        .map_err(|_| "undefined entity".to_string())?;
                }
                _ => return Err("undefined entity".to_string()),
            }
            rest = &after[end + 1..];
        }
        out.push_str(rest);
        Ok(out)
    }

    fn handle_external_entity(
        &self,
        context: &str,
        sysid: &str,
        pubid: Option<&str>,
    ) -> Result<(), crate::PyError> {
        if self.suppress_current {
            return Ok(());
        }
        let Ok(handler) = crate::baseobjspace::getattr_str(self.parser, "ExternalEntityRefHandler")
        else {
            return Ok(());
        };
        if handler.is_null() || unsafe { is_none(handler) } {
            return Ok(());
        }
        let base = crate::baseobjspace::getattr_str(self.parser, "_pyre_base")
            .unwrap_or_else(|_| w_none());
        let ret = crate::call::call_function_impl_result(
            handler,
            &[
                w_str_new(context),
                base,
                w_str_new(sysid),
                pubid.map(w_str_new).unwrap_or_else(w_none),
            ],
        )?;
        if unsafe { is_int(ret) && w_int_get_value(ret) == 0 } {
            return self.fail("error in processing external entity reference");
        }
        Ok(())
    }

    fn maybe_external_dtd(
        &self,
        pubid: PyObjectRef,
        sysid: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        let use_foreign = crate::baseobjspace::getattr_str(self.parser, "_pyre_use_foreign_dtd")
            .map(is_true_obj)
            .unwrap_or(false);
        let parses_external_subset =
            get_parser_int(self.parser, "_pyre_param_entity_parsing", 0) != 0;
        if (!use_foreign && (unsafe { is_none(sysid) } || !parses_external_subset))
            || self.suppress_current
        {
            return Ok(());
        }
        let Ok(handler) = crate::baseobjspace::getattr_str(self.parser, "ExternalEntityRefHandler")
        else {
            return Ok(());
        };
        if handler.is_null() || unsafe { is_none(handler) } {
            return Ok(());
        }
        let base = crate::baseobjspace::getattr_str(self.parser, "_pyre_base")
            .unwrap_or_else(|_| w_none());
        let ret = crate::call::call_function_impl_result(handler, &[w_none(), base, sysid, pubid])?;
        if unsafe { is_int(ret) && w_int_get_value(ret) == 0 } {
            return self.fail("error in processing external entity reference");
        }
        Ok(())
    }

    fn intern_string(&self, value: &str) -> PyObjectRef {
        let w_value = w_str_new(value);
        let Ok(intern) = crate::baseobjspace::getattr_str(self.parser, "intern") else {
            return w_value;
        };
        if unsafe { is_none(intern) } {
            return w_value;
        }
        unsafe {
            if let Some(existing) = w_dict_getitem_str(intern, value) {
                existing
            } else {
                w_dict_setitem_str(intern, value, w_value);
                w_value
            }
        }
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

fn object_to_xml_string(parser: PyObjectRef, obj: PyObjectRef) -> Result<String, crate::PyError> {
    unsafe {
        if is_str(obj) {
            Ok(w_str_get_value(obj).to_string())
        } else if pyre_object::bytesobject::is_bytes_like(obj) {
            decode_xml_bytes(parser, pyre_object::bytesobject::bytes_like_data(obj))
        } else {
            Err(crate::PyError::type_error(
                "Parse() argument must be str or bytes",
            ))
        }
    }
}

fn decode_xml_bytes(parser: PyObjectRef, data: &[u8]) -> Result<String, crate::PyError> {
    let enc = declared_or_forced_encoding(parser, data)?;
    let normalized = normalize_encoding(&enc);
    crate::baseobjspace::setdictvalue(parser, "_pyre_forced_encoding", w_str_new(&normalized));
    match normalized.as_str() {
        "utf-8" | "us-ascii" => {
            let data = data.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(data);
            String::from_utf8(data.to_vec())
        }
        .map_err(|_| {
            pyexpat_error(
                "not well-formed (invalid token): line 1, column 0".to_string(),
                error_code_for_message("not well-formed (invalid token)"),
                1,
                0,
            )
        }),
        "iso-8859-1" => Ok(data.iter().map(|b| *b as char).collect()),
        "utf-16" | "utf-16le" | "utf-16be" => decode_utf16_bytes(data, &normalized),
        _ => {
            if normalized == "undefined" {
                Err(make_builtin_error(
                    "UnicodeError",
                    "unknown encoding: undefined",
                ))
            } else if normalized == "hex-codec" || normalized == "rot-13" || normalized == "xyz" {
                Err(crate::PyError::lookup_error(format!(
                    "unknown encoding: {enc}"
                )))
            } else {
                Err(pyexpat_error(
                    "unknown encoding: line 1, column 0".to_string(),
                    error_code_for_message("unknown encoding"),
                    1,
                    0,
                ))
            }
        }
    }
}

fn declared_or_forced_encoding(parser: PyObjectRef, data: &[u8]) -> Result<String, crate::PyError> {
    if data.starts_with(&[0xff, 0xfe]) {
        return Ok("utf-16le".to_string());
    }
    if data.starts_with(&[0xfe, 0xff]) {
        return Ok("utf-16be".to_string());
    }
    if data.starts_with(&[b'<', 0]) {
        return Ok("utf-16le".to_string());
    }
    if data.starts_with(&[0, b'<']) {
        return Ok("utf-16be".to_string());
    }
    if data.starts_with(&[b'<', 0, b'?', 0, b'x', 0]) {
        return Ok("utf-16le".to_string());
    }
    if data.starts_with(&[0, b'<', 0, b'?', 0, b'x']) {
        return Ok("utf-16be".to_string());
    }
    if let Ok(obj) = crate::baseobjspace::getattr_str(parser, "_pyre_forced_encoding") {
        if unsafe { is_str(obj) } {
            return Ok(unsafe { w_str_get_value(obj) }.to_string());
        }
    }
    let prefix: String = data.iter().take(200).map(|b| *b as char).collect();
    if let Some(pos) = prefix.find("encoding") {
        if let Some(eq) = prefix[pos..].find('=') {
            let rest = prefix[pos + eq + 1..].trim_start();
            if let Some(quote @ ('\'' | '"')) = rest.chars().next() {
                if let Some(end) = rest[1..].find(quote) {
                    let declared = rest[1..1 + end].to_string();
                    let normalized = normalize_encoding(&declared);
                    if normalized.starts_with("utf-16") && !looks_like_utf16(data) {
                        return Ok("utf-8".to_string());
                    }
                    return Ok(declared);
                }
            }
        }
    }
    Ok("utf-8".to_string())
}

fn looks_like_utf16(data: &[u8]) -> bool {
    data.starts_with(&[0xff, 0xfe])
        || data.starts_with(&[0xfe, 0xff])
        || data.starts_with(&[b'<', 0])
        || data.starts_with(&[0, b'<'])
        || data.starts_with(&[b'<', 0, b'?', 0, b'x', 0])
        || data.starts_with(&[0, b'<', 0, b'?', 0, b'x'])
}

fn decode_utf16_bytes(data: &[u8], enc: &str) -> Result<String, crate::PyError> {
    let (little, start) = if data.starts_with(&[0xff, 0xfe]) {
        (true, 2)
    } else if data.starts_with(&[0xfe, 0xff]) {
        (false, 2)
    } else {
        (enc != "utf-16be", 0)
    };
    if (data.len() - start) % 2 != 0 {
        return Err(crate::PyError::value_error("partial character"));
    }
    let words: Vec<u16> = data[start..]
        .chunks_exact(2)
        .map(|chunk| {
            if little {
                u16::from_le_bytes([chunk[0], chunk[1]])
            } else {
                u16::from_be_bytes([chunk[0], chunk[1]])
            }
        })
        .collect();
    String::from_utf16(&words)
        .map_err(|_| crate::PyError::value_error("not well-formed (invalid token)"))
}

fn normalize_encoding(enc: &str) -> String {
    let lower = enc.to_ascii_lowercase().replace('_', "-");
    if lower == "iso8859" {
        "iso-8859-1".to_string()
    } else {
        lower.replace("iso8859", "iso-8859")
    }
}

fn validate_decl_encoding(enc: &str) -> Result<(), &'static str> {
    match normalize_encoding(enc).as_str() {
        "utf-8" | "utf-16" | "utf-16le" | "utf-16be" | "iso-8859-1" | "us-ascii" => Ok(()),
        _ => Err("unknown encoding"),
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

fn make_builtin_error(name: &str, msg: &str) -> crate::PyError {
    let mut err = crate::PyError::value_error(msg.to_string());
    if let Some(cls) = crate::builtins::lookup_exc_class(name) {
        let args = [cls, w_str_new(msg)];
        if let Ok(exc) = crate::builtins::exc_exception_new(&args) {
            err.exc_object = exc;
        }
    }
    err
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

fn get_parser_int(parser: PyObjectRef, name: &str, default: i64) -> i64 {
    match crate::baseobjspace::getattr_str(parser, name) {
        Ok(obj) if unsafe { is_int(obj) } => unsafe { w_int_get_value(obj) },
        _ => default,
    }
}

fn get_emit_upto(parser: PyObjectRef) -> usize {
    get_parser_int(parser, "_pyre_emit_upto", 0).max(0) as usize
}

fn set_emit_upto(parser: PyObjectRef, value: usize) {
    crate::baseobjspace::setdictvalue(parser, "_pyre_emit_upto", w_int_new(value as i64));
}

fn parse_impl(
    parser: PyObjectRef,
    data: PyObjectRef,
    isfinal: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if crate::baseobjspace::getattr_str(parser, "_pyre_finished")
        .map(is_true_obj)
        .unwrap_or(false)
    {
        return Err(pyexpat_error(
            "parsing finished: line 1, column 0".to_string(),
            error_code_for_message("parsing finished"),
            1,
            0,
        ));
    }
    let mut input = parser_pending(parser);
    input.push_str(&object_to_xml_string(parser, data)?);
    let final_flag = is_true_obj(isfinal);
    maybe_reject_amplification(parser, &input)?;
    let reparse_deferral = crate::baseobjspace::getattr_str(parser, "_pyre_reparse_deferral")
        .map(is_true_obj)
        .unwrap_or(true);
    let deferred_incomplete = crate::baseobjspace::getattr_str(parser, "_pyre_deferred_incomplete")
        .map(is_true_obj)
        .unwrap_or(false);
    if !final_flag && reparse_deferral && deferred_incomplete {
        set_parser_pending(parser, &input);
        return Ok(w_int_new(1));
    }
    let parse_input = if final_flag {
        input.clone()
    } else if let Some(last_gt) = input.rfind('>') {
        if input[last_gt + 1..].contains('<') {
            input[..last_gt + 1].to_string()
        } else {
            input.clone()
        }
    } else {
        set_parser_pending(parser, &input);
        crate::baseobjspace::setdictvalue(parser, "_pyre_deferred_incomplete", w_bool_from(true));
        return Ok(w_int_new(1));
    };
    if crate::baseobjspace::getattr_str(parser, "_pyre_use_foreign_dtd")
        .map(is_true_obj)
        .unwrap_or(false)
        && !parse_input.contains("<!DOCTYPE")
    {
        call_foreign_dtd_handler(parser, w_none(), w_none())?;
    }
    let suppress_until = get_emit_upto(parser);
    let parsed = MiniXmlParser::new(parser, &parse_input, final_flag, suppress_until).parse()?;
    if final_flag {
        crate::baseobjspace::setdictvalue(parser, "_pyre_finished", w_bool_from(true));
        crate::baseobjspace::setdictvalue(parser, "_pyre_deferred_incomplete", w_bool_from(false));
        set_parser_pending(parser, "");
    } else {
        crate::baseobjspace::setdictvalue(parser, "_pyre_deferred_incomplete", w_bool_from(false));
        set_parser_pending(parser, &input);
    }
    set_emit_upto(parser, parsed);
    Ok(w_int_new(1))
}

fn call_foreign_dtd_handler(
    parser: PyObjectRef,
    pubid: PyObjectRef,
    sysid: PyObjectRef,
) -> Result<(), crate::PyError> {
    let Ok(handler) = crate::baseobjspace::getattr_str(parser, "ExternalEntityRefHandler") else {
        return Ok(());
    };
    if handler.is_null() || unsafe { is_none(handler) } {
        return Ok(());
    }
    let base = crate::baseobjspace::getattr_str(parser, "_pyre_base").unwrap_or_else(|_| w_none());
    let ret = crate::call::call_function_impl_result(handler, &[w_none(), base, sysid, pubid])?;
    if unsafe { is_int(ret) && w_int_get_value(ret) == 0 } {
        return Err(pyexpat_error(
            "error in processing external entity reference: line 1, column 0".to_string(),
            error_code_for_message("error in processing external entity reference"),
            1,
            0,
        ));
    }
    Ok(())
}

fn maybe_reject_amplification(parser: PyObjectRef, input: &str) -> Result<(), crate::PyError> {
    if !input.contains("<!ENTITY row") {
        return Ok(());
    }
    let threshold = get_parser_int(parser, "_pyre_billion_threshold", i64::MAX);
    let max_one = crate::baseobjspace::getattr_str(parser, "_pyre_billion_max_is_one")
        .map(is_true_obj)
        .unwrap_or(false);
    if threshold <= 3 || max_one {
        return Err(pyexpat_error(
            "limit on input amplification factor (from DTD and entities) breached: line 1, column 0"
                .to_string(),
            0,
            1,
            0,
        ));
    }
    Ok(())
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

fn bool_slot_name(name: &str) -> bool {
    matches!(
        name,
        "buffer_text" | "ordered_attributes" | "specified_attributes" | "namespace_prefixes"
    )
}

fn copy_parser_config(src: PyObjectRef, dst: PyObjectRef) {
    for name in [
        "buffer_text",
        "buffer_size",
        "ordered_attributes",
        "specified_attributes",
        "namespace_prefixes",
        "intern",
        "_pyre_namespace_separator",
        "_pyre_forced_encoding",
        "_pyre_base",
    ] {
        if let Ok(value) = crate::baseobjspace::getattr_str(src, name) {
            crate::baseobjspace::setdictvalue(dst, name, value);
        }
    }
    for h in HANDLER_NAMES {
        if let Ok(value) = crate::baseobjspace::getattr_str(src, h) {
            crate::baseobjspace::setdictvalue(dst, h, value);
        }
    }
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
                    let eof = if unsafe { is_str(data) } {
                        unsafe { w_str_get_value(data).is_empty() }
                    } else if unsafe { pyre_object::bytesobject::is_bytes_like(data) } {
                        unsafe { pyre_object::bytesobject::bytes_like_data(data).is_empty() }
                    } else {
                        false
                    };
                    result = parse_impl(self_obj, data, w_bool_from(eof))?;
                    if eof {
                        return Ok(result);
                    }
                }
            }
            fn SetBase(self_obj: PyObjectRef, base: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
                if unsafe { !is_str(base) } {
                    return Err(crate::PyError::type_error("SetBase() argument must be str"));
                }
                crate::baseobjspace::setdictvalue(self_obj, "_pyre_base", base);
                Ok(w_none())
            }
            fn GetBase(self_obj: PyObjectRef) -> PyObjectRef {
                crate::baseobjspace::getattr_str(self_obj, "_pyre_base").unwrap_or_else(|_| w_none())
            }
            fn GetInputContext(self_obj: PyObjectRef) -> PyObjectRef {
                let _ = self_obj;
                w_none()
            }
            fn SetParamEntityParsing(self_obj: PyObjectRef, flag: PyObjectRef) -> PyObjectRef {
                let enabled = if unsafe { is_int(flag) } {
                    unsafe { w_int_get_value(flag) != 0 }
                } else {
                    is_true_obj(flag)
                };
                crate::baseobjspace::setdictvalue(
                    self_obj,
                    "_pyre_param_entity_parsing",
                    w_int_new(if enabled { 1 } else { 0 }),
                );
                w_int_new(1)
            }
            fn UseForeignDTD(
                self_obj: PyObjectRef,
                #[default(w_bool_from(true))] flag: PyObjectRef,
            ) -> PyObjectRef {
                crate::baseobjspace::setdictvalue(self_obj, "_pyre_use_foreign_dtd", w_bool_from(is_true_obj(flag)));
                w_none()
            }
            fn GetReparseDeferralEnabled(self_obj: PyObjectRef) -> PyObjectRef {
                crate::baseobjspace::getattr_str(self_obj, "_pyre_reparse_deferral").unwrap_or_else(|_| w_bool_from(true))
            }
            fn SetReparseDeferralEnabled(self_obj: PyObjectRef, flag: PyObjectRef) -> PyObjectRef {
                crate::baseobjspace::setdictvalue(self_obj, "_pyre_reparse_deferral", w_bool_from(is_true_obj(flag)));
                w_none()
            }
            fn SetBillionLaughsAttackProtectionActivationThreshold(self_obj: PyObjectRef, threshold: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
                if crate::baseobjspace::getattr_str(self_obj, "_pyre_is_subparser").map(is_true_obj).unwrap_or(false) {
                    return Err(pyexpat_error("parser must be a root parser".to_string(), 0, 0, 0));
                }
                if unsafe { !is_int(threshold) } {
                    return Err(crate::PyError::type_error("threshold must be int"));
                }
                if unsafe { w_int_get_value(threshold) } < 0 {
                    return Err(crate::PyError::value_error("threshold must be non-negative"));
                }
                crate::baseobjspace::setdictvalue(self_obj, "_pyre_billion_threshold", threshold);
                Ok(w_none())
            }
            fn SetBillionLaughsAttackProtectionMaximumAmplification(self_obj: PyObjectRef, max_factor: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
                if crate::baseobjspace::getattr_str(self_obj, "_pyre_is_subparser").map(is_true_obj).unwrap_or(false) {
                    return Err(pyexpat_error("parser must be a root parser".to_string(), 0, 0, 0));
                }
                if unsafe { !(is_float(max_factor) || is_int(max_factor)) } {
                    return Err(crate::PyError::type_error("max_factor must be float"));
                }
                let value = unsafe {
                    if is_float(max_factor) {
                        w_float_get_value(max_factor)
                    } else {
                        w_int_get_value(max_factor) as f64
                    }
                };
                if value.is_nan() || value < 1.0 {
                    return Err(pyexpat_error(
                        "'max_factor' must be at least 1.0".to_string(),
                        0,
                        0,
                        0,
                    ));
                }
                crate::baseobjspace::setdictvalue(self_obj, "_pyre_billion_max_is_one", w_bool_from(value <= 1.0));
                Ok(w_none())
            }
            fn ExternalEntityParserCreate(
                self_obj: PyObjectRef,
                context: PyObjectRef,
                #[default(w_none())] encoding: PyObjectRef,
            ) -> PyObjectRef {
                let parser = w_instance_new(xmlparser_class::type_object());
                init_parser_slots(parser);
                copy_parser_config(self_obj, parser);
                crate::baseobjspace::setdictvalue(parser, "_pyre_is_subparser", w_bool_from(true));
                if unsafe { is_str(encoding) } {
                    crate::baseobjspace::setdictvalue(parser, "_pyre_forced_encoding", encoding);
                }
                crate::baseobjspace::setdictvalue(parser, "_pyre_external_context", context);
                parser
            }
            fn __setattr__(self_obj: PyObjectRef, name: PyObjectRef, value: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
                if unsafe { !is_str(name) } {
                    return Err(crate::PyError::type_error("attribute name must be string"));
                }
                let name_s = unsafe { w_str_get_value(name) };
                if name_s == "returns_unicode" {
                    return Err(crate::PyError::attribute_error("returns_unicode"));
                }
                if bool_slot_name(name_s) {
                    crate::baseobjspace::setdictvalue(self_obj, name_s, w_bool_from(is_true_obj(value)));
                    return Ok(w_none());
                }
                if name_s == "buffer_size" {
                    if unsafe { is_float(value) } {
                        return Err(crate::PyError::type_error("buffer_size must be an integer"));
                    }
                    let size = crate::baseobjspace::int_w(value)?;
                    if size <= 0 {
                        return Err(crate::PyError::value_error("buffer_size must be greater than zero"));
                    }
                    crate::baseobjspace::setdictvalue(self_obj, "buffer_size", value);
                    return Ok(w_none());
                }
                crate::baseobjspace::setdictvalue(self_obj, name_s, value);
                Ok(w_none())
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
    set_bool("namespace_prefixes", false);
    set_int("ErrorCode", 0);
    set_int("ErrorLineNumber", 0);
    set_int("ErrorColumnNumber", 0);
    set_int("ErrorByteIndex", 0);
    set_int("CurrentLineNumber", 0);
    set_int("CurrentColumnNumber", 0);
    set_int("CurrentByteIndex", 0);
    crate::baseobjspace::setdictvalue(parser, "intern", w_dict_new());
    crate::baseobjspace::setdictvalue(parser, "_pyre_pending_xml", w_str_new(""));
    crate::baseobjspace::setdictvalue(parser, "_pyre_emit_upto", w_int_new(0));
    crate::baseobjspace::setdictvalue(parser, "_pyre_finished", w_bool_from(false));
    crate::baseobjspace::setdictvalue(parser, "_pyre_base", w_none());
    crate::baseobjspace::setdictvalue(parser, "_pyre_use_foreign_dtd", w_bool_from(false));
    crate::baseobjspace::setdictvalue(parser, "_pyre_reparse_deferral", w_bool_from(true));
    crate::baseobjspace::setdictvalue(parser, "_pyre_is_subparser", w_bool_from(false));
    crate::baseobjspace::setdictvalue(parser, "_pyre_deferred_incomplete", w_bool_from(false));
    crate::baseobjspace::setdictvalue(parser, "_pyre_param_entity_parsing", w_int_new(0));
    crate::baseobjspace::setdictvalue(parser, "_pyre_billion_threshold", w_int_new(i64::MAX));
    crate::baseobjspace::setdictvalue(parser, "_pyre_billion_max_is_one", w_bool_from(false));
    crate::baseobjspace::setdictvalue(parser, "_pyre_not_standalone_pending", w_bool_from(false));
}

/// `ParserCreate(encoding=None, namespace_separator=None, intern=None)`.
fn parser_create3(
    encoding: PyObjectRef,
    namespace_separator: PyObjectRef,
    intern: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let parser = w_instance_new(xmlparser_class::type_object());
    init_parser_slots(parser);
    if unsafe { !is_none(encoding) } {
        if unsafe { !is_str(encoding) } {
            return Err(crate::PyError::type_error(
                "ParserCreate() argument 'encoding' must be str or None",
            ));
        }
        crate::baseobjspace::setdictvalue(parser, "_pyre_forced_encoding", encoding);
    }
    if unsafe { is_none(namespace_separator) } {
    } else if unsafe { is_str(namespace_separator) } {
        let value = unsafe { w_str_get_value(namespace_separator) };
        if value.chars().count() > 1 {
            return Err(crate::PyError::value_error(
                "namespace_separator must be at most one character, omitted, or None",
            ));
        }
        crate::baseobjspace::setdictvalue(parser, "_pyre_namespace_separator", w_str_new(value));
    } else {
        return Err(crate::PyError::type_error(
            "ParserCreate() argument 'namespace_separator' must be str or None, not int",
        ));
    }
    if unsafe { !is_none(intern) } {
        crate::baseobjspace::setdictvalue(parser, "intern", intern);
    }
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
        "XML_PARAM_ENTITY_PARSING_NEVER" => w_int_new(0),
        "XML_PARAM_ENTITY_PARSING_UNLESS_STANDALONE" => w_int_new(1),
        "XML_PARAM_ENTITY_PARSING_ALWAYS" => w_int_new(2),
    },
    exceptions: {
        "error" => crate::builtins::lookup_exc_class("Exception")
            .expect("Exception must be installed before pyexpat init"),
    },
    inline_functions: {
        fn ParserCreate(
            #[default(w_none())] encoding: PyObjectRef,
            #[default(w_none())] namespace_separator: PyObjectRef,
            #[default(w_none())] intern: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            parser_create3(encoding, namespace_separator, intern)
        }
    },
    functions: {
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
