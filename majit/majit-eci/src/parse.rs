//! `external_compilation_info!` input grammar.

use syn::parse::{Parse, ParseStream};
use syn::{Ident, LitBool, LitStr, Token, Visibility};

use crate::{_ATTRIBUTES, ExternalCompilationInfo};

pub struct EciInput {
    pub vis: Visibility,
    pub name: Ident,
    pub fields: Vec<(Ident, EciValue)>,
}

pub enum EciValue {
    Strs(Vec<LitStr>),
    Bool(bool),
}

impl Parse for EciInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let vis = input.parse()?;
        input.parse::<Token![const]>()?;
        let name: Ident = input.parse()?;
        input.parse::<Token![=]>()?;
        let body;
        syn::braced!(body in input);
        let mut fields = Vec::new();
        while !body.is_empty() {
            let key: Ident = body.parse()?;
            body.parse::<Token![:]>()?;
            let value = if body.peek(LitBool) {
                EciValue::Bool(body.parse::<LitBool>()?.value())
            } else {
                let list;
                syn::bracketed!(list in body);
                let mut strs = Vec::new();
                while !list.is_empty() {
                    strs.push(list.parse::<LitStr>()?);
                    if list.is_empty() {
                        break;
                    }
                    list.parse::<Token![,]>()?;
                }
                EciValue::Strs(strs)
            };
            fields.push((key, value));
            let _ = body.parse::<Token![,]>();
        }
        let _ = input.parse::<Token![;]>();
        Ok(EciInput { vis, name, fields })
    }
}

impl EciInput {
    pub fn to_eci(&self) -> syn::Result<ExternalCompilationInfo> {
        let mut eci = ExternalCompilationInfo::default();
        for (key, value) in &self.fields {
            let name = key.to_string();
            match value {
                EciValue::Bool(value) => {
                    if name != "use_cpp_linker" {
                        return Err(syn::Error::new(key.span(), "only use_cpp_linker is a bool"));
                    }
                    eci.use_cpp_linker = *value;
                }
                EciValue::Strs(strs) => {
                    if !_ATTRIBUTES.contains(&name.as_str()) {
                        return Err(syn::Error::new(
                            key.span(),
                            format!("unknown ExternalCompilationInfo field `{name}`"),
                        ));
                    }
                    let values: Vec<String> = strs.iter().map(LitStr::value).collect();
                    set_strings(&mut eci, &name, values);
                }
            }
        }
        Ok(eci)
    }
}

fn set_strings(eci: &mut ExternalCompilationInfo, name: &str, values: Vec<String>) {
    match name {
        "pre_include_bits" => eci.pre_include_bits = values,
        "includes" => eci.includes = values,
        "include_dirs" => eci.include_dirs = values,
        "post_include_bits" => eci.post_include_bits = values,
        "libraries" => eci.libraries = values,
        "library_dirs" => eci.library_dirs = values,
        "separate_module_sources" => eci.separate_module_sources = values,
        "separate_module_files" => eci.separate_module_files = values,
        "compile_extra" => eci.compile_extra = values,
        "link_extra" => eci.link_extra = values,
        "frameworks" => eci.frameworks = values,
        "link_files" => eci.link_files = values,
        "testonly_libraries" => eci.testonly_libraries = values,
        other => panic!("unknown ExternalCompilationInfo field `{other}`"),
    }
}
