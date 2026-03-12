/// Interpreter for tinyframe — port of rpython/jit/tl/tinyframe/tinyframe.py.
///
/// Register-based interpreter with:
/// - Object system: Int, Func, CombinedFunc
/// - Frame with virtualizable registers
/// - Opcodes: ADD, INTROSPECT, PRINT, CALL, LOAD, LOAD_FUNCTION, RETURN,
///   JUMP, JUMP_IF_ABOVE
use std::fmt;

// Opcodes — same values as the Python version (0-indexed from the list)
pub const ADD: u8 = 0;
pub const INTROSPECT: u8 = 1;
pub const PRINT: u8 = 2;
pub const CALL: u8 = 3;
pub const LOAD: u8 = 4;
pub const LOAD_FUNCTION: u8 = 5;
pub const RETURN: u8 = 6;
pub const JUMP: u8 = 7;
pub const JUMP_IF_ABOVE: u8 = 8;

/// A function's code object.
#[derive(Clone, Debug)]
pub struct Code {
    pub code: Vec<u8>,
    pub regno: usize,
    pub functions: Vec<Code>,
    pub name: String,
}

/// Runtime object — either an integer or a function.
#[derive(Clone, Debug)]
pub enum Object {
    Int(i64),
    Func(FuncObj),
    CombinedFunc(Box<CombinedFuncObj>),
}

#[derive(Clone, Debug)]
pub struct FuncObj {
    pub code_idx: usize, // index into Code.functions
}

#[derive(Clone, Debug)]
pub struct CombinedFuncObj {
    pub outer: Object,
    pub inner: Object,
}

impl Object {
    pub fn as_int(&self) -> i64 {
        match self {
            Object::Int(v) => *v,
            _ => panic!("expected Int"),
        }
    }

    pub fn gt(&self, other: &Object) -> bool {
        match (self, other) {
            (Object::Int(a), Object::Int(b)) => a > b,
            _ => panic!("gt requires Int operands"),
        }
    }

    pub fn add(&self, other: &Object) -> Object {
        match (self, other) {
            (Object::Int(a), Object::Int(b)) => Object::Int(a + b),
            _ => Object::CombinedFunc(Box::new(CombinedFuncObj {
                outer: self.clone(),
                inner: other.clone(),
            })),
        }
    }

    pub fn repr(&self, code: &Code) -> String {
        match self {
            Object::Int(v) => v.to_string(),
            Object::Func(f) => format!("<function {}>", code.functions[f.code_idx].name),
            Object::CombinedFunc(cf) => format!(
                "<function {}({})>",
                cf.outer.repr(code),
                cf.inner.repr(code)
            ),
        }
    }

    fn call(&self, arg: Object, root_code: &Code) -> Object {
        match self {
            Object::Func(f) => {
                let func_code = &root_code.functions[f.code_idx];
                let mut frame = Frame::new(func_code);
                frame.registers[0] = Some(arg);
                frame.interpret(root_code)
            }
            Object::CombinedFunc(cf) => {
                let inner_result = cf.inner.call(arg, root_code);
                cf.outer.call(inner_result, root_code)
            }
            Object::Int(_) => panic!("cannot call Int"),
        }
    }
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Object::Int(v) => write!(f, "{v}"),
            Object::Func(func) => write!(f, "<function #{}>", func.code_idx),
            Object::CombinedFunc(_) => write!(f, "<combined function>"),
        }
    }
}

/// A stack frame with registers.
pub struct Frame<'a> {
    pub code: &'a Code,
    pub registers: Vec<Option<Object>>,
}

impl<'a> Frame<'a> {
    pub fn new(code: &'a Code) -> Self {
        Frame {
            code,
            registers: vec![None; code.regno],
        }
    }

    pub fn interpret(&mut self, root_code: &Code) -> Object {
        let code = &self.code.code;
        let mut i: usize = 0;

        loop {
            if i >= code.len() {
                break;
            }
            let opcode = code[i];

            if opcode == LOAD {
                let val = code[i + 1] as i64;
                let reg = code[i + 2] as usize;
                self.registers[reg] = Some(Object::Int(val));
                i += 3;
            } else if opcode == ADD {
                let r1 = code[i + 1] as usize;
                let r2 = code[i + 2] as usize;
                let r3 = code[i + 3] as usize;
                let arg1 = self.registers[r1].as_ref().unwrap();
                let arg2 = self.registers[r2].as_ref().unwrap();
                let result = arg1.add(arg2);
                self.registers[r3] = Some(result);
                i += 4;
            } else if opcode == RETURN {
                let r = code[i + 1] as usize;
                return self.registers[r].clone().unwrap();
            } else if opcode == JUMP_IF_ABOVE {
                let r1 = code[i + 1] as usize;
                let r2 = code[i + 2] as usize;
                let tgt = code[i + 3] as usize;
                let arg0 = self.registers[r1].as_ref().unwrap();
                let arg1 = self.registers[r2].as_ref().unwrap();
                if arg0.gt(arg1) {
                    i = tgt;
                } else {
                    i += 4;
                }
            } else if opcode == LOAD_FUNCTION {
                let func_idx = code[i + 1] as usize;
                let reg = code[i + 2] as usize;
                self.registers[reg] = Some(Object::Func(FuncObj { code_idx: func_idx }));
                i += 3;
            } else if opcode == CALL {
                let r_func = code[i + 1] as usize;
                let r_arg = code[i + 2] as usize;
                let r_result = code[i + 3] as usize;
                let func = self.registers[r_func].as_ref().unwrap().clone();
                let arg = self.registers[r_arg].as_ref().unwrap().clone();
                let result = func.call(arg, root_code);
                self.registers[r_result] = Some(result);
                i += 4;
            } else if opcode == PRINT {
                let r = code[i + 1] as usize;
                let val = self.registers[r].as_ref().unwrap();
                println!("{}", val.repr(root_code));
                i += 2;
            } else if opcode == INTROSPECT {
                let r_arg = code[i + 1] as usize;
                let r_result = code[i + 2] as usize;
                let source = self.registers[r_arg].as_ref().unwrap().as_int() as usize;
                let val = self.registers[source].clone().unwrap();
                self.registers[r_result] = Some(val);
                i += 3;
            } else {
                panic!("unimplemented opcode {opcode}");
            }
        }

        panic!("fell off end of code without RETURN");
    }
}

pub fn interpret(code: &Code) -> Object {
    let mut frame = Frame::new(code);
    frame.interpret(code)
}

/// Parse/compile a tinyframe program from text source.
pub fn compile(source: &str) -> Code {
    let mut parser = Parser::new();
    parser.compile(source)
}

struct Parser {
    code: Vec<u8>,
    max_regno: usize,
    functions: Vec<(String, Code)>,
    labels: std::collections::HashMap<String, usize>,
    name: Option<String>,
}

impl Parser {
    fn new() -> Self {
        Parser {
            code: Vec::new(),
            max_regno: 0,
            functions: Vec::new(),
            labels: std::collections::HashMap::new(),
            name: None,
        }
    }

    fn compile(&mut self, source: &str) -> Code {
        for line in source.lines() {
            // Strip comments
            let line = if let Some(pos) = line.find('#') {
                &line[..pos]
            } else {
                line
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if line.ends_with(':') {
                // Function/section label
                self.finish_current_code();
                self.name = Some(line[..line.len() - 1].to_string());
                continue;
            }

            if line.starts_with('@') {
                // Label
                self.labels.insert(line[1..].to_string(), self.code.len());
                continue;
            }

            let first_space = line.find(' ').unwrap();
            let opcode_str = &line[..first_space];
            let args = line[first_space + 1..].trim();

            match opcode_str {
                "ADD" => self.compile_add(args),
                "INTROSPECT" => self.compile_introspect(args),
                "PRINT" => self.compile_print(args),
                "CALL" => self.compile_call(args),
                "LOAD" => self.compile_load(args),
                "LOAD_FUNCTION" => self.compile_load_function(args),
                "RETURN" => self.compile_return(args),
                "JUMP" => self.compile_jump(args),
                "JUMP_IF_ABOVE" => self.compile_jump_if_above(args),
                _ => panic!("unknown opcode: {opcode_str}"),
            }
        }

        // The last section must be "main"
        assert_eq!(
            self.name.as_deref(),
            Some("main"),
            "last function must be 'main'"
        );

        // Sort functions by insertion order
        let functions: Vec<Code> = self
            .functions
            .iter()
            .map(|(_, code)| code.clone())
            .collect();

        Code {
            code: self.code.clone(),
            regno: self.max_regno + 1,
            functions,
            name: "main".to_string(),
        }
    }

    fn finish_current_code(&mut self) {
        if let Some(name) = self.name.take() {
            let code = Code {
                code: self.code.clone(),
                regno: self.max_regno + 1,
                functions: Vec::new(),
                name: name.clone(),
            };
            self.functions.push((name, code));
            self.code.clear();
            self.labels.clear();
            self.max_regno = 0;
        }
    }

    fn rint(&mut self, arg: &str) -> u8 {
        let arg = arg.trim();
        assert!(arg.starts_with('r'), "expected register: {arg}");
        let no: usize = arg[1..].parse().unwrap();
        self.max_regno = self.max_regno.max(no);
        no as u8
    }

    fn compile_add(&mut self, args: &str) {
        let (args, result) = args.split_once("=>").unwrap();
        let parts: Vec<&str> = args.trim().split_whitespace().collect();
        let r0 = self.rint(parts[0]);
        let r1 = self.rint(parts[1]);
        let r2 = self.rint(result.trim());
        self.code.extend_from_slice(&[ADD, r0, r1, r2]);
    }

    fn compile_load(&mut self, args: &str) {
        let (val, result) = args.split_once("=>").unwrap();
        let val: u8 = val.trim().parse().unwrap();
        let r = self.rint(result.trim());
        self.code.extend_from_slice(&[LOAD, val, r]);
    }

    fn compile_print(&mut self, args: &str) {
        let r = self.rint(args.trim());
        self.code.extend_from_slice(&[PRINT, r]);
    }

    fn compile_return(&mut self, args: &str) {
        let r = self.rint(args.trim());
        self.code.extend_from_slice(&[RETURN, r]);
    }

    fn compile_jump_if_above(&mut self, args: &str) {
        let parts: Vec<&str> = args.split_whitespace().collect();
        let r0 = self.rint(parts[0]);
        let r1 = self.rint(parts[1]);
        let label = parts[2].trim_start_matches('@');
        let target = *self
            .labels
            .get(label)
            .unwrap_or_else(|| panic!("unknown label: {label}"));
        self.code
            .extend_from_slice(&[JUMP_IF_ABOVE, r0, r1, target as u8]);
    }

    fn compile_load_function(&mut self, args: &str) {
        let (name, result) = args.split_once("=>").unwrap();
        let name = name.trim();
        let func_idx = self
            .functions
            .iter()
            .position(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("unknown function: {name}"));
        let r = self.rint(result.trim());
        self.code
            .extend_from_slice(&[LOAD_FUNCTION, func_idx as u8, r]);
    }

    fn compile_call(&mut self, args: &str) {
        let (args, result) = args.split_once("=>").unwrap();
        let parts: Vec<&str> = args.trim().split_whitespace().collect();
        let r0 = self.rint(parts[0]);
        let r1 = self.rint(parts[1]);
        let r2 = self.rint(result.trim());
        self.code.extend_from_slice(&[CALL, r0, r1, r2]);
    }

    fn compile_introspect(&mut self, args: &str) {
        let (arg, result) = args.split_once("=>").unwrap();
        let r0 = self.rint(arg.trim());
        let r1 = self.rint(result.trim());
        self.code.extend_from_slice(&[INTROSPECT, r0, r1]);
    }

    fn compile_jump(&mut self, _args: &str) {
        panic!("JUMP not implemented");
    }
}

pub fn disassemble(code: &Code) -> Vec<u8> {
    code.code.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_simple() {
        let code = compile(
            "
        main:
        LOAD 0 => r1
        LOAD 1 => r0 # comment
        # other comment
        ADD r0 r1 => r2
        PRINT r2
        ",
        );
        assert_eq!(
            disassemble(&code),
            vec![LOAD, 0, 1, LOAD, 1, 0, ADD, 0, 1, 2, PRINT, 2]
        );
    }

    #[test]
    fn test_return() {
        let code = compile(
            "
        main:
        LOAD 0 => r1
        LOAD 1 => r0
        ADD r0 r1 => r2
        RETURN r2
        ",
        );
        let res = interpret(&code);
        assert_eq!(res.as_int(), 1);
    }

    #[test]
    fn test_loop() {
        let code = compile(
            "
        main:
        LOAD 1 => r1
        LOAD 100 => r2
        LOAD 0 => r0
        @l1
        ADD r0 r1 => r0
        JUMP_IF_ABOVE r2 r0 @l1
        RETURN r0
        ",
        );
        let ret = interpret(&code);
        assert_eq!(ret.as_int(), 100);
    }

    #[test]
    fn test_function() {
        let code = compile(
            "
        func:
        LOAD 1 => r1
        ADD r0 r1 => r1
        RETURN r1
        main:
        LOAD_FUNCTION func => r0
        LOAD 1 => r1
        CALL r0 r1 => r2
        RETURN r2
        ",
        );
        let ret = interpret(&code);
        assert_eq!(ret.as_int(), 2); // 1 + 1
    }

    #[test]
    fn test_function_combination() {
        let code = compile(
            "
        inner:
        LOAD 2 => r1
        ADD r1 r0 => r0
        RETURN r0
        outer:
        LOAD 1 => r1
        ADD r1 r0 => r2
        RETURN r2
        main:
        LOAD_FUNCTION inner => r0
        LOAD_FUNCTION outer => r1
        ADD r1 r0 => r2
        LOAD 1 => r3
        CALL r2 r3 => r4
        RETURN r4
        ",
        );
        let ret = interpret(&code);
        assert_eq!(ret.as_int(), 4); // outer(inner(1)) = outer(3) = 4
    }

    #[test]
    fn test_introspect() {
        let code = compile(
            "
        main:
        LOAD 100 => r0
        LOAD 0 => r1
        INTROSPECT r1 => r2
        RETURN r0
        ",
        );
        let res = interpret(&code);
        assert_eq!(res.as_int(), 100);
    }

    #[test]
    fn test_loop_sum() {
        // Sum from 1 to N: LOAD 0 => r1 (acc), LOAD 1 => r2 (step)
        // loop: ADD r2 r1 => r1, JUMP_IF_ABOVE r0 r1 @loop, RETURN r1
        let code = compile(
            "
        main:
        LOAD 0 => r1
        LOAD 1 => r2
        @add
        ADD r2 r1 => r1
        JUMP_IF_ABOVE r0 r1 @add
        RETURN r1
        ",
        );
        let mut frame = Frame::new(&code);
        frame.registers[0] = Some(Object::Int(10));
        let ret = frame.interpret(&code);
        assert_eq!(ret.as_int(), 10);
    }
}
