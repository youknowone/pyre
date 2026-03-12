/// Interpreter for TLC — Toy Language with Cons Cells.
///
/// Port of rpython/jit/tl/tlc.py.
///
/// Stack-based interpreter with boxed values (IntObj, NilObj, ConsObj),
/// OO features (ClassDescr, Class, InstanceObj), and recursive CALL/SEND.
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

// ── Opcodes ──
// Identical values to tlopcode.py

pub const NOP: u8 = 1;
pub const PUSH: u8 = 2;
pub const POP: u8 = 3;
pub const SWAP: u8 = 4;
pub const ROLL: u8 = 5;
pub const PICK: u8 = 6;
pub const PUT: u8 = 7;
pub const ADD: u8 = 8;
pub const SUB: u8 = 9;
pub const MUL: u8 = 10;
pub const DIV: u8 = 11;
pub const EQ: u8 = 12;
pub const NE: u8 = 13;
pub const LT: u8 = 14;
pub const LE: u8 = 15;
pub const GT: u8 = 16;
pub const GE: u8 = 17;
pub const BR_COND: u8 = 18;
pub const BR_COND_STK: u8 = 19;
pub const CALL: u8 = 20;
pub const RETURN: u8 = 21;
pub const PUSHARG: u8 = 22;
pub const INVALID: u8 = 23;
pub const NIL: u8 = 24;
pub const CONS: u8 = 25;
pub const CAR: u8 = 26;
pub const CDR: u8 = 27;
pub const NEW: u8 = 28;
pub const GETATTR: u8 = 29;
pub const SETATTR: u8 = 30;
pub const SEND: u8 = 31;
pub const PUSHARGN: u8 = 32;
pub const PRINT: u8 = 33;
pub const DUMP: u8 = 34;
pub const BR: u8 = 35;

/// Opcode name table for the assembler.
fn opcode_names() -> HashMap<&'static str, u8> {
    let mut m = HashMap::new();
    m.insert("NOP", NOP);
    m.insert("PUSH", PUSH);
    m.insert("POP", POP);
    m.insert("SWAP", SWAP);
    m.insert("ROLL", ROLL);
    m.insert("PICK", PICK);
    m.insert("PUT", PUT);
    m.insert("ADD", ADD);
    m.insert("SUB", SUB);
    m.insert("MUL", MUL);
    m.insert("DIV", DIV);
    m.insert("EQ", EQ);
    m.insert("NE", NE);
    m.insert("LT", LT);
    m.insert("LE", LE);
    m.insert("GT", GT);
    m.insert("GE", GE);
    m.insert("BR_COND", BR_COND);
    m.insert("BR_COND_STK", BR_COND_STK);
    m.insert("CALL", CALL);
    m.insert("RETURN", RETURN);
    m.insert("PUSHARG", PUSHARG);
    m.insert("INVALID", INVALID);
    m.insert("NIL", NIL);
    m.insert("CONS", CONS);
    m.insert("CAR", CAR);
    m.insert("CDR", CDR);
    m.insert("NEW", NEW);
    m.insert("GETATTR", GETATTR);
    m.insert("SETATTR", SETATTR);
    m.insert("SEND", SEND);
    m.insert("PUSHARGN", PUSHARGN);
    m.insert("PRINT", PRINT);
    m.insert("DUMP", DUMP);
    m.insert("BR", BR);
    m
}

// ── Object system ──

/// A boxed value — the base type for TLC objects.
/// InstanceObj uses Rc<RefCell<>> for reference semantics (shared mutation).
#[derive(Clone, Debug)]
pub enum Obj {
    Int(i64),
    Nil,
    Cons(Rc<ConsCell>),
    Instance(Rc<RefCell<InstanceData>>),
}

#[derive(Clone, Debug)]
pub struct ConsCell {
    pub car: Obj,
    pub cdr: Obj,
}

#[derive(Clone, Debug)]
pub struct InstanceData {
    pub cls: Class,
    pub values: Vec<Obj>,
}

/// ClassDescr — describes attributes and methods of a class.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassDescr {
    pub attributes: Vec<String>,
    pub methods: Vec<(String, usize)>, // (method_name, pc)
}

/// Class — cached from ClassDescr, with fast lookup maps.
#[derive(Clone, Debug)]
pub struct Class {
    pub attributes: HashMap<String, usize>,
    pub methods: HashMap<String, usize>,
}

impl Class {
    fn from_descr(descr: &ClassDescr) -> Self {
        let mut attributes = HashMap::new();
        for (i, name) in descr.attributes.iter().enumerate() {
            attributes.insert(name.clone(), i);
        }
        let mut methods = HashMap::new();
        for (name, pc) in &descr.methods {
            methods.insert(name.clone(), *pc);
        }
        Class {
            attributes,
            methods,
        }
    }
}

/// ConstantPool — stores class descriptors and interned strings.
#[derive(Clone, Debug)]
pub struct ConstantPool {
    pub classdescrs: Vec<ClassDescr>,
    pub strings: Vec<String>,
}

impl ConstantPool {
    pub fn new() -> Self {
        ConstantPool {
            classdescrs: Vec::new(),
            strings: Vec::new(),
        }
    }

    pub fn add_classdescr(
        &mut self,
        attributes: Vec<String>,
        methods: Vec<(String, usize)>,
    ) -> usize {
        let idx = self.classdescrs.len();
        self.classdescrs.push(ClassDescr {
            attributes,
            methods,
        });
        idx
    }

    pub fn add_string(&mut self, s: &str) -> usize {
        if let Some(idx) = self.strings.iter().position(|x| x == s) {
            return idx;
        }
        let idx = self.strings.len();
        self.strings.push(s.to_string());
        idx
    }
}

impl Default for ConstantPool {
    fn default() -> Self {
        Self::new()
    }
}

// ── Object operations ──

impl Obj {
    pub fn is_true(&self) -> bool {
        match self {
            Obj::Int(v) => *v != 0,
            Obj::Nil => false,
            Obj::Cons(_) => true,
            Obj::Instance(_) => true,
        }
    }

    pub fn int_o(&self) -> i64 {
        match self {
            Obj::Int(v) => *v,
            _ => panic!("TypeError: expected int"),
        }
    }

    pub fn to_string_repr(&self) -> String {
        match self {
            Obj::Int(v) => v.to_string(),
            Obj::Nil => "nil".to_string(),
            Obj::Cons(_) => "<ConsObj>".to_string(),
            Obj::Instance(_) => "<Object>".to_string(),
        }
    }

    pub fn add(&self, other: &Obj) -> Obj {
        match (self, other) {
            (Obj::Int(a), Obj::Int(b)) => Obj::Int(a + b),
            (Obj::Int(_), _) | (_, Obj::Int(_)) => panic!("TypeError: cannot add int and list"),
            _ => self.as_lisp_concat(other),
        }
    }

    pub fn sub(&self, other: &Obj) -> Obj {
        Obj::Int(self.int_o() - other.int_o())
    }

    pub fn mul(&self, other: &Obj) -> Obj {
        Obj::Int(self.int_o() * other.int_o())
    }

    pub fn div(&self, other: &Obj) -> Obj {
        match self {
            Obj::Cons(_) | Obj::Nil => {
                let n = other.int_o();
                if n < 0 {
                    panic!("IndexError");
                }
                self.nth(n as usize)
            }
            Obj::Int(a) => Obj::Int(a / other.int_o()),
            _ => panic!("TypeError"),
        }
    }

    pub fn eq_obj(&self, other: &Obj) -> bool {
        match (self, other) {
            (Obj::Int(a), Obj::Int(b)) => a == b,
            (Obj::Nil, Obj::Nil) => true,
            (Obj::Cons(a), Obj::Cons(b)) => a.car.eq_obj(&b.car) && a.cdr.eq_obj(&b.cdr),
            (Obj::Instance(a), Obj::Instance(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }

    pub fn lt_obj(&self, other: &Obj) -> bool {
        self.int_o() < other.int_o()
    }

    pub fn car(&self) -> Obj {
        match self {
            Obj::Cons(c) => c.car.clone(),
            _ => panic!("TypeError: car on non-cons"),
        }
    }

    pub fn cdr(&self) -> Obj {
        match self {
            Obj::Cons(c) => c.cdr.clone(),
            _ => panic!("TypeError: cdr on non-cons"),
        }
    }

    fn nth(&self, n: usize) -> Obj {
        match self {
            Obj::Cons(c) => {
                if n == 0 {
                    c.car.clone()
                } else {
                    c.cdr.nth(n - 1)
                }
            }
            Obj::Nil => panic!("IndexError"),
            _ => panic!("TypeError"),
        }
    }

    fn as_lisp_concat(&self, other: &Obj) -> Obj {
        match self {
            Obj::Nil => other.clone(),
            Obj::Cons(c) => Obj::Cons(Rc::new(ConsCell {
                car: c.car.clone(),
                cdr: c.cdr.as_lisp_concat(other),
            })),
            _ => panic!("TypeError: concat on non-lisp"),
        }
    }

    pub fn getattr(&self, name: &str) -> Obj {
        match self {
            Obj::Instance(inst) => {
                let inst = inst.borrow();
                let i = inst.cls.attributes[name];
                inst.values[i].clone()
            }
            _ => panic!("TypeError: getattr on non-instance"),
        }
    }

    pub fn setattr(&self, name: &str, value: Obj) {
        match self {
            Obj::Instance(inst) => {
                let mut inst = inst.borrow_mut();
                let i = inst.cls.attributes[name];
                inst.values[i] = value;
            }
            _ => panic!("TypeError: setattr on non-instance"),
        }
    }

    pub fn send(&self, name: &str) -> usize {
        match self {
            Obj::Instance(inst) => {
                let inst = inst.borrow();
                inst.cls.methods[name]
            }
            _ => panic!("TypeError: send on non-instance"),
        }
    }
}

impl fmt::Display for Obj {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_repr())
    }
}

// ── Frame ──

struct Frame {
    args: Vec<Obj>,
    pc: usize,
    stack: Vec<Obj>,
}

// ── Interpreter ──

fn char2int(c: u8) -> i64 {
    c as i8 as i64
}

/// Interpret bytecode, returning the integer result.
/// Entry point: wraps inputarg in IntObj, calls interp_eval.
pub fn interp(code: &[u8], pc: usize, inputarg: i64, pool: &ConstantPool) -> i64 {
    let args = vec![Obj::Int(inputarg)];
    match interp_eval(code, pc, args, pool) {
        Some(obj) => obj.int_o(),
        None => 0,
    }
}

/// Evaluate bytecode starting at pc with the given args and pool.
/// Returns the top of stack when RETURN is hit (or None if stack is empty).
pub fn interp_eval(code: &[u8], pc: usize, args: Vec<Obj>, pool: &ConstantPool) -> Option<Obj> {
    let mut frame = Frame {
        args,
        pc,
        stack: Vec::new(),
    };
    let mut pc = frame.pc;

    while pc < code.len() {
        let opcode = code[pc];
        pc += 1;
        let stack = &mut frame.stack;

        if opcode == NOP {
            // no-op
        } else if opcode == NIL {
            stack.push(Obj::Nil);
        } else if opcode == CONS {
            let car = stack.pop().unwrap();
            let cdr = stack.pop().unwrap();
            stack.push(Obj::Cons(Rc::new(ConsCell { car, cdr })));
        } else if opcode == CAR {
            let obj = stack.pop().unwrap();
            stack.push(obj.car());
        } else if opcode == CDR {
            let obj = stack.pop().unwrap();
            stack.push(obj.cdr());
        } else if opcode == PUSH {
            stack.push(Obj::Int(char2int(code[pc])));
            pc += 1;
        } else if opcode == POP {
            stack.pop();
        } else if opcode == SWAP {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(a);
            stack.push(b);
        } else if opcode == ROLL {
            let r = char2int(code[pc]);
            if r < -1 {
                let i = stack.len() as i64 + r;
                if i < 0 {
                    panic!("IndexError");
                }
                let val = stack.pop().unwrap();
                stack.insert(i as usize, val);
            } else if r > 1 {
                let i = stack.len() as i64 - r;
                if i < 0 {
                    panic!("IndexError");
                }
                let val = stack.remove(i as usize);
                stack.push(val);
            }
            pc += 1;
        } else if opcode == PICK {
            let i = char2int(code[pc]) as usize;
            let n = stack.len() - 1 - i;
            stack.push(stack[n].clone());
            pc += 1;
        } else if opcode == PUT {
            let i = char2int(code[pc]) as usize;
            let val = stack.pop().unwrap();
            let n = stack.len() - 1 - i;
            stack[n] = val;
            pc += 1;
        } else if opcode == ADD {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(b.add(&a));
        } else if opcode == SUB {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(b.sub(&a));
        } else if opcode == MUL {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(b.mul(&a));
        } else if opcode == DIV {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(b.div(&a));
        } else if opcode == EQ {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(Obj::Int(if b.eq_obj(&a) { 1 } else { 0 }));
        } else if opcode == NE {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(Obj::Int(if !b.eq_obj(&a) { 1 } else { 0 }));
        } else if opcode == LT {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(Obj::Int(if b.lt_obj(&a) { 1 } else { 0 }));
        } else if opcode == LE {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(Obj::Int(if !a.lt_obj(&b) { 1 } else { 0 }));
        } else if opcode == GT {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(Obj::Int(if a.lt_obj(&b) { 1 } else { 0 }));
        } else if opcode == GE {
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            stack.push(Obj::Int(if !b.lt_obj(&a) { 1 } else { 0 }));
        } else if opcode == BR {
            let old_pc = pc;
            pc = ((pc as i64) + char2int(code[pc]) + 1) as usize;
            let _ = old_pc;
        } else if opcode == BR_COND {
            let cond = stack.pop().unwrap();
            if cond.is_true() {
                pc = ((pc as i64) + char2int(code[pc]) + 1) as usize;
            } else {
                pc += 1;
            }
        } else if opcode == BR_COND_STK {
            let offset = stack.pop().unwrap().int_o();
            let cond = stack.pop().unwrap();
            if cond.is_true() {
                pc = ((pc as i64) + offset) as usize;
            }
        } else if opcode == CALL {
            let offset = char2int(code[pc]);
            pc += 1;
            let call_pc = ((pc as i64) + offset) as usize;
            let res = interp_eval(code, call_pc, vec![Obj::Int(0)], pool);
            if let Some(r) = res {
                stack.push(r);
            }
        } else if opcode == RETURN {
            break;
        } else if opcode == PUSHARG {
            stack.push(frame.args[0].clone());
        } else if opcode == PUSHARGN {
            let idx = char2int(code[pc]) as usize;
            pc += 1;
            stack.push(frame.args[idx].clone());
        } else if opcode == NEW {
            let idx = char2int(code[pc]) as usize;
            pc += 1;
            let descr = &pool.classdescrs[idx];
            let cls = Class::from_descr(descr);
            let num_attrs = cls.attributes.len();
            stack.push(Obj::Instance(Rc::new(RefCell::new(InstanceData {
                cls,
                values: vec![Obj::Nil; num_attrs],
            }))));
        } else if opcode == GETATTR {
            let idx = char2int(code[pc]) as usize;
            pc += 1;
            let name = &pool.strings[idx];
            let a = stack.pop().unwrap();
            stack.push(a.getattr(name));
        } else if opcode == SETATTR {
            let idx = char2int(code[pc]) as usize;
            pc += 1;
            let name = &pool.strings[idx];
            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            b.setattr(name, a);
        } else if opcode == SEND {
            let idx = char2int(code[pc]) as usize;
            pc += 1;
            let mut num_args = char2int(code[pc]) as usize;
            pc += 1;
            num_args += 1; // include self
            let name = &pool.strings[idx];
            let mut meth_args: Vec<Obj> = vec![Obj::Nil; num_args];
            let mut i = num_args;
            while i > 0 {
                i -= 1;
                meth_args[i] = stack.pop().unwrap();
            }
            let meth_pc = meth_args[0].send(name);
            let res = interp_eval(code, meth_pc, meth_args, pool);
            if let Some(r) = res {
                stack.push(r);
            }
        } else if opcode == PRINT {
            let a = stack.pop().unwrap();
            println!("{}", a.to_string_repr());
        } else if opcode == DUMP {
            let parts: Vec<String> = stack.iter().map(|o| o.to_string_repr()).collect();
            println!("[{}]", parts.join(", "));
        } else {
            panic!("unknown opcode: {opcode}");
        }
    }

    if frame.stack.is_empty() {
        None
    } else {
        Some(frame.stack.pop().unwrap())
    }
}

// ── Assembler (port of tlopcode.compile) ──

/// Compile TLC assembly source to bytecode.
pub fn compile(source: &str, pool: &mut ConstantPool) -> Vec<u8> {
    let names = opcode_names();
    let mut bytecode: Vec<i32> = Vec::new();
    let mut labels: HashMap<String, usize> = HashMap::new();
    let mut label_usage: Vec<(String, usize)> = Vec::new();
    let mut method_usage: Vec<(usize, Vec<(String, String)>)> = Vec::new(); // (descr_idx, methods)

    for line in source.lines() {
        // Strip comments
        let mut s = line;
        for comment_start in [";", "#", "//"] {
            if let Some(pos) = s.find(comment_start) {
                s = &s[..pos];
            }
        }
        let s = s.trim();
        if s.is_empty() {
            continue;
        }

        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts[0].ends_with(':') {
            let label = &parts[0][..parts[0].len() - 1];
            assert!(!label.contains(','), "label cannot contain comma");
            labels.insert(label.to_string(), bytecode.len());
            continue;
        }

        let opcode = *names
            .get(parts[0])
            .unwrap_or_else(|| panic!("unknown opcode: {}", parts[0]));
        bytecode.push(opcode as i32);

        if parts.len() > 1 {
            let arg = parts[1];
            if let Ok(n) = arg.parse::<i32>() {
                bytecode.push(n);
            } else if parts[0] == "NEW" {
                // Class descriptor: "attr1,attr2,meth=label"
                let items: Vec<&str> = arg.split(',').filter(|x| !x.is_empty()).collect();
                let mut attributes = Vec::new();
                let mut methods = Vec::new();
                let mut method_labels = Vec::new();
                for item in &items {
                    let item = item.trim();
                    if let Some(eq_pos) = item.find('=') {
                        let methname = &item[..eq_pos];
                        let label = &item[eq_pos + 1..];
                        methods.push((methname.to_string(), 0usize)); // pc resolved later
                        method_labels.push((methname.to_string(), label.to_string()));
                    } else {
                        attributes.push(item.to_string());
                    }
                }
                let idx = pool.add_classdescr(attributes, methods);
                method_usage.push((idx, method_labels));
                bytecode.push(idx as i32);
            } else if parts[0] == "GETATTR" || parts[0] == "SETATTR" {
                let idx = pool.add_string(arg);
                bytecode.push(idx as i32);
            } else if parts[0] == "SEND" {
                // "methodname/num_args"
                let (methname, num_args_str) = arg.split_once('/').unwrap();
                let idx = pool.add_string(methname);
                bytecode.push(idx as i32);
                bytecode.push(num_args_str.parse::<i32>().unwrap());
            } else {
                // Label reference (for BR, BR_COND, CALL)
                label_usage.push((arg.to_string(), bytecode.len()));
                bytecode.push(0);
            }
        }
    }

    // Resolve label references
    for (label, pc) in &label_usage {
        let target = labels[label];
        let offset = target as i32 - *pc as i32 - 1;
        assert!(
            (-128..=127).contains(&offset),
            "label offset {offset} out of range"
        );
        bytecode[*pc] = offset;
    }

    // Resolve method labels in class descriptors
    for (descr_idx, method_list) in &method_usage {
        let descr = &mut pool.classdescrs[*descr_idx];
        for (methname, label) in method_list {
            let pc = labels[label];
            for (name, method_pc) in &mut descr.methods {
                if name == methname {
                    *method_pc = pc;
                }
            }
        }
    }

    bytecode.iter().map(|&b| (b & 0xff) as u8).collect()
}

/// Convert bytecode list to byte string (for test compatibility).
pub fn list2bytecode(ops: &[u8]) -> Vec<u8> {
    ops.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_pool() {
        let mut pool = ConstantPool::new();
        let bytecode = compile(
            "
            NEW foo,bar,meth=f
          f:
            RETURN
        ",
            &mut pool,
        );
        let expected = list2bytecode(&[NEW, 0, RETURN]);
        assert_eq!(expected, bytecode);
        assert_eq!(pool.classdescrs.len(), 1);
        let descr = &pool.classdescrs[0];
        assert_eq!(descr.attributes, vec!["foo", "bar"]);
        assert_eq!(descr.methods.len(), 1);
        assert_eq!(descr.methods[0].0, "meth");
        assert_eq!(descr.methods[0].1, 2);
    }

    #[test]
    fn test_unconditional_branch() {
        let bytecode = compile(
            "
        main:
            BR target
            PUSH 123
            RETURN
        target:
            PUSH 42
            RETURN
        ",
            &mut ConstantPool::new(),
        );
        let res = interp(&bytecode, 0, 0, &ConstantPool::new());
        assert_eq!(res, 42);
    }

    #[test]
    fn test_basic_cons_cell() {
        let bytecode = compile(
            "
            NIL
            PUSHARG
            CONS
            PUSH 1
            CONS
            CDR
            CAR
        ",
            &mut ConstantPool::new(),
        );
        let res = interp(&bytecode, 0, 42, &ConstantPool::new());
        assert_eq!(res, 42);
    }

    #[test]
    fn test_nth() {
        let bytecode = compile(
            "
            NIL
            PUSH 4
            CONS
            PUSH 2
            CONS
            PUSH 1
            CONS
            PUSHARG
            DIV
        ",
            &mut ConstantPool::new(),
        );
        let pool = ConstantPool::new();
        assert_eq!(interp(&bytecode, 0, 0, &pool), 1);
        assert_eq!(interp(&bytecode, 0, 1, &pool), 2);
        assert_eq!(interp(&bytecode, 0, 2, &pool), 4);
    }

    #[test]
    #[should_panic(expected = "IndexError")]
    fn test_nth_out_of_bounds() {
        let bytecode = compile(
            "
            NIL
            PUSH 4
            CONS
            PUSH 2
            CONS
            PUSH 1
            CONS
            PUSHARG
            DIV
        ",
            &mut ConstantPool::new(),
        );
        interp(&bytecode, 0, 3, &ConstantPool::new());
    }

    #[test]
    fn test_concat() {
        let bytecode = compile(
            "
            NIL
            PUSH 4
            CONS
            PUSH 2
            CONS
            NIL
            PUSH 5
            CONS
            PUSH 3
            CONS
            PUSH 1
            CONS
            ADD
            PUSHARG
            DIV
        ",
            &mut ConstantPool::new(),
        );
        let pool = ConstantPool::new();
        for (i, n) in [2, 4, 1, 3, 5].iter().enumerate() {
            assert_eq!(interp(&bytecode, 0, i as i64, &pool), *n);
        }
    }

    #[test]
    fn test_new_obj() {
        let mut pool = ConstantPool::new();
        let bytecode = compile(
            "
            NEW foo,bar
        ",
            &mut pool,
        );
        let obj = interp_eval(&bytecode, 0, vec![Obj::Nil], &pool);
        let obj = obj.unwrap();
        match &obj {
            Obj::Instance(inst) => {
                let inst = inst.borrow();
                assert_eq!(inst.values.len(), 2);
                let mut keys: Vec<&String> = inst.cls.attributes.keys().collect();
                keys.sort();
                assert_eq!(keys, vec!["bar", "foo"]);
            }
            _ => panic!("expected Instance"),
        }
    }

    #[test]
    fn test_setattr() {
        let mut pool = ConstantPool::new();
        let bytecode = compile(
            "
            NEW foo,bar
            PICK 0
            PUSH 42
            SETATTR foo
        ",
            &mut pool,
        );
        let obj = interp_eval(&bytecode, 0, vec![Obj::Nil], &pool).unwrap();
        match &obj {
            Obj::Instance(inst) => {
                let inst = inst.borrow();
                let foo_idx = inst.cls.attributes["foo"];
                assert_eq!(inst.values[foo_idx].int_o(), 42);
                let bar_idx = inst.cls.attributes["bar"];
                assert!(matches!(inst.values[bar_idx], Obj::Nil));
            }
            _ => panic!("expected Instance"),
        }
    }

    #[test]
    fn test_getattr() {
        let mut pool = ConstantPool::new();
        let bytecode = compile(
            "
            NEW foo,bar
            PICK 0
            PUSH 42
            SETATTR bar
            GETATTR bar
        ",
            &mut pool,
        );
        let res = interp_eval(&bytecode, 0, vec![Obj::Nil], &pool).unwrap();
        assert_eq!(res.int_o(), 42);
    }

    #[test]
    fn test_obj_truth() {
        let mut pool = ConstantPool::new();
        let bytecode = compile(
            "
            NEW foo,bar
            BR_COND true
            PUSH 12
            PUSH 1
            BR_COND exit
        true:
            PUSH 42
        exit:
            RETURN
        ",
            &mut pool,
        );
        let res = interp_eval(&bytecode, 0, vec![Obj::Nil], &pool).unwrap();
        assert_eq!(res.int_o(), 42);
    }

    #[test]
    fn test_obj_equality() {
        let mut pool = ConstantPool::new();
        let bytecode = compile(
            "
            NEW foo,bar
            NEW foo,bar
            EQ
        ",
            &mut pool,
        );
        let res = interp_eval(&bytecode, 0, vec![Obj::Nil], &pool).unwrap();
        assert_eq!(res.int_o(), 0);
    }

    #[test]
    fn test_method() {
        let mut pool = ConstantPool::new();
        let bytecode = compile(
            "
            NEW foo,meth=meth
            PICK 0
            PUSH 42
            SETATTR foo
            SEND meth/0
            RETURN
        meth:
            PUSHARG
            GETATTR foo
            RETURN
        ",
            &mut pool,
        );
        let res = interp_eval(&bytecode, 0, vec![Obj::Nil], &pool).unwrap();
        assert_eq!(res.int_o(), 42);
    }

    #[test]
    fn test_method_arg() {
        let mut pool = ConstantPool::new();
        let bytecode = compile(
            "
            NEW foo,meth=meth
            PICK 0
            PUSH 40
            SETATTR foo
            PUSH 2
            SEND meth/1
            RETURN
        meth:
            PUSHARG
            GETATTR foo
            PUSHARGN 1
            ADD
            RETURN
        ",
            &mut pool,
        );
        let res = interp_eval(&bytecode, 0, vec![Obj::Nil], &pool).unwrap();
        assert_eq!(res.int_o(), 42);
    }

    #[test]
    fn test_call_without_return_value() {
        let mut pool = ConstantPool::new();
        let bytecode = compile(
            "
            CALL foo
            PUSH 42
            RETURN
        foo:
            RETURN
        ",
            &mut pool,
        );
        let res = interp_eval(&bytecode, 0, vec![Obj::Nil], &pool).unwrap();
        assert_eq!(res.int_o(), 42);
    }

    fn compile_and_run(source: &str, n: i64) -> i64 {
        let mut pool = ConstantPool::new();
        let bytecode = compile(source, &mut pool);
        let args = vec![Obj::Int(n)];
        let res = interp_eval(&bytecode, 0, args, &pool).unwrap();
        res.int_o()
    }

    #[test]
    fn test_fibo() {
        let source = include_str!("../../../../rpython/jit/tl/fibo.tlc.src");
        assert_eq!(compile_and_run(source, 1), 1);
        assert_eq!(compile_and_run(source, 2), 1);
        assert_eq!(compile_and_run(source, 3), 2);
        assert_eq!(compile_and_run(source, 7), 13);
    }

    #[test]
    fn test_accumulator() {
        let source = include_str!("../../../../rpython/jit/tl/accumulator.tlc.src");
        assert_eq!(compile_and_run(source, 0), 0);
        assert_eq!(compile_and_run(source, 1), 0);
        assert_eq!(compile_and_run(source, 10), 45); // sum(0..10)
        assert_eq!(compile_and_run(source, 20), 190); // sum(0..20)
        assert_eq!(compile_and_run(source, -1), 1);
        assert_eq!(compile_and_run(source, -2), 2);
        assert_eq!(compile_and_run(source, -10), 10);
    }

    #[test]
    fn test_binarytree() {
        let source = include_str!("../../../../rpython/jit/tl/binarytree.tlc.src");
        assert_eq!(compile_and_run(source, 20), 1);
        assert_eq!(compile_and_run(source, 10), 1);
        assert_eq!(compile_and_run(source, 15), 1);
        assert_eq!(compile_and_run(source, 30), 1);
        assert_eq!(compile_and_run(source, 1), 0);
        assert_eq!(compile_and_run(source, 40), 0);
        assert_eq!(compile_and_run(source, 12), 0);
        assert_eq!(compile_and_run(source, 27), 0);
    }
}
