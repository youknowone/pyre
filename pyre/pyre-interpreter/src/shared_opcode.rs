use crate::PyError;

type OpcodeResult<T> = Result<T, PyError>;

pub trait SharedOpcodeHandler {
    type Value: Copy;

    fn push_value(&mut self, value: Self::Value) -> OpcodeResult<()>;
    fn pop_value(&mut self) -> OpcodeResult<Self::Value>;
    fn peek_at(&mut self, depth: usize) -> OpcodeResult<Self::Value>;
    fn guard_nonnull_value(&mut self, value: Self::Value) -> OpcodeResult<()> {
        let _ = value;
        Ok(())
    }

    fn make_function(&mut self, code_obj: Self::Value) -> OpcodeResult<Self::Value>;
    fn call_callable(
        &mut self,
        callable: Self::Value,
        args: &[Self::Value],
    ) -> OpcodeResult<Self::Value>;
    fn build_list(&mut self, items: &[Self::Value]) -> OpcodeResult<Self::Value>;
    fn build_tuple(&mut self, items: &[Self::Value]) -> OpcodeResult<Self::Value>;
    fn build_map(&mut self, items: &[Self::Value]) -> OpcodeResult<Self::Value>;
    fn store_subscr(
        &mut self,
        obj: Self::Value,
        key: Self::Value,
        value: Self::Value,
    ) -> OpcodeResult<()>;
    fn list_append(&mut self, list: Self::Value, value: Self::Value) -> OpcodeResult<()>;
    fn unpack_sequence(&mut self, seq: Self::Value, count: usize)
    -> OpcodeResult<Vec<Self::Value>>;
    fn load_attr(&mut self, obj: Self::Value, name: &str) -> OpcodeResult<Self::Value>;
    fn store_attr(&mut self, obj: Self::Value, name: &str, value: Self::Value) -> OpcodeResult<()>;
}

fn pop_n<H: SharedOpcodeHandler + ?Sized>(
    handler: &mut H,
    count: usize,
) -> OpcodeResult<Vec<H::Value>> {
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(handler.pop_value()?);
    }
    values.reverse();
    Ok(values)
}

pub fn opcode_make_function<H: SharedOpcodeHandler + ?Sized>(handler: &mut H) -> OpcodeResult<()> {
    let code_obj = handler.pop_value()?;
    let func = handler.make_function(code_obj)?;
    handler.push_value(func)
}

pub fn opcode_call<H: SharedOpcodeHandler + ?Sized>(
    handler: &mut H,
    nargs: usize,
) -> OpcodeResult<()> {
    // Specialize common arities to avoid Vec heap allocation per call.
    // null_or_self is always discarded here — self-binding for instance
    // method calls is handled in the interpreter's OpcodeStepExecutor::call
    // override, NOT in this shared path, to avoid trace/concrete divergence.
    match nargs {
        0 => {
            let _null_or_self = handler.pop_value()?;
            let callable = handler.pop_value()?;
            let result = handler.call_callable(callable, &[])?;
            handler.push_value(result)
        }
        1 => {
            let a0 = handler.pop_value()?;
            let _null_or_self = handler.pop_value()?;
            let callable = handler.pop_value()?;
            let result = handler.call_callable(callable, &[a0])?;
            handler.push_value(result)
        }
        2 => {
            let a1 = handler.pop_value()?;
            let a0 = handler.pop_value()?;
            let _null_or_self = handler.pop_value()?;
            let callable = handler.pop_value()?;
            let result = handler.call_callable(callable, &[a0, a1])?;
            handler.push_value(result)
        }
        3 => {
            let a2 = handler.pop_value()?;
            let a1 = handler.pop_value()?;
            let a0 = handler.pop_value()?;
            let _null_or_self = handler.pop_value()?;
            let callable = handler.pop_value()?;
            let result = handler.call_callable(callable, &[a0, a1, a2])?;
            handler.push_value(result)
        }
        _ => {
            let args = pop_n(handler, nargs)?;
            let _null_or_self = handler.pop_value()?;
            let callable = handler.pop_value()?;
            let result = handler.call_callable(callable, &args)?;
            handler.push_value(result)
        }
    }
}

pub fn opcode_build_list<H: SharedOpcodeHandler + ?Sized>(
    handler: &mut H,
    size: usize,
) -> OpcodeResult<()> {
    let items = pop_n(handler, size)?;
    let list = handler.build_list(&items)?;
    handler.push_value(list)
}

pub fn opcode_build_tuple<H: SharedOpcodeHandler + ?Sized>(
    handler: &mut H,
    size: usize,
) -> OpcodeResult<()> {
    let items = pop_n(handler, size)?;
    let tuple = handler.build_tuple(&items)?;
    handler.push_value(tuple)
}

pub fn opcode_build_map<H: SharedOpcodeHandler + ?Sized>(
    handler: &mut H,
    size: usize,
) -> OpcodeResult<()> {
    let items = pop_n(handler, size * 2)?;
    let dict = handler.build_map(&items)?;
    handler.push_value(dict)
}

pub fn opcode_store_subscr<H: SharedOpcodeHandler + ?Sized>(handler: &mut H) -> OpcodeResult<()> {
    let key = handler.pop_value()?;
    let obj = handler.pop_value()?;
    let value = handler.pop_value()?;
    handler.store_subscr(obj, key, value)
}

pub fn opcode_list_append<H: SharedOpcodeHandler + ?Sized>(
    handler: &mut H,
    depth: usize,
) -> OpcodeResult<()> {
    if depth == 0 {
        return Err(stack_underflow_error("list append"));
    }
    let value = handler.pop_value()?;
    let list = handler.peek_at(depth - 1)?;
    handler.list_append(list, value)
}

pub fn opcode_unpack_sequence<H: SharedOpcodeHandler + ?Sized>(
    handler: &mut H,
    count: usize,
) -> OpcodeResult<()> {
    let seq = handler.pop_value()?;
    let items = handler.unpack_sequence(seq, count)?;
    for item in items.into_iter().rev() {
        handler.push_value(item)?;
    }
    Ok(())
}

pub fn opcode_load_attr<H: SharedOpcodeHandler + ?Sized>(
    handler: &mut H,
    name: &str,
) -> OpcodeResult<()> {
    let obj = handler.pop_value()?;
    let attr = handler.load_attr(obj, name)?;
    handler.push_value(attr)
}

pub fn opcode_store_attr<H: SharedOpcodeHandler + ?Sized>(
    handler: &mut H,
    name: &str,
) -> OpcodeResult<()> {
    let obj = handler.pop_value()?;
    let value = handler.pop_value()?;
    handler.store_attr(obj, name, value)
}

pub fn stack_underflow_error(context: &str) -> PyError {
    PyError::type_error(format!("stack underflow during {context}"))
}
