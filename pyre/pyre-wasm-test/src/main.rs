use pyre_interpreter::*;

fn main() {
    pyre_interpreter::importing::install_builtin_modules();

    let source = "print(1 + 2)";
    let code = match compile_source(source, Mode::Exec) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("SyntaxError: {e}");
            return;
        }
    };

    let execution_context = std::rc::Rc::new(PyExecutionContext::default());
    let mut frame = pyframe::PyFrame::new_with_context(code, execution_context);

    eprintln!("[wasi-test] calling eval_with_jit...");
    match pyre_jit::eval::eval_with_jit(&mut frame) {
        Ok(result) => {
            if !result.is_null() && !unsafe { pyre_object::is_none(result) } {
                println!("{}", PyDisplay(result));
            }
            eprintln!("[wasi-test] success");
        }
        Err(e) => eprintln!("Error: {e}"),
    }
}
