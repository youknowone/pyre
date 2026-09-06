use majit_backend_wasm_host::*;
use wasmtime::{Caller, Engine, Func, Linker, Module, Ref, Store};

fn with_caller(test: fn(&mut Caller<'_, TraceState>)) {
    let engine = Engine::default();
    let mut store = Store::new(&engine, TraceState::default());
    let mut linker = Linker::new(&engine);
    linker
        .func_wrap("test", "run", move |mut caller: Caller<'_, TraceState>| {
            test(&mut caller)
        })
        .unwrap();
    let guest = Module::new(
        &engine,
        r#"(module
        (import "test" "run" (func $run))
        (memory (export "memory") 1)
        (table (export "__indirect_function_table") 2 funcref)
        (func (export "main") call $run))"#,
    )
    .unwrap();
    let instance = linker.instantiate(&mut store, &guest).unwrap();
    store.data_mut().memory = instance.get_memory(&mut store, "memory");
    store.data_mut().table = instance.get_table(&mut store, "__indirect_function_table");
    store.data_mut().trace_base = 2;
    instance
        .get_typed_func::<(), ()>(&mut store, "main")
        .unwrap()
        .call(&mut store, ())
        .unwrap();
}

#[test]
fn trace_pairs_publish_replace_and_free() {
    with_caller(|caller| {
        let module = Module::new(
            caller.engine(),
            r#"(module
            (func (export "trace") (param i32) (result i32) i32.const 7)
            (func (export "trace_wide") (param i32) (result i32) i32.const 8))"#,
        )
        .unwrap();
        let (trace, wide, _) = instantiate(caller, &module, |_, _, _| Ok(())).unwrap();
        let slot = publish(caller, trace, wide).unwrap();
        assert_eq!(slot, 2);
        assert_eq!(table(caller).unwrap().size(&*caller), 4);
        assert_eq!(execute(caller, slot, 0).unwrap(), 7);
        assert!(replace(caller, slot, trace, None).is_err());
        assert_eq!(execute(caller, slot, 0).unwrap(), 7);
        assert!(replace(caller, 0, trace, wide).is_err());
        assert!(execute(caller, 0, 0).is_err());
        free(caller, slot).unwrap();
        assert!(execute(caller, slot, 0).is_err());
        assert!(matches!(
            table(caller).unwrap().get(&mut *caller, slot as u64 + 1),
            Some(Ref::Func(None))
        ));
    });
}

#[test]
fn reflective_call_uses_declared_width_and_zero_extends_result() {
    with_caller(|caller| {
        let target = Func::wrap(&mut *caller, |value: i32, wide: i64| -> i32 {
            assert_eq!(value, -1);
            assert_eq!(wide, 1_i64 << 40);
            -2
        });
        table(caller)
            .unwrap()
            .set(&mut *caller, 1, Ref::Func(Some(target)))
            .unwrap();
        let mem = memory(caller).unwrap();
        let area = CALL_RESULT_OFS as usize;
        mem.write(&mut *caller, CALL_FUNC_OFS as usize, &1_u32.to_le_bytes())
            .unwrap();
        mem.write(
            &mut *caller,
            CALL_ARGS_OFS as usize,
            &(-1_i64).to_le_bytes(),
        )
        .unwrap();
        mem.write(
            &mut *caller,
            CALL_ARGS_OFS as usize + 8,
            &(1_i64 << 40).to_le_bytes(),
        )
        .unwrap();
        residual_call(caller, 0, area as u32, |slot, live| {
            assert_eq!(slot, 1);
            assert!(live);
        })
        .unwrap();
        let mut result = [0; 8];
        mem.read(&*caller, area, &mut result).unwrap();
        assert_eq!(i64::from_le_bytes(result), 0xffff_fffe);
        mem.write(&mut *caller, CALL_FUNC_OFS as usize, &0_u32.to_le_bytes())
            .unwrap();
        residual_call(caller, 0, area as u32, |_, live| assert!(!live)).unwrap();
        mem.read(&*caller, area, &mut result).unwrap();
        assert_eq!(result, [0; 8]);
    });
}
