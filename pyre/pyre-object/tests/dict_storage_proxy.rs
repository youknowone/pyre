use pyre_object::dictmultiobject::{
    StrategyKind, register_dict_storage_store_hook, w_dict_get_strategy, w_dict_is_empty_strategy,
    w_dict_new_with_storage_proxy,
};
use pyre_object::{
    PyObjectRef, w_dict_getitem_str, w_dict_setitem_str, w_int_get_value, w_int_new,
};

#[derive(Default)]
struct ProxyStorage {
    entries: Vec<(String, PyObjectRef)>,
}

unsafe fn store_proxy_value(storage: *mut u8, name: &str, value: PyObjectRef) {
    let storage = unsafe { &mut *(storage as *mut ProxyStorage) };
    match storage.entries.iter_mut().find(|(key, _)| key == name) {
        Some((_, stored_value)) => *stored_value = value,
        None => storage.entries.push((name.to_owned(), value)),
    }
}

unsafe fn hash_str_bytes(ptr: *const u8, len: usize) -> i64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    unsafe { std::slice::from_raw_parts(ptr, len) }.hash(&mut hasher);
    hasher.finish() as i64
}

#[test]
fn proxy_bridged_unicode_setitem_forwards_to_storage() {
    pyre_object::dict_eq_hook::register_hash_str_hook(hash_str_bytes);
    register_dict_storage_store_hook(store_proxy_value);

    let mut storage = ProxyStorage::default();
    let dict = w_dict_new_with_storage_proxy(&mut storage as *mut ProxyStorage as *mut u8);
    assert!(unsafe { w_dict_is_empty_strategy(dict) });

    let value = w_int_new(1);
    unsafe {
        w_dict_setitem_str(dict, "x", value);
    }

    assert_eq!(
        unsafe { w_dict_get_strategy(dict).strategy_kind() },
        StrategyKind::Unicode,
    );
    let (stored_key, stored_value) = storage
        .entries
        .first()
        .expect("proxy-bridged Unicode setitem must forward to storage");
    assert_eq!(stored_key, "x");
    assert_eq!(*stored_value, value);
    assert_eq!(unsafe { w_int_get_value(*stored_value) }, 1);
    assert_eq!(
        unsafe { w_int_get_value(w_dict_getitem_str(dict, "x").unwrap()) },
        1,
    );
}
