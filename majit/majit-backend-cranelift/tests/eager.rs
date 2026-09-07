fn new_backend() -> Box<dyn majit_backend::Backend> {
    Box::new(majit_backend_cranelift::CraneliftBackend::new())
}

#[path = "../../majit-backend/tests/support/eager.rs"]
mod conformance;
