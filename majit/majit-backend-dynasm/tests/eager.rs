fn new_backend() -> Box<dyn majit_backend::Backend> {
    Box::new(majit_backend_dynasm::runner::DynasmBackend::new())
}

#[path = "../../majit-backend/tests/support/eager.rs"]
mod conformance;
