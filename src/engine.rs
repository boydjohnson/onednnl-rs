use {
    crate::error::DnnlError,
    onednnl_sys::{
        dnnl_engine_create, dnnl_engine_destroy, dnnl_engine_get_count, dnnl_engine_get_kind,
        dnnl_engine_kind_t, dnnl_engine_t, dnnl_status_t,
    },
    std::sync::Arc,
};

#[derive(Debug)]
pub struct Engine {
    pub(crate) handle: dnnl_engine_t,
}

impl Engine {
    pub const GPU: dnnl_engine_kind_t::Type = dnnl_engine_kind_t::dnnl_gpu;

    pub const ANY: dnnl_engine_kind_t::Type = dnnl_engine_kind_t::dnnl_any_engine;

    pub const CPU: dnnl_engine_kind_t::Type = dnnl_engine_kind_t::dnnl_cpu;

    /// Create an Engine of a specific kind
    ///
    /// ```
    /// use onednnl::engine::Engine;
    ///
    /// let engine = Engine::new(Engine::CPU, 0);
    ///
    /// assert!(engine.is_ok());
    /// ```
    pub fn new(kind: dnnl_engine_kind_t::Type, index: usize) -> Result<Arc<Self>, DnnlError> {
        let mut handle: dnnl_engine_t = std::ptr::null_mut();
        let status = unsafe { dnnl_engine_create(&mut handle, kind, index) };
        if status == dnnl_status_t::dnnl_success {
            Ok(Arc::new(Self { handle }))
        } else {
            Err(status.into())
        }
    }

    /// Get the kind of an Engine
    ///
    /// ```
    /// use onednnl::engine::Engine;
    ///
    /// let engine = Engine::new(Engine::CPU, 0);
    ///
    /// assert!(engine.is_ok());
    ///
    /// let engine = engine.unwrap();
    ///
    /// assert_eq!(engine.get_kind(), Ok(Engine::CPU));
    /// ```
    pub fn get_kind(self: &Arc<Self>) -> Result<dnnl_engine_kind_t::Type, DnnlError> {
        let mut kind: dnnl_engine_kind_t::Type = 0; // Initialize a variable to store the kind
        let status = unsafe { dnnl_engine_get_kind(self.handle, &mut kind) }; // Pass a mutable reference

        if status == dnnl_status_t::dnnl_success {
            Ok(kind) // Return the kind if successful
        } else {
            Err(status.into()) // Convert the status into a DnnlError
        }
    }

    /// Get the count of an Engine of a particular type
    ///
    /// ```
    /// use onednnl::engine::Engine;
    ///
    /// assert!(Engine::get_count(Engine::CPU) > 0);
    /// ```
    pub fn get_count(kind: dnnl_engine_kind_t::Type) -> usize {
        unsafe { dnnl_engine_get_count(kind) }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { dnnl_engine_destroy(self.handle) };
    }
}

unsafe impl Sync for Engine {}
unsafe impl Send for Engine {}
