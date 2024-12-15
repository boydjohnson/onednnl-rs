use {
    crate::{engine::Engine, error::DnnlError},
    onednnl_sys::{
        dnnl_status_t, dnnl_stream_create, dnnl_stream_destroy, dnnl_stream_flags_t, dnnl_stream_t,
        dnnl_stream_wait,
    },
    std::sync::Arc,
};

#[derive(Debug)]
pub struct Stream {
    handle: dnnl_stream_t,
    engine: Arc<Engine>,
}

impl Stream {
    pub const DEFAULT: dnnl_stream_flags_t::Type = dnnl_stream_flags_t::dnnl_stream_default_flags;

    pub const IN_ORDER: dnnl_stream_flags_t::Type = dnnl_stream_flags_t::dnnl_stream_in_order;

    pub const OUT_OF_ORDER: dnnl_stream_flags_t::Type =
        dnnl_stream_flags_t::dnnl_stream_out_of_order;

    /// Create a new Stream
    ///
    /// ```
    /// use onednnl::engine::Engine;
    /// use onednnl::stream::Stream;
    ///
    /// let engine = Engine::new(Engine::CPU, 0).unwrap();
    ///
    /// let stream = Stream::new(engine);
    ///
    /// assert!(stream.is_ok());
    /// ```
    pub fn new(engine: Arc<Engine>) -> Result<Self, DnnlError> {
        Self::new_with_flags(engine, Self::DEFAULT)
    }

    /// Create a new Stream with non-default flags
    ///
    /// ```
    /// use onednnl::engine::Engine;
    /// use onednnl::stream::Stream;
    ///
    /// let engine = Engine::new(Engine::CPU, 0).unwrap();
    ///
    /// let stream = Stream::new_with_flags(engine, Stream::IN_ORDER);
    ///
    /// assert!(stream.is_ok());
    /// ```
    pub fn new_with_flags(
        engine: Arc<Engine>,
        flags: dnnl_stream_flags_t::Type,
    ) -> Result<Self, DnnlError> {
        let mut handle: dnnl_stream_t = std::ptr::null_mut();
        let status = unsafe { dnnl_stream_create(&mut handle, engine.handle, flags) };
        if status == dnnl_status_t::dnnl_success {
            Ok(Self { handle, engine })
        } else {
            Err(status.into())
        }
    }

    /// Wait for all computations in the Stream to complete
    pub fn wait(&self) -> Result<(), DnnlError> {
        let status = unsafe { dnnl_stream_wait(self.handle) };

        if status == dnnl_status_t::dnnl_success {
            Ok(())
        } else {
            Err(status.into())
        }
    }

    /// Get the Engine associated with this Stream
    pub fn get_engine(&self) -> Arc<Engine> {
        Arc::clone(&self.engine)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { dnnl_stream_destroy(self.handle) };
    }
}
