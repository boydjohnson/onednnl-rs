use {
    super::{config::PrimitiveConfig, Direction, Operation, PropType},
    crate::{engine::Engine, error::DnnlError},
    onednnl_sys::{dnnl_primitive_desc_destroy, dnnl_primitive_desc_t},
    std::sync::Arc,
};

pub struct PrimitiveDescriptor {
    pub handle: dnnl_primitive_desc_t,
}

impl PrimitiveDescriptor {
    /// Creates a new `PrimitiveDescriptor`.
    ///
    /// # Example
    ///
    /// ```
    /// use {
    ///     onednnl::{
    ///         engine::Engine,
    ///         memory::{descriptor::MemoryDescriptor, format_tag::x},
    ///         primitive::{
    ///             attributes::PrimitiveAttributes, descriptor::PrimitiveDescriptor, Forward,
    ///             PropForwardInference,
    ///         },
    ///         primitives::binary::{ForwardBinary, ForwardBinaryConfig},
    ///     },
    ///     onednnl_sys::{dnnl_alg_kind_t, dnnl_data_type_t::dnnl_f32},
    /// };
    ///
    /// let engine = Engine::new(Engine::CPU, 0).unwrap();
    ///
    /// let src0_desc = MemoryDescriptor::new::<1, x>([15], dnnl_f32).unwrap();
    /// let src1_desc = MemoryDescriptor::new::<1, x>([15], dnnl_f32).unwrap();
    /// let dst_desc = MemoryDescriptor::new::<1, x>([15], dnnl_f32).unwrap();
    ///
    /// // Define a forward binary config
    /// let binary_config = ForwardBinaryConfig {
    ///     alg_kind: dnnl_alg_kind_t::dnnl_binary_add, // Example: addition operation
    ///     src0_desc: &src0_desc,
    ///     src1_desc: &src1_desc,
    ///     dst_desc: &dst_desc,
    ///     attr: &PrimitiveAttributes::new().unwrap(),
    /// };
    ///
    /// // Create a new PrimitiveDescriptor for the forward binary operation
    /// let primitive_descriptor = PrimitiveDescriptor::new::<
    ///     Forward,
    ///     _,
    ///     ForwardBinary<PropForwardInference>,
    /// >(binary_config, engine);
    ///
    /// assert!(primitive_descriptor.is_ok());
    /// ```
    pub fn new<'a, D: Direction, P: PropType<D>, O: Operation<'a, D, P>>(
        config: O::OperationConfig,
        engine: Arc<Engine>,
    ) -> Result<Self, DnnlError> {
        config.create_primitive_desc(engine)
    }
}

impl Drop for PrimitiveDescriptor {
    fn drop(&mut self) {
        unsafe { dnnl_primitive_desc_destroy(self.handle) };
    }
}
