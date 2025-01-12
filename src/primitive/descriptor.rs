use {
    super::{config::PrimitiveConfig, Direction, Operation, PropType},
    crate::{engine::Engine, error::DnnlError},
    onednnl_sys::{dnnl_primitive_desc_destroy, dnnl_primitive_desc_t},
    std::{marker::PhantomData, sync::Arc},
};

pub struct PrimitiveDescriptor<
    'a,
    D: Direction,
    P: PropType<D>,
    C: PrimitiveConfig<'a, D, P> + Sized,
> {
    pub handle: dnnl_primitive_desc_t,
    pub config: C,

    pub(crate) _marker_a: PhantomData<&'a ()>,
    pub(crate) _marker_d: PhantomData<D>,
    pub(crate) _marker_p: PhantomData<P>,
}

impl<'a, D: Direction, P: PropType<D>, C: PrimitiveConfig<'a, D, P> + Sized>
    PrimitiveDescriptor<'a, D, P, C>
{
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
    ///     src0_desc: src0_desc,
    ///     src1_desc: src1_desc,
    ///     dst_desc: dst_desc,
    ///     attr: PrimitiveAttributes::new().unwrap(),
    /// };
    ///
    /// // Create a new PrimitiveDescriptor for the forward binary operation
    /// let primitive_descriptor = PrimitiveDescriptor::<_, _, ForwardBinaryConfig>::new::<
    ///     ForwardBinary<PropForwardInference>,
    /// >(binary_config, engine);
    ///
    /// assert!(primitive_descriptor.is_ok());
    /// ```
    pub fn new<O: Operation<'a, D, P, OperationConfig = C>>(
        config: O::OperationConfig,
        engine: Arc<Engine>,
    ) -> Result<PrimitiveDescriptor<'a, D, P, C>, DnnlError> {
        config.create_primitive_desc(engine)
    }
}

impl<'a, D: Direction, P: PropType<D>, C: PrimitiveConfig<'a, D, P>> Drop
    for PrimitiveDescriptor<'a, D, P, C>
{
    fn drop(&mut self) {
        unsafe { dnnl_primitive_desc_destroy(self.handle) };
    }
}
