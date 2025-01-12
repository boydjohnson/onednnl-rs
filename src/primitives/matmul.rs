use {
    crate::{
        memory::descriptor::MemoryDescriptor,
        primitive::{
            attributes::PrimitiveAttributes, config::PrimitiveConfig,
            descriptor::PrimitiveDescriptor, Forward, Operation, OperationType, PropType,
        },
    },
    onednnl_sys::{dnnl_matmul_primitive_desc_create, dnnl_status_t},
    std::marker::PhantomData,
};

pub struct ForwardMatMulConfig {
    pub src_desc: MemoryDescriptor,
    pub weights_desc: MemoryDescriptor,
    pub bias_desc: MemoryDescriptor,
    pub dst_desc: MemoryDescriptor,
    pub attr: PrimitiveAttributes,
}

impl<'a, P: PropType<Forward>> PrimitiveConfig<'a, Forward, P> for ForwardMatMulConfig {
    fn create_primitive_desc(
        self,
        engine: std::sync::Arc<crate::engine::Engine>,
    ) -> Result<
        crate::primitive::descriptor::PrimitiveDescriptor<'a, Forward, P, ForwardMatMulConfig>,
        crate::error::DnnlError,
    > {
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            dnnl_matmul_primitive_desc_create(
                &mut handle,
                engine.handle,
                self.src_desc.handle,
                self.weights_desc.handle,
                self.bias_desc.handle,
                self.dst_desc.handle,
                self.attr.handle,
            )
        };
        if status == dnnl_status_t::dnnl_success {
            Ok(PrimitiveDescriptor::<'_, Forward, P, ForwardMatMulConfig> {
                handle,
                config: self,

                _marker_a: PhantomData,
                _marker_d: PhantomData,
                _marker_p: PhantomData,
            })
        } else {
            Err(status.into())
        }
    }
}

pub struct ForwardMatMul<P: PropType<Forward>> {
    pub prop_type: P,
}

impl<P: PropType<Forward>> Operation<'_, Forward, P> for ForwardMatMul<P> {
    const TYPE: OperationType = OperationType::MatMul;

    type OperationConfig = ForwardMatMulConfig;
}
