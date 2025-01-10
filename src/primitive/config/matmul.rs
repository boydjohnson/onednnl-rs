use {
    super::PrimitiveConfig,
    crate::{
        memory::descriptor::MemoryDescriptor,
        primitive::{
            attributes::PrimitiveAttributes, descriptor::PrimitiveDescriptor, Forward, Operation,
            OperationType, PropType,
        },
    },
    onednnl_sys::{dnnl_matmul_primitive_desc_create, dnnl_status_t},
};

pub struct ForwardMatMulConfig<'a> {
    pub src_desc: &'a MemoryDescriptor,
    pub weights_desc: &'a MemoryDescriptor,
    pub bias_desc: &'a MemoryDescriptor,
    pub dst_desc: &'a MemoryDescriptor,
    pub attr: &'a PrimitiveAttributes,
}

impl<'a, P: PropType<Forward>> PrimitiveConfig<'a, Forward, P> for ForwardMatMulConfig<'a> {
    fn create_primitive_desc(
        &self,
        engine: std::sync::Arc<crate::engine::Engine>,
    ) -> Result<crate::primitive::descriptor::PrimitiveDescriptor, crate::error::DnnlError> {
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
            Ok(PrimitiveDescriptor { handle })
        } else {
            Err(status.into())
        }
    }
}

pub struct ForwardMatMul<P: PropType<Forward>> {
    pub prop_type: P,
}

impl<'a, P: PropType<Forward>> Operation<'a, Forward, P> for ForwardMatMul<P> {
    const TYPE: OperationType = OperationType::MatMul;

    type OperationConfig = ForwardMatMulConfig<'a>;
}
