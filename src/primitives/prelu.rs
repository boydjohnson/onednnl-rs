use crate::{
    memory::descriptor::MemoryDescriptor,
    onednnl_sys::{dnnl_prelu_forward_primitive_desc_create, dnnl_status_t},
    primitive::{
        attributes::PrimitiveAttributes, config::PrimitiveConfig, descriptor::PrimitiveDescriptor,
        Forward, Operation, OperationType, PropType,
    },
};

pub struct ForwardPreluConfig<'a> {
    pub src_desc: &'a MemoryDescriptor,
    weights_desc: &'a MemoryDescriptor,
    dst_desc: &'a MemoryDescriptor,
    attr: &'a PrimitiveAttributes,
}

impl<'a, P: PropType<Forward>> PrimitiveConfig<'a, Forward, P> for ForwardPreluConfig<'a> {
    fn create_primitive_desc(
        &self,
        engine: std::sync::Arc<crate::engine::Engine>,
    ) -> Result<crate::primitive::descriptor::PrimitiveDescriptor, crate::error::DnnlError> {
        let mut handle = std::ptr::null_mut();

        let status = unsafe {
            dnnl_prelu_forward_primitive_desc_create(
                &mut handle,
                engine.handle,
                P::KIND,
                self.src_desc.handle,
                self.weights_desc.handle,
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

pub struct ForwardPrelu<P: PropType<Forward>> {
    pub prop_type: P,
}

impl<'a, P: PropType<Forward>> Operation<'a, Forward, P> for ForwardPrelu<P> {
    const TYPE: crate::primitive::OperationType = OperationType::PRelu;
    type OperationConfig = ForwardPreluConfig<'a>;
}
