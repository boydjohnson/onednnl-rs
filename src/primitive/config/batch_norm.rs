use {
    crate::{
        memory::descriptor::MemoryDescriptor,
        primitive::{
            attributes::PrimitiveAttributes, descriptor::PrimitiveDescriptor, Forward, Operation,
            OperationType, PropType,
        },
    },
    onednnl_sys::{dnnl_batch_normalization_forward_primitive_desc_create, dnnl_status_t},
    std::ffi::c_uint,
};

use super::PrimitiveConfig;

pub struct ForwardBatchNormConfig<'a> {
    src_desc: &'a MemoryDescriptor,
    dst_desc: &'a MemoryDescriptor,
    epsilon: f32,
    flags: c_uint,
    attr: &'a PrimitiveAttributes,
}

impl<'a, P: PropType<Forward>> PrimitiveConfig<'a, Forward, P> for ForwardBatchNormConfig<'a> {
    fn create_primitive_desc(
        &self,
        engine: std::sync::Arc<crate::engine::Engine>,
    ) -> Result<crate::primitive::descriptor::PrimitiveDescriptor, crate::error::DnnlError> {
        let mut handle = std::ptr::null_mut();

        let status = unsafe {
            dnnl_batch_normalization_forward_primitive_desc_create(
                &mut handle,
                engine.handle,
                P::KIND,
                self.src_desc.handle,
                self.dst_desc.handle,
                self.epsilon,
                self.flags,
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

pub struct ForwardBatchNorm<P: PropType<Forward>> {
    pub prop_type: P,
}

impl<'a, P: PropType<Forward>> Operation<'a, Forward, P> for ForwardBatchNorm<P> {
    const TYPE: OperationType = OperationType::BatchNormalization;

    type OperationConfig = ForwardBatchNormConfig<'a>;
}
