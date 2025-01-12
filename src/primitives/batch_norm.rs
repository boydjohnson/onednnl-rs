use {
    crate::{
        memory::descriptor::MemoryDescriptor,
        primitive::{
            attributes::PrimitiveAttributes, config::PrimitiveConfig,
            descriptor::PrimitiveDescriptor, Forward, Operation, OperationType, PropType,
        },
    },
    onednnl_sys::{dnnl_batch_normalization_forward_primitive_desc_create, dnnl_status_t},
    std::{ffi::c_uint, marker::PhantomData},
};

pub struct ForwardBatchNormConfig {
    src_desc: MemoryDescriptor,
    dst_desc: MemoryDescriptor,
    epsilon: f32,
    flags: c_uint,
    attr: PrimitiveAttributes,
}

impl<'a, P: PropType<Forward>> PrimitiveConfig<'a, Forward, P> for ForwardBatchNormConfig {
    fn create_primitive_desc(
        self,
        engine: std::sync::Arc<crate::engine::Engine>,
    ) -> Result<
        crate::primitive::descriptor::PrimitiveDescriptor<'a, Forward, P, ForwardBatchNormConfig>,
        crate::error::DnnlError,
    > {
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
            Ok(
                PrimitiveDescriptor::<'a, Forward, P, ForwardBatchNormConfig> {
                    handle,
                    config: self,
                    _marker_a: PhantomData,
                    _marker_d: PhantomData,
                    _marker_p: PhantomData,
                },
            )
        } else {
            Err(status.into())
        }
    }
}

pub struct ForwardBatchNorm<P: PropType<Forward>> {
    pub prop_type: P,
}

impl<P: PropType<Forward>> Operation<'_, Forward, P> for ForwardBatchNorm<P> {
    const TYPE: OperationType = OperationType::BatchNormalization;

    type OperationConfig = ForwardBatchNormConfig;
}
