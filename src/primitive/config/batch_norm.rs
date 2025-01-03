use std::ffi::c_uint;

use onednnl_sys::{
    const_dnnl_primitive_attr_t, dnnl_batch_normalization_forward_primitive_desc_create,
    dnnl_status_t,
};

use crate::{
    memory::descriptor::MemoryDescriptor,
    primitive::{descriptor::PrimitiveDescriptor, Forward, PropType},
};

use super::PrimitiveConfig;

pub struct ForwardBatchNormConfig<'a> {
    src_desc: &'a MemoryDescriptor,
    dst_desc: &'a MemoryDescriptor,
    epsilon: f32,
    flags: c_uint,
    attr: const_dnnl_primitive_attr_t,
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
                self.attr,
            )
        };

        if status == dnnl_status_t::dnnl_success {
            Ok(PrimitiveDescriptor { handle })
        } else {
            Err(status.into())
        }
    }
}
