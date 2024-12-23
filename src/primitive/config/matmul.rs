use onednnl_sys::{dnnl_matmul_primitive_desc_create, dnnl_primitive_attr_t, dnnl_status_t};

use crate::{
    memory::descriptor::MemoryDescriptor,
    primitive::{descriptor::PrimitiveDescriptor, Forward, PropType},
};

use super::PrimitiveConfig;

pub struct ForwardMatMulConfig<'a> {
    pub src_desc: &'a MemoryDescriptor,
    pub weights_desc: &'a MemoryDescriptor,
    pub bias_desc: &'a MemoryDescriptor,
    pub dst_desc: &'a MemoryDescriptor,
    pub attr: dnnl_primitive_attr_t,
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
