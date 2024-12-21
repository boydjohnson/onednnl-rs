use {
    super::PrimitiveConfig,
    crate::{
        memory::descriptor::MemoryDescriptor,
        primitive::{descriptor::PrimitiveDescriptor, Forward, PropForwardInference},
    },
    onednnl_sys::{
        dnnl_alg_kind_t, dnnl_binary_primitive_desc_create, dnnl_primitive_attr_t, dnnl_status_t,
    },
};

pub struct ForwardBinaryConfig<'a> {
    pub alg_kind: dnnl_alg_kind_t::Type,
    pub src0_desc: &'a MemoryDescriptor,
    pub src1_desc: &'a MemoryDescriptor,
    pub dst_desc: &'a MemoryDescriptor,
    pub attr: dnnl_primitive_attr_t,
}

impl<'a> PrimitiveConfig<'a, Forward, PropForwardInference> for ForwardBinaryConfig<'a> {
    fn create_primitive_desc(
        &self,
        engine: std::sync::Arc<crate::engine::Engine>,
    ) -> Result<crate::primitive::descriptor::PrimitiveDescriptor, crate::error::DnnlError> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            dnnl_binary_primitive_desc_create(
                &mut handle,
                engine.handle,
                self.alg_kind,
                self.src0_desc.handle,
                self.src1_desc.handle,
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
