use {
    super::PrimitiveConfig,
    crate::{
        memory::descriptor::MemoryDescriptor,
        primitive::{descriptor::PrimitiveDescriptor, Forward, PropType},
    },
    onednnl_sys::{
        dnnl_alg_kind_t, dnnl_binary_primitive_desc_create, dnnl_primitive_attr_t, dnnl_status_t,
    },
};

pub struct ForwardBinaryConfig<'a> {
    alg_kind: dnnl_alg_kind_t::Type,
    src0_desc: &'a MemoryDescriptor,
    src1_desc: &'a MemoryDescriptor,
    dst_desc: &'a MemoryDescriptor,
    attr: dnnl_primitive_attr_t,
}

impl<'a, P: PropType<Forward>> PrimitiveConfig<'a, Forward, P> for ForwardBinaryConfig<'a> {
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
