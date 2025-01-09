use {
    super::PrimitiveConfig,
    crate::{
        memory::descriptor::MemoryDescriptor,
        primitive::{
            attributes::PrimitiveAttributes, descriptor::PrimitiveDescriptor, Forward, PropType,
        },
    },
    onednnl_sys::{dnnl_inner_product_forward_primitive_desc_create, dnnl_status_t},
};

pub struct ForwardInnerProductConfig<'a> {
    pub src_desc: &'a MemoryDescriptor,
    pub weights_desc: &'a MemoryDescriptor,
    pub bias_desc: &'a MemoryDescriptor,
    pub dst_desc: &'a MemoryDescriptor,
    pub attr: &'a PrimitiveAttributes,
}

impl<'a, P: PropType<Forward>> PrimitiveConfig<'a, Forward, P> for ForwardInnerProductConfig<'a> {
    fn create_primitive_desc(
        &self,
        engine: std::sync::Arc<crate::engine::Engine>,
    ) -> Result<crate::primitive::descriptor::PrimitiveDescriptor, crate::error::DnnlError> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            dnnl_inner_product_forward_primitive_desc_create(
                &mut handle,
                engine.handle,
                P::KIND,
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
