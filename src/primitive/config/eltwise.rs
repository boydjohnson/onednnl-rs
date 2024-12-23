use {
    super::PrimitiveConfig,
    crate::{
        memory::descriptor::MemoryDescriptor,
        primitive::{descriptor::PrimitiveDescriptor, Forward, PropType},
    },
    onednnl_sys::{
        dnnl_alg_kind_t, dnnl_eltwise_forward_primitive_desc_create, dnnl_primitive_attr_t,
        dnnl_status_t,
    },
};

pub struct ForwardEltwiseConfig<'a> {
    pub alg_kind: dnnl_alg_kind_t::Type,
    pub src_desc: &'a MemoryDescriptor,
    pub dst_desc: &'a MemoryDescriptor,
    pub alpha: f32,
    pub beta: f32,
    pub attr: dnnl_primitive_attr_t,
}

impl<'a, P: PropType<Forward>> PrimitiveConfig<'a, Forward, P> for ForwardEltwiseConfig<'a> {
    fn create_primitive_desc(
        &self,
        engine: std::sync::Arc<crate::engine::Engine>,
    ) -> Result<crate::primitive::descriptor::PrimitiveDescriptor, crate::error::DnnlError> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            dnnl_eltwise_forward_primitive_desc_create(
                &mut handle,
                engine.handle,
                P::KIND,
                self.alg_kind,
                self.src_desc.handle,
                self.dst_desc.handle,
                self.alpha,
                self.beta,
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
