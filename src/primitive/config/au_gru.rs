use {
    super::PrimitiveConfig,
    crate::{
        engine::Engine,
        error::DnnlError,
        memory::descriptor::MemoryDescriptor,
        primitive::{descriptor::PrimitiveDescriptor, ForwardAuGru},
    },
    onednnl_sys::{
        dnnl_augru_forward_primitive_desc_create, dnnl_primitive_attr_t, dnnl_prop_kind_t,
        dnnl_rnn_direction_t, dnnl_status_t,
    },
    std::{ffi::c_uint, sync::Arc},
};

pub struct AuGruConfig<'a> {
    prop_kind: dnnl_prop_kind_t::Type,
    direction: dnnl_rnn_direction_t::Type,
    src_layer_desc: &'a MemoryDescriptor,
    src_iter_desc: &'a MemoryDescriptor,
    attention_desc: &'a MemoryDescriptor,
    weights_layer_desc: &'a MemoryDescriptor,
    weights_iter_desc: &'a MemoryDescriptor,
    bias_desc: &'a MemoryDescriptor,
    dst_layer_desc: &'a MemoryDescriptor,
    dst_iter_desc: &'a MemoryDescriptor,
    flags: c_uint,
    attr: dnnl_primitive_attr_t,
}

impl<'a> PrimitiveConfig<ForwardAuGru> for AuGruConfig<'a> {
    fn create_primitive_desc(&self, engine: Arc<Engine>) -> Result<PrimitiveDescriptor, DnnlError> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            dnnl_augru_forward_primitive_desc_create(
                &mut handle,
                engine.handle,
                self.prop_kind,
                self.direction,
                self.src_layer_desc.handle,
                self.src_iter_desc.handle,
                self.attention_desc.handle,
                self.weights_layer_desc.handle,
                self.weights_iter_desc.handle,
                self.bias_desc.handle,
                self.dst_layer_desc.handle,
                self.dst_iter_desc.handle,
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
