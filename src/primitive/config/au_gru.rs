use {
    super::PrimitiveConfig,
    crate::{
        engine::Engine,
        error::DnnlError,
        memory::descriptor::MemoryDescriptor,
        primitive::{
            attributes::PrimitiveAttributes, descriptor::PrimitiveDescriptor, Backward, Forward,
            Operation, OperationType, PropType,
        },
    },
    onednnl_sys::{
        dnnl_augru_backward_primitive_desc_create, dnnl_augru_forward_primitive_desc_create,
        dnnl_primitive_attr_t, dnnl_rnn_direction_t, dnnl_status_t,
    },
    std::{ffi::c_uint, sync::Arc},
};

pub struct ForwardAuGruConfig<'a> {
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
    attr: &'a PrimitiveAttributes,
}

impl<'a, P: PropType<Forward>> PrimitiveConfig<'a, Forward, P> for ForwardAuGruConfig<'a> {
    fn create_primitive_desc(&self, engine: Arc<Engine>) -> Result<PrimitiveDescriptor, DnnlError> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            dnnl_augru_forward_primitive_desc_create(
                &mut handle,
                engine.handle,
                P::KIND,
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

pub struct BackwardAuGruConfig<'a> {
    direction: dnnl_rnn_direction_t::Type,
    src_layer_desc: &'a MemoryDescriptor,
    src_iter_desc: &'a MemoryDescriptor,
    attention_desc: &'a MemoryDescriptor,
    weights_layer_desc: &'a MemoryDescriptor,
    weights_iter_desc: &'a MemoryDescriptor,
    bias_desc: &'a MemoryDescriptor,
    dst_layer_desc: &'a MemoryDescriptor,
    dst_iter_desc: &'a MemoryDescriptor,
    diff_src_layer_desc: &'a MemoryDescriptor,
    diff_src_iter_desc: &'a MemoryDescriptor,
    diff_attention_desc: &'a MemoryDescriptor,
    diff_weights_layer_desc: &'a MemoryDescriptor,
    diff_weights_iter_desc: &'a MemoryDescriptor,
    diff_bias_desc: &'a MemoryDescriptor,
    diff_dst_layer_desc: &'a MemoryDescriptor,
    diff_dst_iter_desc: &'a MemoryDescriptor,
    flags: c_uint,
    hint_fwd_pd: &'a PrimitiveDescriptor,
    attr: dnnl_primitive_attr_t,
}

impl<'a, P: PropType<Backward>> PrimitiveConfig<'a, Backward, P> for BackwardAuGruConfig<'a> {
    fn create_primitive_desc(&self, engine: Arc<Engine>) -> Result<PrimitiveDescriptor, DnnlError> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            dnnl_augru_backward_primitive_desc_create(
                &mut handle,
                engine.handle,
                P::KIND,
                self.direction,
                self.src_layer_desc.handle,
                self.src_iter_desc.handle,
                self.attention_desc.handle,
                self.weights_layer_desc.handle,
                self.weights_iter_desc.handle,
                self.bias_desc.handle,
                self.dst_layer_desc.handle,
                self.dst_iter_desc.handle,
                self.diff_src_layer_desc.handle,
                self.diff_src_iter_desc.handle,
                self.diff_attention_desc.handle,
                self.diff_weights_layer_desc.handle,
                self.diff_weights_iter_desc.handle,
                self.diff_bias_desc.handle,
                self.diff_dst_layer_desc.handle,
                self.diff_dst_iter_desc.handle,
                self.flags,
                self.hint_fwd_pd.handle,
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

pub struct ForwardAuGru<P: PropType<Forward>> {
    pub prop_type: P,
}

impl<'a, P: PropType<Forward>> Operation<'a, Forward, P> for ForwardAuGru<P> {
    const TYPE: OperationType = OperationType::Augru;
    type OperationConfig = ForwardAuGruConfig<'a>;
}

pub struct BackwardAuGru<P: PropType<Backward>> {
    pub prop_type: P,
}

impl<'a, P: PropType<Backward>> Operation<'a, Backward, P> for BackwardAuGru<P> {
    const TYPE: OperationType = OperationType::Augru;
    type OperationConfig = BackwardAuGruConfig<'a>;
}
