use {
    crate::{
        engine::Engine,
        error::DnnlError,
        memory::descriptor::MemoryDescriptor,
        primitive::{
            attributes::PrimitiveAttributes, config::PrimitiveConfig,
            descriptor::PrimitiveDescriptor, Backward, Forward, Operation, OperationType,
            PropForwardTraining, PropType,
        },
    },
    onednnl_sys::{
        dnnl_augru_backward_primitive_desc_create, dnnl_augru_forward_primitive_desc_create,
        dnnl_rnn_direction_t, dnnl_status_t,
    },
    std::{ffi::c_uint, marker::PhantomData, sync::Arc},
};

pub struct ForwardAuGruConfig {
    direction: dnnl_rnn_direction_t::Type,
    src_layer_desc: MemoryDescriptor,
    src_iter_desc: MemoryDescriptor,
    attention_desc: MemoryDescriptor,
    weights_layer_desc: MemoryDescriptor,
    weights_iter_desc: MemoryDescriptor,
    bias_desc: MemoryDescriptor,
    dst_layer_desc: MemoryDescriptor,
    dst_iter_desc: MemoryDescriptor,
    flags: c_uint,
    attr: PrimitiveAttributes,
}

impl<'a, P: PropType<Forward>> PrimitiveConfig<'a, Forward, P> for ForwardAuGruConfig {
    fn create_primitive_desc(
        self,
        engine: Arc<Engine>,
    ) -> Result<PrimitiveDescriptor<'a, Forward, P, ForwardAuGruConfig>, DnnlError> {
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
            Ok(PrimitiveDescriptor::<'a, Forward, P, ForwardAuGruConfig> {
                handle,
                config: self,

                _marker_a: PhantomData,
                _marker_d: PhantomData,
                _marker_p: PhantomData,
            })
        } else {
            Err(status.into())
        }
    }
}

pub struct BackwardAuGruConfig<'a> {
    direction: dnnl_rnn_direction_t::Type,
    src_layer_desc: MemoryDescriptor,
    src_iter_desc: MemoryDescriptor,
    attention_desc: MemoryDescriptor,
    weights_layer_desc: MemoryDescriptor,
    weights_iter_desc: MemoryDescriptor,
    bias_desc: MemoryDescriptor,
    dst_layer_desc: MemoryDescriptor,
    dst_iter_desc: MemoryDescriptor,
    diff_src_layer_desc: MemoryDescriptor,
    diff_src_iter_desc: MemoryDescriptor,
    diff_attention_desc: MemoryDescriptor,
    diff_weights_layer_desc: MemoryDescriptor,
    diff_weights_iter_desc: MemoryDescriptor,
    diff_bias_desc: MemoryDescriptor,
    diff_dst_layer_desc: MemoryDescriptor,
    diff_dst_iter_desc: MemoryDescriptor,
    flags: c_uint,
    hint_fwd_pd: &'a PrimitiveDescriptor<'a, Forward, PropForwardTraining, ForwardAuGruConfig>,
    attr: PrimitiveAttributes,
}

impl<'a, P: PropType<Backward>> PrimitiveConfig<'a, Backward, P> for BackwardAuGruConfig<'a> {
    fn create_primitive_desc(
        self,
        engine: Arc<Engine>,
    ) -> Result<PrimitiveDescriptor<'a, Backward, P, BackwardAuGruConfig<'a>>, DnnlError> {
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
                self.attr.handle,
            )
        };

        if status == dnnl_status_t::dnnl_success {
            Ok(
                PrimitiveDescriptor::<'a, Backward, P, BackwardAuGruConfig<'a>> {
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

pub struct ForwardAuGru<P: PropType<Forward>> {
    pub prop_type: P,
}

impl<P: PropType<Forward>> Operation<'_, Forward, P> for ForwardAuGru<P> {
    const TYPE: OperationType = OperationType::Augru;
    type OperationConfig = ForwardAuGruConfig;
}

pub struct BackwardAuGru<P: PropType<Backward>> {
    pub prop_type: P,
}

impl<'a, P: PropType<Backward>> Operation<'a, Backward, P> for BackwardAuGru<P> {
    const TYPE: OperationType = OperationType::Augru;
    type OperationConfig = BackwardAuGruConfig<'a>;
}
