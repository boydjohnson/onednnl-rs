use {
    crate::{
        memory::descriptor::MemoryDescriptor,
        primitive::{
            attributes::PrimitiveAttributes, config::PrimitiveConfig,
            descriptor::PrimitiveDescriptor, Backward, Forward, Operation, OperationType, PropType,
        },
    },
    onednnl_sys::{
        dnnl_alg_kind_t, dnnl_eltwise_backward_primitive_desc_create,
        dnnl_eltwise_forward_primitive_desc_create, dnnl_primitive_desc_t, dnnl_status_t,
    },
};

pub struct ForwardEltwiseConfig<'a> {
    pub alg_kind: dnnl_alg_kind_t::Type,
    pub src_desc: &'a MemoryDescriptor,
    pub dst_desc: &'a MemoryDescriptor,
    pub alpha: f32,
    pub beta: f32,
    pub attr: &'a PrimitiveAttributes,
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

pub struct Unary;

impl Unary {
    pub const TANH: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_tanh;
    pub const TANH_USE_DST_FOR_BWD: dnnl_alg_kind_t::Type =
        dnnl_alg_kind_t::dnnl_eltwise_tanh_use_dst_for_bwd;
    pub const ABS: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_abs;
    pub const CLIP: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_clip;
    pub const CLIP_V2: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_clip_v2;
    pub const CLIP_V2_USE_DST_FOR_BWD: dnnl_alg_kind_t::Type =
        dnnl_alg_kind_t::dnnl_eltwise_clip_v2_use_dst_for_bwd;
    pub const ELU: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_elu;
    pub const ELU_USE_DST_FOR_BWD: dnnl_alg_kind_t::Type =
        dnnl_alg_kind_t::dnnl_eltwise_elu_use_dst_for_bwd;
    pub const EXP: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_exp;
    pub const EXP_USE_DST_FOR_BWD: dnnl_alg_kind_t::Type =
        dnnl_alg_kind_t::dnnl_eltwise_exp_use_dst_for_bwd;
    pub const GELU_ERF: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_gelu_erf;
    pub const GELU_TANH: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_gelu_tanh;
    pub const HARDSIGMOID: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_hardsigmoid;
    pub const HARDSWISH: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_hardswish;
    pub const LINEAR: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_linear;
    pub const LOG: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_log;
    pub const LOGISTIC: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_logistic;
    pub const LOGISTIC_USE_DST_FOR_BWD: dnnl_alg_kind_t::Type =
        dnnl_alg_kind_t::dnnl_eltwise_logistic_use_dst_for_bwd;
    pub const MISH: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_mish;
    pub const POW: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_pow;
    pub const RELU: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_relu;
    pub const RELU_USE_DST_FOR_BWD: dnnl_alg_kind_t::Type =
        dnnl_alg_kind_t::dnnl_eltwise_relu_use_dst_for_bwd;
    pub const ROUND: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_round;
    pub const SOFT_RELU: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_soft_relu;
    pub const SQRT: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_sqrt;
    pub const SQRT_USE_DST_FOR_BWD: dnnl_alg_kind_t::Type =
        dnnl_alg_kind_t::dnnl_eltwise_sqrt_use_dst_for_bwd;
    pub const SQUARE: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_square;
    pub const SWISH: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_eltwise_swish;
}

pub struct BackwardEltwiseConfig<'a> {
    pub alg_kind: dnnl_alg_kind_t::Type,
    pub diff_src_desc: &'a MemoryDescriptor,
    pub diff_dest_desc: &'a MemoryDescriptor,
    pub data_desc: &'a MemoryDescriptor,
    pub alpha: f32,
    pub beta: f32,
    pub forward_hint_desc: dnnl_primitive_desc_t,
    pub attr: &'a PrimitiveAttributes,
}

impl<'a, P: PropType<Backward>> PrimitiveConfig<'a, Backward, P> for BackwardEltwiseConfig<'a> {
    fn create_primitive_desc(
        &self,
        engine: std::sync::Arc<crate::engine::Engine>,
    ) -> Result<crate::primitive::descriptor::PrimitiveDescriptor, crate::error::DnnlError> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            dnnl_eltwise_backward_primitive_desc_create(
                &mut handle,
                engine.handle,
                self.alg_kind,
                self.diff_src_desc.handle,
                self.diff_dest_desc.handle,
                self.data_desc.handle,
                self.alpha,
                self.beta,
                self.forward_hint_desc,
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

pub struct ForwardEltwise<P: PropType<Forward>> {
    pub prop_type: P,
}

impl<'a, P: PropType<Forward>> Operation<'a, Forward, P> for ForwardEltwise<P> {
    const TYPE: OperationType = OperationType::Eltwise;

    type OperationConfig = ForwardEltwiseConfig<'a>;
}

pub struct BackwardEltwise<T: PropType<Backward>> {
    pub prop_type: T,
}

impl<'a, P: PropType<Backward>> Operation<'a, Backward, P> for BackwardEltwise<P> {
    const TYPE: OperationType = OperationType::Eltwise;
    type OperationConfig = BackwardEltwiseConfig<'a>;
}
