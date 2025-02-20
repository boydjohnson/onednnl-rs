use {
    crate::{
        memory::descriptor::MemoryDescriptor,
        primitive::{
            attributes::PrimitiveAttributes, config::PrimitiveConfig,
            descriptor::PrimitiveDescriptor, Forward, Operation, OperationType,
            PropForwardInference,
        },
    },
    onednnl_sys::{dnnl_alg_kind_t, dnnl_reduction_primitive_desc_create, dnnl_status_t},
    std::marker::PhantomData,
};

pub struct ForwardReductionConfig {
    pub alg_kind: dnnl_alg_kind_t::Type,
    pub src_desc: MemoryDescriptor,
    pub dst_desc: MemoryDescriptor,
    pub p: f32,
    pub eps: f32,
    pub attr: PrimitiveAttributes,
}

impl<'a> PrimitiveConfig<'a, Forward, PropForwardInference> for ForwardReductionConfig {
    fn create_primitive_desc(
        self,
        engine: std::sync::Arc<crate::engine::Engine>,
    ) -> Result<
        crate::primitive::descriptor::PrimitiveDescriptor<
            'a,
            Forward,
            PropForwardInference,
            ForwardReductionConfig,
        >,
        crate::error::DnnlError,
    > {
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            dnnl_reduction_primitive_desc_create(
                &mut handle,
                engine.handle,
                self.alg_kind,
                self.src_desc.handle,
                self.dst_desc.handle,
                self.p,
                self.eps,
                self.attr.handle,
            )
        };

        if status == dnnl_status_t::dnnl_success {
            Ok(PrimitiveDescriptor {
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

pub struct Reduction;

impl Reduction {
    pub const MAX: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_reduction_max;
    pub const MIN: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_reduction_min;
    pub const MUL: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_reduction_mul;
    pub const SUM: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_reduction_sum;
    pub const MEAN: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_reduction_mean;
    pub const NORM_LP_MAX: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_reduction_norm_lp_max;
    pub const NORM_LP_SUM: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_reduction_norm_lp_sum;
    pub const NORM_LP_POWER_P_MAX: dnnl_alg_kind_t::Type =
        dnnl_alg_kind_t::dnnl_reduction_norm_lp_power_p_max;
    pub const NORM_LP_POWER_P_SUM: dnnl_alg_kind_t::Type =
        dnnl_alg_kind_t::dnnl_reduction_norm_lp_power_p_sum;
}

pub struct ForwardReduction {
    pub prop_type: PropForwardInference,
}

impl Operation<'_, Forward, PropForwardInference> for ForwardReduction {
    const TYPE: OperationType = OperationType::Reduction;

    type OperationConfig = ForwardReductionConfig;
}
