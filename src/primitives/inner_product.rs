use {
    crate::{
        memory::descriptor::MemoryDescriptor,
        primitive::{
            attributes::PrimitiveAttributes, config::PrimitiveConfig,
            descriptor::PrimitiveDescriptor, Backward, Forward, Operation, OperationType,
            PropBackwardData, PropBackwardWeights, PropForwardTraining, PropType,
        },
    },
    onednnl_sys::{
        dnnl_inner_product_backward_data_primitive_desc_create,
        dnnl_inner_product_backward_weights_primitive_desc_create,
        dnnl_inner_product_forward_primitive_desc_create, dnnl_status_t,
    },
    std::marker::PhantomData,
};

pub struct ForwardInnerProductConfig {
    pub src_desc: MemoryDescriptor,
    pub weights_desc: MemoryDescriptor,
    pub bias_desc: MemoryDescriptor,
    pub dst_desc: MemoryDescriptor,
    pub attr: PrimitiveAttributes,
}

impl<'a, P: PropType<Forward>> PrimitiveConfig<'a, Forward, P> for ForwardInnerProductConfig {
    fn create_primitive_desc(
        self,
        engine: std::sync::Arc<crate::engine::Engine>,
    ) -> Result<
        crate::primitive::descriptor::PrimitiveDescriptor<
            'a,
            Forward,
            P,
            ForwardInnerProductConfig,
        >,
        crate::error::DnnlError,
    > {
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
            Ok(
                PrimitiveDescriptor::<'_, Forward, P, ForwardInnerProductConfig> {
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

pub struct BackwardWeightsInnerProductConfig<'a> {
    pub src_desc: MemoryDescriptor,
    pub diff_weights_desc: MemoryDescriptor,
    pub diff_bias_desc: MemoryDescriptor,
    pub diff_dst_desc: MemoryDescriptor,
    pub hint_fwd_pd:
        &'a PrimitiveDescriptor<'a, Forward, PropForwardTraining, ForwardInnerProductConfig>,
    pub attr: PrimitiveAttributes,
}

impl<'a> PrimitiveConfig<'a, Backward, PropBackwardWeights>
    for BackwardWeightsInnerProductConfig<'a>
{
    fn create_primitive_desc(
        self,
        engine: std::sync::Arc<crate::engine::Engine>,
    ) -> Result<
        PrimitiveDescriptor<
            'a,
            Backward,
            PropBackwardWeights,
            BackwardWeightsInnerProductConfig<'a>,
        >,
        crate::error::DnnlError,
    > {
        let mut handle = std::ptr::null_mut();

        let status = unsafe {
            dnnl_inner_product_backward_weights_primitive_desc_create(
                &mut handle,
                engine.handle,
                self.src_desc.handle,
                self.diff_weights_desc.handle,
                self.diff_bias_desc.handle,
                self.diff_dst_desc.handle,
                self.hint_fwd_pd.handle,
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

pub struct BackwardDataInnerProductConfig<'a> {
    pub diff_src_desc: MemoryDescriptor,
    pub weights_desc: MemoryDescriptor,
    pub diff_dst_desc: MemoryDescriptor,
    pub hint_fwd_pd:
        &'a PrimitiveDescriptor<'a, Forward, PropForwardTraining, ForwardInnerProductConfig>,
    pub attr: PrimitiveAttributes,
}

impl<'a> PrimitiveConfig<'a, Backward, PropBackwardData> for BackwardDataInnerProductConfig<'a> {
    fn create_primitive_desc(
        self,
        engine: std::sync::Arc<crate::engine::Engine>,
    ) -> Result<
        PrimitiveDescriptor<'a, Backward, PropBackwardData, BackwardDataInnerProductConfig<'a>>,
        crate::error::DnnlError,
    > {
        let mut handle = std::ptr::null_mut();

        let status = unsafe {
            dnnl_inner_product_backward_data_primitive_desc_create(
                &mut handle,
                engine.handle,
                self.diff_src_desc.handle,
                self.weights_desc.handle,
                self.diff_dst_desc.handle,
                self.hint_fwd_pd.handle,
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

pub struct BackwardWeightsInnerProduct;

impl<'a> Operation<'a, Backward, PropBackwardWeights> for BackwardWeightsInnerProduct {
    const TYPE: OperationType = OperationType::InnerProduct;
    type OperationConfig = BackwardWeightsInnerProductConfig<'a>;
}

pub struct BackwardDataInnerProduct;

impl<'a> Operation<'a, Backward, PropBackwardData> for BackwardDataInnerProduct {
    const TYPE: OperationType = OperationType::InnerProduct;
    type OperationConfig = BackwardDataInnerProductConfig<'a>;
}

pub struct ForwardInnerProduct<P: PropType<Forward>> {
    pub prop_type: P,
}

impl<P: PropType<Forward>> Operation<'_, Forward, P> for ForwardInnerProduct<P> {
    const TYPE: OperationType = OperationType::InnerProduct;
    type OperationConfig = ForwardInnerProductConfig;
}
