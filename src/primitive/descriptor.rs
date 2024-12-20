use {
    super::{
        config::{au_gru::AuGruConfig, PrimitiveConfig},
        Direction, DirectionT, Operation, PropType,
    },
    crate::{engine::Engine, error::DnnlError},
    onednnl_sys::dnnl_primitive_desc_t,
    std::sync::Arc,
};

pub struct PrimitiveDescriptor {
    pub(crate) handle: dnnl_primitive_desc_t,
}

impl PrimitiveDescriptor {
    pub fn new<'a, D: Direction, P: PropType<D>, O: Operation<'a, D, P>>(
        config: O::OperationConfig,
        engine: Arc<Engine>,
    ) -> Result<Self, DnnlError> {
        match O::TYPE {
            super::OperationType::Augru => config.create_primitive_desc(engine),
            super::OperationType::BatchNormalization => todo!(),
            super::OperationType::Binary => todo!(),
            super::OperationType::Concat => todo!(),
            super::OperationType::Convolution => todo!(),
            super::OperationType::Deconvolution => todo!(),
            super::OperationType::Eltwise => todo!(),
            super::OperationType::GroupNormalization => todo!(),
            super::OperationType::Gru => todo!(),
            super::OperationType::InnerProduct => todo!(),
            super::OperationType::LayerNormalization => todo!(),
            super::OperationType::LbrAuGru => todo!(),
            super::OperationType::Lrn => todo!(),
            super::OperationType::Lstm => todo!(),
            super::OperationType::MatMul => todo!(),
            super::OperationType::PRelu => todo!(),
            super::OperationType::Shuffle => todo!(),
            super::OperationType::Softmax => todo!(),
            super::OperationType::VanillaRnn => todo!(),
        }
    }
}
