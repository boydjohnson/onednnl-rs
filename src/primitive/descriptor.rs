use {super::Operation, crate::error::DnnlError, onednnl_sys::dnnl_primitive_desc_t};

pub struct PrimitiveDescriptor {
    pub(crate) handle: dnnl_primitive_desc_t,
}

impl PrimitiveDescriptor {
    pub fn new<O: Operation>() -> Result<Self, DnnlError> {
        match (O::DIRECTION, O::TYPE) {
            (super::Direction::Forward, super::OperationType::Augru) => {
                todo!()
            }
            (super::Direction::Forward, super::OperationType::BatchNormalization) => todo!(),
            (super::Direction::Forward, super::OperationType::Binary) => todo!(),
            (super::Direction::Forward, super::OperationType::Concat) => todo!(),
            (super::Direction::Forward, super::OperationType::Convolution) => todo!(),
            (super::Direction::Forward, super::OperationType::Deconvolution) => todo!(),
            (super::Direction::Forward, super::OperationType::Eltwise) => todo!(),
            (super::Direction::Forward, super::OperationType::GroupNormalization) => todo!(),
            (super::Direction::Forward, super::OperationType::Gru) => todo!(),
            (super::Direction::Forward, super::OperationType::InnerProduct) => todo!(),
            (super::Direction::Forward, super::OperationType::LayerNormalization) => todo!(),
            (super::Direction::Forward, super::OperationType::LbrAuGru) => todo!(),
            (super::Direction::Forward, super::OperationType::Lrn) => todo!(),
            (super::Direction::Forward, super::OperationType::Lstm) => todo!(),
            (super::Direction::Forward, super::OperationType::MatMul) => todo!(),
            (super::Direction::Forward, super::OperationType::PRelu) => todo!(),
            (super::Direction::Forward, super::OperationType::Shuffle) => todo!(),
            (super::Direction::Forward, super::OperationType::Softmax) => todo!(),
            (super::Direction::Forward, super::OperationType::VanillaRnn) => todo!(),
            (super::Direction::Backward, super::OperationType::Augru) => todo!(),
            (super::Direction::Backward, super::OperationType::BatchNormalization) => todo!(),
            (super::Direction::Backward, super::OperationType::Binary) => todo!(),
            (super::Direction::Backward, super::OperationType::Concat) => todo!(),
            (super::Direction::Backward, super::OperationType::Convolution) => todo!(),
            (super::Direction::Backward, super::OperationType::Deconvolution) => todo!(),
            (super::Direction::Backward, super::OperationType::Eltwise) => todo!(),
            (super::Direction::Backward, super::OperationType::GroupNormalization) => todo!(),
            (super::Direction::Backward, super::OperationType::Gru) => todo!(),
            (super::Direction::Backward, super::OperationType::InnerProduct) => todo!(),
            (super::Direction::Backward, super::OperationType::LayerNormalization) => todo!(),
            (super::Direction::Backward, super::OperationType::LbrAuGru) => todo!(),
            (super::Direction::Backward, super::OperationType::Lrn) => todo!(),
            (super::Direction::Backward, super::OperationType::Lstm) => todo!(),
            (super::Direction::Backward, super::OperationType::MatMul) => todo!(),
            (super::Direction::Backward, super::OperationType::PRelu) => todo!(),
            (super::Direction::Backward, super::OperationType::Shuffle) => todo!(),
            (super::Direction::Backward, super::OperationType::Softmax) => todo!(),
            (super::Direction::Backward, super::OperationType::VanillaRnn) => todo!(),
        }
    }
}
