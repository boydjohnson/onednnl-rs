pub mod attributes;
pub mod config;
pub mod descriptor;

pub enum Direction {
    Forward,
    Backward,
}

pub enum OperationType {
    Augru,
    BatchNormalization,
    Binary,
    Concat,
    Convolution,
    Deconvolution,
    Eltwise,
    GroupNormalization,
    Gru,
    InnerProduct,
    LayerNormalization,
    LbrAuGru,
    Lrn,
    Lstm,
    MatMul,
    PRelu,
    Shuffle,
    Softmax,
    VanillaRnn,
}

pub trait Operation {
    const DIRECTION: Direction;

    const TYPE: OperationType;
}

macro_rules! impl_operation {
    ($t:ident, $dir:expr, $op:expr) => {
        pub struct $t;

        impl Operation for $t {
            const DIRECTION: Direction = $dir;
            const TYPE: OperationType = $op;
        }
    };
}

impl_operation!(ForwardAuGru, Direction::Forward, OperationType::Augru);
impl_operation!(BackwardAuGru, Direction::Backward, OperationType::Augru);

impl_operation!(
    ForwardBatchNorm,
    Direction::Forward,
    OperationType::BatchNormalization
);
impl_operation!(
    BackwardBatchNorm,
    Direction::Backward,
    OperationType::BatchNormalization
);

impl_operation!(ForwardBinary, Direction::Forward, OperationType::Binary);
impl_operation!(BackwardBinary, Direction::Backward, OperationType::Binary);

impl_operation!(ForwardConcat, Direction::Forward, OperationType::Concat);
impl_operation!(BackwardConcat, Direction::Backward, OperationType::Concat);

impl_operation!(
    ForwardConvolution,
    Direction::Forward,
    OperationType::Convolution
);
impl_operation!(
    BackwardConvolution,
    Direction::Backward,
    OperationType::Convolution
);

impl_operation!(
    ForwardDeconvolution,
    Direction::Forward,
    OperationType::Deconvolution
);
impl_operation!(
    BackwardDeconvolution,
    Direction::Backward,
    OperationType::Deconvolution
);

impl_operation!(ForwardEltwise, Direction::Forward, OperationType::Eltwise);
impl_operation!(BackwardEltwise, Direction::Backward, OperationType::Eltwise);

impl_operation!(
    ForwardGroupNorm,
    Direction::Forward,
    OperationType::GroupNormalization
);
impl_operation!(
    BackwardGroupNorm,
    Direction::Backward,
    OperationType::GroupNormalization
);

impl_operation!(ForwardGru, Direction::Forward, OperationType::Gru);
impl_operation!(BackwardGru, Direction::Backward, OperationType::Gru);

impl_operation!(
    ForwardInnerProduct,
    Direction::Forward,
    OperationType::InnerProduct
);
impl_operation!(
    BackwardInnerProduct,
    Direction::Backward,
    OperationType::InnerProduct
);

impl_operation!(
    ForwardLayerNorm,
    Direction::Forward,
    OperationType::LayerNormalization
);
impl_operation!(
    BackwardLayerNorm,
    Direction::Backward,
    OperationType::LayerNormalization
);

impl_operation!(ForwardLbrAuGru, Direction::Forward, OperationType::LbrAuGru);
impl_operation!(
    BackwardLbrAuGru,
    Direction::Backward,
    OperationType::LbrAuGru
);

impl_operation!(ForwardLrn, Direction::Forward, OperationType::Lrn);
impl_operation!(BackwardLrn, Direction::Backward, OperationType::Lrn);

impl_operation!(ForwardLstm, Direction::Forward, OperationType::Lstm);
impl_operation!(BackwardLstm, Direction::Backward, OperationType::Lstm);

impl_operation!(ForwardMatMul, Direction::Forward, OperationType::MatMul);
impl_operation!(BackwardMatMul, Direction::Backward, OperationType::MatMul);

impl_operation!(ForwardPRelu, Direction::Forward, OperationType::PRelu);
impl_operation!(BackwardPRelu, Direction::Backward, OperationType::PRelu);

impl_operation!(ForwardShuffle, Direction::Forward, OperationType::Shuffle);
impl_operation!(BackwardShuffle, Direction::Backward, OperationType::Shuffle);

impl_operation!(
    ForwardVanillaRnn,
    Direction::Forward,
    OperationType::VanillaRnn
);
impl_operation!(
    BackwardVanillaRnn,
    Direction::Backward,
    OperationType::VanillaRnn
);
