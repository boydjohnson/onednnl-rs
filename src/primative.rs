pub mod attributes;
pub mod descriptor;

pub enum Direction {
    Forward,
    Backwards,
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
    PReLu,
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
impl_operation!(BackwardsAuGru, Direction::Backwards, OperationType::Augru);

impl_operation!(
    ForwardBatchNorm,
    Direction::Forward,
    OperationType::BatchNormalization
);
impl_operation!(
    BackwardsBatchNorm,
    Direction::Backwards,
    OperationType::BatchNormalization
);

impl_operation!(ForwardBinary, Direction::Forward, OperationType::Binary);
impl_operation!(BackwardsBinary, Direction::Backwards, OperationType::Binary);

impl_operation!(ForwardConcat, Direction::Forward, OperationType::Concat);
impl_operation!(BackwardsConcat, Direction::Backwards, OperationType::Concat);

impl_operation!(
    ForwardConvolution,
    Direction::Forward,
    OperationType::Convolution
);
impl_operation!(
    BackwardsConvolution,
    Direction::Backwards,
    OperationType::Convolution
);
