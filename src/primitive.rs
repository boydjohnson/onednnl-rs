use {
    config::{
        au_gru::{BackwardAuGruConfig, ForwardAuGruConfig},
        PrimitiveConfig,
    },
    onednnl_sys::dnnl_prop_kind_t,
};

pub mod attributes;
pub mod config;
pub mod descriptor;

pub trait Direction {
    const KIND: DirectionT;
}

pub enum DirectionT {
    Forward,
    Backward,
}

pub struct Forward;

pub struct Backward;

impl Direction for Forward {
    const KIND: DirectionT = DirectionT::Forward;
}

impl Direction for Backward {
    const KIND: DirectionT = DirectionT::Backward;
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

#[derive(Debug, Copy, Clone)]
pub enum PropForward {
    Training,
    Inference,
}

pub enum PropBackward {
    Backward,
    Weights,
    Bias,
}

pub struct PropAny;

pub trait Operation<'a, D: Direction, P: PropType<D>> {
    const TYPE: OperationType;

    type OperationConfig: PrimitiveConfig<'a, D, P>;
}

pub trait PropType<D> {
    const KIND: dnnl_prop_kind_t::Type;
}

impl PropType<Forward> for PropForward {
    const KIND: dnnl_prop_kind_t::Type = dnnl_prop_kind_t::dnnl_forward;
}

impl PropType<Backward> for PropBackward {
    const KIND: dnnl_prop_kind_t::Type = dnnl_prop_kind_t::dnnl_backward;
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

// pub struct BatchNorm<D: Direction, P: PropType<D>> {
//     pub direction: D,
//     pub prop_type: P,
// }

// impl<D: Direction, P: PropType<D>> Operation<D, P> for BatchNorm<D, P> {
//     const TYPE: OperationType = OperationType::BatchNormalization;

//     pub type OperationConfig = BatchNormConfig<D, P>;
// }

// pub struct Binary<D: Direction, P: PropType<D>> {
//     pub direction: D,
//     pub prop_type: P,
// }

// impl<D: Direction, P: PropType<D>> Operation for Binary<D, P> {
//     const TYPE: OperationType = OperationType::Binary;
// }

// impl_operation!(ForwardConcat, Direction::Forward, OperationType::Concat);
// impl_operation!(BackwardConcat, Direction::Backward, OperationType::Concat);

// impl_operation!(
//     ForwardConvolution,
//     Direction::Forward,
//     OperationType::Convolution
// );
// impl_operation!(
//     BackwardConvolution,
//     Direction::Backward,
//     OperationType::Convolution
// );

// impl_operation!(
//     ForwardDeconvolution,
//     Direction::Forward,
//     OperationType::Deconvolution
// );
// impl_operation!(
//     BackwardDeconvolution,
//     Direction::Backward,
//     OperationType::Deconvolution
// );

// impl_operation!(ForwardEltwise, Direction::Forward, OperationType::Eltwise);
// impl_operation!(BackwardEltwise, Direction::Backward, OperationType::Eltwise);

// impl_operation!(
//     ForwardGroupNorm,
//     Direction::Forward,
//     OperationType::GroupNormalization
// );
// impl_operation!(
//     BackwardGroupNorm,
//     Direction::Backward,
//     OperationType::GroupNormalization
// );

// impl_operation!(ForwardGru, Direction::Forward, OperationType::Gru);
// impl_operation!(BackwardGru, Direction::Backward, OperationType::Gru);

// impl_operation!(
//     ForwardInnerProduct,
//     Direction::Forward,
//     OperationType::InnerProduct
// );
// impl_operation!(
//     BackwardInnerProduct,
//     Direction::Backward,
//     OperationType::InnerProduct
// );

// impl_operation!(
//     ForwardLayerNorm,
//     Direction::Forward,
//     OperationType::LayerNormalization
// );
// impl_operation!(
//     BackwardLayerNorm,
//     Direction::Backward,
//     OperationType::LayerNormalization
// );

// impl_operation!(ForwardLbrAuGru, Direction::Forward, OperationType::LbrAuGru);
// impl_operation!(
//     BackwardLbrAuGru,
//     Direction::Backward,
//     OperationType::LbrAuGru
// );

// impl_operation!(ForwardLrn, Direction::Forward, OperationType::Lrn);
// impl_operation!(BackwardLrn, Direction::Backward, OperationType::Lrn);

// impl_operation!(ForwardLstm, Direction::Forward, OperationType::Lstm);
// impl_operation!(BackwardLstm, Direction::Backward, OperationType::Lstm);

// impl_operation!(ForwardMatMul, Direction::Forward, OperationType::MatMul);
// impl_operation!(BackwardMatMul, Direction::Backward, OperationType::MatMul);

// impl_operation!(ForwardPRelu, Direction::Forward, OperationType::PRelu);
// impl_operation!(BackwardPRelu, Direction::Backward, OperationType::PRelu);

// impl_operation!(ForwardShuffle, Direction::Forward, OperationType::Shuffle);
// impl_operation!(BackwardShuffle, Direction::Backward, OperationType::Shuffle);

// impl_operation!(
//     ForwardVanillaRnn,
//     Direction::Forward,
//     OperationType::VanillaRnn
// );
// impl_operation!(
//     BackwardVanillaRnn,
//     Direction::Backward,
//     OperationType::VanillaRnn
// );
