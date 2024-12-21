use {
    crate::{engine::Engine, error::DnnlError},
    config::{
        au_gru::{BackwardAuGruConfig, ForwardAuGruConfig},
        batch_norm::ForwardBatchNormConfig,
        binary::ForwardBinaryConfig,
        PrimitiveConfig,
    },
    descriptor::PrimitiveDescriptor,
    onednnl_sys::{
        dnnl_primitive_create, dnnl_primitive_destroy, dnnl_primitive_t, dnnl_prop_kind_t,
        dnnl_status_t,
    },
    std::sync::Arc,
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
pub struct PropForwardTraining;

#[derive(Debug, Clone, Copy)]
pub struct PropForwardInference;

#[derive(Debug, Clone, Copy)]
pub struct PropBackward;

#[derive(Debug, Clone, Copy)]
pub struct PropBackwardBias;

#[derive(Debug, Clone, Copy)]
pub struct PropBackwardWeights;

#[derive(Debug, Clone, Copy)]
pub struct PropAny;

pub trait Operation<'a, D: Direction, P: PropType<D>> {
    const TYPE: OperationType;

    type OperationConfig: PrimitiveConfig<'a, D, P>;
}

pub trait PropType<D> {
    const KIND: dnnl_prop_kind_t::Type;
}

impl PropType<Forward> for PropForwardInference {
    const KIND: dnnl_prop_kind_t::Type = dnnl_prop_kind_t::dnnl_forward;
}

impl PropType<Forward> for PropForwardTraining {
    const KIND: dnnl_prop_kind_t::Type = dnnl_prop_kind_t::dnnl_forward_inference;
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

pub struct ForwardBinary<P: PropType<Forward>> {
    pub prop_type: P,
}

impl<'a> Operation<'a, Forward, PropForwardInference> for ForwardBinary<PropForwardInference> {
    const TYPE: OperationType = OperationType::Binary;

    type OperationConfig = ForwardBinaryConfig<'a>;
}

pub struct ForwardBatchNorm<P: PropType<Forward>> {
    pub prop_type: P,
}

impl<'a, P: PropType<Forward>> Operation<'a, Forward, P> for ForwardBatchNorm<P> {
    const TYPE: OperationType = OperationType::BatchNormalization;

    type OperationConfig = ForwardBatchNormConfig<'a>;
}

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

pub struct Primitive {
    pub(crate) handle: dnnl_primitive_t,
    pub desc: PrimitiveDescriptor,
    pub engine: Arc<Engine>,
}

impl Primitive {
    /// Creates a new `Primitive`.
    ///
    /// # Example
    ///
    /// ```
    /// use onednnl::primitive::{Forward, PropForwardInference};
    /// use onednnl::engine::Engine;
    /// use onednnl::primitive::ForwardBinary;
    /// use onednnl::primitive::config::binary::ForwardBinaryConfig;
    /// use onednnl::primitive::Primitive;
    /// use onednnl_sys::dnnl_alg_kind_t;
    /// use onednnl::memory::format_tag::x;
    /// use onednnl_sys::dnnl_data_type_t::dnnl_f32;
    /// use onednnl::memory::descriptor::MemoryDescriptor;
    ///
    ///
    ///
    /// let engine = Engine::new(Engine::CPU, 0).unwrap();
    ///
    /// let src0_desc = MemoryDescriptor::new::<1, x>([15], dnnl_f32).unwrap();
    /// let src1_desc = MemoryDescriptor::new::<1, x>([15], dnnl_f32).unwrap();
    /// let dst_desc = MemoryDescriptor::new::<1, x>([15], dnnl_f32).unwrap();
    ///
    /// // Define a forward binary config
    /// let binary_config = ForwardBinaryConfig {
    ///     alg_kind: dnnl_alg_kind_t::dnnl_binary_add, // Example: addition operation
    ///     src0_desc: &src0_desc,
    ///     src1_desc: &src1_desc,
    ///     dst_desc: &dst_desc,
    ///     attr: std::ptr::null_mut(), // Default attributes
    /// };
    ///
    /// let primitive = Primitive::new::<_, PropForwardInference, ForwardBinary<_>>(binary_config, engine);
    ///
    /// assert!(primitive.is_ok());
    /// ```
    pub fn new<'a, D: Direction, P: PropType<D>, O: Operation<'a, D, P>>(
        config: O::OperationConfig,
        engine: Arc<Engine>,
    ) -> Result<Primitive, DnnlError> {
        let desc = config.create_primitive_desc(engine.clone())?;
        Self::from_descriptor(desc, engine)
    }

    pub fn from_descriptor(
        desc: PrimitiveDescriptor,
        engine: Arc<Engine>,
    ) -> Result<Primitive, DnnlError> {
        let mut handle = std::ptr::null_mut();

        let status = unsafe { dnnl_primitive_create(&mut handle, desc.handle) };

        if status == dnnl_status_t::dnnl_success {
            Ok(Self {
                handle,
                desc,
                engine,
            })
        } else {
            Err(status.into())
        }
    }
}

impl Drop for Primitive {
    fn drop(&mut self) {
        unsafe {
            dnnl_primitive_destroy(self.handle);
        }
    }
}
