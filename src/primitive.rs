use {
    crate::{engine::Engine, error::DnnlError, memory::Memory, stream::Stream},
    config::{
        au_gru::{BackwardAuGruConfig, ForwardAuGruConfig},
        batch_norm::ForwardBatchNormConfig,
        binary::ForwardBinaryConfig,
        eltwise::{BackwardEltwiseConfig, ForwardEltwiseConfig},
        inner_product::{
            BackwardDataInnerProductConfig, BackwardWeightsInnerProductConfig,
            ForwardInnerProductConfig,
        },
        matmul::ForwardMatMulConfig,
        reduction::ForwardReductionConfig,
        PrimitiveConfig,
    },
    descriptor::PrimitiveDescriptor,
    onednnl_sys::{
        dnnl_exec_arg_t, dnnl_primitive_create, dnnl_primitive_destroy, dnnl_primitive_execute,
        dnnl_primitive_t, dnnl_prop_kind_t, dnnl_status_t,
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
    Reduction,
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
pub struct PropBackwardData;

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
    const KIND: dnnl_prop_kind_t::Type = dnnl_prop_kind_t::dnnl_forward_inference;
}

impl PropType<Forward> for PropForwardTraining {
    const KIND: dnnl_prop_kind_t::Type = dnnl_prop_kind_t::dnnl_forward_training;
}

impl PropType<Backward> for PropBackward {
    const KIND: dnnl_prop_kind_t::Type = dnnl_prop_kind_t::dnnl_backward;
}

impl PropType<Backward> for PropBackwardWeights {
    const KIND: dnnl_prop_kind_t::Type = dnnl_prop_kind_t::dnnl_backward_weights;
}

impl PropType<Backward> for PropBackwardData {
    const KIND: dnnl_prop_kind_t::Type = dnnl_prop_kind_t::dnnl_backward_data;
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

pub struct ForwardEltwise<P: PropType<Forward>> {
    pub prop_type: P,
}

impl<'a, P: PropType<Forward>> Operation<'a, Forward, P> for ForwardEltwise<P> {
    const TYPE: OperationType = OperationType::Eltwise;

    type OperationConfig = ForwardEltwiseConfig<'a>;
}

pub struct ForwardMatMul<P: PropType<Forward>> {
    pub prop_type: P,
}

impl<'a, P: PropType<Forward>> Operation<'a, Forward, P> for ForwardMatMul<P> {
    const TYPE: OperationType = OperationType::MatMul;

    type OperationConfig = ForwardMatMulConfig<'a>;
}

pub struct ForwardReduction {
    pub prop_type: PropForwardInference,
}

impl<'a> Operation<'a, Forward, PropForwardInference> for ForwardReduction {
    const TYPE: OperationType = OperationType::Reduction;

    type OperationConfig = ForwardReductionConfig<'a>;
}

pub struct BackwardEltwise<T: PropType<Backward>> {
    pub prop_type: T,
}

impl<'a, P: PropType<Backward>> Operation<'a, Backward, P> for BackwardEltwise<P> {
    const TYPE: OperationType = OperationType::Eltwise;
    type OperationConfig = BackwardEltwiseConfig<'a>;
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

impl<'a, P: PropType<Forward>> Operation<'a, Forward, P> for ForwardInnerProduct<P> {
    const TYPE: OperationType = OperationType::InnerProduct;
    type OperationConfig = ForwardInnerProductConfig<'a>;
}

pub struct Primitive {
    pub handle: dnnl_primitive_t,
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
    /// use onednnl::primitive::attributes::PrimitiveAttributes;
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
    ///     attr: &PrimitiveAttributes::new().unwrap(),
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

    pub fn execute<T>(&self, stream: &Stream, args: Vec<ExecArg<'_, T>>) -> Result<(), DnnlError> {
        let c_args: Vec<dnnl_exec_arg_t> = args
            .iter()
            .map(|arg| dnnl_exec_arg_t {
                arg: arg.index,
                memory: arg.mem.handle,
            })
            .collect();

        let status = unsafe {
            dnnl_primitive_execute(
                self.handle,
                stream.handle,
                c_args.len() as i32,
                c_args.as_ptr(),
            )
        };

        if status == dnnl_status_t::dnnl_success {
            Ok(())
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

pub struct ExecArg<'a, T> {
    pub index: i32,
    pub mem: &'a Memory<T>,
}
