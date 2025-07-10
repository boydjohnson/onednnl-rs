use {
    super::spec::OpSpec,
    crate::{
        error::DnnlError,
        graph::{
            op::OneDNNGraphOp,
            ops_builders::{
                abs::AbsSpec, abs_backward::AbsBackwardSpec, add::AddSpec, avg_pool::AvgPoolSpec,
                avg_pool_backward::AvgPoolBackwardSpec,
                batch_norm_inference::BatchNormInferenceSpec, clamp::ClampSpec,
                convolution::ConvolutionSpec, elu::EluSpec, end::EndSpec, exp::ExpSpec,
                matmul::MatMulSpec, reorder::ReorderSpec, softmax::SoftmaxSpec,
                static_reshape::StaticReshapeSpec,
            },
            spec::{AttrValue, RequiredAttrs},
            tensor::logical::LogicalTensor,
        },
    },
    std::marker::PhantomData,
};

pub mod abs;
pub mod abs_backward;
pub mod add;
pub mod avg_pool;
pub mod avg_pool_backward;
pub mod batch_norm_inference;
pub mod bias_add;
pub mod clamp;
pub mod concat;
pub mod convolution;
pub mod divide;
pub mod elu;
pub mod end;
pub mod exp;
pub mod gelu;
pub mod matmul;
pub mod max_pool;
pub mod multiply;
pub mod reorder;
pub mod sigmoid;
pub mod softmax;
pub mod static_reshape;
pub mod subtract;

pub type OpAttrKind = onednnl_sys::dnnl_graph_op_attr_t::Type;

pub struct OpBuilder<K: OpSpec> {
    id: usize,
    inputs: Vec<LogicalTensor>,
    outputs: Vec<LogicalTensor>,
    attrs: Vec<(OpAttrKind, AttrValue)>,
    _marker: PhantomData<K>,
}

impl<K: OpSpec> OpBuilder<K> {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            inputs: vec![],
            outputs: vec![],
            attrs: vec![],
            _marker: PhantomData,
        }
    }

    pub fn required(mut self, r: impl Into<RequiredAttrs>) -> Self {
        if let RequiredAttrs::Some(iter) = r.into() {
            for (k, v) in iter {
                self.attrs.push((k, v));
            }
        }
        self
    }

    pub fn with_input(mut self, t: LogicalTensor) -> Self {
        self.inputs.push(t);
        self
    }

    pub fn with_output(mut self, t: LogicalTensor) -> Self {
        self.outputs.push(t);
        self
    }

    pub fn with_extra_attr(
        mut self,
        key: impl Into<OpAttrKind>,
        val: impl Into<AttrValue>,
    ) -> Self {
        self.attrs.push((key.into(), val.into()));
        self
    }

    pub fn build(self, verbose_name: &str) -> Result<OneDNNGraphOp, DnnlError> {
        let mut op = OneDNNGraphOp::new(self.id, K::KIND, verbose_name)?;

        for (k, v) in self.attrs {
            op.set_attribute(&k, &v)?;
        }

        for t in self.inputs {
            op.add_input(&t)?;
        }
        for t in self.outputs {
            op.add_output(&t)?;
        }

        Ok(op)
    }
}

pub type AbsOpBuilder = OpBuilder<AbsSpec>;
pub type AbsBackwardOpBuilder = OpBuilder<AbsBackwardSpec>;
pub type AddOpBuilder = OpBuilder<AddSpec>;
pub type MatMulOpBuilder = OpBuilder<MatMulSpec>;
pub type ClampOpBuilder = OpBuilder<ClampSpec>;
pub type AvgPoolOpBuilder = OpBuilder<AvgPoolSpec>;
pub type ConvOpBuilder = OpBuilder<ConvolutionSpec>;
pub type SoftmaxOpBuilder = OpBuilder<SoftmaxSpec>;
pub type EndOpBuilder = OpBuilder<EndSpec>;
pub type StaticReshapeOpBuilder = OpBuilder<StaticReshapeSpec>;
pub type ReorderOpBuilder = OpBuilder<ReorderSpec>;
pub type AvgPoolBackwardOpBuilder = OpBuilder<AvgPoolBackwardSpec>;
pub type BatchNormInferenceOpBuilder = OpBuilder<BatchNormInferenceSpec>;
pub type EluOpBuilder = OpBuilder<EluSpec>;
pub type ExpOpBuilder = OpBuilder<ExpSpec>;
