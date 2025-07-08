use onednnl_sys::dnnl_graph_op_attr_t;

use crate::graph::{
    op::{OneDNNGraphOp, OneDNNGraphOpType},
    spec::OpSpec,
};

pub struct SoftmaxSpec;

impl OpSpec for SoftmaxSpec {
    const KIND: OneDNNGraphOpType = OneDNNGraphOp::SOFTMAX;
}

impl SoftmaxSpec {
    pub const AXIS: dnnl_graph_op_attr_t::Type = dnnl_graph_op_attr_t::dnnl_graph_op_attr_axis;
}
