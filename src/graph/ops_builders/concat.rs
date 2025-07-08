use onednnl_sys::dnnl_graph_op_attr_t::dnnl_graph_op_attr_axis;

use crate::graph::{
    op::{OneDNNGraphOp, OneDNNGraphOpType},
    spec::{OpSpec, RequiredAttrs},
};

pub struct ConcatSpec;

impl OpSpec for ConcatSpec {
    const KIND: OneDNNGraphOpType = OneDNNGraphOp::CONCAT;
}

#[derive(Debug, Clone, Copy)]
pub struct ConcatAttrs {
    pub axis: i64,
}

impl From<ConcatAttrs> for RequiredAttrs {
    fn from(attrs: ConcatAttrs) -> Self {
        RequiredAttrs::Some(vec![(dnnl_graph_op_attr_axis, vec![attrs.axis].into())])
    }
}
