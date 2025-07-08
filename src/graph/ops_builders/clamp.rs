use onednnl_sys::dnnl_graph_op_attr_t::{dnnl_graph_op_attr_max, dnnl_graph_op_attr_min};

use crate::graph::{
    op::{OneDNNGraphOp, OneDNNGraphOpType},
    spec::{OpSpec, RequiredAttrs},
};

pub struct ClampSpec;

impl OpSpec for ClampSpec {
    const KIND: OneDNNGraphOpType = OneDNNGraphOp::CLAMP;
}

#[derive(Debug, Clone, Copy)]
pub struct ClampAttrs {
    pub min: f32,
    pub max: f32,
}

impl From<ClampAttrs> for RequiredAttrs {
    fn from(attrs: ClampAttrs) -> Self {
        RequiredAttrs::Some(vec![
            (dnnl_graph_op_attr_min, vec![attrs.min].into()),
            (dnnl_graph_op_attr_max, vec![attrs.max].into()),
        ])
    }
}
