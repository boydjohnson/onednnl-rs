use onednnl_sys::dnnl_graph_op_attr_t::{
    dnnl_graph_op_attr_shape, dnnl_graph_op_attr_special_zero,
};

use crate::graph::{
    op::{OneDNNGraphOp, OneDNNGraphOpType},
    spec::{OpSpec, RequiredAttrs},
};

pub struct StaticReshapeSpec;

impl OpSpec for StaticReshapeSpec {
    const KIND: OneDNNGraphOpType = OneDNNGraphOp::STATIC_RESHAPE;
}

pub struct StaticReshapeAttrs {
    pub shape: Vec<i64>,
    pub special_zero: u8,
}

impl From<StaticReshapeAttrs> for RequiredAttrs {
    fn from(attrs: StaticReshapeAttrs) -> Self {
        RequiredAttrs::Some(vec![
            (dnnl_graph_op_attr_shape, attrs.shape.into()),
            (
                dnnl_graph_op_attr_special_zero,
                vec![attrs.special_zero].into(),
            ),
        ])
    }
}
