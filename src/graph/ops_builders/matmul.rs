use onednnl_sys::dnnl_graph_op_attr_t::{
    dnnl_graph_op_attr_transpose_a, dnnl_graph_op_attr_transpose_b,
};

use crate::graph::{
    op::{OneDNNGraphOp, OneDNNGraphOpType},
    ops_builders::OpAttrKind,
    spec::{OpSpec, RequiredAttrs},
};

pub struct MatMulSpec;

impl OpSpec for MatMulSpec {
    const KIND: OneDNNGraphOpType = OneDNNGraphOp::MATMUL;
}

impl MatMulSpec {
    pub const TRANSPOSE_A: OpAttrKind = dnnl_graph_op_attr_transpose_a;
    pub const TRANSPOSE_B: OpAttrKind = dnnl_graph_op_attr_transpose_b;
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulAttrs {
    pub transpose_a: bool,
    pub transpose_b: bool,
}

impl From<MatMulAttrs> for RequiredAttrs {
    fn from(attrs: MatMulAttrs) -> Self {
        RequiredAttrs::Some(vec![
            (
                dnnl_graph_op_attr_transpose_a,
                vec![attrs.transpose_a as u8].into(),
            ),
            (
                dnnl_graph_op_attr_transpose_b,
                vec![attrs.transpose_b as u8].into(),
            ),
        ])
    }
}
