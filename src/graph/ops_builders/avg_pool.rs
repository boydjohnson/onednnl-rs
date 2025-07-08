use onednnl_sys::dnnl_graph_op_attr_t::{
    self, dnnl_graph_op_attr_auto_pad, dnnl_graph_op_attr_data_format,
    dnnl_graph_op_attr_exclude_pad, dnnl_graph_op_attr_kernel, dnnl_graph_op_attr_pads_begin,
    dnnl_graph_op_attr_pads_end, dnnl_graph_op_attr_rounding_type, dnnl_graph_op_attr_strides,
};

use crate::graph::{
    op::{OneDNNGraphOp, OneDNNGraphOpType},
    spec::{OpSpec, RequiredAttrs},
};

pub struct AvgPoolSpec;

impl OpSpec for AvgPoolSpec {
    const KIND: OneDNNGraphOpType = OneDNNGraphOp::AVG_POOL;
}

impl AvgPoolSpec {
    pub const ROUNDING_TYPE: dnnl_graph_op_attr_t::Type = dnnl_graph_op_attr_rounding_type;
    pub const AUTO_PAD: dnnl_graph_op_attr_t::Type = dnnl_graph_op_attr_auto_pad;
    pub const DATA_FORMAT: dnnl_graph_op_attr_t::Type = dnnl_graph_op_attr_data_format;
}

#[derive(Debug, Clone)]
pub struct AvgPoolAttrs {
    pub strides: Vec<i64>,
    pub pads_begin: Vec<i64>,
    pub pads_end: Vec<i64>,
    pub exclude_pad: bool,
    pub kernel: Vec<i64>,
}

impl From<AvgPoolAttrs> for RequiredAttrs {
    fn from(attrs: AvgPoolAttrs) -> Self {
        RequiredAttrs::Some(vec![
            (dnnl_graph_op_attr_strides, attrs.strides.into()),
            (dnnl_graph_op_attr_pads_begin, attrs.pads_begin.into()),
            (dnnl_graph_op_attr_pads_end, attrs.pads_end.into()),
            (
                dnnl_graph_op_attr_exclude_pad,
                vec![attrs.exclude_pad as u8].into(),
            ),
            (dnnl_graph_op_attr_kernel, attrs.kernel.into()),
        ])
    }
}
