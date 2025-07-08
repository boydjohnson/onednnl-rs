use {
    crate::graph::{
        op::{OneDNNGraphOp, OneDNNGraphOpType},
        spec::{OpSpec, RequiredAttrs},
    },
    onednnl_sys::dnnl_graph_op_attr_t::{
        self, dnnl_graph_op_attr_dilations, dnnl_graph_op_attr_pads_begin,
        dnnl_graph_op_attr_pads_end, dnnl_graph_op_attr_strides,
    },
};

pub struct ConvolutionSpec;

impl ConvolutionSpec {
    pub const GROUPS: dnnl_graph_op_attr_t::Type = dnnl_graph_op_attr_t::dnnl_graph_op_attr_groups;
    pub const AUTO_PAD: dnnl_graph_op_attr_t::Type =
        dnnl_graph_op_attr_t::dnnl_graph_op_attr_auto_pad;
    pub const DATA_FORMAT: dnnl_graph_op_attr_t::Type =
        dnnl_graph_op_attr_t::dnnl_graph_op_attr_data_format;
    pub const WEIGHTS_FORMAT: dnnl_graph_op_attr_t::Type =
        dnnl_graph_op_attr_t::dnnl_graph_op_attr_weights_format;
}

impl OpSpec for ConvolutionSpec {
    const KIND: OneDNNGraphOpType = OneDNNGraphOp::CONVOLUTION;
}

#[derive(Debug, Clone)]
pub struct ConvolutionAttrs {
    pub strides: Vec<i64>,
    pub pads_begin: Vec<i64>,
    pub pads_end: Vec<i64>,
    pub dilations: Vec<i64>,
}

impl From<ConvolutionAttrs> for RequiredAttrs {
    fn from(attrs: ConvolutionAttrs) -> Self {
        RequiredAttrs::Some(vec![
            (dnnl_graph_op_attr_strides, attrs.strides.into()),
            (dnnl_graph_op_attr_pads_begin, attrs.pads_begin.into()),
            (dnnl_graph_op_attr_pads_end, attrs.pads_end.into()),
            (dnnl_graph_op_attr_dilations, attrs.dilations.into()),
        ])
    }
}
