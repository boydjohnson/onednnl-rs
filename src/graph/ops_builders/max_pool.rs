use {
    crate::graph::{
        op::OneDNNGraphOpType,
        spec::{OpSpec, RequiredAttrs},
    },
    onednnl_sys::{
        dnnl_graph_op_attr_t::{
            self, dnnl_graph_op_attr_auto_pad, dnnl_graph_op_attr_data_format,
            dnnl_graph_op_attr_dilations, dnnl_graph_op_attr_kernel, dnnl_graph_op_attr_pads_begin,
            dnnl_graph_op_attr_pads_end, dnnl_graph_op_attr_rounding_type,
            dnnl_graph_op_attr_strides,
        },
        dnnl_graph_op_kind_t::dnnl_graph_op_max_pool,
    },
};

pub struct MaxPoolSpec;

impl OpSpec for MaxPoolSpec {
    const KIND: OneDNNGraphOpType = dnnl_graph_op_max_pool;
}

pub struct MaxPoolAttrs {
    pub strides: Vec<i64>,
    pub pads_begin: Vec<i64>,
    pub pads_end: Vec<i64>,
    pub kernel: Vec<i64>,
}

impl From<MaxPoolAttrs> for RequiredAttrs {
    fn from(attrs: MaxPoolAttrs) -> Self {
        RequiredAttrs::Some(vec![
            (dnnl_graph_op_attr_strides, attrs.strides.into()),
            (dnnl_graph_op_attr_pads_begin, attrs.pads_begin.into()),
            (dnnl_graph_op_attr_pads_end, attrs.pads_end.into()),
            (dnnl_graph_op_attr_kernel, attrs.kernel.into()),
        ])
    }
}

impl MaxPoolSpec {
    pub const DILATIONS: dnnl_graph_op_attr_t::Type = dnnl_graph_op_attr_dilations;
    pub const ROUNDING_TYPE: dnnl_graph_op_attr_t::Type = dnnl_graph_op_attr_rounding_type;
    pub const AUTO_PAD: dnnl_graph_op_attr_t::Type = dnnl_graph_op_attr_auto_pad;
    pub const DATA_FORMAT: dnnl_graph_op_attr_t::Type = dnnl_graph_op_attr_data_format;
}
