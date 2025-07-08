use onednnl_sys::{dnnl_graph_op_attr_t::dnnl_graph_op_attr_data_format, dnnl_graph_op_kind_t};

use crate::graph::{
    op::OneDNNGraphOpType,
    spec::{OpSpec, RequiredAttrs},
};

pub struct AvgPoolBackwardSpec;

impl OpSpec for AvgPoolBackwardSpec {
    const KIND: OneDNNGraphOpType = dnnl_graph_op_kind_t::dnnl_graph_op_avg_pool_backward;
}

pub struct AvgPoolBackwardAttrs {
    pub data_format: String,
}

impl From<AvgPoolBackwardAttrs> for RequiredAttrs {
    fn from(attrs: AvgPoolBackwardAttrs) -> Self {
        RequiredAttrs::Some(vec![(
            dnnl_graph_op_attr_data_format,
            attrs.data_format.into(),
        )])
    }
}
