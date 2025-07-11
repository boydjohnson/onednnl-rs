use {
    crate::graph::{op::OneDNNGraphOpType, spec::OpSpec},
    onednnl_sys::{dnnl_graph_op_attr_t, dnnl_graph_op_kind_t::dnnl_graph_op_multiply},
};

pub struct MultiplySpec;

impl OpSpec for MultiplySpec {
    const KIND: OneDNNGraphOpType = dnnl_graph_op_multiply;
}

impl MultiplySpec {
    pub const AUTO_BROADCAST: dnnl_graph_op_attr_t::Type =
        dnnl_graph_op_attr_t::dnnl_graph_op_attr_auto_broadcast;
}
