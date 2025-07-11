use {
    crate::graph::{op::OneDNNGraphOpType, spec::OpSpec},
    onednnl_sys::{dnnl_graph_op_attr_t, dnnl_graph_op_kind_t::dnnl_graph_op_subtract},
};

pub struct SubtractSpec;

impl OpSpec for SubtractSpec {
    const KIND: OneDNNGraphOpType = dnnl_graph_op_subtract;
}

impl SubtractSpec {
    pub const AUTO_BROADCAST: dnnl_graph_op_attr_t::Type =
        dnnl_graph_op_attr_t::dnnl_graph_op_attr_auto_broadcast;
}
