use {
    crate::graph::{op::OneDNNGraphOpType, spec::OpSpec},
    onednnl_sys::dnnl_graph_op_kind_t::dnnl_graph_op_exp,
};

pub struct ExpSpec;

impl OpSpec for ExpSpec {
    const KIND: OneDNNGraphOpType = dnnl_graph_op_exp;
}
