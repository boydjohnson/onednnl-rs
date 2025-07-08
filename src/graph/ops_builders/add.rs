use {
    crate::graph::{
        op::{OneDNNGraphOp, OneDNNGraphOpType},
        spec::OpSpec,
    },
    onednnl_sys::dnnl_graph_op_attr_t,
};

pub struct AddSpec;

impl OpSpec for AddSpec {
    const KIND: OneDNNGraphOpType = OneDNNGraphOp::ADD;
}

impl AddSpec {
    /// Possible values of "none" and "numpy"
    pub const AUTO_BROADCAST: dnnl_graph_op_attr_t::Type =
        dnnl_graph_op_attr_t::dnnl_graph_op_attr_auto_broadcast;
}
