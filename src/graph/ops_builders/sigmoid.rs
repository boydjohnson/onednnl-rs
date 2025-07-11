use onednnl_sys::dnnl_graph_op_kind_t::dnnl_graph_op_sigmoid;

use crate::graph::{op::OneDNNGraphOpType, spec::OpSpec};

pub struct SigmoidSpec;

impl OpSpec for SigmoidSpec {
    const KIND: OneDNNGraphOpType = dnnl_graph_op_sigmoid;
}
