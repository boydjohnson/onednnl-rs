use onednnl_sys::{dnnl_graph_op_attr_t, dnnl_graph_op_kind_t};

use crate::graph::spec::OpSpec;

pub struct DivideSpec;

impl OpSpec for DivideSpec {
    const KIND: dnnl_graph_op_kind_t::Type = dnnl_graph_op_kind_t::dnnl_graph_op_divide;
}

impl DivideSpec {
    pub const AUTO_BROADCAST: dnnl_graph_op_attr_t::Type =
        dnnl_graph_op_attr_t::dnnl_graph_op_attr_auto_broadcast;
}
