use crate::graph::spec::OpSpec;

pub struct EndSpec;

impl OpSpec for EndSpec {
    const KIND: onednnl_sys::dnnl_graph_op_kind_t::Type =
        onednnl_sys::dnnl_graph_op_kind_t::dnnl_graph_op_end;
}
