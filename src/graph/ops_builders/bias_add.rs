use {
    crate::graph::spec::OpSpec,
    onednnl_sys::{dnnl_graph_op_attr_t, dnnl_graph_op_kind_t},
};

pub struct BiasAddSpec;

impl OpSpec for BiasAddSpec {
    const KIND: dnnl_graph_op_kind_t::Type = dnnl_graph_op_kind_t::dnnl_graph_op_bias_add;
}

impl BiasAddSpec {
    pub const DATA_FORMAT: dnnl_graph_op_attr_t::Type =
        dnnl_graph_op_attr_t::dnnl_graph_op_attr_data_format;
}
