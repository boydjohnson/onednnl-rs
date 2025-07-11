use {
    crate::graph::spec::{OpSpec, RequiredAttrs},
    onednnl_sys::{dnnl_graph_op_attr_t::dnnl_graph_op_attr_mode, dnnl_graph_op_kind_t},
};

pub struct GeluSpec;

impl OpSpec for GeluSpec {
    const KIND: dnnl_graph_op_kind_t::Type = dnnl_graph_op_kind_t::dnnl_graph_op_gelu;
}

pub struct GeluAttrs {
    pub mode: String,
}

impl From<GeluAttrs> for RequiredAttrs {
    fn from(attrs: GeluAttrs) -> Self {
        RequiredAttrs::Some(vec![(dnnl_graph_op_attr_mode, attrs.mode.into())])
    }
}
