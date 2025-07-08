use onednnl_sys::{
    dnnl_graph_op_attr_t::dnnl_graph_op_attr_epsilon, dnnl_graph_op_kind_t::dnnl_graph_op_elu,
};

use crate::graph::{
    op::OneDNNGraphOpType,
    spec::{OpSpec, RequiredAttrs},
};

pub struct EluSpec;

#[derive(Debug, Clone, Copy)]
pub struct EluAttrs {
    pub alpha: f32,
}

impl OpSpec for EluSpec {
    const KIND: OneDNNGraphOpType = dnnl_graph_op_elu;
}

impl From<EluAttrs> for RequiredAttrs {
    fn from(attrs: EluAttrs) -> Self {
        RequiredAttrs::Some(vec![(dnnl_graph_op_attr_epsilon, vec![attrs.alpha].into())])
    }
}
