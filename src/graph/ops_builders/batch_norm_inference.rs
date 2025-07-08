use onednnl_sys::dnnl_graph_op_attr_t::dnnl_graph_op_attr_epsilon;

use crate::graph::spec::RequiredAttrs;

pub struct BatchNormInferenceSpec;

pub struct BatchNormInferenceAttrs {
    pub epsilon: f32,
}

impl BatchNormInferenceSpec {
    pub const DATA_FORMAT: onednnl_sys::dnnl_graph_op_attr_t::Type =
        onednnl_sys::dnnl_graph_op_attr_t::dnnl_graph_op_attr_data_format;
}

impl From<BatchNormInferenceAttrs> for RequiredAttrs {
    fn from(attrs: BatchNormInferenceAttrs) -> Self {
        RequiredAttrs::Some(vec![(
            dnnl_graph_op_attr_epsilon,
            vec![attrs.epsilon].into(),
        )])
    }
}
