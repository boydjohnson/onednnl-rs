use onednnl_sys::{dnnl_graph_op_attr_t, dnnl_graph_op_kind_t};

pub trait OpSpec {
    const KIND: dnnl_graph_op_kind_t::Type;
}

pub enum RequiredAttrs {
    None,
    Some(Vec<(dnnl_graph_op_attr_t::Type, AttrValue)>),
}

#[derive(Debug, Clone)]
pub enum AttrValue {
    Bool(Vec<u8>),
    Int(Vec<i64>),
    Float(Vec<f32>),
    Str(String),
}

impl From<Vec<u8>> for AttrValue {
    fn from(value: Vec<u8>) -> Self {
        AttrValue::Bool(value)
    }
}

impl From<Vec<i64>> for AttrValue {
    fn from(value: Vec<i64>) -> Self {
        AttrValue::Int(value)
    }
}

impl From<Vec<f32>> for AttrValue {
    fn from(value: Vec<f32>) -> Self {
        AttrValue::Float(value)
    }
}

impl From<String> for AttrValue {
    fn from(value: String) -> Self {
        AttrValue::Str(value)
    }
}
