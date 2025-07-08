use crate::graph::{
    op::{OneDNNGraphOp, OneDNNGraphOpType},
    spec::OpSpec,
};

pub struct ReorderSpec;

impl OpSpec for ReorderSpec {
    const KIND: OneDNNGraphOpType = OneDNNGraphOp::REORDER;
}
