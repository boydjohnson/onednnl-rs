use crate::graph::{
    op::{OneDNNGraphOp, OneDNNGraphOpType},
    spec::OpSpec,
};

pub struct AbsSpec;

impl OpSpec for AbsSpec {
    const KIND: OneDNNGraphOpType = OneDNNGraphOp::ABS;
}
