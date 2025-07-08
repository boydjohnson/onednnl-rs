use crate::graph::{
    op::{OneDNNGraphOp, OneDNNGraphOpType},
    spec::OpSpec,
};

pub struct AbsBackwardSpec;

impl OpSpec for AbsBackwardSpec {
    const KIND: OneDNNGraphOpType = OneDNNGraphOp::ABS_BACKWARD;
}
