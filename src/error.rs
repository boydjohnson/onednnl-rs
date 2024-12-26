use onednnl_sys::dnnl_status_t;

#[derive(Debug, PartialEq)]
pub enum DnnlError {
    InvalidArguments,
    OutOfMemory,
    Unsupported,
    Unknown,
    InvalidDataType,
    InvalidGraph,
    InvalidGraphOp,
    RuntimeError,
    InvalidShape,
    LastImplReached,
    NotRequired,
    Success,
    InvalidLayout,
    NonNullViolated,
    InvalidQueryOutput,
}

impl From<dnnl_status_t::Type> for DnnlError {
    fn from(status: dnnl_status_t::Type) -> Self {
        match status {
            dnnl_status_t::dnnl_invalid_arguments => Self::InvalidArguments,
            dnnl_status_t::dnnl_out_of_memory => Self::OutOfMemory,
            dnnl_status_t::dnnl_unimplemented => Self::Unsupported,
            dnnl_status_t::dnnl_invalid_data_type => Self::InvalidDataType,
            dnnl_status_t::dnnl_invalid_graph => Self::InvalidGraph,
            dnnl_status_t::dnnl_invalid_graph_op => Self::InvalidGraphOp,
            dnnl_status_t::dnnl_runtime_error => Self::RuntimeError,
            dnnl_status_t::dnnl_invalid_shape => Self::InvalidShape,
            dnnl_status_t::dnnl_last_impl_reached => Self::LastImplReached,
            dnnl_status_t::dnnl_not_required => Self::NotRequired,
            dnnl_status_t::dnnl_success => Self::Success,
            _ => Self::Unknown,
        }
    }
}
