use onednnl_sys::dnnl_status_t;

#[derive(Debug)]
pub enum DnnlError {
    InvalidArguments,
    OutOfMemory,
    Unsupported,
    Unknown,
}

impl From<dnnl_status_t::Type> for DnnlError {
    fn from(status: dnnl_status_t::Type) -> Self {
        match status {
            dnnl_status_t::dnnl_invalid_arguments => Self::InvalidArguments,
            dnnl_status_t::dnnl_out_of_memory => Self::OutOfMemory,
            dnnl_status_t::dnnl_unimplemented => Self::Unsupported,
            _ => Self::Unknown,
        }
    }
}
