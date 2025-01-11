pub mod engine;
pub mod error;
pub mod memory;
pub mod primitive;
pub mod primitives;
pub mod stream;

use error::DnnlError;
pub use onednnl_sys;

pub fn set_primitive_cache_capacity(capacity: std::ffi::c_int) -> Result<(), DnnlError> {
    let status = unsafe { onednnl_sys::dnnl_set_primitive_cache_capacity(capacity) };

    if status == onednnl_sys::dnnl_status_t::dnnl_success {
        Ok(())
    } else {
        Err(status.into())
    }
}
