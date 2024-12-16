use onednnl_sys::{
    dnnl_data_type_t, dnnl_dim_t, dnnl_memory_desc_create_with_blob,
    dnnl_memory_desc_create_with_tag, dnnl_memory_desc_destroy, dnnl_memory_desc_t, dnnl_status_t,
};

#[derive(Debug)]
pub struct MemoryDescriptor {
    handle: dnnl_memory_desc_t,
}

use crate::error::DnnlError;

use super::format_tag::FormatTag;

impl MemoryDescriptor {
    /// Create a new MemoryDescriptor
    /// ```
    /// use onednnl::memory::descriptor::MemoryDescriptor;
    /// use onednnl::memory::format_tag::{x, ab};
    /// use onednnl_sys::dnnl_data_type_t::dnnl_f32;
    ///
    ///
    /// let md_1d = MemoryDescriptor::new::<1, x>(&[15], dnnl_f32);
    ///
    /// assert!(md_1d.is_ok());
    ///
    /// let md_2d = MemoryDescriptor::new::<2, ab>(&[2, 3], dnnl_f32);
    ///
    /// assert!(md_2d.is_ok());
    ///
    /// ```
    pub fn new<const NDIMS: usize, T: FormatTag<NDIMS>>(
        dims: &[dnnl_dim_t],
        data_type: dnnl_data_type_t::Type,
    ) -> Result<Self, DnnlError> {
        let mut handle: dnnl_memory_desc_t = std::ptr::null_mut();
        let status = unsafe {
            dnnl_memory_desc_create_with_tag(
                &mut handle,
                NDIMS as i32,
                dims.as_ptr(),
                data_type,
                T::TAG,
            )
        };

        if status == dnnl_status_t::dnnl_success {
            Ok(Self { handle })
        } else {
            Err(status.into())
        }
    }

    pub unsafe fn new_from_blob(blob: *mut u8) -> Result<Self, DnnlError> {
        let mut handle: dnnl_memory_desc_t = std::ptr::null_mut();
        let status = dnnl_memory_desc_create_with_blob(&mut handle, blob);

        if status == dnnl_status_t::dnnl_success {
            Ok(Self { handle })
        } else {
            Err(status.into())
        }
    }
}

impl Drop for MemoryDescriptor {
    fn drop(&mut self) {
        unsafe {
            dnnl_memory_desc_destroy(self.handle);
        }
    }
}
