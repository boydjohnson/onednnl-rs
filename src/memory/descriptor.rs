use onednnl_sys::{
    dnnl_data_type_t, dnnl_dim_t, dnnl_memory_desc_clone, dnnl_memory_desc_create_with_blob,
    dnnl_memory_desc_create_with_tag, dnnl_memory_desc_destroy, dnnl_memory_desc_equal,
    dnnl_memory_desc_get_blob, dnnl_memory_desc_get_size, dnnl_memory_desc_t, dnnl_status_t,
};

#[derive(Debug)]
pub struct MemoryDescriptor {
    pub(crate) handle: dnnl_memory_desc_t,
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
    /// let md_1d = MemoryDescriptor::new::<1, x>([15], dnnl_f32);
    ///
    /// assert!(md_1d.is_ok());
    ///
    /// let md_2d = MemoryDescriptor::new::<2, ab>([2, 3], dnnl_f32);
    ///
    /// assert!(md_2d.is_ok());
    ///
    /// ```
    pub fn new<const NDIMS: usize, T: FormatTag<NDIMS>>(
        dims: [dnnl_dim_t; NDIMS],
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

    pub fn new_from_blob(blob: *mut u8) -> Result<Self, DnnlError> {
        let mut handle: dnnl_memory_desc_t = std::ptr::null_mut();
        let status = unsafe { dnnl_memory_desc_create_with_blob(&mut handle, blob) };

        if status == dnnl_status_t::dnnl_success {
            Ok(Self { handle })
        } else {
            Err(status.into())
        }
    }

    /// Clones the memory descriptor.
    ///
    /// ```
    /// use onednnl::memory::descriptor::MemoryDescriptor;
    /// use onednnl::memory::format_tag::{x, ab};
    /// use onednnl_sys::dnnl_data_type_t::dnnl_f32;
    ///
    ///
    /// let md_1d = MemoryDescriptor::new::<1, x>([15], dnnl_f32);
    ///
    /// assert!(md_1d.is_ok());
    ///
    /// let md_1d = md_1d.unwrap();
    /// let md_1d_2 = md_1d.clone_desc();
    ///
    /// assert!(md_1d_2.is_ok());
    /// ```
    pub fn clone_desc(&self) -> Result<Self, DnnlError> {
        let mut cloned_handle: dnnl_memory_desc_t = std::ptr::null_mut();
        let status = unsafe { dnnl_memory_desc_clone(&mut cloned_handle, self.handle) };

        if status == dnnl_status_t::dnnl_success {
            Ok(Self {
                handle: cloned_handle,
            })
        } else {
            Err(status.into())
        }
    }

    /// Checks if two memory descriptors are equal.
    /// ```
    /// use onednnl::memory::descriptor::MemoryDescriptor;
    /// use onednnl::memory::format_tag::{x, ab};
    /// use onednnl_sys::dnnl_data_type_t::dnnl_f32;
    ///
    ///
    /// let md_1d = MemoryDescriptor::new::<1, x>([15], dnnl_f32);
    ///
    /// assert!(md_1d.is_ok());
    ///
    /// let md_1d = md_1d.unwrap();
    /// let md_1d_2 = md_1d.clone_desc();
    ///
    /// assert!(md_1d_2.is_ok());
    ///
    /// let md_1d_2 = md_1d_2.unwrap();
    ///
    /// assert!(md_1d.equal(&md_1d_2));
    ///
    /// assert_eq!(md_1d, md_1d_2);
    ///
    /// ```
    pub fn equal(&self, other: &Self) -> bool {
        let is_equal = unsafe { dnnl_memory_desc_equal(self.handle, other.handle) };
        is_equal == 1
    }

    /// Retrieves the blob associated with the memory descriptor.
    /// ```
    /// use onednnl::memory::descriptor::MemoryDescriptor;
    /// use onednnl::memory::format_tag::{x, ab};
    /// use onednnl_sys::dnnl_data_type_t::dnnl_f32;
    ///
    ///
    /// let md_1d = MemoryDescriptor::new::<1, x>([15], dnnl_f32);
    ///
    /// assert!(md_1d.is_ok());
    ///
    /// let md_1d = md_1d.unwrap();
    ///
    /// let blob = md_1d.get_blob();
    /// dbg!(&blob);
    /// assert!(blob.is_ok());
    ///
    /// let mut blob = blob.unwrap();
    ///
    /// let md_1d_2 = MemoryDescriptor::new_from_blob(blob.as_mut_ptr());
    ///
    /// assert!(md_1d_2.is_ok());
    ///
    /// let md_1d_2 = md_1d_2.unwrap();
    ///
    /// assert!(md_1d_2.equal(&md_1d));
    ///
    /// ```
    pub fn get_blob(&self) -> Result<Vec<u8>, DnnlError> {
        let mut size: usize = 0;
        let status =
            unsafe { dnnl_memory_desc_get_blob(std::ptr::null_mut(), &mut size, self.handle) };

        if status != dnnl_status_t::dnnl_success {
            return Err(status.into());
        }
        let mut blob_data = vec![0u8; size];
        let blob_ptr = blob_data.as_mut_ptr();

        let status = unsafe { dnnl_memory_desc_get_blob(blob_ptr, &mut size, self.handle) };

        if status != dnnl_status_t::dnnl_success {
            return Err(status.into());
        }

        blob_data.resize(size, 0);

        Ok(blob_data)
    }

    /// Gets the size in bytes of the memory described by the descriptor.
    ///
    /// **Note**
    /// This is not the same as the size of the MemoryDescriptor blob.
    ///
    /// ```
    /// use onednnl::memory::descriptor::MemoryDescriptor;
    /// use onednnl::memory::format_tag::{x, ab};
    /// use onednnl_sys::dnnl_data_type_t::dnnl_f32;
    ///
    ///
    /// let md_1d = MemoryDescriptor::new::<1, x>([15], dnnl_f32);
    ///
    /// assert!(md_1d.is_ok());
    ///
    /// let md_1d = md_1d.unwrap();
    ///
    ///
    /// let size = md_1d.get_size();
    ///
    /// assert!(size > 0);
    ///
    /// ```
    pub fn get_size(&self) -> usize {
        let size = unsafe { dnnl_memory_desc_get_size(self.handle) };
        size
    }
}

impl PartialEq for MemoryDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.equal(other)
    }
}

impl Drop for MemoryDescriptor {
    fn drop(&mut self) {
        unsafe {
            dnnl_memory_desc_destroy(self.handle);
        }
    }
}
