use {
    onednnl_sys::{
        dnnl_data_type_t, dnnl_dim_t, dnnl_memory_desc_clone, dnnl_memory_desc_create_with_blob,
        dnnl_memory_desc_create_with_tag, dnnl_memory_desc_destroy, dnnl_memory_desc_equal,
        dnnl_memory_desc_get_blob, dnnl_memory_desc_get_size, dnnl_memory_desc_query,
        dnnl_memory_desc_t, dnnl_query_t, dnnl_status_t,
    },
    std::{
        alloc::{alloc, Layout},
        ffi::c_void,
    },
};

#[derive(Debug)]
pub struct MemoryDescriptor {
    pub(crate) handle: dnnl_memory_desc_t,
}

use {super::format_tag::FormatTag, crate::error::DnnlError};

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

    /// Create a new MemoryDescriptor
    /// ```
    /// use onednnl::memory::descriptor::MemoryDescriptor;
    /// use onednnl_sys::dnnl_data_type_t::dnnl_f32;
    ///
    ///
    /// let md = MemoryDescriptor::new_any(&[15, 15], dnnl_f32);
    ///
    /// assert!(md.is_ok());
    /// ```
    pub fn new_any(dims: &[i64], data_type: dnnl_data_type_t::Type) -> Result<Self, DnnlError> {
        let mut handle: dnnl_memory_desc_t = std::ptr::null_mut();
        let status = unsafe {
            dnnl_memory_desc_create_with_tag(
                &mut handle,
                dims.len() as i32,
                dims.as_ptr(),
                data_type,
                dnnl_format_tag_any,
            )
        };

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
        unsafe { dnnl_memory_desc_get_size(self.handle) }
    }

    // Queries the memory descriptor for various pieces of information.
    ///
    /// # Arguments
    ///
    /// * `query` - The query to perform.
    ///
    /// # Returns
    ///
    /// A `Result` containing the query output on success or a `DnnlError` on failure.
    ///
    /// ```
    /// use onednnl::memory::descriptor::MemoryDescriptor;
    /// use onednnl::memory::format_tag::abcd;
    /// use onednnl::memory::descriptor::{NDimsQuery, DimsQuery, DataTypeQuery};
    /// use onednnl_sys::dnnl_data_type_t;
    ///
    /// let md = MemoryDescriptor::new::<4, abcd>([1, 3, 228, 228], dnnl_data_type_t::dnnl_f32).unwrap();
    ///
    /// assert_eq!(md.query::<NDimsQuery>(), Ok(4));
    /// assert_eq!(md.query::<DimsQuery>(), Ok(vec![1, 3, 228, 228]));
    /// assert_eq!(md.query::<DataTypeQuery>(), Ok(dnnl_data_type_t::dnnl_f32));
    /// ```
    pub fn query<Q: Query>(&self) -> Result<Q::Output, DnnlError> {
        Q::execute(self.handle)
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

const DNNL_MAX_NDIMS: usize = 12;

pub struct DataType;

impl DataType {
    pub const F32: dnnl_data_type_t::Type = dnnl_data_type_t::dnnl_f32;
    pub const F64: dnnl_data_type_t::Type = dnnl_data_type_t::dnnl_f64;
}

/// Trait representing a query to be performed
pub trait Query {
    /// The output type associated with the query.
    type Output;

    /// The query (what) value.
    const QUERY: dnnl_query_t::Type;

    /// Executes the query on the given memory descriptor and retrieves the result.
    ///
    /// # Arguments
    ///
    /// * `handle` - A reference to the `dnnl_memory_desc_t` handle.
    ///
    /// # Returns
    ///
    /// A `Result` containing the query output on success or a `DnnlError` on failure.
    fn execute(handle: dnnl_memory_desc_t) -> Result<Self::Output, DnnlError>;
}

/// Query type for retrieving the number of dimensions.
pub struct NDimsQuery;

impl Query for NDimsQuery {
    type Output = i32;
    const QUERY: dnnl_query_t::Type = dnnl_query_t::dnnl_query_ndims_s32;

    fn execute(handle: dnnl_memory_desc_t) -> Result<Self::Output, DnnlError> {
        unsafe {
            let mut result: i32 = 0;
            let status =
                dnnl_memory_desc_query(handle, Self::QUERY, &mut result as *mut i32 as *mut c_void);
            if status != dnnl_status_t::dnnl_success {
                return Err(status.into());
            }
            Ok(result)
        }
    }
}

/// Query type for retrieving the dimensions.
use onednnl_sys::{
    dnnl_format_tag_t::dnnl_format_tag_any, dnnl_query_t::dnnl_query_dims as QUERY_DIMS,
};

pub struct DimsQuery;

impl Query for DimsQuery {
    type Output = Vec<dnnl_dim_t>;
    const QUERY: dnnl_query_t::Type = QUERY_DIMS;

    fn execute(handle: dnnl_memory_desc_t) -> Result<Self::Output, DnnlError> {
        let ndims = NDimsQuery::execute(handle)? as usize;

        if ndims <= 0 || ndims > DNNL_MAX_NDIMS {
            return Err(DnnlError::InvalidQueryOutput);
        }

        let layout = Layout::array::<dnnl_dim_t>(ndims).unwrap();
        let mut ptr = unsafe { alloc(layout) as *mut dnnl_dim_t };

        let ptr_to_ptr = &mut ptr as *mut *mut dnnl_dim_t;

        let status =
            unsafe { dnnl_memory_desc_query(handle, Self::QUERY, ptr_to_ptr as *mut c_void) };

        if status == dnnl_status_t::dnnl_success {
            let dims = unsafe { std::slice::from_raw_parts(ptr, ndims) };

            Ok(dims.to_vec())
        } else {
            Err(status.into())
        }
    }
}

/// Query type for retrieving the data type.
pub struct DataTypeQuery;

impl Query for DataTypeQuery {
    type Output = dnnl_data_type_t::Type;
    const QUERY: dnnl_query_t::Type = dnnl_query_t::dnnl_query_data_type;

    fn execute(handle: dnnl_memory_desc_t) -> Result<Self::Output, DnnlError> {
        unsafe {
            let mut dtype: dnnl_data_type_t::Type = 0;
            let status = dnnl_memory_desc_query(
                handle,
                Self::QUERY,
                &mut dtype as *mut dnnl_data_type_t::Type as *mut c_void,
            );
            if status != dnnl_status_t::dnnl_success {
                return Err(status.into());
            }
            Ok(dtype)
        }
    }
}
