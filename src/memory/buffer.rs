use {
    crate::error::DnnlError,
    std::{
        alloc::{alloc, alloc_zeroed, dealloc, Layout},
        ptr::NonNull,
    },
};

#[derive(Debug)]
pub struct AlignedBuffer<T> {
    pub(crate) ptr: NonNull<T>,
    pub size: usize,
    pub layout: Layout,
}

impl<T> AlignedBuffer<T> {
    pub fn new(data: &[T]) -> Result<Self, DnnlError>
    where
        T: Copy,
    {
        let length = data.len();

        let layout = Layout::array::<T>(length).map_err(|_| DnnlError::InvalidLayout)?;
        let buffer_ptr = unsafe { alloc(layout) as *mut T };
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buffer_ptr, length);
        }

        Ok(Self {
            ptr: NonNull::new(buffer_ptr).ok_or(DnnlError::NonNullViolated)?,
            size: length,
            layout,
        })
    }

    /// Returns an immutable slice to the buffer's data.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    /// Returns a mutable slice to the buffer's data.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Allocates an aligned buffer and initializes it with zeros.
    pub fn zeroed(length: usize) -> Result<Self, DnnlError> {
        let layout = Layout::array::<T>(length).map_err(|_| DnnlError::InvalidLayout)?;

        let buffer_ptr = unsafe { alloc_zeroed(layout) as *mut T };
        let buffer_ptr = NonNull::new(buffer_ptr).ok_or(DnnlError::NonNullViolated)?;

        Ok(Self {
            ptr: buffer_ptr,
            size: length,
            layout,
        })
    }
}

impl<T> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}
