use {
    crate::{engine::Engine, error::DnnlError},
    buffer::AlignedBuffer,
    descriptor::MemoryDescriptor,
    onednnl_sys::{
        dnnl_memory, dnnl_memory_create, dnnl_memory_destroy, dnnl_memory_t, dnnl_status_t,
    },
    std::{ffi::c_void, sync::Arc},
};

/// Memory without an underlying buffer
const DNNL_MEMORY_NONE: *mut c_void = std::ptr::null_mut();

/// Memory with library allocated buffer
const DNNL_MEMORY_ALLOCATE: *mut c_void = (-1isize) as *mut c_void;

pub mod buffer;
pub mod descriptor;
#[allow(non_camel_case_types)]
pub mod format_tag;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum BufferType {
    UserAllocated,
    LibraryAllocated,
    None,
}

#[derive(Debug)]
pub struct Memory {
    pub(crate) handle: dnnl_memory_t,
    pub engine: Arc<Engine>,
    pub buffer_type: BufferType,
    pub desc: MemoryDescriptor,
}

impl Memory {
    /// Creates a new memory object with a user-allocated buffer.
    ///
    /// This function initializes a `Memory` instance using a buffer provided by the user.
    /// The library does **not** take ownership of the buffer; therefore, the user is responsible
    /// for ensuring that the buffer remains valid for the lifetime of the `Memory` object.
    ///
    /// # Safety
    ///
    /// - The user must ensure that the buffer remains valid for the lifetime of the `Memory` object.
    /// - The buffer must be properly aligned and sized according to `memory_desc`.
    ///
    /// # Parameters
    ///
    /// - `engine`: An `Arc<Engine>` instance representing the engine to associate with this memory.
    /// - `desc`: A `MemoryDescriptor` describing the memory layout.
    /// - `buffer`: A raw pointer to the user-allocated buffer.
    ///
    /// # Returns
    ///
    /// - `Ok(Memory)` if the memory object is successfully created.
    /// - `Err(DnnlError)` if the creation fails.
    ///
    /// # Example
    ///
    /// ```
    /// use std::ffi::c_void;
    /// use std::sync::Arc;
    /// use onednnl::{engine::Engine, memory::Memory, memory::descriptor::MemoryDescriptor, error::DnnlError};
    /// use onednnl_sys::dnnl_data_type_t::dnnl_f32;
    /// use onednnl::memory::format_tag::abcdef;
    /// use onednnl::memory::buffer::AlignedBuffer;
    ///
    /// let engine = Arc::new(Engine::new(Engine::CPU, 0).unwrap());
    ///
    ///     
    /// let dims = [1, 3, 224, 224, 112, 112];
    ///
    ///     
    /// let mem_desc = MemoryDescriptor::new::<6, abcdef>(dims, dnnl_f32).unwrap();
    /// let buffer = AlignedBuffer::<f32>::zeroed(dims.iter().copied().product::<i64>() as usize).unwrap();
    /// let memory = Memory::new_with_user_buffer(Arc::clone(&engine), mem_desc, &buffer);
    /// assert!(memory.is_ok());
    /// ```
    pub fn new_with_user_buffer<T>(
        engine: Arc<Engine>,
        desc: MemoryDescriptor,
        buffer: &AlignedBuffer<T>,
    ) -> Result<Self, DnnlError> {
        let mut handle = std::ptr::null_mut::<dnnl_memory>();
        let status = unsafe {
            dnnl_memory_create(
                &mut handle,
                desc.handle,
                engine.handle,
                buffer.ptr.as_ptr() as *mut c_void,
            )
        };

        if status == dnnl_status_t::dnnl_success {
            Ok(Memory {
                handle,
                engine,
                buffer_type: BufferType::UserAllocated,
                desc,
            })
        } else {
            Err(status.into())
        }
    }

    /// Creates a new memory object with a library-allocated buffer.
    /// # Example
    ///
    /// ```
    /// use std::ffi::c_void;
    /// use std::sync::Arc;
    /// use onednnl::{engine::Engine, memory::Memory, memory::descriptor::MemoryDescriptor, error::DnnlError};
    /// use onednnl_sys::dnnl_data_type_t::dnnl_f32;
    /// use onednnl::memory::format_tag::abcd;
    ///
    /// let engine = Arc::new(Engine::new(Engine::CPU, 0).unwrap());
    ///
    ///     
    /// let dims = [1, 3, 224, 224];
    ///
    ///     
    /// let mem_desc = MemoryDescriptor::new::<4, abcd>(dims, dnnl_f32).unwrap();
    ///
    ///     
    /// let memory = Memory::new_with_library_buffer(Arc::clone(&engine), mem_desc);
    /// assert!(memory.is_ok());
    /// ```
    pub fn new_with_library_buffer(
        engine: Arc<Engine>,
        desc: MemoryDescriptor,
    ) -> Result<Self, DnnlError> {
        let mut handle = std::ptr::null_mut::<dnnl_memory>();
        let status = unsafe {
            dnnl_memory_create(
                &mut handle,
                desc.handle,
                engine.handle,
                DNNL_MEMORY_ALLOCATE,
            )
        };

        if status == dnnl_status_t::dnnl_success {
            Ok(Self {
                handle,
                buffer_type: BufferType::LibraryAllocated,
                engine,
                desc,
            })
        } else {
            Err(status.into())
        }
    }

    /// Creates a new memory object without an underlying buffer.
    /// # Example
    ///
    /// ```
    /// use std::ffi::c_void;
    /// use std::sync::Arc;
    /// use onednnl::{engine::Engine, memory::Memory, memory::descriptor::MemoryDescriptor, error::DnnlError};
    /// use onednnl_sys::dnnl_data_type_t::dnnl_f32;
    /// use onednnl::memory::format_tag::abcdef;
    ///
    /// let engine = Arc::new(Engine::new(Engine::CPU, 0).unwrap());
    ///
    ///     
    /// let dims = [1, 3, 224, 224, 112, 112];
    ///
    ///     
    /// let mem_desc = MemoryDescriptor::new::<6, abcdef>(dims, dnnl_f32).unwrap();
    ///
    ///     
    /// let memory = Memory::new_without_buffer(Arc::clone(&engine), mem_desc);
    /// assert!(memory.is_ok());
    /// ```
    pub fn new_without_buffer(
        engine: Arc<Engine>,
        desc: MemoryDescriptor,
    ) -> Result<Self, DnnlError> {
        let mut handle = std::ptr::null_mut::<dnnl_memory>();
        let status = unsafe {
            dnnl_memory_create(&mut handle, desc.handle, engine.handle, DNNL_MEMORY_NONE)
        };

        if status == dnnl_status_t::dnnl_success {
            Ok(Self {
                handle,
                buffer_type: BufferType::None,
                engine,
                desc,
            })
        } else {
            Err(status.into())
        }
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        unsafe { dnnl_memory_destroy(self.handle) };
    }
}
