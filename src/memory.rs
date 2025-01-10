#![allow(non_snake_case)]
use {
    crate::{engine::Engine, error::DnnlError},
    buffer::AlignedBuffer,
    descriptor::MemoryDescriptor,
    onednnl_sys::{
        dnnl_data_type_size,
        dnnl_data_type_t::{self, dnnl_f32},
        dnnl_engine_kind_t, dnnl_memory, dnnl_memory_create, dnnl_memory_destroy,
        dnnl_memory_get_data_handle, dnnl_memory_t, dnnl_status_t, DNNL_GPU_RUNTIME,
        DNNL_RUNTIME_OCL, DNNL_RUNTIME_SYCL,
    },
    std::{ffi::c_void, sync::Arc},
};

/// Get the size for a data type.
pub fn data_type_size(ty: dnnl_data_type_t::Type) -> usize {
    unsafe { dnnl_data_type_size(ty) }
}

/// Memory without an underlying buffer
const DNNL_MEMORY_NONE: *mut c_void = std::ptr::null_mut();

/// Memory with library allocated buffer
const DNNL_MEMORY_ALLOCATE: *mut c_void = (usize::MAX) as *mut c_void;

pub mod buffer;
pub mod descriptor;
#[allow(non_camel_case_types)]
pub mod format_tag;

#[derive(Debug)]
pub enum BufferType<T> {
    UserAllocated(AlignedBuffer<T>),
    LibraryAllocated,
    None,
}

#[derive(Debug)]
pub struct Memory<T> {
    pub(crate) handle: dnnl_memory_t,
    pub engine: Arc<Engine>,
    pub buffer_type: BufferType<T>,
    pub desc: MemoryDescriptor,
}

impl<T> Memory<T> {
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
    /// use {
    ///     onednnl::{
    ///         engine::Engine,
    ///         error::DnnlError,
    ///         memory::{
    ///             buffer::AlignedBuffer, data_type_size, descriptor::MemoryDescriptor,
    ///             format_tag::abcdef, Memory,
    ///         },
    ///     },
    ///     onednnl_sys::dnnl_data_type_t::dnnl_f32,
    ///     std::{ffi::c_void, sync::Arc},
    /// };
    ///
    /// let engine = Arc::new(Engine::new(Engine::CPU, 0).unwrap());
    ///
    /// let dims = [1, 3, 224, 224, 112, 112];
    ///
    /// let mem_desc = MemoryDescriptor::new::<6, abcdef>(dims, dnnl_f32).unwrap();
    /// let mut buffer =
    ///     AlignedBuffer::<f32>::zeroed(mem_desc.get_size() / data_type_size(dnnl_f32)).unwrap();
    /// let memory = Memory::new_with_user_buffer(Arc::clone(&engine), mem_desc, buffer);
    /// assert!(memory.is_ok());
    /// ```
    pub fn new_with_user_buffer(
        engine: Arc<Engine>,
        desc: MemoryDescriptor,
        buffer: AlignedBuffer<T>,
    ) -> Result<Self, DnnlError> {
        let mut handle = std::ptr::null_mut::<dnnl_memory>();

        let status = match engine.get_kind() {
            Ok(dnnl_engine_kind_t::dnnl_cpu) => unsafe {
                dnnl_memory_create(
                    &mut handle,
                    desc.handle,
                    engine.handle,
                    buffer.ptr.as_ptr() as *mut c_void,
                )
            },
            Ok(dnnl_engine_kind_t::dnnl_gpu) => {
                if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL {
                    todo!("Add SYCL interop")
                } else if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL {
                    todo!("Add OCL interop")
                } else {
                    todo!("Return Error for lack of a GPU Runtime")
                }
            }
            Ok(dnnl_engine_kind_t::dnnl_any_engine) => {
                todo!("Add DNNL ANY interop")
            }
            Ok(_) => {
                panic!("Unexpected engine kind type type")
            }
            Err(e) => return Err(e),
        };

        if status == dnnl_status_t::dnnl_success {
            Ok(Memory {
                handle,
                engine,
                buffer_type: BufferType::UserAllocated(buffer),
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
    /// use {
    ///     onednnl::{
    ///         engine::Engine,
    ///         error::DnnlError,
    ///         memory::{descriptor::MemoryDescriptor, format_tag::abcd, Memory},
    ///     },
    ///     onednnl_sys::dnnl_data_type_t::dnnl_f32,
    ///     std::{ffi::c_void, sync::Arc},
    /// };
    ///
    /// let engine = Arc::new(Engine::new(Engine::CPU, 0).unwrap());
    ///
    /// let dims = [1, 3, 224, 224];
    ///
    /// let mem_desc = MemoryDescriptor::new::<4, abcd>(dims, dnnl_f32).unwrap();
    ///
    /// let memory = Memory::<f32>::new_with_library_buffer(Arc::clone(&engine), mem_desc);
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
    /// use {
    ///     onednnl::{
    ///         engine::Engine,
    ///         error::DnnlError,
    ///         memory::{descriptor::MemoryDescriptor, format_tag::abcdef, Memory},
    ///     },
    ///     onednnl_sys::dnnl_data_type_t::dnnl_f32,
    ///     std::{ffi::c_void, sync::Arc},
    /// };
    ///
    /// let engine = Arc::new(Engine::new(Engine::CPU, 0).unwrap());
    ///
    /// let dims = [1, 3, 224, 224, 112, 112];
    ///
    /// let mem_desc = MemoryDescriptor::new::<6, abcdef>(dims, dnnl_f32).unwrap();
    ///
    /// let memory = Memory::<f32>::new_without_buffer(Arc::clone(&engine), mem_desc);
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

    pub fn to_vec(&self) -> Result<Vec<T>, DnnlError>
    where
        T: Clone,
    {
        match self.engine.get_kind() {
            Ok(Engine::CPU) => match &self.buffer_type {
                BufferType::UserAllocated(buffer) => Ok(buffer.as_slice().to_vec()),
                BufferType::LibraryAllocated => {
                    let mut buffer_ptr = std::ptr::null_mut();

                    let status = unsafe {
                        dnnl_memory_get_data_handle(
                            self.handle,
                            &mut buffer_ptr as *mut *mut _ as *mut *mut c_void,
                        )
                    };

                    if status == dnnl_status_t::dnnl_success {
                        Ok(unsafe {
                            std::slice::from_raw_parts(
                                buffer_ptr as *const T,
                                self.desc.get_size() / data_type_size(dnnl_f32),
                            )
                        }
                        .to_vec())
                    } else {
                        Err(status.into())
                    }
                }
                BufferType::None => todo!("return error"),
            },
            Ok(Engine::GPU) => {
                todo!("Return the right data")
            }
            Ok(dnnl_engine_kind_t::dnnl_any_engine) => {
                todo!("Return the right data")
            }
            Ok(t) => {
                panic!("Received incorrect engine_kind_t: {}", t)
            }
            Err(e) => Err(e),
        }
    }
}

impl<T> Drop for Memory<T> {
    fn drop(&mut self) {
        unsafe { dnnl_memory_destroy(self.handle) };
    }
}

unsafe impl<T> Sync for Memory<T> {}
unsafe impl<T> Send for Memory<T> {}
