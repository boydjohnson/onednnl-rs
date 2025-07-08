use {
    crate::{
        engine, error::DnnlError, graph::tensor::logical::LogicalTensor,
        memory::DNNL_MEMORY_ALLOCATE,
    },
    onednnl_sys::{
        dnnl_graph_logical_tensor_t, dnnl_graph_tensor_create, dnnl_graph_tensor_destroy,
        dnnl_graph_tensor_get_data_handle, dnnl_graph_tensor_t, dnnl_status_t,
    },
    std::os::raw::c_void,
};

#[derive(Debug, Clone)]
pub struct Tensor {
    pub(crate) handle: dnnl_graph_tensor_t,
}

impl Tensor {
    pub fn new(
        logical_tensor: &LogicalTensor,
        engine: &engine::Engine,
        data: &[f32],
    ) -> Result<Self, DnnlError> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            dnnl_graph_tensor_create(
                &mut handle,
                &logical_tensor.handle as *const dnnl_graph_logical_tensor_t,
                engine.handle,
                data.as_ptr() as *mut c_void,
            )
        };
        if status != dnnl_status_t::dnnl_success {
            Err(status.into())
        } else {
            Ok(Self { handle })
        }
    }

    pub fn new_library_allocated(
        logical_tensor: &LogicalTensor,
        engine: &engine::Engine,
    ) -> Result<Self, DnnlError> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            dnnl_graph_tensor_create(
                &mut handle,
                &logical_tensor.handle as *const dnnl_graph_logical_tensor_t,
                engine.handle,
                DNNL_MEMORY_ALLOCATE,
            )
        };
        if status != dnnl_status_t::dnnl_success {
            Err(status.into())
        } else {
            Ok(Self { handle })
        }
    }

    pub fn get_data_handle(&self, size: usize) -> Result<Vec<f32>, DnnlError> {
        let mut data_handle: *mut c_void = std::ptr::null_mut();

        let status = unsafe { dnnl_graph_tensor_get_data_handle(self.handle, &mut data_handle) };

        if status != dnnl_status_t::dnnl_success {
            return Err(status.into());
        }

        // 4. Allocate a Rust Vec to copy the data into.
        let mut rust_buffer = vec![0.0f32; size];

        unsafe {
            std::ptr::copy_nonoverlapping(
                data_handle as *const f32,
                rust_buffer.as_mut_ptr(),
                size,
            );
        }
        Ok(rust_buffer)
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe { dnnl_graph_tensor_destroy(self.handle) };
    }
}
