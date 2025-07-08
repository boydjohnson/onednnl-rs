use {
    crate::error::DnnlError,
    onednnl_sys::{
        dnnl_data_type_t, dnnl_dims_t, dnnl_graph_layout_type_t, dnnl_graph_logical_tensor_init,
        dnnl_graph_logical_tensor_init_with_dims, dnnl_graph_logical_tensor_t,
        dnnl_graph_tensor_property_t, dnnl_status_t,
    },
    std::mem::MaybeUninit,
};

pub struct LogicalTensor {
    pub(crate) handle: onednnl_sys::dnnl_graph_logical_tensor_t,
}

impl LogicalTensor {
    pub fn create(
        tid: usize,
        dtype: dnnl_data_type_t::Type,
        ndims: i32,
        layout: dnnl_graph_layout_type_t::Type,
        property: dnnl_graph_tensor_property_t::Type,
    ) -> Result<Self, DnnlError> {
        // allocate uninitialized LT
        let mut lt = MaybeUninit::<dnnl_graph_logical_tensor_t>::uninit();
        let status = unsafe {
            dnnl_graph_logical_tensor_init(lt.as_mut_ptr(), tid, dtype, ndims, layout, property)
        };
        if status != dnnl_status_t::dnnl_success {
            return Err(status.into());
        }
        // assume_init is now safe
        Ok(LogicalTensor {
            handle: unsafe { lt.assume_init() },
        })
    }

    pub fn id(&self) -> usize {
        self.handle.id
    }

    pub fn new_with_dims(
        tid: usize,
        dtype: dnnl_data_type_t::Type,
        dims: &[i64],
        layout: dnnl_graph_layout_type_t::Type,
        property: dnnl_graph_tensor_property_t::Type,
    ) -> Result<Self, DnnlError> {
        let ndims = dims.len() as i32;
        let mut c_dims: dnnl_dims_t = [0; 12];
        for (i, &dim) in dims.iter().enumerate() {
            c_dims[i] = dim;
        }

        let mut lt = MaybeUninit::<dnnl_graph_logical_tensor_t>::uninit();
        let status = unsafe {
            dnnl_graph_logical_tensor_init_with_dims(
                lt.as_mut_ptr(),
                tid,
                dtype,
                ndims,
                c_dims.as_ptr(),
                layout,
                property,
            )
        };
        if status != dnnl_status_t::dnnl_success {
            return Err(status.into());
        }
        Ok(LogicalTensor {
            handle: unsafe { lt.assume_init() },
        })
    }

    pub fn get_dims(&self) -> Vec<i64> {
        let ndims = self.handle.ndims;
        let dims = self.handle.dims;

        dims.iter().take(ndims as usize).map(|&dim| dim).collect()
    }
}
