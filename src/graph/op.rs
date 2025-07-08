use {
    crate::{
        error::DnnlError,
        graph::{spec::AttrValue, tensor::logical::LogicalTensor},
    },
    onednnl_sys::{
        dnnl_graph_op_add_input, dnnl_graph_op_add_output, dnnl_graph_op_attr_t,
        dnnl_graph_op_create, dnnl_graph_op_destroy, dnnl_graph_op_kind_t,
        dnnl_graph_op_set_attr_bool, dnnl_graph_op_set_attr_f32, dnnl_graph_op_set_attr_s64,
        dnnl_graph_op_set_attr_str, dnnl_graph_op_t, dnnl_status_t,
    },
    std::ffi::CString,
};

pub struct OneDNNGraphOp {
    pub(crate) handle: dnnl_graph_op_t,
}

pub type OneDNNGraphOpType = dnnl_graph_op_kind_t::Type;

impl OneDNNGraphOp {
    pub const ABS: OneDNNGraphOpType = dnnl_graph_op_kind_t::dnnl_graph_op_abs;
    pub const ABS_BACKWARD: OneDNNGraphOpType = dnnl_graph_op_kind_t::dnnl_graph_op_abs_backward;
    pub const ADD: OneDNNGraphOpType = dnnl_graph_op_kind_t::dnnl_graph_op_add;
    pub const AVG_POOL: OneDNNGraphOpType = dnnl_graph_op_kind_t::dnnl_graph_op_avg_pool;
    pub const AVG_POOL_BACKWARD: OneDNNGraphOpType =
        dnnl_graph_op_kind_t::dnnl_graph_op_avg_pool_backward;
    pub const CONVOLUTION: OneDNNGraphOpType = dnnl_graph_op_kind_t::dnnl_graph_op_convolution;
    pub const CLAMP: OneDNNGraphOpType = dnnl_graph_op_kind_t::dnnl_graph_op_clamp;
    pub const CONCAT: OneDNNGraphOpType = dnnl_graph_op_kind_t::dnnl_graph_op_concat;
    pub const MATMUL: OneDNNGraphOpType = dnnl_graph_op_kind_t::dnnl_graph_op_matmul;
    pub const SOFTMAX: OneDNNGraphOpType = dnnl_graph_op_kind_t::dnnl_graph_op_softmax;
    pub const STATIC_RESHAPE: OneDNNGraphOpType =
        dnnl_graph_op_kind_t::dnnl_graph_op_static_reshape;
    pub const REORDER: OneDNNGraphOpType = dnnl_graph_op_kind_t::dnnl_graph_op_reorder;

    pub fn new(
        id: usize,
        kind: OneDNNGraphOpType,
        verbose_name: impl AsRef<str>,
    ) -> Result<Self, DnnlError> {
        let c_string = CString::new(verbose_name.as_ref()).unwrap();

        let mut handle = std::ptr::null_mut();
        let status = unsafe { dnnl_graph_op_create(&mut handle, id, kind, c_string.as_ptr()) };
        if status == dnnl_status_t::dnnl_success {
            Ok(Self { handle })
        } else {
            Err(status.into())
        }
    }

    pub fn add_input(&mut self, tensor: &LogicalTensor) -> Result<(), DnnlError> {
        let status = unsafe { dnnl_graph_op_add_input(self.handle, &tensor.handle) };
        if status == dnnl_status_t::dnnl_success {
            Ok(())
        } else {
            Err(status.into())
        }
    }

    pub fn add_output(&mut self, tensor: &LogicalTensor) -> Result<(), DnnlError> {
        let status = unsafe { dnnl_graph_op_add_output(self.handle, &tensor.handle) };
        if status == dnnl_status_t::dnnl_success {
            Ok(())
        } else {
            Err(status.into())
        }
    }

    pub fn set_attribute(
        &mut self,
        name: &dnnl_graph_op_attr_t::Type,
        value: &AttrValue,
    ) -> Result<(), DnnlError> {
        let status = match value {
            AttrValue::Bool(value) => unsafe {
                dnnl_graph_op_set_attr_bool(self.handle, *name, value.as_ptr(), value.len())
            },
            AttrValue::Int(value) => unsafe {
                dnnl_graph_op_set_attr_s64(self.handle, *name, value.as_ptr(), value.len())
            },
            AttrValue::Float(value) => unsafe {
                dnnl_graph_op_set_attr_f32(self.handle, *name, value.as_ptr(), value.len())
            },
            AttrValue::Str(value) => {
                let l = value.len();
                let c_string = CString::new(value.as_str()).unwrap();
                unsafe { dnnl_graph_op_set_attr_str(self.handle, *name, c_string.as_ptr(), l) }
            }
        };
        if status == dnnl_status_t::dnnl_success {
            Ok(())
        } else {
            Err(status.into())
        }
    }
}

impl Drop for OneDNNGraphOp {
    fn drop(&mut self) {
        unsafe { dnnl_graph_op_destroy(self.handle) };
    }
}
