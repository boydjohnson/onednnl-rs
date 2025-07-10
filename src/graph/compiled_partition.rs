use onednnl_sys::{
    dnnl_graph_compiled_partition_create, dnnl_graph_compiled_partition_destroy,
    dnnl_graph_compiled_partition_execute, dnnl_graph_compiled_partition_query_logical_tensor,
    dnnl_graph_compiled_partition_t, dnnl_status_t,
};

use crate::{
    error::DnnlError,
    graph::{
        partition::OneDNNGraphPartition,
        tensor::{logical::LogicalTensor, tensor::Tensor},
    },
    stream::Stream,
};

pub struct CompiledPartition {
    pub(crate) handle: dnnl_graph_compiled_partition_t,
    pub(crate) partition: OneDNNGraphPartition,
}

impl CompiledPartition {
    pub fn create(partition: OneDNNGraphPartition) -> Result<Self, DnnlError> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe { dnnl_graph_compiled_partition_create(&mut handle, partition.handle) };
        if status != dnnl_status_t::dnnl_success {
            return Err(status.into());
        }
        Ok(CompiledPartition { handle, partition })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        inputs: &[Tensor],
        outputs: &[&mut Tensor],
    ) -> Result<(), DnnlError> {
        // Collect the input tensor handles into a vector. This ensures the collection
        // of pointers has a stable memory location that lives long enough for the C call.
        let mut input_handles: Vec<_> = inputs.iter().map(|t| t.handle as *const _).collect();

        // Do the same for the output tensor handles.
        let mut output_handles: Vec<_> = outputs.iter().map(|t| t.handle as *const _).collect();

        // The C API expects the number of inputs/outputs as an integer type.
        let num_inputs = input_handles.len();
        let num_outputs = output_handles.len();

        let status = unsafe {
            // Now, we pass pointers to our vectors' data, which are guaranteed to be
            // valid for the duration of this call.
            dnnl_graph_compiled_partition_execute(
                self.handle,
                stream.handle,
                num_inputs,
                input_handles.as_mut_ptr(),
                num_outputs,
                output_handles.as_mut_ptr(), // The C API uses these handles to find the output buffers
            )
        };

        if status != dnnl_status_t::dnnl_success {
            return Err(status.into());
        }

        Ok(())
    }

    pub fn query_logical_tensor(&self, index: usize) -> Result<LogicalTensor, DnnlError> {
        let mut logical_tensor = std::mem::MaybeUninit::uninit();
        let status = unsafe {
            dnnl_graph_compiled_partition_query_logical_tensor(
                self.handle,
                index,
                logical_tensor.as_mut_ptr(),
            )
        };

        if status != dnnl_status_t::dnnl_success {
            return Err(status.into());
        }

        let lt = unsafe {
            LogicalTensor {
                handle: logical_tensor.assume_init(),
            }
        };
        Ok(lt)
    }
}

impl Drop for CompiledPartition {
    fn drop(&mut self) {
        unsafe {
            dnnl_graph_compiled_partition_destroy(self.handle);
        }
    }
}
