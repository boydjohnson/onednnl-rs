use {
    crate::{
        engine::Engine,
        error::DnnlError,
        graph::{
            compiled_partition::CompiledPartition, op::OneDNNGraphOp,
            tensor::logical::LogicalTensor,
        },
    },
    onednnl_sys::{
        dnnl_engine_kind_t, dnnl_graph_logical_tensor_t, dnnl_graph_partition_compile,
        dnnl_graph_partition_create_with_op, dnnl_graph_partition_destroy,
        dnnl_graph_partition_get_id, dnnl_graph_partition_get_input_ports,
        dnnl_graph_partition_get_input_ports_num, dnnl_graph_partition_get_output_ports,
        dnnl_graph_partition_get_output_ports_num, dnnl_graph_partition_is_supported,
        dnnl_graph_partition_t, dnnl_status_t,
    },
};

pub struct OneDNNGraphPartition {
    pub(crate) handle: dnnl_graph_partition_t,
}

impl OneDNNGraphPartition {
    pub fn create(engine: dnnl_engine_kind_t::Type, op: &OneDNNGraphOp) -> Result<Self, DnnlError> {
        let mut handle = std::ptr::null_mut();

        let status = unsafe {
            dnnl_graph_partition_create_with_op(&mut handle, op.handle as *const _, engine)
        };
        if status != dnnl_status_t::dnnl_success {
            Err(status.into())
        } else {
            Ok(Self { handle })
        }
    }

    pub fn id(&self) -> Result<usize, DnnlError> {
        let mut output = 0;
        let status = unsafe { dnnl_graph_partition_get_id(self.handle, &mut output) };
        if status != dnnl_status_t::dnnl_success {
            Err(status.into())
        } else {
            Ok(output)
        }
    }

    pub fn get_output_ports_num(&self) -> Result<usize, DnnlError> {
        let mut num = 0;

        let status = unsafe { dnnl_graph_partition_get_output_ports_num(self.handle, &mut num) };
        if status != dnnl_status_t::dnnl_success {
            Err(status.into())
        } else {
            Ok(num)
        }
    }

    pub fn get_input_ports_num(&self) -> Result<usize, DnnlError> {
        let mut num = 0;

        let status = unsafe { dnnl_graph_partition_get_input_ports_num(self.handle, &mut num) };
        if status != dnnl_status_t::dnnl_success {
            return Err(status.into());
        }
        Ok(num)
    }

    pub fn get_input_ports(&self) -> Result<Vec<LogicalTensor>, DnnlError> {
        let num = self.get_input_ports_num()? as usize;

        // 1) Reserve space for `num` dnnl_graph_logical_tensor_t values
        let mut raw_ports = Vec::<dnnl_graph_logical_tensor_t>::with_capacity(num);

        // 2) Call the C API to fill them in
        let status = unsafe {
            dnnl_graph_partition_get_input_ports(self.handle, num, raw_ports.as_mut_ptr())
        };
        if status != dnnl_status_t::dnnl_success {
            return Err(status.into());
        }

        // 3) *now* those slots are initialized—tell Rust how many valid entries there are
        unsafe { raw_ports.set_len(num) };

        // 4) Wrap each one in your safe type
        Ok(raw_ports
            .into_iter()
            .map(|handle| LogicalTensor { handle })
            .collect())
    }

    pub fn get_output_ports(&self) -> Result<Vec<LogicalTensor>, DnnlError> {
        let num = self.get_output_ports_num()? as usize;
        let mut raw_ports = Vec::<dnnl_graph_logical_tensor_t>::with_capacity(num);

        let status = unsafe {
            dnnl_graph_partition_get_output_ports(self.handle, num, raw_ports.as_mut_ptr())
        };
        if status != dnnl_status_t::dnnl_success {
            return Err(status.into());
        }

        unsafe { raw_ports.set_len(num) };
        Ok(raw_ports
            .into_iter()
            .map(|handle| LogicalTensor { handle })
            .collect())
    }

    pub fn is_supported(&self) -> Result<bool, DnnlError> {
        let mut supported = 0;
        let status = unsafe { dnnl_graph_partition_is_supported(self.handle, &mut supported) };
        if status != dnnl_status_t::dnnl_success {
            return Err(status.into());
        }
        Ok(supported != 0)
    }

    pub fn compile(self, engine: &Engine) -> Result<CompiledPartition, DnnlError> {
        let in_num = self.get_input_ports_num()?;
        let out_num = self.get_output_ports_num()?;
        let input_logical_tensors = self.get_input_ports()?;

        let output_logical_tensors = self.get_output_ports()?;

        let mut inputs = input_logical_tensors
            .iter()
            .map(|e| &e.handle as *const dnnl_graph_logical_tensor_t)
            .collect::<Vec<_>>();
        let mut outputs = output_logical_tensors
            .iter()
            .map(|e| &e.handle as *const dnnl_graph_logical_tensor_t)
            .collect::<Vec<_>>();

        let cp = CompiledPartition::create(self)?;

        let status = unsafe {
            dnnl_graph_partition_compile(
                cp.partition.handle,
                cp.handle,
                in_num,
                inputs.as_mut_ptr(),
                out_num,
                outputs.as_mut_ptr(),
                engine.handle,
            )
        };
        if status != dnnl_status_t::dnnl_success {
            return Err(status.into());
        }
        Ok(cp)
    }
}

impl Drop for OneDNNGraphPartition {
    fn drop(&mut self) {
        unsafe {
            dnnl_graph_partition_destroy(self.handle);
        }
    }
}
