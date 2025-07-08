use {
    super::op::OneDNNGraphOp,
    crate::{error::DnnlError, graph::partition::OneDNNGraphPartition},
    onednnl_sys::{
        dnnl_graph_add_op, dnnl_graph_graph_create, dnnl_graph_graph_create_with_fpmath_mode,
        dnnl_graph_graph_destroy, dnnl_graph_graph_filter, dnnl_graph_graph_finalize,
        dnnl_graph_graph_get_fpmath_mode, dnnl_graph_graph_get_partition_num,
        dnnl_graph_graph_get_partitions, dnnl_graph_graph_is_finalized, dnnl_graph_graph_t,
        dnnl_status_t,
    },
};

pub struct OneDNNGraph {
    handle: dnnl_graph_graph_t,
    ops: Vec<OneDNNGraphOp>,
}

impl OneDNNGraph {
    pub fn new(engine_type: onednnl_sys::dnnl_engine_kind_t::Type) -> Result<Self, DnnlError> {
        let mut handle: dnnl_graph_graph_t = std::ptr::null_mut();
        let status = unsafe { dnnl_graph_graph_create(&mut handle, engine_type) };
        if status == dnnl_status_t::dnnl_success {
            Ok(Self {
                handle,
                ops: Vec::new(),
            })
        } else {
            Err(status.into())
        }
    }

    pub fn new_with_fpmath_mode(
        engine_type: onednnl_sys::dnnl_engine_kind_t::Type,
        fp_mode: onednnl_sys::dnnl_fpmath_mode_t::Type,
    ) -> Result<Self, DnnlError> {
        let mut handle: dnnl_graph_graph_t = std::ptr::null_mut();
        let status =
            unsafe { dnnl_graph_graph_create_with_fpmath_mode(&mut handle, engine_type, fp_mode) };
        if status == dnnl_status_t::dnnl_success {
            Ok(Self {
                handle,
                ops: Vec::new(),
            })
        } else {
            Err(status.into())
        }
    }

    pub fn filter(
        &self,
        policy: onednnl_sys::dnnl_graph_partition_policy_t::Type,
    ) -> Result<(), DnnlError> {
        let status = unsafe { dnnl_graph_graph_filter(self.handle, policy) };
        if status == dnnl_status_t::dnnl_success {
            Ok(())
        } else {
            Err(status.into())
        }
    }

    pub fn finalize(&self) -> Result<(), DnnlError> {
        let status = unsafe { dnnl_graph_graph_finalize(self.handle) };
        if status == dnnl_status_t::dnnl_success {
            Ok(())
        } else {
            Err(status.into())
        }
    }

    pub fn ops(&self) -> &[OneDNNGraphOp] {
        &self.ops
    }

    pub fn is_finalized(&self) -> Result<bool, DnnlError> {
        let mut is_finalized = 0;
        let status = unsafe { dnnl_graph_graph_is_finalized(self.handle, &mut is_finalized) };
        if status == dnnl_status_t::dnnl_success {
            Ok(is_finalized != 0)
        } else {
            Err(status.into())
        }
    }

    pub fn get_fpmath_mode(
        &self,
    ) -> Result<(onednnl_sys::dnnl_fpmath_mode_t::Type, i32), DnnlError> {
        let mut mode = onednnl_sys::dnnl_fpmath_mode_t::dnnl_fpmath_mode_strict;
        let mut apply_to_int = 0;

        let status =
            unsafe { dnnl_graph_graph_get_fpmath_mode(self.handle, &mut mode, &mut apply_to_int) };
        if status == dnnl_status_t::dnnl_success {
            Ok((mode, apply_to_int))
        } else {
            Err(status.into())
        }
    }
    pub fn get_partition_num(&self) -> Result<usize, DnnlError> {
        let mut num = 0;

        let status = unsafe { dnnl_graph_graph_get_partition_num(self.handle, &mut num) };
        if status == dnnl_status_t::dnnl_success {
            Ok(num)
        } else {
            Err(status.into())
        }
    }

    pub fn get_partitions(&self) -> Result<Vec<OneDNNGraphPartition>, DnnlError> {
        let num = self.get_partition_num()?;
        let mut partitions = Vec::with_capacity(num);

        let status =
            unsafe { dnnl_graph_graph_get_partitions(self.handle, num, partitions.as_mut_ptr()) };
        if status == dnnl_status_t::dnnl_success {
            unsafe { partitions.set_len(partitions.capacity()) };
            Ok(partitions
                .into_iter()
                .map(|p| OneDNNGraphPartition { handle: p })
                .collect())
        } else {
            Err(status.into())
        }
    }

    pub fn add_op(&mut self, op: OneDNNGraphOp) -> Result<(), DnnlError> {
        let status = unsafe { dnnl_graph_add_op(self.handle, op.handle) };
        self.ops.push(op);
        if status == dnnl_status_t::dnnl_success {
            Ok(())
        } else {
            Err(status.into())
        }
    }
}

impl Drop for OneDNNGraph {
    fn drop(&mut self) {
        unsafe { dnnl_graph_graph_destroy(self.handle) };
    }
}
