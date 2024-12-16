use onednnl_sys::{dnnl_memory_desc_destroy, dnnl_memory_desc_t};

#[derive(Debug)]
pub struct MemoryDescriptor {
    handle: dnnl_memory_desc_t,
}

impl Drop for MemoryDescriptor {
    fn drop(&mut self) {
        unsafe {
            dnnl_memory_desc_destroy(self.handle);
        }
    }
}
