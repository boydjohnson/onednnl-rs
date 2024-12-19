use {
    super::{descriptor::PrimitiveDescriptor, Operation},
    crate::{engine::Engine, error::DnnlError},
    std::sync::Arc,
};

pub mod au_gru;

pub trait PrimitiveConfig<O: Operation> {
    fn create_primitive_desc(&self, engine: Arc<Engine>) -> Result<PrimitiveDescriptor, DnnlError>;
}
