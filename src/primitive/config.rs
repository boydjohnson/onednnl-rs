use {
    super::{descriptor::PrimitiveDescriptor, Direction, PropType},
    crate::{engine::Engine, error::DnnlError},
    std::sync::Arc,
};

pub mod au_gru;
pub mod batch_norm;
pub mod binary;
pub mod eltwise;
pub mod inner_product;

pub trait PrimitiveConfig<'a, D: Direction, P: PropType<D>> {
    fn create_primitive_desc(&self, engine: Arc<Engine>) -> Result<PrimitiveDescriptor, DnnlError>;
}
