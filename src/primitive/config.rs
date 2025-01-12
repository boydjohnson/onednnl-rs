use {
    super::{descriptor::PrimitiveDescriptor, Direction, PropType},
    crate::{engine::Engine, error::DnnlError},
    std::sync::Arc,
};

pub trait PrimitiveConfig<'a, D: Direction, P: PropType<D>>: Sized {
    fn create_primitive_desc(
        self,
        engine: Arc<Engine>,
    ) -> Result<PrimitiveDescriptor<'a, D, P, Self>, DnnlError>;
}
