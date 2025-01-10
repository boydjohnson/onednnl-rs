use {
    crate::{
        memory::descriptor::MemoryDescriptor,
        primitive::{
            attributes::PrimitiveAttributes, config::PrimitiveConfig,
            descriptor::PrimitiveDescriptor, Forward, Operation, OperationType,
            PropForwardInference, PropType,
        },
    },
    onednnl_sys::{dnnl_alg_kind_t, dnnl_binary_primitive_desc_create, dnnl_status_t},
};

pub struct ForwardBinaryConfig<'a> {
    pub alg_kind: dnnl_alg_kind_t::Type,
    pub src0_desc: &'a MemoryDescriptor,
    pub src1_desc: &'a MemoryDescriptor,
    pub dst_desc: &'a MemoryDescriptor,
    pub attr: &'a PrimitiveAttributes,
}

impl<'a> PrimitiveConfig<'a, Forward, PropForwardInference> for ForwardBinaryConfig<'a> {
    fn create_primitive_desc(
        &self,
        engine: std::sync::Arc<crate::engine::Engine>,
    ) -> Result<crate::primitive::descriptor::PrimitiveDescriptor, crate::error::DnnlError> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            dnnl_binary_primitive_desc_create(
                &mut handle,
                engine.handle,
                self.alg_kind,
                self.src0_desc.handle,
                self.src1_desc.handle,
                self.dst_desc.handle,
                self.attr.handle,
            )
        };

        if status == dnnl_status_t::dnnl_success {
            Ok(PrimitiveDescriptor { handle })
        } else {
            Err(status.into())
        }
    }
}

pub struct Binary;

impl Binary {
    pub const ADD: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_binary_add;
    pub const DIV: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_binary_div;
    pub const EQ: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_binary_eq;
    pub const GT: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_binary_gt;
    pub const GE: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_binary_ge;
    pub const LE: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_binary_le;
    pub const LT: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_binary_lt;
    pub const MAX: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_binary_max;
    pub const MIN: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_binary_min;
    pub const MUL: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_binary_mul;
    pub const NE: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_binary_ne;
    pub const SUB: dnnl_alg_kind_t::Type = dnnl_alg_kind_t::dnnl_binary_sub;
}

pub struct ForwardBinary<P: PropType<Forward>> {
    pub prop_type: P,
}

impl<'a> Operation<'a, Forward, PropForwardInference> for ForwardBinary<PropForwardInference> {
    const TYPE: OperationType = OperationType::Binary;

    type OperationConfig = ForwardBinaryConfig<'a>;
}
