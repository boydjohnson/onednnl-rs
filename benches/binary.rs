#![feature(test)]

extern crate test;

use {
    onednnl::{
        engine::Engine,
        memory::{
            buffer::AlignedBuffer,
            data_type_size,
            descriptor::{DataType, MemoryDescriptor},
            format_tag::x,
            Memory,
        },
        primitive::{attributes::PrimitiveAttributes, ExecArg, Primitive, PropForwardInference},
        primitives::binary::{Binary, ForwardBinary, ForwardBinaryConfig},
        set_primitive_cache_capacity,
        stream::Stream,
    },
    onednnl_sys::{dnnl_data_type_t::dnnl_f32, DNNL_ARG_DST, DNNL_ARG_SRC_0, DNNL_ARG_SRC_1},
    std::sync::Arc,
    test::Bencher,
};

#[bench]
fn binary_add(b: &mut Bencher) {
    let engine = Engine::new(Engine::CPU, 0).unwrap();

    set_primitive_cache_capacity(2).unwrap();

    let stream = Arc::new(Stream::new(engine.clone()).unwrap());

    let src0_desc = MemoryDescriptor::new::<1, x>([3], DataType::F32).unwrap();
    let src1_desc = MemoryDescriptor::new::<1, x>([3], DataType::F32).unwrap();
    let dst_desc = MemoryDescriptor::new::<1, x>([3], DataType::F32).unwrap();

    let binary_config = ForwardBinaryConfig {
        alg_kind: Binary::ADD,
        src0_desc: src0_desc.clone_desc().unwrap(),
        src1_desc: src1_desc.clone_desc().unwrap(),
        dst_desc: dst_desc.clone_desc().unwrap(),
        attr: PrimitiveAttributes::new().unwrap(),
    };

    let primitive = Primitive::<_, PropForwardInference, ForwardBinaryConfig>::new::<
        ForwardBinary<_>,
    >(binary_config, engine.clone());
    assert!(primitive.is_ok());
    let primitive = primitive.unwrap();

    let s0_buffer = AlignedBuffer::new(&[4.0f32, 5.0, 6.0]).unwrap().into();

    // Allocate and initialize memory
    let src0_memory = Memory::new_with_user_buffer(engine.clone(), src0_desc, s0_buffer).unwrap();

    let s1_buffer = AlignedBuffer::new(&[1.0f32, 2.0, 3.0]).unwrap().into();

    let src1_memory = Memory::new_with_user_buffer(engine.clone(), src1_desc, s1_buffer).unwrap();

    let output = AlignedBuffer::<f32>::zeroed(dst_desc.get_size() / data_type_size(dnnl_f32))
        .unwrap()
        .into();

    let dst_memory = Memory::new_with_user_buffer(engine.clone(), dst_desc, output).unwrap();

    b.iter(|| {
        // Create the primitive

        // Configure the binary operation

        // Execute the primitive
        let args = vec![
            ExecArg {
                index: DNNL_ARG_SRC_0 as i32,
                mem: &src0_memory,
            },
            ExecArg {
                index: DNNL_ARG_SRC_1 as i32,
                mem: &src1_memory,
            },
            ExecArg {
                index: DNNL_ARG_DST as i32,
                mem: &dst_memory,
            },
        ];

        let result = primitive.execute(&stream, args);

        assert!(stream.wait().is_ok());

        assert_eq!(result, Ok(()));

        assert_eq!(dst_memory.to_vec(), Ok(vec![5.0, 7.0, 9.0]));
    });
}
