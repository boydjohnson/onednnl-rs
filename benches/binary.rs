#![feature(test)]

extern crate test;

use {
    onednnl::{
        engine::Engine,
        memory::{
            buffer::AlignedBuffer,
            descriptor::{DataType, MemoryDescriptor},
            format_tag::x,
            Memory,
        },
        primitive::{
            config::{
                binary::{Binary, ForwardBinaryConfig},
            },
            ExecArg, ForwardBinary, Primitive, PropForwardInference,
        },
        stream::Stream,
    },
    onednnl_sys::{DNNL_ARG_DST, DNNL_ARG_SRC_0, DNNL_ARG_SRC_1},
    std::sync::Arc,
    test::Bencher,
};

#[bench]
fn binary_add(b: &mut Bencher) {
    let engine = Engine::new(Engine::CPU, 0).unwrap();

    let stream = Arc::new(Stream::new(engine.clone()).unwrap());

    let src0_desc = MemoryDescriptor::new::<1, x>([3], DataType::F32).unwrap();
    let src1_desc = MemoryDescriptor::new::<1, x>([3], DataType::F32).unwrap();
    let dst_desc = MemoryDescriptor::new::<1, x>([3], DataType::F32).unwrap();

    let binary_config = ForwardBinaryConfig {
        alg_kind: Binary::ADD,
        src0_desc: &src0_desc,
        src1_desc: &src1_desc,
        dst_desc: &dst_desc,
        attr: std::ptr::null_mut(),
    };

    let primitive =
        Primitive::new::<_, PropForwardInference, ForwardBinary<_>>(binary_config, engine.clone());
    assert!(primitive.is_ok());
    let primitive = primitive.unwrap();

    let mut s0_buffer = AlignedBuffer::new(&[4.0f32, 5.0, 6.0]).unwrap().into();

    // Allocate and initialize memory
    let src0_memory =
        Memory::new_with_user_buffer(engine.clone(), src0_desc, &mut s0_buffer).unwrap();

    let mut s1_buffer = AlignedBuffer::new(&[1.0f32, 2.0, 3.0]).unwrap().into();

    let src1_memory =
        Memory::new_with_user_buffer(engine.clone(), src1_desc, &mut s1_buffer).unwrap();

    let mut output = AlignedBuffer::<f32>::zeroed(3).unwrap().into();

    let dst_memory = Memory::new_with_user_buffer(engine.clone(), dst_desc, &mut output).unwrap();

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

        assert_eq!(output.to_vec::<f32>(), vec![5.0, 7.0, 9.0]);
    });
}
