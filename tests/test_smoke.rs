use {
    onednnl::{
        engine::Engine,
        memory::{buffer::AlignedBuffer, descriptor::MemoryDescriptor, format_tag::x, Memory},
        primitive::{
            config::binary::ForwardBinaryConfig, ExecArg, ForwardBinary, Primitive,
            PropForwardInference,
        },
        stream::Stream,
    },
    onednnl_sys::{
        dnnl_alg_kind_t, dnnl_data_type_t::dnnl_f32, DNNL_ARG_DST, DNNL_ARG_SRC_0, DNNL_ARG_SRC_1,
    },
};

#[test]
pub fn test_smoke_binary_add() {
    let engine = Engine::new(Engine::CPU, 0).unwrap();

    let src0_desc = MemoryDescriptor::new::<1, x>([3], dnnl_f32).unwrap();
    let src1_desc = MemoryDescriptor::new::<1, x>([3], dnnl_f32).unwrap();
    let dst_desc = MemoryDescriptor::new::<1, x>([3], dnnl_f32).unwrap();

    let binary_config = ForwardBinaryConfig {
        alg_kind: dnnl_alg_kind_t::dnnl_binary_add,
        src0_desc: &src0_desc,
        src1_desc: &src1_desc,
        dst_desc: &dst_desc,
        attr: std::ptr::null_mut(),
    };

    // Create the primitive
    let primitive =
        Primitive::new::<_, PropForwardInference, ForwardBinary<_>>(binary_config, engine.clone());
    assert!(primitive.is_ok());
    let primitive = primitive.unwrap();

    let s0_buffer = AlignedBuffer::new(&[4.0f32, 5.0, 6.0]).unwrap();

    // Allocate and initialize memory
    let src0_memory = Memory::new_with_user_buffer(engine.clone(), src0_desc, &s0_buffer).unwrap();

    let s1_buffer = AlignedBuffer::new(&[1.0f32, 2.0, 3.0]).unwrap();

    let src1_memory = Memory::new_with_user_buffer(engine.clone(), src1_desc, &s1_buffer).unwrap();

    let output = AlignedBuffer::<f32>::zeroed(3).unwrap();

    let dst_memory = Memory::new_with_user_buffer(engine.clone(), dst_desc, &output).unwrap();

    // Configure the binary operation

    // Execute the primitive
    let stream = Stream::new(engine.clone()).unwrap();
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

    assert_eq!(output.as_slice(), &[5.0, 7.0, 9.0]);
}
