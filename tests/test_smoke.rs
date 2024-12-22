use {
    onednnl::{
        engine::Engine,
        memory::{descriptor::MemoryDescriptor, format_tag::x, Memory},
        primitive::{
            config::binary::ForwardBinaryConfig, ExecArg, ForwardBinary, Primitive,
            PropForwardInference,
        },
        stream::Stream,
    },
    onednnl_sys::{
        dnnl_alg_kind_t, dnnl_data_type_t::dnnl_f32, DNNL_ARG_DST, DNNL_ARG_SRC_0, DNNL_ARG_SRC_1,
    },
    std::ffi::c_void,
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

    use std::alloc::{alloc, Layout};

    // Allocate aligned memory for `src0`
    let layout = Layout::array::<f32>(3).unwrap();
    let s0_ptr = unsafe { alloc(layout) as *mut f32 };
    let s0 = unsafe { std::slice::from_raw_parts_mut(s0_ptr, 3) };
    s0.copy_from_slice(&[1.0, 2.0, 3.0]);

    // Allocate and initialize memory
    let src0_memory =
        Memory::new_with_user_buffer(engine.clone(), src0_desc, s0.as_mut_ptr() as *mut c_void)
            .unwrap();

    let layout = Layout::array::<f32>(3).unwrap();
    let s1_ptr = unsafe { alloc(layout) as *mut f32 };
    let s1 = unsafe { std::slice::from_raw_parts_mut(s1_ptr, 3) };
    s1.copy_from_slice(&[4.0, 5.0, 6.0]);

    let src1_memory =
        Memory::new_with_user_buffer(engine.clone(), src1_desc, s1.as_mut_ptr() as *mut c_void)
            .unwrap();

    let mut output = vec![0.0f32; 3].into_boxed_slice();

    let dst_memory =
        Memory::new_with_user_buffer(engine.clone(), dst_desc, output.as_mut_ptr() as *mut c_void)
            .unwrap();

    // Configure the binary operation

    // Execute the primitive
    let stream = Stream::new(engine.clone()).unwrap();
    let args = vec![
        ExecArg {
            index: DNNL_ARG_SRC_0 as i32,
            mem: src0_memory,
        },
        ExecArg {
            index: DNNL_ARG_SRC_1 as i32,
            mem: src1_memory,
        },
        ExecArg {
            index: DNNL_ARG_DST as i32,
            mem: dst_memory,
        },
    ];

    let result = primitive.execute(&stream, args);

    assert!(stream.wait().is_ok());

    assert_eq!(result, Ok(()));

    assert_eq!(output, vec![5.0, 7.0, 9.0].into());
}
