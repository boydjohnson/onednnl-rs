use onednnl::{
    engine::Engine,
    memory::{
        buffer::AlignedBuffer,
        data_type_size,
        descriptor::{new_plain_descriptor, DataType},
        Memory,
    },
    onednnl_sys::{DNNL_ARG_DIFF_DST, DNNL_ARG_DIFF_SRC, DNNL_ARG_DST, DNNL_ARG_SRC},
    primitive::{
        attributes::PrimitiveAttributes, Backward, ExecArg, Primitive, PropBackward,
        PropForwardTraining,
    },
    primitives::eltwise::{
        BackwardEltwise, BackwardEltwiseConfig, ForwardEltwise, ForwardEltwiseConfig, Unary,
    },
    stream::Stream,
};

#[test]
fn test_relu_forward_backward() {
    // 1. Create an engine (CPU in this example)
    let engine = Engine::new(Engine::CPU, 0).unwrap();

    // ---------------------------------------------------
    // 2. Prepare input data (shape = [2, 3])
    //    We'll intentionally include negative values to test ReLU clamping at 0.
    let src_data: Vec<f32> = vec![-1.0f32, 2.0, -3.0, 4.0, 0.0, 5.0];
    let dims = [2, 3];

    // 2a. Create a memory descriptor for src
    let src_md = new_plain_descriptor(2, dims.to_vec(), DataType::F32);

    let dst_md = new_plain_descriptor(2, dims.to_vec(), DataType::F32);

    let forward_config = ForwardEltwiseConfig {
        alg_kind: Unary::RELU, // ReLU forward
        src_desc: src_md.clone_desc().unwrap(),
        dst_desc: dst_md.clone_desc().unwrap(),
        alpha: 0.0,
        beta: 0.0,
        attr: PrimitiveAttributes::new().unwrap(), // no special attributes
    };

    // 3b. Create the forward primitive
    let mut fwd_prim = Primitive::<_, PropForwardTraining, ForwardEltwiseConfig>::new::<
        ForwardEltwise<_>,
    >(forward_config, engine.clone())
    .unwrap();

    // 3c. Allocate memory for the forward result

    let a_buffer =
        AlignedBuffer::zeroed(dst_md.get_size() / data_type_size(DataType::F32)).unwrap();

    let dst_mem = Memory::new_with_user_buffer(engine.clone(), dst_md, a_buffer).unwrap();

    let buffer = AlignedBuffer::new(&src_data).unwrap();

    let src_mem = Memory::new_with_user_buffer(engine.clone(), src_md, buffer).unwrap();

    let stream = Stream::new(engine.clone()).unwrap();

    // 3d. Execute forward ReLU
    let fwd_desc = fwd_prim
        .execute(
            &stream,
            vec![
                ExecArg {
                    index: DNNL_ARG_SRC as i32,
                    mem: &src_mem,
                },
                ExecArg {
                    index: DNNL_ARG_DST as i32,
                    mem: &dst_mem,
                },
            ],
        )
        .unwrap()
        .unwrap();

    stream.wait().unwrap();

    // ---------------------------------------------------
    // 4. Validate Forward Output
    //
    //    ReLU(x) = max(0, x). So for:
    //       -1.0 -> 0.0
    //        2.0 -> 2.0
    //       -3.0 -> 0.0
    //        4.0 -> 4.0
    //        0.0 -> 0.0
    //        5.0 -> 5.0
    let forward_result = dst_mem.to_vec().unwrap();

    let expected_forward: Vec<f32> = vec![0.0, 2.0, 0.0, 4.0, 0.0, 5.0];
    assert_eq!(
        forward_result, expected_forward,
        "Forward ReLU output mismatch"
    );

    // ---------------------------------------------------
    // 5. Backward ReLU Configuration
    //
    //    We'll define "diff_dst" as if the gradient from the next layer is all 1.0:
    //    shape = [2, 3], so all ones => [1,1,1,1,1,1].
    let diff_dst_data = AlignedBuffer::new(&vec![1.0; src_data.len()]).unwrap();
    let diff_dst_md = src_mem.desc.clone_desc().unwrap();

    //    We'll store the result of the backward pass (the gradient w.r.t src) in diff_src
    let diff_src_md = src_mem.desc.clone_desc().unwrap();

    //    We also need a "forward hint descriptor", from the forward pass

    let bwd_config = BackwardEltwiseConfig {
        alg_kind: Unary::RELU_USE_DST_FOR_BWD,
        diff_src_desc: diff_src_md.clone_desc().unwrap(),
        diff_dest_desc: diff_dst_md.clone_desc().unwrap(),
        data_desc: dst_mem.desc.clone_desc().unwrap(), // "data_desc" is typically the forward data or forward dst
        alpha: 0.0,
        beta: 0.0,
        forward_hint_desc: &fwd_desc,
        attr: PrimitiveAttributes::new().unwrap(),
    };

    // 5b. Create the backward primitive
    let mut bwd_prim = Primitive::<Backward, PropBackward, BackwardEltwiseConfig>::new::<
        BackwardEltwise<_>,
    >(bwd_config, engine.clone())
    .unwrap();

    let diff_dst_mem =
        Memory::new_with_user_buffer(engine.clone(), diff_dst_md, diff_dst_data).unwrap();

    let a_buffer =
        AlignedBuffer::zeroed(diff_src_md.get_size() / data_type_size(DataType::F32)).unwrap();

    let diff_src_mem = Memory::new_with_user_buffer(engine.clone(), diff_src_md, a_buffer).unwrap();

    // 5c. Execute backward ReLU
    //
    //    We'll pass:
    //      - "diff_dst" as input (DNNL_ARG_DIFF_DST),
    //      - "dst" from forward pass if using *_USE_DST_FOR_BWD variant
    //      - "diff_src" as output.
    bwd_prim
        .execute(
            &stream,
            vec![
                // "diff_dst" as input gradient
                ExecArg {
                    index: DNNL_ARG_DIFF_DST as i32,
                    mem: &diff_dst_mem,
                },
                // "dst" from the forward pass if using *_USE_DST_FOR_BWD
                ExecArg {
                    index: DNNL_ARG_DST as i32,
                    mem: &dst_mem,
                },
                // "diff_src" as output gradient (w.r.t. src)
                ExecArg {
                    index: DNNL_ARG_DIFF_SRC as i32,
                    mem: &diff_src_mem,
                },
            ],
        )
        .unwrap();

    stream.wait().unwrap();

    // ---------------------------------------------------
    // 6. Validate Backward Gradient
    //
    //    The rule for ReLU backward:
    //      dX = dY if Y > 0, else 0    ( for standard ReLU ).
    //
    //    Our forward output was [0,2,0,4,0,5].
    //    The "diff_dst" is all 1.0 => [1,1,1,1,1,1].
    //    => diff_src = [0,1,0,1,0,1].
    let backward_result = diff_src_mem.to_vec().unwrap();

    let expected_backward = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    assert_eq!(
        backward_result, expected_backward,
        "Backward ReLU gradient mismatch"
    );
}
