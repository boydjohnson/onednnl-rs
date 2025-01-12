use onednnl::primitive::Backward;

#[test]
fn test_inner_product_nchw_to_nc_backprop() {
    use onednnl::{
        engine::Engine,
        memory::{
            buffer::AlignedBuffer,
            descriptor::{new_plain_descriptor, DataType},
            Memory,
        },
        onednnl_sys::{
            DNNL_ARG_BIAS, DNNL_ARG_DIFF_BIAS, DNNL_ARG_DIFF_DST, DNNL_ARG_DIFF_SRC,
            DNNL_ARG_DIFF_WEIGHTS, DNNL_ARG_DST, DNNL_ARG_SRC, DNNL_ARG_WEIGHTS,
        },
        primitive::{
            attributes::PrimitiveAttributes, ExecArg, Primitive, PropBackwardData,
            PropBackwardWeights, PropForwardTraining,
        },
        primitives::inner_product::{
            BackwardDataInnerProduct, BackwardDataInnerProductConfig, BackwardWeightsInnerProduct,
            BackwardWeightsInnerProductConfig, ForwardInnerProduct, ForwardInnerProductConfig,
        },
        stream::Stream,
    };

    // 1. Create an engine (CPU)
    let engine = Engine::new(Engine::CPU, 0).unwrap();
    let stream = Stream::new(engine.clone()).unwrap();

    // ---------------------------------------------------
    // 2. Prepare input shapes/dimensions
    //
    //    As in the C++ example:
    //      N = 3, IC = 3, IH = 227, IW = 227, OC = 96
    //    We'll do an inner product from [N, IC, IH, IW] => [N, OC]
    //    Weights = [OC, IC, IH, IW]
    //    Bias = [OC]
    //
    let n: i64 = 15;
    let ic: i64 = 3;
    let ih: i64 = 227;
    let iw: i64 = 227;
    let oc: i64 = 96;

    let src_dims = [n, ic, ih, iw]; // shape [3, 3, 227, 227]
    let weights_dims = [oc, ic, ih, iw]; // shape [96, 3, 227, 227]
    let bias_dims = [oc]; // shape [96]
    let dst_dims = [n, oc]; // shape [3, 96]

    // 2a. Create memory descriptors (plain / row-major)
    let src_md = new_plain_descriptor(4, src_dims.to_vec(), DataType::F32);
    let weights_md = new_plain_descriptor(4, weights_dims.to_vec(), DataType::F32);
    let bias_md = new_plain_descriptor(1, bias_dims.to_vec(), DataType::F32);
    let dst_md = new_plain_descriptor(2, dst_dims.to_vec(), DataType::F32);

    // 2b. Allocate some input data (all same, just for demonstration).
    let src_len = (n * ic * ih * iw) as usize;
    let weights_len = (oc * ic * ih * iw) as usize;
    let bias_len = oc as usize;
    let dst_len = (n * oc) as usize;

    let src_data = vec![0.5_f32; src_len];
    let weights_data = vec![0.1_f32; weights_len];
    let bias_data = vec![0.0_f32; bias_len];
    let dst_data = vec![0.0_f32; dst_len]; // Will hold forward output

    // Wrap them in user buffers
    let src_buf = AlignedBuffer::new(&src_data).unwrap();
    let src_mem =
        Memory::new_with_user_buffer(engine.clone(), src_md.clone_desc().unwrap(), src_buf)
            .unwrap();

    let weights_buf = AlignedBuffer::new(&weights_data).unwrap();
    let weights_mem = Memory::new_with_user_buffer(
        engine.clone(),
        weights_md.clone_desc().unwrap(),
        weights_buf,
    )
    .unwrap();

    let bias_buf = AlignedBuffer::new(&bias_data).unwrap();
    let bias_mem =
        Memory::new_with_user_buffer(engine.clone(), bias_md.clone_desc().unwrap(), bias_buf)
            .unwrap();

    let dst_buf = AlignedBuffer::new(&dst_data).unwrap();
    let dst_mem =
        Memory::new_with_user_buffer(engine.clone(), dst_md.clone_desc().unwrap(), dst_buf)
            .unwrap();

    // ---------------------------------------------------
    // 3. Forward Inner Product
    let fwd_config = ForwardInnerProductConfig {
        src_desc: src_md.clone_desc().unwrap(),
        weights_desc: weights_md.clone_desc().unwrap(),
        bias_desc: bias_md.clone_desc().unwrap(),
        dst_desc: dst_md.clone_desc().unwrap(),
        attr: PrimitiveAttributes::new().unwrap(),
    };

    // 3a. Create the forward primitive
    let mut fwd_prim = Primitive::<_, PropForwardTraining, _>::new::<ForwardInnerProduct<_>>(
        fwd_config,
        engine.clone(),
    )
    .unwrap();

    // 3b. Execute forward
    let fwd_desc = fwd_prim
        .execute(
            &stream,
            vec![
                ExecArg {
                    index: DNNL_ARG_SRC as i32,
                    mem: &src_mem,
                },
                ExecArg {
                    index: DNNL_ARG_WEIGHTS as i32,
                    mem: &weights_mem,
                },
                ExecArg {
                    index: DNNL_ARG_BIAS as i32,
                    mem: &bias_mem,
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

    // 3c. Print a few forward outputs
    let forward_result = dst_mem.to_vec().unwrap();
    println!("\n== Forward Pass ==");
    println!("Forward output shape = [{}, {}]", n, oc);
    println!(
        "First few elements: {:?}",
        &forward_result[..8.min(forward_result.len())]
    );

    // ---------------------------------------------------
    // 4. Backward Weights: compute gradient w.r.t. weights and bias
    //
    //    We'll define diff_dst as shape = [N, OC], typically the gradient
    //    from the next layer. For demonstration, fill with 1.0.
    let diff_dst_data = vec![1.0_f32; dst_len];
    let diff_dst_buf = AlignedBuffer::new(&diff_dst_data).unwrap();
    let diff_dst_mem =
        Memory::new_with_user_buffer(engine.clone(), dst_md.clone_desc().unwrap(), diff_dst_buf)
            .unwrap();

    // We'll store diff_weights in a new user buffer
    let diff_weights_buf = AlignedBuffer::zeroed(weights_len).unwrap();
    let diff_weights_mem = Memory::new_with_user_buffer(
        engine.clone(),
        weights_md.clone_desc().unwrap(),
        diff_weights_buf,
    )
    .unwrap();

    // We'll store diff_bias in a new user buffer
    let diff_bias_buf = AlignedBuffer::zeroed(bias_len).unwrap();
    let diff_bias_mem =
        Memory::new_with_user_buffer(engine.clone(), bias_md.clone_desc().unwrap(), diff_bias_buf)
            .unwrap();

    let bwd_weights_config = BackwardWeightsInnerProductConfig {
        src_desc: src_md.clone_desc().unwrap(),
        diff_weights_desc: weights_md.clone_desc().unwrap(),
        diff_bias_desc: bias_md.clone_desc().unwrap(),
        diff_dst_desc: dst_md.clone_desc().unwrap(),
        hint_fwd_pd: &fwd_desc, // from the forward primitive
        attr: PrimitiveAttributes::new().unwrap(),
    };

    // 4a. Create backward-weights primitive
    let mut bwd_weights_prim = Primitive::<Backward, PropBackwardWeights, _>::new::<
        BackwardWeightsInnerProduct,
    >(bwd_weights_config, engine.clone())
    .unwrap();

    // 4b. Execute backward-weights
    bwd_weights_prim
        .execute(
            &stream,
            vec![
                ExecArg {
                    index: DNNL_ARG_SRC as i32,
                    mem: &src_mem,
                },
                ExecArg {
                    index: DNNL_ARG_DIFF_DST as i32,
                    mem: &diff_dst_mem,
                },
                ExecArg {
                    index: DNNL_ARG_DIFF_WEIGHTS as i32,
                    mem: &diff_weights_mem,
                },
                ExecArg {
                    index: DNNL_ARG_DIFF_BIAS as i32,
                    mem: &diff_bias_mem,
                },
            ],
        )
        .unwrap();
    stream.wait().unwrap();

    // 4c. Print a few backward-weights outputs
    let diff_weights_result = diff_weights_mem.to_vec().unwrap();
    let diff_bias_result = diff_bias_mem.to_vec().unwrap();
    println!("\n== Backward Weights ==");
    println!(
        "diff_weights: First few elements = {:?}",
        &diff_weights_result[..8.min(diff_weights_result.len())]
    );
    println!(
        "diff_bias:   First few elements = {:?}",
        &diff_bias_result[..8.min(diff_bias_result.len())]
    );

    // ---------------------------------------------------
    // 5. Backward Data: compute gradient w.r.t. src
    //
    //    We'll produce diff_src from:
    //       - diff_dst + the original weights.
    //    The shape is the same as src_dims: [N, IC, IH, IW].
    let diff_src_buf = AlignedBuffer::zeroed(src_len).unwrap();
    let diff_src_mem =
        Memory::new_with_user_buffer(engine.clone(), src_md.clone_desc().unwrap(), diff_src_buf)
            .unwrap();

    let bwd_data_config = BackwardDataInnerProductConfig {
        diff_src_desc: src_md.clone_desc().unwrap(),
        weights_desc: weights_md.clone_desc().unwrap(),
        diff_dst_desc: dst_md.clone_desc().unwrap(),
        hint_fwd_pd: &fwd_desc, // from forward pass
        attr: PrimitiveAttributes::new().unwrap(),
    };

    // 5a. Create backward-data primitive
    let mut bwd_data_prim = Primitive::<Backward, PropBackwardData, _>::new::<
        BackwardDataInnerProduct,
    >(bwd_data_config, engine.clone())
    .unwrap();

    // 5b. Execute backward-data
    bwd_data_prim
        .execute(
            &stream,
            vec![
                ExecArg {
                    index: DNNL_ARG_DIFF_DST as i32,
                    mem: &diff_dst_mem,
                },
                ExecArg {
                    index: DNNL_ARG_WEIGHTS as i32,
                    mem: &weights_mem,
                },
                ExecArg {
                    index: DNNL_ARG_DIFF_SRC as i32,
                    mem: &diff_src_mem,
                },
            ],
        )
        .unwrap();
    stream.wait().unwrap();

    // 5c. Print a few backward-data outputs
    let diff_src_result = diff_src_mem.to_vec().unwrap();
    println!("\n== Backward Data ==");
    println!(
        "diff_src shape = [N, IC, IH, IW] = [{}, {}, {}, {}]",
        n, ic, ih, iw
    );
    println!(
        "diff_src: First few elements = {:?}",
        &diff_src_result[..8.min(diff_src_result.len())]
    );
}
