use {
    onednnl::{
        engine::Engine,
        memory::{
            buffer::AlignedBuffer,
            data_type_size,
            descriptor::{
                new_plain_descriptor, DataType, DataTypeQuery, DimsQuery, MemoryDescriptor,
                NDimsQuery,
            },
            format_tag::{abc, abcd, x},
            Memory,
        },
        primitive::{
            attributes::PrimitiveAttributes, ExecArg, Primitive, PropBackward, PropBackwardData,
            PropBackwardWeights, PropForwardInference, PropForwardTraining,
        },
        primitives::{
            binary::{Binary, ForwardBinary, ForwardBinaryConfig},
            eltwise::{
                BackwardEltwise, BackwardEltwiseConfig, ForwardEltwise, ForwardEltwiseConfig, Unary,
            },
            matmul::{ForwardMatMul, ForwardMatMulConfig},
            reduction::{ForwardReduction, ForwardReductionConfig, Reduction},
        },
        stream::Stream,
    },
    onednnl_sys::{
        dnnl_data_type_t::dnnl_f32, DNNL_ARG_BIAS, DNNL_ARG_DIFF_DST, DNNL_ARG_DIFF_SRC,
        DNNL_ARG_DST, DNNL_ARG_SRC, DNNL_ARG_SRC_0, DNNL_ARG_SRC_1, DNNL_ARG_WEIGHTS,
    },
    std::sync::Arc,
};

#[test]
pub fn test_smoke_binary_add() {
    let engine = Engine::new(Engine::CPU, 0).unwrap();

    let src0_desc = MemoryDescriptor::new::<1, x>([3], dnnl_f32).unwrap();
    let src1_desc = MemoryDescriptor::new::<1, x>([3], dnnl_f32).unwrap();
    let dst_desc = MemoryDescriptor::new::<1, x>([3], dnnl_f32).unwrap();

    let binary_config = ForwardBinaryConfig {
        alg_kind: Binary::ADD,
        src0_desc: &src0_desc,
        src1_desc: &src1_desc,
        dst_desc: &dst_desc,
        attr: &PrimitiveAttributes::new().unwrap(),
    };

    // Create the primitive
    let primitive =
        Primitive::new::<_, PropForwardInference, ForwardBinary<_>>(binary_config, engine.clone());
    assert!(primitive.is_ok());
    let primitive = primitive.unwrap();

    let s0_buffer = AlignedBuffer::new(&[4.0f32, 5.0, 6.0]).unwrap().into();

    // Allocate and initialize memory
    let src0_memory = Memory::new_with_user_buffer(engine.clone(), src0_desc, s0_buffer).unwrap();

    let s1_buffer = AlignedBuffer::new(&[1.0f32, 2.0, 3.0]).unwrap().into();

    let src1_memory = Memory::new_with_user_buffer(engine.clone(), src1_desc, s1_buffer).unwrap();

    let dst_memory = Memory::new_with_library_buffer(engine.clone(), dst_desc).unwrap();

    // Configure the binary operation

    // Execute the primitive
    let stream = Arc::new(Stream::new(engine.clone()).unwrap());
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
}

#[test]
pub fn test_smoke_matmul() {
    // Step 1: Initialize the oneDNN Engine
    let engine = Engine::new(Engine::CPU, 0).expect("Failed to create oneDNN engine");

    // Step 2: Define Memory Descriptors
    // For MatMul: src [batch, M, K], weights [batch, K, N], bias [batch, N], dst [batch, M, N]
    // Here, we'll disable bias by passing a zeroed MemoryDescriptor

    let src_shape = [1, 2, 3]; // [batch, M, K] where M=2, K=3
    let weights_shape = [1, 3, 2]; // [batch, K, N] where N=2
    let dst_shape = [1, 2, 2]; // [batch, M, N]

    // Create MemoryDescriptors with specific format tags instead of AnyFormat
    // Assuming 'Abc' is a predefined format tag in your implementation
    let src_desc = MemoryDescriptor::new::<3, abc>(src_shape, DataType::F32)
        .expect("Failed to create src memory descriptor");
    let weights_desc = MemoryDescriptor::new::<3, abc>(weights_shape, DataType::F32)
        .expect("Failed to create weights memory descriptor");
    let dst_desc = MemoryDescriptor::new::<3, abc>(dst_shape, DataType::F32)
        .expect("Failed to create destination memory descriptor");

    // Step 3: Allocate Aligned Buffers
    // Initialize src and weights with sample data, dst with zeros
    let src_buffer = AlignedBuffer::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("Failed to allocate src buffer")
        .into();
    let weights_buffer = AlignedBuffer::new(&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0])
        .expect("Failed to allocate weights buffer")
        .into();
    let output_buffer =
        AlignedBuffer::<f32>::zeroed(dst_desc.get_size() / data_type_size(dnnl_f32))
            .expect("Failed to allocate output buffer")
            .into();

    let zero_bias_desc = MemoryDescriptor::new::<3, abc>([1, 2, 2], DataType::F32).unwrap();

    // Step 4: Configure the MatMul Operation
    // Set up ForwardMatMulConfig with references to the memory descriptors
    let matmul_config = ForwardMatMulConfig {
        src_desc: &src_desc,
        weights_desc: &weights_desc,
        bias_desc: &zero_bias_desc, // Disables bias
        dst_desc: &dst_desc,
        attr: &PrimitiveAttributes::new().unwrap(),
    };

    // Step 5: Create and Configure the MatMul Primitive
    // Instantiate the matmul primitive using the configuration
    let primitive =
        Primitive::new::<_, PropForwardInference, ForwardMatMul<_>>(matmul_config, engine.clone())
            .expect("Failed to create MatMul primitive");

    // Step 6: Create Memory Objects
    // Wrap the buffers into oneDNN Memory objects
    let src_memory = Memory::new_with_user_buffer(engine.clone(), src_desc, src_buffer)
        .expect("Failed to create src memory");
    let weights_memory = Memory::new_with_user_buffer(engine.clone(), weights_desc, weights_buffer)
        .expect("Failed to create weights memory");

    // Since we are disabling bias, create a Memory object without a buffer
    let bias_memory = Memory::new_without_buffer(engine.clone(), zero_bias_desc)
        .expect("Failed to create bias memory (disabled)");

    let dst_memory = Memory::new_with_user_buffer(engine.clone(), dst_desc, output_buffer)
        .expect("Failed to create destination memory");

    // Step 7: Create a Stream
    // A stream is required to execute the primitive
    let stream = Arc::new(Stream::new(engine.clone()).expect("Failed to create stream"));

    // Step 8: Define Execution Arguments
    // Map each argument kind to its corresponding Memory object
    let args = vec![
        ExecArg {
            index: DNNL_ARG_SRC_0 as i32,
            mem: &src_memory,
        },
        ExecArg {
            index: DNNL_ARG_WEIGHTS as i32,
            mem: &weights_memory,
        },
        ExecArg {
            index: DNNL_ARG_BIAS as i32,
            mem: &bias_memory, // Bias is disabled, but required by the API
        },
        ExecArg {
            index: DNNL_ARG_DST as i32,
            mem: &dst_memory,
        },
    ];

    // Step 9: Execute the MatMul Primitive
    // Pass the stream and arguments to execute the operation
    primitive
        .execute(&stream, args)
        .expect("Failed to execute MatMul primitive");

    // Step 10: Wait for the Stream to Complete
    // Ensure that the operation has finished
    stream.wait().expect("Failed to wait on the stream");

    // Step 11: Verify the Output
    // Calculate the expected result manually and compare
    // Expected Calculation:
    // src = [1, 2, 3], [4, 5, 6] (batch=1, M=2, K=3)
    // weights = [7, 8], [9, 10], [11, 12] (batch=1, K=3, N=2)
    // matmul = [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [7 + 18 + 33, 8 + 20 + 36] = [58, 64]
    //          [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [28 + 45 + 66, 32 + 50 + 72] = [139, 154]
    // Since batch=1, the expected output is [58, 64, 139, 154]

    let expected = vec![58.0f32, 64.0, 139.0, 154.0];
    assert_eq!(
        dst_memory.to_vec(),
        Ok(expected),
        "MatMul output does not match expected results"
    );
}

#[test]
pub fn test_smoke_memory_desc() {
    let md = MemoryDescriptor::new::<4, abcd>([1, 3, 228, 228], dnnl_f32).unwrap();

    let ndims = md.query::<NDimsQuery>().unwrap();
    println!("Number of dimensions: {}", ndims);

    let data_type = md.query::<DataTypeQuery>().unwrap();
    println!("Data type: {:?}", data_type);

    let dims = md.query::<DimsQuery>().unwrap();
    println!("Dimensions: {:?}", dims);

    assert_eq!(ndims, 4);
    assert_eq!(data_type, dnnl_f32);
    assert_eq!(dims, vec![1, 3, 228, 228]);
}

#[test]
pub fn test_reduction_smoke() {
    let engine = Engine::new(Engine::CPU, 0).unwrap();

    let src_desc = MemoryDescriptor::new::<1, x>([3], dnnl_f32).unwrap();
    let dst_desc = MemoryDescriptor::new::<1, x>([1], dnnl_f32).unwrap();

    let reduction_config = ForwardReductionConfig {
        alg_kind: Reduction::SUM,
        src_desc: &src_desc,
        dst_desc: &dst_desc,
        p: 0.0,
        eps: 0.0,
        attr: &PrimitiveAttributes::new().unwrap(),
    };

    // Create the primitive
    let primitive = Primitive::new::<_, PropForwardInference, ForwardReduction>(
        reduction_config,
        engine.clone(),
    );
    assert!(primitive.is_ok());
    let primitive = primitive.unwrap();

    let src_buffer = AlignedBuffer::new(&[1.0f32, 2.0, 3.0]).unwrap().into();

    // Allocate and initialize memory
    let src_memory = Memory::new_with_user_buffer(engine.clone(), src_desc, src_buffer).unwrap();

    let output = AlignedBuffer::<f32>::zeroed(dst_desc.get_size() / data_type_size(dnnl_f32))
        .unwrap()
        .into();

    let dst_memory = Memory::new_with_user_buffer(engine.clone(), dst_desc, output).unwrap();

    // Execute the primitive
    let stream = Arc::new(Stream::new(engine.clone()).unwrap());
    let args = vec![
        ExecArg {
            index: DNNL_ARG_SRC as i32,
            mem: &src_memory,
        },
        ExecArg {
            index: DNNL_ARG_DST as i32,
            mem: &dst_memory,
        },
    ];

    let result = primitive.execute(&stream, args);

    assert!(stream.wait().is_ok());

    assert_eq!(result, Ok(()));

    assert_eq!(dst_memory.to_vec(), Ok(vec![6.0]));
}

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
        src_desc: &src_md,
        dst_desc: &dst_md,
        alpha: 0.0,
        beta: 0.0,
        attr: &PrimitiveAttributes::new().unwrap(), // no special attributes
    };

    // 3b. Create the forward primitive
    let fwd_prim =
        Primitive::new::<_, PropForwardTraining, ForwardEltwise<_>>(forward_config, engine.clone())
            .unwrap();

    // 3c. Allocate memory for the forward result

    let a_buffer =
        AlignedBuffer::zeroed(dst_md.get_size() / data_type_size(DataType::F32)).unwrap();

    let dst_mem = Memory::new_with_user_buffer(engine.clone(), dst_md, a_buffer).unwrap();

    let buffer = AlignedBuffer::new(&src_data).unwrap();

    let src_mem = Memory::new_with_user_buffer(engine.clone(), src_md, buffer).unwrap();

    let stream = Stream::new(engine.clone()).unwrap();

    // 3d. Execute forward ReLU
    fwd_prim
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
    let forward_hint_desc = fwd_prim.desc.handle; // The C-level primitive_desc handle

    let bwd_config = BackwardEltwiseConfig {
        alg_kind: Unary::RELU_USE_DST_FOR_BWD,
        diff_src_desc: &diff_src_md,
        diff_dest_desc: &diff_dst_md,
        data_desc: &dst_mem.desc, // "data_desc" is typically the forward data or forward dst
        alpha: 0.0,
        beta: 0.0,
        forward_hint_desc,
        attr: &PrimitiveAttributes::new().unwrap(),
    };

    // 5b. Create the backward primitive
    let bwd_prim = Primitive::new::<_, PropBackward, BackwardEltwise<PropBackward>>(
        bwd_config,
        engine.clone(),
    )
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

#[test]
fn test_inner_product_nchw_to_nc_backprop() {
    use onednnl::{
        onednnl_sys::{
            DNNL_ARG_BIAS, DNNL_ARG_DIFF_BIAS, DNNL_ARG_DIFF_DST, DNNL_ARG_DIFF_SRC,
            DNNL_ARG_DIFF_WEIGHTS, DNNL_ARG_DST, DNNL_ARG_SRC, DNNL_ARG_WEIGHTS,
        },
        primitives::inner_product::{
            BackwardDataInnerProduct, BackwardDataInnerProductConfig, BackwardWeightsInnerProduct,
            BackwardWeightsInnerProductConfig, ForwardInnerProduct, ForwardInnerProductConfig,
        },
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
        src_desc: &src_md,
        weights_desc: &weights_md,
        bias_desc: &bias_md,
        dst_desc: &dst_md,
        attr: &PrimitiveAttributes::new().unwrap(),
    };

    // 3a. Create the forward primitive
    let fwd_prim = Primitive::new::<_, PropForwardTraining, ForwardInnerProduct<_>>(
        fwd_config,
        engine.clone(),
    )
    .unwrap();

    // 3b. Execute forward
    fwd_prim
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
        src_desc: &src_md,
        diff_weights_desc: &weights_md,
        diff_bias_desc: &bias_md,
        diff_dst_desc: &dst_md,
        hint_fwd_pd: &fwd_prim.desc, // from the forward primitive
        attr: &PrimitiveAttributes::new().unwrap(),
    };

    // 4a. Create backward-weights primitive
    let bwd_weights_prim = Primitive::new::<_, PropBackwardWeights, BackwardWeightsInnerProduct>(
        bwd_weights_config,
        engine.clone(),
    )
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
        diff_src_desc: &src_md,
        weights_desc: &weights_md,
        diff_dst_desc: &dst_md,
        hint_fwd_pd: &fwd_prim.desc, // from forward pass
        attr: &PrimitiveAttributes::new().unwrap(),
    };

    // 5a. Create backward-data primitive
    let bwd_data_prim = Primitive::new::<_, PropBackwardData, BackwardDataInnerProduct>(
        bwd_data_config,
        engine.clone(),
    )
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
