use {
    onednnl::{
        engine::Engine,
        memory::{
            buffer::AlignedBuffer,
            descriptor::{DataType, DataTypeQuery, DimsQuery, MemoryDescriptor, NDimsQuery},
            format_tag::{abc, abcd, x},
            Memory,
        },
        primitive::{
            config::{
                binary::{Binary, ForwardBinaryConfig},
                matmul::ForwardMatMulConfig,
            },
            ExecArg, ForwardBinary, ForwardMatMul, Primitive, PropForwardInference,
        },
        stream::Stream,
    },
    onednnl_sys::{
        dnnl_data_type_t::dnnl_f32, DNNL_ARG_BIAS, DNNL_ARG_DST, DNNL_ARG_SRC_0, DNNL_ARG_SRC_1,
        DNNL_ARG_WEIGHTS,
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
        attr: std::ptr::null_mut(),
    };

    // Create the primitive
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

    assert_eq!(output.to_vec::<f32>(), vec![5.0, 7.0, 9.0]);
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
    let mut src_buffer = AlignedBuffer::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("Failed to allocate src buffer")
        .into();
    let mut weights_buffer = AlignedBuffer::new(&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0])
        .expect("Failed to allocate weights buffer")
        .into();
    let mut output_buffer = AlignedBuffer::<f32>::zeroed(4)
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
        attr: std::ptr::null_mut(), // No special attributes
    };

    // Step 5: Create and Configure the MatMul Primitive
    // Instantiate the matmul primitive using the configuration
    let primitive =
        Primitive::new::<_, PropForwardInference, ForwardMatMul<_>>(matmul_config, engine.clone())
            .expect("Failed to create MatMul primitive");

    // Step 6: Create Memory Objects
    // Wrap the buffers into oneDNN Memory objects
    let src_memory = Memory::new_with_user_buffer(engine.clone(), src_desc, &mut src_buffer)
        .expect("Failed to create src memory");
    let weights_memory =
        Memory::new_with_user_buffer(engine.clone(), weights_desc, &mut weights_buffer)
            .expect("Failed to create weights memory");

    // Since we are disabling bias, create a Memory object without a buffer
    let bias_memory = Memory::new_without_buffer(engine.clone(), zero_bias_desc)
        .expect("Failed to create bias memory (disabled)");

    let dst_memory = Memory::new_with_user_buffer(engine.clone(), dst_desc, &mut output_buffer)
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
        output_buffer.to_vec::<f32>(),
        expected,
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
