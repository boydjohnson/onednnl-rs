# To be done

## Primitives completed / to be done

| Primitive         | Forward | Backward | Test | Bench |
| ------------------ | :------: | :------: | :--: | :---: |
| `au_gru`          |    ✅    |    ❌    |  ❌  |  ❌   |
| `batch_norm`      |    ✅    |    ❌    |  ❌  |  ❌   |
| `binary`          |    ✅    |    ⬜    |  ✅  |  ✅   |
| `concat`          |    ❌    |    ⬜    |  ❌  |  ❌   | 
| `convolution`     |    ❌    |    ❌    |  ❌  |  ❌   |
| `deconvolution`   |    ❌    |    ❌    |  ❌  |  ❌   |
| `eltwise`         |    ✅    |    ✅    |  ✅  |  ❌   |
| `gemm`            |    ❌    |    ⬜    |  ❌  |  ❌   | 
| `group_norm`      |    ❌    |    ❌    |  ❌  |  ❌   |
| `gru`             |    ❌    |    ❌    |  ❌  |  ❌   |
| `inner_product`   |    ✅    |    ❌    |  ❌  |  ❌   |
| `layer_norm`      |    ❌    |    ❌    |  ❌  |  ❌   |
| `lbr_augru`       |    ❌    |    ❌    |  ❌  |  ❌   | 
| `lbr_gru`         |    ❌    |    ❌    |  ❌  |  ❌   | 
| `lrn`             |    ❌    |    ❌    |  ❌  |  ❌   |
| `lstm`            |    ❌    |    ❌    |  ❌  |  ❌   |
| `matmul`          |    ✅    |    ❌    |  ✅  |  ❌   |
| `pooling`         |    ❌    |    ❌    |  ❌  |  ❌   |
| `prelu`           |    ❌    |    ❌    |  ❌  |  ❌   |
| `reduction`       |    ✅    |    ⬜    |  ✅  |  ❌   | 
| `reorder`         |    ❌    |    ⬜    |  ❌  |  ❌   | 
| `resampling`      |    ❌    |    ❌    |  ❌  |  ❌   |
| `shuffle`         |    ❌    |    ❌    |  ❌  |  ❌   |
| `softmax`         |    ❌    |    ❌    |  ❌  |  ❌   |
| `sum`             |    ❌    |    ⬜    |  ❌  |  ❌   | 
| `vanilla_rnn`     |    ❌    |    ❌    |  ❌  |  ❌   |

## Known Issues

- Missing support for GPU (SYCL and OpenCL)
- Missing Primitive attrs
- Missing postops
