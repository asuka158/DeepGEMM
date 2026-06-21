# DeepGEMM

DeepGEMM 是一个统一的高性能张量核心（Tensor Core）内核库，它将现代大语言模型的关键计算原语汇聚到一个统一、内聚的 CUDA 代码库中——包括 GEMM（FP8、FP4、BF16）、带通信重叠的融合 MoE（Mega MoE）、用于闪电索引器（lightning indexer）的 MQA 评分、HyperConnection（HC）等。所有内核都通过一个轻量级的即时编译（Just-In-Time，JIT）模块在运行时编译，安装期间无需进行任何 CUDA 编译。

DeepGEMM 借鉴了 [CUTLASS](https://github.com/nvidia/cutlass) 和 [CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute) 的一些理念，但避免了对它们的模板或代数体系的过度依赖。该库的设计追求简洁，仅包含数量有限的核心内核函数，使其成为学习 NVIDIA GPU 内核优化技术的一份干净且易于上手的资源。

尽管设计轻量，DeepGEMM 在各种矩阵形状上的性能都能匹敌甚至超越专家级调优的库。

## 新闻

- 2026.04.16：Mega MoE、FP8xFP4 GEMM、FP4 索引器、PDL、更快的 JIT 编译等。
    - 更多细节请参阅 [#304](https://github.com/deepseek-ai/DeepGEMM/pull/304)。
    - 关于 Mega MoE 的基准测试，请参阅 [#316](https://github.com/deepseek-ai/DeepGEMM/pull/316)。
- 2025.09.28：DeepGEMM 现已支持用于 DeepSeek v3.2 闪电索引器的评分内核（加权 ReLU MQA logits）。
    - 更多细节请参阅 [#200](https://github.com/deepseek-ai/DeepGEMM/pull/200)。
- 2025.07.20：DeepGEMM 现已同时支持 SM90/SM100，并进行了全面重构，采用低 CPU 开销的 JIT CPP 模块。
    - NVRTC 与编译后 SASS 优化均已禁用。
    - NVRTC 将在后续支持。
    - 由于 NVCC 12.9 会自动进行 FFMA 交错（interleaving），所有编译后优化都将不再被支持。
    - 更多细节请参阅 [#112](https://github.com/deepseek-ai/DeepGEMM/pull/112)。
- 2025.05.14：DeepGEMM 现已为稠密和 MoE 反向传播提供权重梯度内核！详情请见 [#95](https://github.com/deepseek-ai/DeepGEMM/pull/95)。
- 2025.05.07：DeepGEMM 现已支持 NVRTC，编译速度最高提升 10 倍！详情请见 [#94](https://github.com/deepseek-ai/DeepGEMM/pull/94)。请使用 `DG_JIT_USE_NVRTC=1` 来启用它（在某些情况下可能会有性能损失）。
- 2025.04.18：DeepGEMM 现已在 H800 上达到高达 **1550 TFLOPS**！详情请见 [#74](https://github.com/deepseek-ai/DeepGEMM/pull/74)、[#78](https://github.com/deepseek-ai/DeepGEMM/pull/78)、[#81](https://github.com/deepseek-ai/DeepGEMM/pull/81)、[#86](https://github.com/deepseek-ai/DeepGEMM/pull/86) 以及 [340d988](https://github.com/deepseek-ai/DeepGEMM/commit/340d9880f4a418d943d34260d20a79f41f4c0526)。

## 快速开始

### 环境要求

- NVIDIA SM90 或 SM100 架构 GPU
- Python 3.8 或更高版本
- 支持 C++20 的编译器
- CUDA Toolkit：
    - SM90 需要 CUDA 12.3 或更高版本
        - **我们强烈推荐使用 12.9 或更高版本以获得最佳性能**
    - SM100 需要 CUDA 12.9 或更高版本
- PyTorch 2.1 或更高版本
- CUTLASS 4.0 或更高版本（可通过 Git 子模块克隆）
- `{fmt}` 库（可通过 Git 子模块克隆）

```bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib:/usr/local/cuda/lib64
export PYTHONPATH=/root/workspace/gb_gemm_bench/DeepGEMM/build/lib.linux-aarch64-cpython-312${PYTHONPATH:+:$PYTHONPATH}
```

### 开发

```bash
# 必须克隆子模块
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM

# 链接一些必要的头文件并构建 CPP JIT 模块
cat develop.sh
./develop.sh
```

### 安装

```bash
cat install.sh
./install.sh
```

然后，在你的 Python 项目中导入 `deep_gemm`，尽情使用吧！

## 接口

#### 注意事项

本库为 NVIDIA GPU 提供了优化的 GEMM 内核，命名约定为：`D = C + A @ B`。输入形状布局为 NT（A 不转置，B 转置）。SM90 实现仅支持 NT 内存布局（行主序、列主序），而 SM100 实现支持所有内存布局（NT、TN、NN、TT）。例如，`fp8_gemm_nt` 会执行 `D = C + A @ B.T`。

对于这两种架构，LHS 缩放因子都要求采用 TMA 对齐且转置的布局。SM90 和 SM100 的缩放因子数据格式有所不同：

- SM90 要求缩放因子采用 FP32 格式。
- SM100 要求缩放因子采用打包的 [UE8M0](https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats) 格式，即将 4 个 UE8M0 打包进一个 `torch.int`。

请注意，诸如输入转置或 FP8 类型转换等操作必须由用户自行处理，请独立地实现它们或将它们融合进前置内核中。虽然本库提供了一些简单的 PyTorch 实用函数，但这些函数可能会带来较慢的性能，因为我们的主要关注点在于优化 GEMM 内核本身。

#### 普通稠密 GEMM（非分组）

要执行基础的非分组 FP8 GEMM，请调用 `fp8_gemm_{nt, nn, tn, tt}` 函数。更多细节请参阅该函数的文档。

#### 分组 GEMM（连续布局）

与 CUTLASS 中传统的分组 GEMM 不同，DeepGEMM 仅对 M 轴进行分组，而 N 和 K 必须保持固定。这一设计专为 MoE 模型中各专家共享相同形状的场景而定制。在训练前向传播或推理预填充（prefilling）阶段，每个专家可能处理数量不等的 token，我们将这些 token 拼接成一个张量，称为“连续”（contiguous）布局。请注意，每个专家段必须对齐到 GEMM 的 M 块大小（`get_mk_alignment_for_contiguous_layout()`）。更多信息请参阅 `m_grouped_fp8_gemm_{nt, nn}_contiguous` 函数的文档。

我们还为 MoE 权重反向传播提供了一个按 K 轴分组的 API（其中 M 和 N 必须保持固定），更多信息请参阅 `k_grouped_fp8_gemm_tn_contiguous`。

#### 分组 GEMM（掩码布局）

在推理解码阶段，当启用 CUDA 图（CUDA graph）且 CPU 无法感知每个专家接收的 token 数量时，我们支持掩码（masked）分组 GEMM。通过提供一个掩码张量，内核只会计算有效的部分。

请使用 `m_grouped_fp8_gemm_nt_masked` 来实现这一目的，并参阅相关文档。一个使用示例是将来自 [DeepEP](https://github.com/deepseek-ai/DeepEP) 的低延迟内核的输出作为输入。

#### 用于索引器的 V3.2 MQA 内核

该内核家族有两个版本：非分页（non-paged，用于预填充）和分页（paged，用于解码）。
以非分页版本 `fp8_mqa_logits` 为例，它有 6 个输入：

- `q`，形状为 `[seq_len, num_heads, head_dim]` 的 E4M3 张量
- `kv`，E4M3 张量（形状为 `[seq_len_kv, head_dim]`），带 float 类型的缩放因子 SF（形状为 `[seq_len_kv]`）
- `weights`，形状为 `[seq_len, num_heads]` 的 float 张量
- `cu_seq_len_k_start` 和 `cu_seq_len_k_end`，形状为 `[seq_len]` 的 int 张量
- `clean_logits`，是否将未填充的 logits 清理为 `-inf`

输出张量的形状为 `[seq_len, seq_len_kv]`，表示 token 到 token 的 logits。
对于 `q` 中的每个 token `i`，它会遍历 `[cu_seq_len_k_start[i], cu_seq_len_k_end[i])` 范围内的所有 token `j`，
并按如下方式计算 logit `out[i, j]`：

```python
kv_j = kv[0][j, :] * kv[1][j].unsqueeze(1)  # [head_dim]
out_ij = q[i, :, :] @ kv_j  # [num_heads]
out_ij = out_ij.relu() * weights[i, :]  # [num_heads]
out_ij = out_ij.sum()  # 标量
```

更多细节以及分页版本 `fp8_paged_mqa_logits`，请参阅 `tests/test_attention.py`。

#### Mega MoE

Mega MoE 将 EP dispatch、linear 1（FP8xFP4）、SwiGLU、linear 2（FP8xFP4）以及 EP combine 融合并重叠到单个超级内核（mega-kernel）中，从而将 NVLink 通信与张量核心计算重叠起来。它需要使用对称内存（symmetric memory）以多进程方式启动。用法：

```python
# 分配对称内存缓冲区
# 注意：需要 PyTorch >= 2.9
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden
)

# 将权重（带 UE8M0 SF 的 FP4）转换为所需布局
transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)

# 在每次调用前将输入拷贝进缓冲区
# 你可以将这些操作融合进前置内核中
buffer.x[:num_tokens].copy_(x_fp8)
buffer.x_sf[:num_tokens].copy_(x_sf)
buffer.topk_idx[:num_tokens].copy_(topk_idx)
buffer.topk_weights[:num_tokens].copy_(topk_weights)

# 运行融合的 mega MoE 内核
y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_fp4_mega_moe(y, transformed_l1, transformed_l2, buffer)
```

关于包含多进程设置与基准测试的完整示例，请参阅 `tests/test_mega_moe.py`。

#### 实用工具

除上述内核外，本库还提供了一些实用函数：

- `deep_gemm.set_num_sms` / `get_num_sms`：设置/获取要使用的最大 SM 数量
- `deep_gemm.set_tc_util` / `get_tc_util`：设置/获取一个近似的张量核心利用率
- `deep_gemm.set_pdl` / `get_pdl`：启用/禁用程序化依赖启动（Programmatic Dependent Launch，PDL）
- `deep_gemm.set_mk_alignment_for_contiguous_layout` / `get_mk_alignment_for_contiguous_layout`：设置/获取连续布局的组级 M/K 对齐
- `deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout`：获取理论上的最小 M/K 对齐
- `deep_gemm.set_ignore_compile_dims`：配置在 JIT 编译期间需要忽略的维度
- `deep_gemm.set_block_size_multiple_of`：将块大小约束为给定值的倍数
- `deep_gemm.transform_sf_into_required_layout`：将缩放因子转换为所需的布局
- `deep_gemm.get_tma_aligned_size`：获取所需的 TMA 对齐大小
- `deep_gemm.get_mn_major_tma_aligned_tensor`：获取一个 MN 主序的 TMA 对齐张量
- `deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor`：获取一个 MN 主序的 TMA 对齐张量（将 FP32 打包为 UE8M0）
- `deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`：按 K 轴分组的 GEMM 打包内核

本库还提供了一些可能有用的环境变量：

- 通用
    - `DG_JIT_DEBUG`：`0` 或 `1`，打印 JIT 调试信息，默认为 `0`
    - `DG_PRINT_CONFIGS`：`0` 或 `1`，为每种形状打印所选配置，默认为 `0`
- JIT 缓存
    - `DG_JIT_CACHE_DIR`：字符串，编译后内核的缓存目录，默认为 `$HOME/.deep_gemm`
- 编译器选择
    - `DG_JIT_USE_NVRTC`：`0` 或 `1`，使用 NVRTC 而非 NVCC（编译更快，但在某些情况下性能可能较低），默认为 `0`
    - `DG_JIT_NVCC_COMPILER`：字符串，NVCC 编译器路径；默认为 `torch.utils.cpp_extension.CUDA_HOME`
    - `DG_JIT_CPP_STANDARD`：整数，C++ 标准版本，默认为 `20`
- 编译器输出
    - `DG_JIT_PRINT_COMPILER_COMMAND`：`0` 或 `1`，打印编译命令，默认为 `0`
    - `DG_JIT_PTXAS_VERBOSE`：`0` 或 `1`，显示详细的 PTXAS 输出，默认为 `0`
    - `DG_JIT_PTXAS_CHECK`：`0` 或 `1`，断言编译后的内核中没有本地内存使用，默认为 `0`
    - `DG_JIT_PRINT_LOAD_TIME`：`0` 或 `1`，打印内核加载时间，默认为 `0`
- 调试与性能分析
    - `DG_JIT_WITH_LINEINFO`：`0` 或 `1`，为性能分析工具嵌入源代码行信息，默认为 `0`
    - `DG_JIT_DUMP_ASM`：`0` 或 `1`，同时转储 PTX 和 SASS，默认为 `0`
    - `DG_JIT_DUMP_PTX`：`0` 或 `1`，转储 PTX 输出，默认为 `0`
    - `DG_JIT_DUMP_SASS`：`0` 或 `1`，转储 SASS 输出，默认为 `0`
    - `DG_COMM_KERNEL_DEBUG`：`0` 或 `1`，在每次 Mega MoE 调用前将对称缓冲区清零以便调试，默认为 `0`
    - `DG_USE_NVIDIA_TOOLS`：`0` 或 `1`，在外部 NVIDIA 工具下运行时跳过内部性能分析，默认为 `0`
- 构建选项
    - `DG_SKIP_CUDA_BUILD`：`0` 或 `1`，安装期间跳过 CUDA 扩展构建，默认为 `0`
    - `DG_FORCE_BUILD`：`0` 或 `1`，强制本地构建而非下载预构建的 wheel 包，默认为 `0`
    - `DG_JIT_USE_RUNTIME_API`：`0` 或 `1`，使用 CUDA Runtime API 加载内核（需要 CUDA runtime >= 12.8），默认为 `0`

更多示例和细节，请参阅[测试代码](tests/test_core.py)或查阅相应的 Python 文档。

## 致谢

DeepGEMM 受 [CUTLASS](https://github.com/nvidia/cutlass) 项目启发。向其开发者致以感谢与敬意！

## 许可证

本代码仓库基于 [MIT 许可证](LICENSE)发布。

## 引用

```bibtex
@misc{deepgemm2025,
      title={DeepGEMM: clean and efficient BLAS kernel library on GPU}, 
      author={Chenggang Zhao and Zhean Xu and Liang Zhao and Jiashi Li and Chenhao Xu and Anyi Xu and Shengyu Liu and Kexing Zhou and Kuai Yu},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/DeepGEMM}},
}
```
