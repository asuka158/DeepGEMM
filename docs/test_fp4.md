# DeepGEMM 4-bit (MXFP4) Dense GEMM 全链路精读 —— 从 `test_fp4.py` 到 tcgen05 内核

> 本文沿着 `tests/test_fp4.py` 中 **`test_gemm()`**（dense / 非分组 GEMM）的完整调用路径，
> 从 Python 测试脚本一路讲到 Blackwell（SM100）的 CUDA 内核与 PTX 指令级硬件细节。
>
> 阅读约定：
> - 每段代码统一采用「**先贴源码 → 逐行中文注释 → 下方展开讲解**」的三段式。
> - 遇到对 Python 新手不直观的语法，用 `💡 Python 语法` 小框单独讲。
> - 遇到 GPU 硬件 / CUDA 实现细节，用 `⚙️ 硬件细节` 小框强调。
> - **贯穿全文的示例规模**：`m = 128, n = 2112, k = 7168`（取自 `nk_list` 的第一项）。
>   后文凡是出现具体 shape 数字，都是用这组规模算出来的，方便你对照。
> - 只覆盖 `test_gemm()`；`test_m_grouped_gemm_contiguous` / `_masked`（分组 GEMM）不在本文范围。

---

## 目录

- [第 0 章 导读与全景](#第-0-章-导读与全景)
- [第 1 章 测试入口 `test_gemm()`](#第-1-章-测试入口-test_gemm)
- [第 2 章 数据生成与量化 `generate_normal()`](#第-2-章-数据生成与量化-generate_normal)
- [第 3 章 Python→C++ 边界与 API 分发](#第-3-章-pythonc-边界与-api-分发)
- [第 4 章 Host 端内核装配 `sm100_fp4_gemm_1d1d`](#第-4-章-host-端内核装配-sm100_fp4_gemm_1d1d)
- [第 5 章 CUDA 内核总体结构](#第-5-章-cuda-内核总体结构)
- [第 6 章 内核主循环——逐 warp 深入](#第-6-章-内核主循环逐-warp-深入)
- [第 7 章 回到测试——校验与基准](#第-7-章-回到测试校验与基准)
- [附录](#附录)

---

## 第 0 章 导读与全景

### 0.1 这份 kernel 在算什么

它算的是一个**矩阵乘法** `D = A @ Bᵀ`（外加可选的累加项 `C`），但 A、B 的元素不是普通的
`float`，而是 **4-bit 浮点数（FP4，E2M1 格式）**，每个元素只占 4 个 bit。为了在如此低的精度下还能
保住数值范围，A、B 都采用 **MXFP4（Microscaling FP4）** 方案：每 **32 个**连续的 FP4 元素共享
一个 **UE8M0**（8-bit 纯指数）缩放因子。最终累加和输出用 `float32`。

为什么要这么做？因为大模型推理/训练的瓶颈往往是**显存带宽**和**张量核吞吐**。把权重和激活从
16-bit 压到 4-bit，显存占用和带宽需求直接降到 1/4，而 Blackwell 张量核原生支持 FP4 的 MMA 指令
（`tcgen05.mma...kind::mxf4`），所以算力也能跑满。代价是精度损失，靠 per-block 的缩放因子把误差控制住。

### 0.2 名词速览（先混个眼熟，后面各章详讲）

| 名词 | 全称 / 含义 | 在本文哪里详讲 |
|------|------------|--------------|
| **FP4 / E2M1** | 4-bit 浮点：1 符号位 + 2 指数位 + 1 尾数位，能表示 `{0,0.5,1,1.5,2,3,4,6}` 这 8 个幅值 | 第 2 章 |
| **MXFP4** | Microscaling FP4：一**块** 32 个 FP4 共享一个缩放因子（OCP 微缩放标准） | 第 2 章 |
| **UE8M0** | Unsigned E8M0：8-bit 纯指数、0 尾数、无符号的缩放因子，本质是一个 2 的幂 | 第 2 章 |
| **VS=32** | Vector Size = 32：每 32 个元素一个缩放因子的"块"大小（= MXFP4 的块） | 第 2 章 |
| **SF / SFA / SFB** | Scale Factor（缩放因子），A 的叫 SFA、B 的叫 SFB | 第 2、3、6 章 |
| **TMA** | Tensor Memory Accelerator：Hopper/Blackwell 的异步批量拷贝引擎（gmem↔smem） | 第 6.2 章 |
| **TMEM** | Tensor Memory：Blackwell 新增的、专给张量核当累加器用的片上内存 | 第 5、6 章 |
| **tcgen05** | 第 5 代张量核（Blackwell）的指令族，MMA 异步写 TMEM | 第 6.4 章 |
| **UMMA** | "U" + MMA，CUTLASS 对 tcgen05 MMA 的封装；smem 描述符喂操作数 | 第 6.4 章 |
| **UTCCP** | 把 SF 从 smem 拷进 TMEM（MMA 要求 SF 在 TMEM 里）的拷贝指令 | 第 6.3、6.4 章 |
| **mbarrier** | shared-memory 异步屏障，带"相位（phase）"，用于生产者-消费者同步 | 第 5、6 章 |
| **swizzle** | 把数据在 smem 里按位异或重排，消除 bank 冲突、配合 TMA/MMA 的地址布局 | 第 4、6 章 |
| **CTA / cluster** | CTA = 线程块；cluster = Blackwell 上一组能共享 smem、协同 MMA 的 CTA | 第 5 章 |

### 0.3 完整调用链路全景图

```
test_gemm()                                          tests/test_fp4.py
  │
  ├─ FP4_FP4.get_recipes()                           tests/generators.py  → recipe_a/b = (1, 32)
  │
  ├─ generate_normal(...)                            tests/generators.py  ← 造数据 + 量化
  │    └─ cast_fp8_fp4_with_major
  │         └─ per_token_cast_to_fp4                 deep_gemm/utils/math.py
  │              ├─ _quantize_to_fp4_e2m1            （E2M1 编码）
  │              ├─ ceil_to_ue8m0 / pack_ue8m0_to_int（UE8M0 缩放）
  │
  ├─ deep_gemm.fp8_fp4_gemm_nt(...)  ===  _C.fp8_fp4_gemm_nt （pybind 绑定）  deep_gemm/__init__.py
  │    │
  │    │  C++ fp8_fp4_gemm_nt                        csrc/apis/gemm.hpp
  │    │    ├─ check_ab_fp8_fp4 / early_return        （形状/类型检查、平凡情形短路）
  │    │    ├─ transform_sf_pair_into_required_layout csrc/apis/layout.hpp  ← SF 布局变换
  │    │    │    └─ get_mn_major_tma_aligned_packed_ue8m0_tensor  (float32 → int32 packed)
  │    │    └─ sm100_fp4_gemm_1d1d                    csrc/jit_kernels/impls/sm100_fp4_gemm_1d1d.hpp
  │    │         ├─ make_fp4_desc / pick_fp4_layout   （问题描述 + tile 启发式）
  │    │         ├─ SM100ArchSpec::get_*_config       （swizzle/stage/线程数）
  │    │         ├─ recompute_stages_for_fp4          （232KB smem 约束）
  │    │         ├─ make_tma_{a,b,cd,sf}_desc         （A/B/D/SFA/SFB 共 5 个 TMA 描述符）
  │    │         └─ SM100FP4Gemm1D1DRuntime::generate → compiler->build(NVCC) → launch
  │    │
  │    └─ CUDA kernel: sm100_fp4_gemm_1d1d_impl<...>  deep_gemm/include/deep_gemm/impls/sm100_fp4_gemm_1d1d.cuh
  │         ├─ warp 0  : TMA 加载生产者（A/B/SFA/SFB → smem）
  │         ├─ warp 1  : MMA 消费者 + UTCCP（SF smem→TMEM）+ tcgen05 MXF4 MMA
  │         ├─ warp 2  : SF 在 smem 里做 warp 转置（喂 UTCCP）
  │         └─ warp 3+ : epilogue（TMEM → smem → TMA 写回 D）
  │
  ├─ calc_diff(d, ref_d)                             deep_gemm/testing/numeric.py  ← 校验精度
  └─ bench_kineto(...) / count_bytes(...)            deep_gemm/testing/bench.py    ← 测性能
```

把这张图记在脑子里，后面每一章都是在放大其中的一跳。

### 0.4 贯穿示例的张量速查表（`m=128, n=2112, k=7168`）

| 张量 | 含义 | shape | dtype | 备注 |
|------|------|-------|-------|------|
| `a` 原始 | A 的 bf16 原始数据 | `(128, 7168)` | `bfloat16` | 量化前 |
| `b` 原始 | B 的 bf16 原始数据 | `(2112, 7168)` | `bfloat16` | 量化前 |
| `a[0]` packed | A 的 FP4（打包后） | `(128, 3584)` | `int8` | `7168/2`，两个 FP4 挤进 1 字节 |
| `a[1]` = SFA | A 的缩放因子 | `(128, 224)` | **存储 `float32`**（逻辑 UE8M0） | `7168/32 = 224` 个块 |
| `b[0]` packed | B 的 FP4（打包后） | `(2112, 3584)` | `int8` | |
| `b[1]` = SFB | B 的缩放因子 | `(2112, 224)` | **存储 `float32`**（逻辑 UE8M0） | |
| `c` | 累加输入 | `None` | — | 本测试 `accumulate=False` |
| `d` | 输出 | `(128, 2112)` | `float32` | 内核写入目标 |
| `ref_d` | 参考答案 | `(128, 2112)` | `float32` | 用 bf16 全精度算的 ground truth |

> **关于 SFA/SFB 的 dtype——务必分清"逻辑类型"与"存储类型"：**
> - **逻辑类型**始终是 **UE8M0**（E8M0）：一个 8-bit 纯指数、值恒为 2 的幂的缩放因子（语义上只有 8 bit）。
> - **此刻（Python 返回时）的存储类型是 `torch.float32`**：每个 UE8M0 被塞进一个**完整的 4 字节 float32
>   容器**（指数字段放 e8m0、23 位尾数全清零），见 [math.py:13-16](../deep_gemm/utils/math.py#L13-L16)
>   的 `ceil_to_ue8m0` 返回 `.view(torch.float)`。所以 `element_size()` 此刻是 **4**。
> - 因此 SFA 这一刻占用 **`128 × 224 × 4 = 114688 字节 ≈ 112 KiB`**（是 ×4，不是 ×1）。
> - **进内核前**，C++ 的 `get_mn_major_tma_aligned_packed_ue8m0_tensor`（第 3.3 节）会把它**压成每个
>   1 字节的 e8m0、4 个打包进 1 个 int32**，dtype 变 `int32`，体积降到 `128×224×1 = 28672 字节 ≈ 28 KiB`
>   （等价 `128×56 int32×4`）。这也是分发条件要求 `sfa.scalar_type()==kInt` 的由来。
>
> 进入内核后，packed FP4 的 int8 同理会被**重解释成 int32**（8 个 FP4 = 32 bit = 1 个 int32，第 4 章）；
> SFA/SFB 则如上变成 **packed int32 的 UE8M0** 并改成 MN-major、TMA 对齐布局（第 3 章）。

---

## 第 1 章 测试入口 `test_gemm()`

### 1.1 模块级常量 `FP4_FP4`

先看文件顶部（`tests/test_fp4.py:13-16`）：

```python
# FP4xFP4 (MXFP4): both operands packed FP4 (E2M1) with VS=32 UE8M0 scales.
# Dispatches to the SM100_MMA_MXF4_SS path; the API rejects bf16 D so we
# always allocate fp32 D below.
FP4_FP4 = QuantConfig((32, 32, True, True))   # 构造一个量化配置对象，传入一个 4 元组
```

逐行讲解：
- 这是一个**模块级（全局）常量**，整个文件共用。`QuantConfig` 来自 `generators.py`（第 2 章细讲），
  它的构造函数接收一个 4 元组 `(gran_k_a, gran_k_b, is_fp4_a, is_fp4_b)`：
  - `gran_k_a = 32`：A 沿 K 方向每 **32** 个元素一个缩放因子（即 VS=32）。
  - `gran_k_b = 32`：B 同理。
  - `is_fp4_a = True`：A 用 FP4 量化。
  - `is_fp4_b = True`：B 用 FP4 量化。
- 所以 `FP4_FP4` 表示"A、B 都是 MXFP4、块大小 32"的配置。注释里点明它会走
  `SM100_MMA_MXF4_SS` 这条内核路径，且**输出 D 必须是 fp32**（API 不接受 bf16 的 D）。

> 💡 **Python 语法：`QuantConfig((32, 32, True, True))` 为什么有两层括号？**
> 外层括号是"函数调用"，内层括号是一个 **tuple（元组）字面量**。也就是说我们只传了**一个**参数——
> 一个 4 元素的元组。如果写成 `QuantConfig(32, 32, True, True)`（一层括号）就是传 4 个参数，
> 那会和 `__init__(self, value)` 只收一个参数的签名不匹配而报错。后面 `__init__` 里再用
> `self.gran_k_a, ... = value` 把这个元组**拆包**成 4 个字段。

### 1.2 `test_gemm()` 主体逐行

源码（`tests/test_fp4.py:19-48`）：

```python
def test_gemm() -> None:                                          # 无返回值的测试函数
    print('Testing GEMM:')                                        # 打印一行标题
    nk_list = [(2112, 7168), (576, 7168), (24576, 1536), (32768, 512),   # 要遍历的 (n, k) 组合
               (7168, 16384), (4096, 7168), (7168, 2048)]
    m_list = [128, 4096]                                          # 要遍历的 m 取值
    kernel_type = KernelType.Kernel1D1D                           # 选 1D1D 内核（见 2.5 节）
    out_dtype = torch.float                                       # 输出 D 用 fp32
    recipe, recipe_a, recipe_b = FP4_FP4.get_recipes()            # 从配置导出 3 个 recipe（见 1.3）

    for m in m_list:                                              # 外层遍历 m
        for n, k in nk_list:                                      # 内层遍历 (n, k)，顺手拆包
            a, b, c, d, ref_d = generate_normal(                  # 造数据 + 量化（第 2 章）
                m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,  # A、B 都是 K-major
                accumulate=False, out_dtype=out_dtype, kernel_type=kernel_type,
                use_ue8m0=True, quant_config=FP4_FP4)             # 用 UE8M0 缩放、FP4xFP4 配置
            deep_gemm.fp8_fp4_gemm_nt(                            # ★ 真正调内核（第 3~6 章）
                a, b, d, c=c, disable_ue8m0_cast=False,
                recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
            diff = calc_diff(d, ref_d)                            # 算内核输出 vs 参考答案的误差
            assert diff < FP4_FP4.max_diff(), \                   # 误差必须小于阈值（0.02），否则报错
                f'{m=}, {n=}, {k=}, {diff:.5f}'

            t = bench_kineto(                                     # 用 kineto profiler 测耗时（第 7 章）
                lambda: deep_gemm.fp8_fp4_gemm_nt(               # 把"调一次内核"包成一个无参函数
                    a, b, d, c=c, disable_ue8m0_cast=False,
                    recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b),
                'sm100_fp4_gemm', suppress_kineto_output=True)    # 只统计名字含 'sm100_fp4_gemm' 的内核
            print(f' > Perf (m={m:6}, n={n:6}, k={k:6}, 1D1D, layout=NT, FP32): '  # 打印性能
                  f'{t * 1e6:6.1f} us | {2 * m * n * k / t / 1e12:4.0f} TFLOPS | '  # us 与 TFLOPS
                  f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')                    # 带宽 GB/s
```

讲解（按重要性而非行号）：

**① 双重 for 循环 = 7×2 = 14 个测试用例。** 外层 2 个 `m`，内层 7 个 `(n, k)`，每个组合都完整跑一遍
"造数据 → 调内核 → 校验 → 测速"。这就是为什么一个 `test_gemm()` 会打印很多行性能数据。

**② `MajorTypeAB.KMajor`** 表示 A、B 在内存里都是**沿 K 维连续**（行优先存 `[MN, K]`）。这点对内核
很关键：FP4 这条路径**只支持 K-major**（第 3 章会看到 C++ 里有 `DG_HOST_ASSERT(major_a == K ...)`）。
名字里的 `nt` 就是 "Normal-Transpose"：`D = A @ Bᵀ`，A 不转置（N）、B 转置（T）。

**③ `accumulate=False, c=c`（而 `c` 其实是 `None`）：** 本测试不做 `D += C` 的累加，所以
`generate_normal` 把 `c` 设成 `None`。内核就是纯粹算 `D = A @ Bᵀ`。

**④ `★` 那一行就是本文的主角**——`deep_gemm.fp8_fp4_gemm_nt(...)`。它会一路下沉到 CUDA 内核。
第 3 章开始就专门拆它。

> 💡 **Python 语法：`for n, k in nk_list`（元组拆包遍历）**
> `nk_list` 是一个"元组的列表"，每个元素形如 `(2112, 7168)`。`for n, k in nk_list` 在每次迭代时，
> 自动把当前元组的两个元素分别绑定到 `n` 和 `k`。等价于 `for item in nk_list: n, k = item`。
> 这是 Python 很常用的"解构"写法。

> 💡 **Python 语法：f-string 里的 `{m=}` 和 `{t * 1e6:6.1f}`**
> - `f'...'` 是 **f-string（格式化字符串）**，`{}` 里可以直接写表达式。
> - `{m=}`（Python 3.8+）是**自带变量名**的简写：若 `m=128`，它会输出 `m=128`。常用于调试。
> - `{t * 1e6:6.1f}`：冒号后是**格式说明**。`6.1f` = 宽度至少 6 个字符、保留 1 位小数的浮点。
>   所以 `t`（单位秒）×1e6 变成微秒后，按 `123.4` 这种样子右对齐打印。
> - `{m:6}` 则是整数右对齐占 6 位。这些只影响**显示对齐**，不改变数值。

> 💡 **Python 语法：`assert 条件, 消息` 与行尾的反斜杠 `\`**
> `assert diff < FP4_FP4.max_diff(), f'...'` 的意思是：如果 `diff < 阈值` 为假，就抛出
> `AssertionError` 并把后面的 f-string 作为错误信息。逗号后面的部分**只在断言失败时**才会用到。
> 行尾的 `\` 是**续行符**：告诉 Python "这一行还没写完，下一行接着算"，于是 `assert ... \` 和下一行
> 的 `f'...'` 其实是同一条语句。

> 💡 **Python 语法：`lambda: deep_gemm.fp8_fp4_gemm_nt(...)`（无参匿名函数 / 闭包）**
> `lambda:` 后面没有参数，冒号后是函数体。它把"调用一次内核"打包成一个**可反复调用的无参函数对象**
> 传给 `bench_kineto`，让 profiler 能循环跑很多次来测平均耗时。它能直接用到外面的 `a, b, d, c, ...`，
> 是因为 lambda 形成了**闭包**——捕获了定义时所在作用域里的这些变量。注意此处所有迭代都复用同一个
> `d` 缓冲区，每次调用都会把上一次的结果覆盖掉，这对测速没影响。

### 1.3 `get_recipes()`：从配置导出"配方"

源码（`tests/generators.py:55-62`）：

```python
def get_recipes(self, is_wgrad: bool = False) -> Tuple[Tuple, Tuple, Tuple]:
    recipe, recipe_a, recipe_b = None, None, None                 # 先全部置空
    if self.is_legacy():                                          # 老式 (128,128,False,False) 配置？
        recipe = (1, 1, 128) if is_wgrad else None
    else:                                                         # 我们的 FP4 配置走这里
        recipe_a = (1, self.gran_k_a)                            # → (1, 32)
        recipe_b = (1, self.gran_k_b) if self.is_fp4_b or is_wgrad else (self.gran_k_b, self.gran_k_b)
                                                                  # is_fp4_b=True → (1, 32)
    return recipe, recipe_a, recipe_b                            # 返回 (None, (1,32), (1,32))
```

讲解：
- `FP4_FP4` 不是 legacy（legacy 专指 `(128,128,False,False)`），所以走 `else` 分支。
- `recipe_a = (1, gran_k_a) = (1, 32)`。这个二元组的含义是 **(gran_mn, gran_k)**：
  - `gran_mn = 1`：在 M（对 A）或 N（对 B）方向，每 **1** 行一个缩放因子（即 per-token / per-row）。
  - `gran_k = 32`：在 K 方向每 **32** 个元素一个缩放因子（即 VS=32）。
- 因为 `is_fp4_b=True`，`recipe_b` 也取 `(1, 32)`。
- 最终 `test_gemm()` 拿到 `recipe=None, recipe_a=(1,32), recipe_b=(1,32)`，原样转交给
  `deep_gemm.fp8_fp4_gemm_nt`。C++ 侧会用这两个 recipe 决定如何变换 SF 布局（第 3 章）。

> 💡 **Python 语法：`A if 条件 else B`（三元表达式）**
> `recipe = (1, 1, 128) if is_wgrad else None` 读作："如果 `is_wgrad` 为真，`recipe` 取
> `(1,1,128)`，否则取 `None`"。它是一个**表达式**（有返回值），不是语句，所以能直接赋值。

### 1.4 `max_diff()`：精度阈值

源码（`tests/generators.py:64-69`）：

```python
def max_diff(self) -> float:
    if self.is_fp4_a and self.is_fp4_b:    # A、B 都是 FP4 → 误差容忍最大
        return 0.02
    if self.is_fp4_a or self.is_fp4_b:     # 只有一个是 FP4
        return 0.01
    return 0.001                           # 纯 FP8
```

讲解：FP4 只有 4 bit，精度天然差，所以 FP4×FP4 用一个相对宽松的阈值 `0.02`。`calc_diff` 返回的是
"1 − 余弦相似度"式的标量（第 7 章），越接近 0 越准。`test_gemm()` 里 `assert diff < 0.02` 就是
在保证内核算出来的 `d` 和全精度参考 `ref_d` 足够接近。

---

## 第 2 章 数据生成与量化 `generate_normal()`

这一章是理解整条路径的**数值地基**：A、B 究竟被编码成了什么样的字节，缩放因子怎么来的。
内核所有的"怪异布局"都是为了高效消费这一章产出的数据。

### 2.1 `generate_normal()` 主体

源码（`tests/generators.py:265-288`）：

```python
def generate_normal(m: int, n: int, k: int,
                    major_a: MajorTypeAB, major_b: MajorTypeAB,
                    accumulate: bool, out_dtype: torch.dtype,
                    kernel_type: KernelType,
                    use_ue8m0: bool = False, use_bf16: bool = False,
                    quant_config: Optional[QuantConfig] = None):
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)        # A: 随机 bf16, (m, k)
    b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)        # B: 随机 bf16, (n, k)
    d = torch.randn((m, n), device='cuda', dtype=out_dtype) * 32 if accumulate else \
        torch.empty((m, n), device='cuda', dtype=out_dtype)             # D: 累加则给随机值否则不初始化
    c = d if accumulate else None                                       # 累加时 C 就是 D；否则 None
    ref_d = (a.float() @ b.float().t() + (c if accumulate else 0)).to(out_dtype)  # 全精度参考答案
    if use_bf16:                                                        # 纯 bf16 路径（本测试不走）
        a = a if major_a.is_k_major() else a.T.contiguous().T
        b = b if major_b.is_k_major() else b.T.contiguous().T
        return a, b, c, d, ref_d
    quant_config = QuantConfig() if quant_config is None else quant_config   # 没传就用默认
    a = cast_fp8_fp4_with_major(a, major_a, quant_config.gran_k_a, quant_config.is_fp4_a, use_ue8m0)  # 量化 A
    b = cast_fp8_fp4_with_major(b, major_b, quant_config.gran_k_b, quant_config.is_fp4_b, use_ue8m0,
                                use_block_cast_for_fp8=not (kernel_type.is_1d1d() and accumulate))    # 量化 B
    return a, b, c, d, ref_d
```

**作用**：造一对随机 bf16 矩阵 → 先用全精度算出参考答案 `ref_d` → 再把 A、B 量化成 (FP4, SF) 对返回。

**入参（本测试的实参）**：`m=128, n=2112, k=7168`，`major_a=major_b=KMajor`，`accumulate=False`，
`out_dtype=torch.float`，`kernel_type=Kernel1D1D`，`use_ue8m0=True`，`quant_config=FP4_FP4`。

**返回**（5 元组，对应第 0.4 节速查表）：
- `a` = `(packed_fp4_a: int8 (128,3584), sfa: float32 (128,224))`
- `b` = `(packed_fp4_b: int8 (2112,3584), sfb: float32 (2112,224))`
- `c` = `None`
- `d` = `float32 (128,2112)`（未初始化，内核会覆盖）
- `ref_d` = `float32 (128,2112)`

逐点讲解：

**① 关键顺序——先算 `ref_d` 再量化。** `ref_d = a.float() @ b.float().t()` 是在 A、B **还是
bf16** 时用 fp32 全精度算出来的"标准答案"。之后才把 `a`、`b` 覆盖成量化结果。所以 `ref_d` 反映的是
"理想中无损的乘法"，而内核吃的是有损的 FP4——两者的差距（`calc_diff`）就是量化误差。

**② `a.float() @ b.float().t()`**：`.float()` 把 bf16 升到 fp32；`@` 是矩阵乘；`.t()` 把 `b` 从
`(n,k)` 转成 `(k,n)`，于是 `(m,k) @ (k,n) = (m,n)`。这正好对应 `D = A @ Bᵀ`。

**③ `a` 这个名字被"偷换了类型"。** 进入函数时 `a` 是一个 bf16 张量；执行到
`a = cast_fp8_fp4_with_major(...)` 后，`a` 变成了一个**二元组 `(packed_fp4, sf)`**。Python 变量没有
固定类型，同名变量可以先后指向不同类型的对象。后面 `deep_gemm.fp8_fp4_gemm_nt(a, b, d, ...)` 收到的
`a`、`b` 就都是这种 `(张量, 张量)` 对——这也解释了为什么 C++ 侧签名是 `std::pair<Tensor, Tensor>`。

> 💡 **Python 语法：`X if cond else Y` 跨多行 + 行尾 `\`**
> `d = torch.randn(...) * 32 if accumulate else \` 换行接 `torch.empty(...)`。这是一个三元表达式被
> `\` 续行拆成两行。读法：`accumulate` 为真时 `d` 是随机值×32，否则是未初始化的空张量。本测试
> `accumulate=False`，所以 `d` 是 `torch.empty(...)`、`c=None`、`ref_d` 不含累加项。

> 💡 **Python 语法：`Optional[QuantConfig] = None` 与 `X if X is not None else Y` 模式**
> 形参默认值 `None` + `quant_config = QuantConfig() if quant_config is None else quant_config` 是
> Python 里非常常见的"可选参数兜底"写法：调用方没传就用默认对象。本测试传了 `FP4_FP4`，所以兜底不触发。

### 2.2 `cast_fp8_fp4_with_major()`：分派到 FP4 量化

源码（`tests/generators.py:233-241`）：

```python
def cast_fp8_fp4_with_major(x: torch.Tensor, major: MajorTypeAB, gran_k: int, is_fp4: bool,
                            use_ue8m0: bool, use_block_cast_for_fp8: bool = False):
    if is_fp4:                                                          # FP4 分支（本测试）
        x_fp4 = per_token_cast_to_fp4(x, use_ue8m0=use_ue8m0, gran_k=gran_k)   # 核心量化
        return x_fp4 if major.is_k_major() else (transpose_packed_fp4(x_fp4[0]).T, x_fp4[1])
    else:                                                              # FP8 分支（不走）
        x_fp8 = per_block_cast_to_fp8(x, use_ue8m0=use_ue8m0, gran_k=gran_k) if use_block_cast_for_fp8 \
                else per_token_cast_to_fp8(x, use_ue8m0=use_ue8m0, gran_k=gran_k)
        return x_fp8 if major.is_k_major() else (x_fp8[0].T.contiguous().T, x_fp8[1])
```

**作用**：根据 `is_fp4` 选 FP4 或 FP8 量化，并根据 major 类型决定是否要做转置。

**入参**：`x` 是 bf16 的 `(m,k)`（A）或 `(n,k)`（B）；`gran_k=32`；`is_fp4=True`；`use_ue8m0=True`。

**返回**：`(packed_fp4 int8, sf float32)` 二元组。

讲解：本测试 `is_fp4=True` 且 `major.is_k_major()` 为真，所以**直接返回 `per_token_cast_to_fp4` 的
结果，不做转置**。那条 `transpose_packed_fp4(...).T` 只在 MN-major 时才需要（本文不涉及）。
真正干活的是 `per_token_cast_to_fp4`。

### 2.3 `per_token_cast_to_fp4()`：FP4 量化全流程 ★

这是本章的核心。源码（`deep_gemm/utils/math.py:85-101`）：

```python
def per_token_cast_to_fp4(x: torch.Tensor, use_ue8m0: bool, gran_k: int = 128,
                          use_packed_ue8m0: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape                                                # n 这里就是 K（列数）
    assert n % 2 == 0                                             # 要能两两打包，K 必须偶数
    assert not use_packed_ue8m0 or use_ue8m0                      # packed 模式必须搭配 ue8m0
    padded_n = align(n, gran_k)                                   # 把 K 向上对齐到 32 的倍数
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)   # 补零到 padded_n
    x_padded[:, :n] = x                                          # 原数据拷进去，尾部是 0
    x_view = x_padded.view(m, -1, gran_k)                        # 重塑成 (m, n/32, 32)：每 32 个一块
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)    # 每块的最大绝对值, (m, n/32)
    sf = x_amax / 6.0                                            # 缩放因子 = 块最大值 / FP4 最大幅值 6
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf                 # 把 sf 向上取整到 2 的幂（UE8M0）
    x_scaled = x_view * (1.0 / sf.unsqueeze(2))                 # 用 sf 把每块缩放进 FP4 动态范围
    codes = _quantize_to_fp4_e2m1(x_scaled).view(m, padded_n)  # 量化成 E2M1 的 4-bit 码, int8 存
    codes2 = codes.view(m, padded_n // 2, 2)                    # 相邻两个码配对, (m, n/2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)  # 打包：低 4 位+高 4 位
    return packed[:, :n // 2].contiguous(), pack_ue8m0_to_int(sf) if use_packed_ue8m0 else sf
```

**作用**：把一个 bf16 矩阵按"每行、每 32 列一块"做 MXFP4 量化，返回 (打包后的 FP4 字节, 每块的缩放因子)。

**入参**：`x` = bf16 `(m,n)`（这里 n=K=7168）；`use_ue8m0=True`；`gran_k=32`；`use_packed_ue8m0=False`。

**返回**：
- `packed`：`int8`，shape `(m, n//2) = (128, 3584)`。每个字节装 2 个 FP4。
- `sf`：`float32`，shape `(m, n//gran_k) = (128, 224)`。注意 `use_packed_ue8m0=False`，所以这里
  **返回的是 float32**（每个值是一个 2 的幂），还没打包成 int32——那一步留给 C++（第 3 章）。

逐步拆解（强烈建议对照具体 shape）：

**Step 1 — 分块。** `x_view = x_padded.view(m, -1, gran_k)` 把 `(128, 7168)` 重塑成
`(128, 224, 32)`。最后一维的 32 个元素就是一个 **MXFP4 块**，它们将共享一个缩放因子。
`-1` 让 PyTorch 自动算出中间维 = `7168/32 = 224`。

**Step 2 — 求每块最大绝对值。** `x_view.abs().float().amax(dim=2)` 沿最后一维（那 32 个元素）取
最大绝对值，得到 `(128, 224)`。`clamp_min(1e-4)` 防止全 0 块导致除 0。

**Step 3 — 定缩放因子。** `sf = x_amax / 6.0`。**为什么除以 6？** 因为 FP4(E2M1) 能表示的最大幅值
就是 `6.0`。把"块最大值 / 6"当缩放因子，意味着缩放后这一块的最大元素大约落在 FP4 的满量程 6.0 附近，
**最大化利用 FP4 那仅有的 8 个幅值档位**，让量化误差最小。

**Step 4 — 缩放因子取整到 2 的幂（UE8M0）。** `sf = ceil_to_ue8m0(sf)`（2.4 节细讲）。MXFP4 标准
规定块缩放因子是 **UE8M0**——只有指数、是个 2 的整数次幂。取整后真正用于缩放的是 `2^e`。

**Step 5 — 缩放。** `x_scaled = x_view * (1.0 / sf.unsqueeze(2))`。`sf` 形状 `(128,224)`，
`unsqueeze(2)` 在末尾插一维变 `(128,224,1)`，于是能**广播**到 `(128,224,32)`：块内 32 个元素同除一个
`sf`。除完后元素大致落在 `[-6, 6]`。

**Step 6 — 量化成 E2M1 码。** `_quantize_to_fp4_e2m1`（2.5 节）把每个缩放后的浮点映射到最近的 FP4
档位，返回 4-bit 码（暂存在 int8 里），reshape 回 `(128, 7168)`。

**Step 7 — 两两打包。** 这是 FP4 特有的"位手术"：
- `codes2 = codes.view(m, padded_n // 2, 2)`：把 7168 个码重排成 `(128, 3584, 2)`——相邻两个码成一对。
- `packed = (codes2[:,:,0] & 0x0F) | ((codes2[:,:,1] & 0x0F) << 4)`：
  - `& 0x0F` 取每个码的低 4 位（FP4 只有 4 bit）。
  - 第 0 个码放进字节的**低 4 位**，第 1 个码 `<< 4` 放进**高 4 位**，再用 `|` 合并。
  - 结果：一个 int8 字节 = `[高4位=码1 | 低4位=码0]`，装下 2 个 FP4。shape 变 `(128, 3584)`。

**Step 8 — 裁掉 padding 并返回。** `packed[:, :n // 2]` 去掉之前 padding 出来的列（本例 K 恰好是 32 的
倍数，没有 padding）。`.contiguous()` 确保内存连续。

> 💡 **Python 语法：`tensor.view(...)`、`-1`、广播、`unsqueeze`**
> - `view` 是**零拷贝**地改变张量的逻辑形状（要求内存连续），元素总数不变。
> - 形状里的 `-1` 表示"这一维你帮我算"——由总元素数和其它维推出来。
> - **广播（broadcasting）**：两个形状不同的张量做逐元素运算时，PyTorch 会把大小为 1 的维自动"复制
>   扩展"到匹配。`(128,224,1)` 和 `(128,224,32)` 相乘，前者沿最后一维广播 32 次。
> - `unsqueeze(2)` 在第 2 维（下标从 0 起）插入一个长度 1 的维，是制造可广播形状的常用手段。

> 💡 **Python 语法：`&`、`|`、`<<` 是按位运算，不是逻辑运算**
> 在张量上，`&`/`|`/`<<` 是**逐元素的按位与/或/左移**（不是 `and`/`or`）。`x & 0x0F` 把每个元素的高位
> 清零只留低 4 位；`x << 4` 把每个元素的 bit 整体左移 4 位。FP4 的打包/解包全靠这套操作。

### 2.4 `ceil_to_ue8m0()` 与 `pack_ue8m0_to_int()`：UE8M0 的位运算

源码（`deep_gemm/utils/math.py:13-22`）：

```python
def ceil_to_ue8m0(x: torch.Tensor):
    bits = x.abs().float().view(torch.int)                  # 把 fp32 的位模式当成 int32 看
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()  # 取 8 位指数；尾数非零则 +1（向上取整）
    return (exp.clamp(1, 254) << 23).view(torch.float)      # 指数夹到合法范围，拼回 fp32（尾数=0）

def pack_ue8m0_to_int(x: torch.Tensor):
    assert x.dtype == torch.float and x.size(-1) % 4 == 0   # 必须 fp32、最后一维是 4 的倍数
    assert (x.view(torch.int) & ((1 << 23) - 1) == 0).all() # 断言尾数确实是 0（纯 UE8M0）
    return (x.view(torch.int) >> 23).to(torch.uint8).view(torch.int)  # 取 8 位指数，4 个挤进 1 个 int32
```

**`ceil_to_ue8m0` 作用**：把任意正的 fp32 缩放因子**向上取整到最近的 2 的整数次幂**（即把尾数清零、
若原本有尾数则指数 +1）。这样 `sf` 就成了一个 UE8M0 值（纯指数）。

**为什么 UE8M0 / 为什么是 2 的幂？**
- IEEE-754 fp32 的位布局是 `[1 符号 | 8 指数 | 23 尾数]`。一个值 = `±1.尾数 × 2^(指数-127)`。
- `view(torch.int)` 不改 bit、只换"看待方式"，于是能用整数位运算抠出指数/尾数字段。
- `(bits >> 23) & 0xFF` 取出 8 位指数；`(bits & 0x7FFFFF).bool().int()` 判断 23 位尾数是否非零
  （非零得 1）。两者相加 = "把尾数丢掉、若有余数就进位"——这正是**向上取整到 2 的幂**。
- 结果只保留指数、尾数=0，所以它就是 `2^e`。**用 2 的幂做缩放因子**，在硬件里乘除都退化成"指数加减"，
  极快且无额外舍入误差——这是 MXFP4 选 UE8M0 的根本原因。

**`pack_ue8m0_to_int` 作用**：把 fp32 形式的 UE8M0（尾数已是 0）压缩成"每 4 个 8-bit 指数挤进 1 个
int32"。`>> 23` 取指数字节，`to(torch.uint8)` 截成 1 字节，再 `view(torch.int)` 把连续 4 个 uint8
重解释为 1 个 int32。

**本测试的微妙之处**：`per_token_cast_to_fp4` 调用时 `use_packed_ue8m0=False`，所以**这里并不打包**，
Python 返回的 SFA/SFB 仍是 `float32 (128,224)` / `(2112,224)`（只不过每个值都是 2 的幂）。真正的
"float32 → packed int32 UE8M0 + 改 MN-major + TMA 对齐" 发生在 C++ 的
`get_mn_major_tma_aligned_packed_ue8m0_tensor` 里（第 3 章）。记住这个分工，后面不会乱。

### 2.5 `_quantize_to_fp4_e2m1()`：E2M1 编码本身

源码（`deep_gemm/utils/math.py:72-82`）：

```python
def _quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs().clamp_max(6.0)                             # 取绝对值并夹到 [0,6]（FP4 最大幅值 6）
    # {0, 0.5, 1, 1.5, 2, 3, 4, 6}                          # FP4(E2M1) 的 8 个可表示幅值
    # midpoints: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0      # 相邻档位的中点（四舍五入边界）
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                              device=x.device, dtype=ax.dtype)
    idx = torch.bucketize(ax, boundaries)                  # 落在第几档(0~7)，即"最近的幅值"的下标
    code = idx.to(torch.uint8)                             # 幅值下标占低 3 位
    sign = (x < 0) & (idx != 0)                            # 负数且非零 → 需要符号位
    code = code | (sign.to(torch.uint8) << 3)             # 符号位放到 bit3
    return code.view(torch.int8)                          # 以 int8 形式返回 4-bit 码
```

**作用**：把缩放后的浮点四舍五入到最近的 FP4 档位，输出 4-bit 码（`[bit3=符号 | bit2..0=幅值下标]`）。

讲解 E2M1 的 8 个幅值与编码：

| 码(低3位) | 二进制 | E2M1 解读(exp,mant) | 数值 |
|-----------|--------|---------------------|------|
| 0 | 000 | 0 | 0.0 |
| 1 | 001 | 次正规 | 0.5 |
| 2 | 010 | 1.0×2⁰ | 1.0 |
| 3 | 011 | 1.5×2⁰ | 1.5 |
| 4 | 100 | 1.0×2¹ | 2.0 |
| 5 | 101 | 1.5×2¹ | 3.0 |
| 6 | 110 | 1.0×2² | 4.0 |
| 7 | 111 | 1.5×2² | 6.0 |

- **E2M1 = 2 指数位 + 1 尾数位**（再加 1 符号位 = 4 bit）。尾数 1 bit 意味着每个 2 的幂区间里只有
  "×1.0" 和 "×1.5" 两档，所以才是上表这 8 个值。最大值 6.0 解释了 2.3 节"除以 6"的由来。
- `torch.bucketize(ax, boundaries)`：给定**升序**的边界数组，返回每个 `ax` 应插入的位置下标。边界取的是
  相邻档位的**中点**（如 0.25 是 0 和 0.5 的中点），于是 bucketize 等价于"四舍五入到最近档位"，下标
  恰好就是上表的码值 0~7。
- 符号位：`sign = (x<0) & (idx!=0)`——只有**负且非零**才置符号位（避免产生 -0）。`<< 3` 把它放到 bit3。
- 返回 `int8`：4-bit 码暂存在一个字节里（高 4 位是 0），等回到 `per_token_cast_to_fp4` 的 Step 7 再
  两两打包。

> ⚙️ **硬件细节：为什么打包成 int8 / 后面又当 int32？**
> 张量核读 FP4 操作数时，是把一串 4-bit 当成**不透明的字节流**——每 8 个 FP4（=32 bit）正好是一个
> int32。Python 这里先两两打包成 int8（2 个 FP4/字节），是为了和这种紧凑布局对齐；到了 C++/内核里
> 再用 `view(int32)`（第 4 章）把 4 个字节看成 1 个 int32，让 TMA 以 int32 为单位搬运、让 MMA 指令以
> 8 个 FP4 为一个 "K 子步" 消费。Python 侧的"位手术"和硬件侧的数据通路是严丝合缝设计的。

### 2.6 小结：A/B/SFA/SFB 的最终形态

经过本章，`deep_gemm.fp8_fp4_gemm_nt` 即将收到的就是：

| 参数 | 内容 | shape | 存储 dtype | 占用 |
|------|------|-------|-----------|------|
| `a.first` | A 的 packed FP4 | `(128, 3584)` | `int8` | 128×3584×1 ≈ 448 KiB |
| `a.second` = SFA | A 的 UE8M0 缩放因子（逻辑 e8m0，此刻装在 fp32 容器里，值都是 2 的幂） | `(128, 224)` | `float32` | 128×224×4 ≈ 112 KiB |
| `b.first` | B 的 packed FP4 | `(2112, 3584)` | `int8` | |
| `b.second` = SFB | B 的 UE8M0 缩放因子（同上，fp32 容器） | `(2112, 224)` | `float32` | |
| `d` | 输出 | `(128, 2112)` | `float32` | |
| `c` | 累加项 | `None` | — | — |

> 再次提醒（呼应 0.4 节）：SFA/SFB 的**逻辑类型是 UE8M0（8 bit）**，但**此刻的存储是 float32（4 字节/个）**，
> 所以一个 e8m0 值在这里占了 4 字节（尾数全 0、纯浪费）。下一章 C++ 会把它压成 1 字节/个的 packed e8m0
> （存进 int32），SFA 体积从 112 KiB 降到约 28 KiB，dtype 也从 `float32` 变成 `int32`。

下一章进入 C++：看这些张量如何被检查、SF 如何被改造成内核要的布局（含上面这步 float32→packed-int32），
以及如何分发到 SM100 实现。

---

## 第 3 章 Python→C++ 边界与 API 分发

### 3.1 `deep_gemm.fp8_fp4_gemm_nt` 其实是个 C++ 函数

很多人以为 `deep_gemm.fp8_fp4_gemm_nt` 是个 Python 函数，其实它是**直接绑定到 C++ 的 pybind11 函数**。
看 `deep_gemm/__init__.py:16,34-39`：

```python
from . import _C                                   # _C 是编译出来的 C++ 扩展模块
...
try:
    from ._C import (
        # FP8 FP4 GEMMs
        fp8_fp4_gemm_nt, fp8_fp4_gemm_nn,          # ← 直接从 _C 里 import 这个名字
        ...
```

讲解：
- `_C` 是 DeepGEMM 的 C++ 扩展（pybind11 模块）。`from ._C import fp8_fp4_gemm_nt` 把一个 **C++ 函数**
  的句柄绑成 Python 名字 `deep_gemm.fp8_fp4_gemm_nt`。
- 也就是说，`test_gemm()` 里那一行调用，**一进去就是 C++**，没有中间 Python 包装层。参数从 Python 对象
  （`(Tensor, Tensor)` 元组、`None`、`(1,32)` 元组等）被 pybind11 自动转换成 C++ 类型。

注册处在 `csrc/apis/gemm.hpp:645-654`：

```cpp
m.def("fp8_fp4_gemm_nt", &fp8_fp4_gemm_nt,              // 把 C++ 函数登记成 Python 可调用名
      py::arg("a"), py::arg("b"), py::arg("d"),         // 位置参数 a, b, d
      py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,        // 带默认值的可选参数
      py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
      py::arg("compiled_dims") = "nk",                  // 默认 "nk"
      py::arg("disable_ue8m0_cast") = false);
```

- `py::arg("c") = std::nullopt` 让 Python 侧 `c=None` 能映射到 C++ 的 `std::optional` 空值。
- `compiled_dims="nk"` 的含义：把 **N 和 K 当作编译期常量**烤进内核（M 当运行期变量）。这能让编译器对
  N、K 做更激进的循环展开/寻址优化。第 4 章生成内核模板参数时会用到它。

### 3.2 C++ `fp8_fp4_gemm_nt`：检查 → 变换 SF → 分发

源码（`csrc/apis/gemm.hpp:51-116`），分段讲。

**(a) 函数签名与 major 检查（51-69 行）：**

```cpp
static void fp8_fp4_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,   // A = (packed_fp4, sfa)
                            const std::pair<torch::Tensor, torch::Tensor>& b,   // B = (packed_fp4, sfb)
                            const torch::Tensor& d,                             // 输出 D
                            const std::optional<torch::Tensor>& c,              // 可选累加 C
                            std::optional<std::tuple<int, int, int>> recipe,    // 旧式三元 recipe
                            std::optional<std::tuple<int, int>> recipe_a,       // (gran_mn, gran_k)=(1,32)
                            std::optional<std::tuple<int, int>> recipe_b,       // (1,32)
                            const std::string& compiled_dims,                   // "nk"
                            const bool& disable_ue8m0_cast) {                   // false
    const auto major_a = get_major_type_ab(a.first);     // 从 stride 推断 A 是 K-major 还是 MN-major
    const auto major_b = get_major_type_ab(b.first);
    if (fp8_requires_k_major()) {                        // SM100 要求 K-major
        DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    }
    check_major_type_cd(d);                              // D/C 必须 N-major
```

讲解：`a` 在 C++ 里就是 `std::pair<Tensor,Tensor>`，`a.first` 是 packed FP4，`a.second` 是 SFA——
正好对应第 2 章 Python 返回的二元组。`get_major_type_ab` 通过看张量 stride 判断主序；FP4 路径强制 K-major。

**(b) 形状/类型检查（71-82 行）：**

```cpp
    const auto arch_major = device_runtime->get_arch_major();        // 10 = SM100/Blackwell
    const auto [m , k ] = check_ab_fp8_fp4(a.first, major_a, arch_major);  // 从 A 拆出 (m, k)
    const auto [n , k_] = check_ab_fp8_fp4(b.first, major_b, arch_major);  // 从 B 拆出 (n, k)
    const auto [m_, n_] = get_shape<2>(d);                          // D 的 (m, n)
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);                // 三者一致性
    if (a.first.scalar_type() == kPackedFP4 and b.first.scalar_type() == kPackedFP4) {
        DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);           // FP4×FP4 必须 fp32 输出
    } else {
        DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    }
```

- `check_ab_fp8_fp4` 会校验 packed FP4 张量的 shape 并返回逻辑上的 `(m, k)`。注意这里的 `k` 是**逻辑
  FP4 个数**（7168），不是打包后的字节数（3584）——后者是内核内部的事。
- `kPackedFP4` 是 DeepGEMM 自定义的 dtype 标签（packed int8 的别名）。两个操作数都是它 → 走 FP4×FP4，
  强制 fp32 的 D（对应第 1 章注释里说的"API 拒绝 bf16 D"）。

> 💡 **C++ 语法：`const auto [m, k] = f(...)`（结构化绑定）**
> 这是 C++17 的"结构化绑定"，等价于 Python 的元组拆包 `m, k = f(...)`。`check_ab_fp8_fp4` 返回一个
> `std::pair`/`tuple`，这里就地拆成两个具名变量。`auto` 让编译器自动推断类型。

**(c) 平凡情形短路 `early_return`（84-86 行）：**

```cpp
    if (early_return(m, n, k, d, c))    // m/n/k 为 0、或纯拷贝 C→D 的退化情形，直接返回
        return;
```

`early_return`（`csrc/apis/gemm.hpp:20-47`）处理：`m==0||n==0` 直接返回；`k==0` 时把 D 清零或拷贝 C；
**有累加且 C≠D 时，先把 C 拷进 D**。关键点：内核**没有单独的 C 加载通路**，累加是通过"epilogue 阶段用
`TMA_REDUCE_ADD` 把结果加到 D 上"实现的，而 D 在此处已被预置成 C。本测试 `c=None`，此函数返回 false，
继续往下走。

**(d) 变换 SF 布局（88-90 行）—— 本章重点：**

```cpp
    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        a.second, b.second, m, n, k, recipe, recipe_a, recipe_b, std::nullopt, std::nullopt, disable_ue8m0_cast);
```

把 Python 来的 float32 SFA/SFB 改造成内核要的布局，并取回 `gran_k_a=gran_k_b=32`。详见 3.3。

**(e) 架构分发（92-115 行）：**

```cpp
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {        // SM90/Hopper 分支（不走）
        ...
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {  // SM100/Blackwell + int SF
        if (a.first.scalar_type() == kPackedFP4 and b.first.scalar_type() == kPackedFP4) {
            DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
            DG_HOST_ASSERT(k % 8 == 0);                                  // 8 个 FP4 拼 1 个 int32，K 须 8 的倍数
            sm100_fp4_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, k,  // ★ 进入第 4 章
                                major_a, major_b, compiled_dims);
        } else {
            sm100_fp8_fp4_gemm_1d1d(...);                                // 混合 FP8×FP4（不走）
        }
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
```

讲解：注意分发条件之一是 **`sfa.scalar_type() == torch::kInt`**——即 SF 必须已经是 int32。这正是 3.3 节
`transform_sf_into_required_layout` 把 float32 转成 packed int32 的结果。两个操作数都是 `kPackedFP4`、
K-major、`k%8==0`，于是调用 `sm100_fp4_gemm_1d1d`（第 4 章的 host 装配函数）。

### 3.3 `transform_sf_into_required_layout`：float32 → packed int32 UE8M0

`transform_sf_pair_into_required_layout`（`csrc/apis/layout.hpp:62-87`）只是对 SFA、SFB 各调一次
`transform_sf_into_required_layout`，并从 recipe 里取出 `gran_k`。核心在后者
（`csrc/apis/layout.hpp:14-60`），只看会命中的分支：

```cpp
static torch::Tensor transform_sf_into_required_layout(const torch::Tensor& sf,
        const int& mn, const int& k,
        const std::variant<std::tuple<int,int,int>, std::tuple<int,int>>& recipe,  // 这里是 (1,32)
        const std::optional<int>& num_groups, const std::optional<bool>& is_sfa,
        const bool& disable_ue8m0_cast) {
    const auto arch_major = device_runtime->get_arch_major();
    int gran_mn, gran_k;
    ...
    } else if (auto p = std::get_if<std::tuple<int, int>>(&recipe)) {   // recipe 是二元组 (1,32)
        DG_HOST_ASSERT(not is_sfa.has_value());
        std::tie(gran_mn, gran_k) = *p;                                 // gran_mn=1, gran_k=32
    }
    ...
    check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups);           // 变换前的合法性检查

    // (FP32, x, gran_k) on SM100: transform to (INT, 1, gran_k), TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kFloat and (gran_k == 32 or gran_k == 128) and arch_major == 10) {
        DG_HOST_ASSERT(not disable_ue8m0_cast);
        const auto broadcasted = gran_mn == 1 ? sf :                   // gran_mn=1 → 不广播，直接用 sf
            sf.index_select(-2, torch::arange(mn, ...).floor_divide_(gran_mn));
        return get_mn_major_tma_aligned_packed_ue8m0_tensor(broadcasted);  // ★ 关键变换
    }
    ...
}
```

**作用**：根据 (dtype, gran_mn, gran_k, arch) 选择正确的 SF 布局变换。本例命中"SM100 + float32 +
gran_k=32"分支，调用 `get_mn_major_tma_aligned_packed_ue8m0_tensor`。

`get_mn_major_tma_aligned_packed_ue8m0_tensor` 做三件事（名字直接拆开看）：
1. **packed_ue8m0**：把 float32 的 UE8M0（第 2.4 节，值都是 2 的幂）抠出 8-bit 指数，**4 个挤进 1 个
   int32**。于是 `(128,224) float32` → 约 `(128,56) int32`（224/4=56）。这就是 3.2(e) 里分发条件要求
   `sfa.scalar_type()==kInt` 的来源。
2. **mn_major**：把 SF 转成 **MN-major**（让同一个 MN 上、不同 K 块的 SF 在内存里相邻）。这样内核用 TMA
   按"一列 SF"搬运时是连续访存。
3. **tma_aligned**：把 MN 维**向上对齐**到 TMA 描述符要求的边界（如 128 的倍数），不足处 padding。
   M=128 本就对齐；N=2112 会按需对齐。

> ⚙️ **硬件细节：为什么 SF 要专门"换布局"？**
> 张量核做 block-scaled MMA 时，要求每个 K 子块的缩放因子按特定排布躺在 **TMEM** 里。而 TMA（搬运引擎）
> 又要求源张量在显存里是**连续、对齐、主序匹配**的，才能高效地批量拷进 smem。Python 产出的 SF 是
> "行优先、float32、每行 224 个"，既不是 int32 也不是内核要的主序。这一步就是把它**一次性预处理**成
> "TMA 友好 + MMA 友好"的 packed-int32-MN-major 布局，省得内核在热路径上反复倒腾。

### 3.4 本章产出

经过 `fp8_fp4_gemm_nt`，进入第 4 章 `sm100_fp4_gemm_1d1d` 的实参是：
- `a`（=`a.first`）：packed FP4 `int8 (128,3584)`
- `sfa`：**packed int32 UE8M0、MN-major、TMA 对齐**（≈ `(128,56) int32` 的等价布局）
- `b`、`sfb`：同理（`(2112,3584) int8` 与对应 int32 SF）
- `c=None`、`d` `(128,2112) float32`、`m=128, n=2112, k=7168`、`major_a=major_b=K`、`compiled_dims="nk"`

---

## 第 4 章 Host 端内核装配 `sm100_fp4_gemm_1d1d`

本章是 host（CPU）侧的"总装车间"：决定 tile 形状、stage 数、swizzle、构造 5 个 TMA 描述符，
**JIT 生成并编译**那个 `.cuh` 内核，最后启动它。文件：
`csrc/jit_kernels/impls/sm100_fp4_gemm_1d1d.hpp`。

> ⚙️ **背景：DeepGEMM 的 JIT 机制（先理解全局）**
> DeepGEMM **不在编译期就把所有 tile/形状组合都编译好**，而是运行时根据具体 `m,n,k` 和 GPU 选好参数，
> 把内核模板**实例化成一段 CUDA 源码字符串**，调 NVCC 现场编译成 cubin，再 launch。好处是：每个具体
> 问题都能用最优 tile 且把 N/K 当编译期常量，省去运行时分支、寄存器更省；首次会有编译开销，之后缓存命中。
> 第 1 章 `bench_kineto` 之前先 `fn()` 跑一次，就是为了触发并预热这次 JIT 编译。

### 4.1 `sm100_fp4_gemm_1d1d`：装配主函数

源码（`csrc/jit_kernels/impls/sm100_fp4_gemm_1d1d.hpp:247-317`），分段。

**(a) 描述符 + 布局 + 配置（254-269 行）：**

```cpp
constexpr int gran_k = 32;                                       // VS=32
const auto desc = make_fp4_desc(GemmType::Normal, m, n, k, 1,    // 问题描述（见 4.2）
                                major_a, major_b, d.scalar_type(),
                                c.has_value(), compiled_dims);
const auto layout = pick_fp4_layout(GemmType::Normal, m, n, k, 1,// tile 启发式（见 4.3）
                                    device_runtime->get_num_sms());
auto config = GemmConfig{
    .layout = layout,
    .storage_config = SM100ArchSpec::get_storage_config(desc, layout),  // load/store 块、swizzle 模式
    .pipeline_config = {},
    .launch_config = SM100ArchSpec::get_launch_config(desc, layout),     // 线程数、SM 数
};
config.pipeline_config = SM100ArchSpec::get_pipeline_config(desc, layout, config.storage_config);
const auto [new_stages, new_smem] = recompute_stages_for_fp4(config, layout.block_m, layout.block_n, k);
config.pipeline_config.num_stages = new_stages;                 // 修正 stage 数（见 4.4）
config.pipeline_config.smem_size = new_smem;
```

**(b) int8 → int32 重解释（271-277 行）—— 一个关键技巧：**

```cpp
const auto a_int32 = a.view(torch::kInt);     // 同一块内存，dtype 标签从 int8 改成 int32
const auto b_int32 = b.view(torch::kInt);
const int k_int32 = k / 8;                    // 7168/8 = 896：K 以 int32 为单位
const int block_k_int32 = config.layout.block_k / 4;   // block_k=128 字节 → 32 个 int32
```

讲解：`a.view(torch::kInt)` **不拷贝、不改字节**，只是把 `(128,3584) int8` 重新看成 `(128,896) int32`
（3584 字节/4 = 896）。为什么要这么做？见下框。

> ⚙️ **硬件细节：FP4 为何以 int32 为搬运/寻址单位？**
> 一个 int32 = 4 字节 = **8 个 FP4**。Blackwell 的 `tcgen05.mma.kind::mxf4` 指令一个 "K 子步" 正好吃
> 一组 64 个 FP4（见第 6 章 `UMMA_K_FP4=64`，即 8 个 int32）。把 packed FP4 当 int32 数组：
> ① TMA 描述符用 INT32 元素类型，按 4 字节对齐搬运，匹配内核里 int32 的 smem 缓冲；
> ② 内核里寻址、swizzle、UMMA 描述符推进都以 int32 为步长，干净利落。
> 所以 host 这里统一把 K、block_k 都换算成 int32 单位（`k_int32=896`、`block_k_int32=32`）。

**(c) 构造 5 个 TMA 描述符（279-297 行）：**

```cpp
const auto tensor_map_a = make_tma_a_desc(major_a, a_int32, m, k_int32,           // A 的 TMA 描述符
        config.storage_config.load_block_m, block_k_int32,
        static_cast<int>(a_int32.stride(get_non_contiguous_dim(major_a))), 1,
        config.storage_config.swizzle_a_mode);
const auto tensor_map_b = make_tma_b_desc(major_b, b_int32, n, k_int32, ...);     // B 的
const auto tensor_map_d = make_tma_cd_desc(d, m, n, ...);                         // D 的（输出）
const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,    // SFA 的
        config.layout.block_m, gran_k, 1, 0);
const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,    // SFB 的
        config.layout.block_n, gran_k, 1, 0);
```

讲解：**TMA 描述符（`CUtensorMap`）** 是一个 128 字节的硬件结构，描述"一个多维张量在显存里长什么样、
每次搬一个多大的 tile、怎么 swizzle"。内核里只要给定 tile 的坐标，TMA 引擎就能异步把那一块搬进 smem。
这里一次性为 A/B/D/SFA/SFB 各建一个，作为 `__grid_constant__` 参数传给内核（第 5 章）。

**(d) JIT 生成 → 编译 → 启动（301-316 行）：**

```cpp
const SM100FP4Gemm1D1DRuntime::Args args = { .gemm_desc = desc, .gemm_config = config,
    .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                              config.pipeline_config.smem_size, config.layout.get_cluster_size()),
    .grouped_layout = nullptr,                          // dense 路径无分组信息
    .tensor_map_a = tensor_map_a, .tensor_map_b = tensor_map_b,
    .tensor_map_sfa = tensor_map_sfa, .tensor_map_sfb = tensor_map_sfb,
    .tensor_map_d = tensor_map_d };
const auto code = SM100FP4Gemm1D1DRuntime::generate(args);          // 生成 CUDA 源码字符串（见 4.5）
const auto runtime = compiler->build("sm100_fp4_gemm_1d1d", code);  // 调 NVCC 编译成 cubin
SM100FP4Gemm1D1DRuntime::launch(runtime, args);                     // 配置 grid/smem 后启动内核
```

### 4.2 `make_fp4_desc`：问题描述 `GemmDesc`

源码（`...hpp:225-245`）：

```cpp
static GemmDesc make_fp4_desc(GemmType gemm_type, int m, int n, int k, int num_groups, ...) {
    return GemmDesc {
        .gemm_type = gemm_type,                  // Normal（dense）
        .kernel_type = KernelType::Kernel1D1D,   // 1D1D：A、B 都是 per-token(1D) 缩放
        .m = m, .n = n, .k = k, .num_groups = num_groups,    // 128, 2112, 7168, 1
        .a_dtype = kPackedFP4, .b_dtype = kPackedFP4,        // 强制两操作数 FP4
        .cd_dtype = cd_dtype,                    // float32
        .major_a = major_a, .major_b = major_b,  // K, K
        .with_accumulation = with_accumulation,  // false（c=None）
        .num_sms = device_runtime->get_num_sms(),
        .compiled_dims = compiled_dims,          // "nk"
        ...
    };
}
```

**作用 / 返回**：把"这个 GEMM 长什么样"打包成一个结构体，后续 tile 选择、配置、代码生成都读它。

> 💡 **C++ 语法：`.field = value` 指派初始化（designated initializers）**
> `GemmDesc{ .gemm_type = ..., .m = m, ... }` 是 C++20 的写法，按**字段名**初始化结构体，可读性高，
> 类似 Python 的关键字参数。未列出的字段取默认值。

**关于 1D1D**：名字里的 "1D1D" 指 **A、B 两边的缩放都是"1 维"粒度**（per-token / per-row，
即 `gran_mn=1`）。对比之下，SM90 上有 "1D2D"（一边 per-token、一边 per-128×128 块）。本测试两边都
`(1,32)`，所以是 1D1D，对应内核文件 `sm100_fp4_gemm_1d1d.cuh`。

### 4.3 `pick_fp4_layout`：tile 启发式

源码（`...hpp:137-222`）。它要定四件事：`block_m`、`block_n`、`cluster`、`swap_ab`。

```cpp
constexpr int block_m = 128;          // 固定！MXF4 的 UMMA_M 就是 128（或 2CTA 时 256）
constexpr int block_k_bytes = 128;    // 固定！32 个 int32 = 256 个 FP4 / 每 K 块
...
for (int bn = 16; bn <= 256; bn += 16) {            // 在 16..256 里搜 block_n
    if (!is_legal(bn)) continue;                    // 受 TMEM 列预算约束
    const int waves = num_waves(bn);                // = ceil(总 tile 数 / SM 数)
    ...
    const int score = est_stages * est_stages * bn; // 评分：流水线越深、bn 越大越好
    ...                                             // 先比 waves（越少越好），再比 score
}
...
// Multicast：M >= 512 时，cluster_m=2（B 多播，2 个 CTA 协同算 M=256 的 MMA）
const bool can_multicast = (m >= 512) && (ceil_div(m, block_m) % 2 == 0)
                       && (num_sms % 2 == 0) && (gemm_type == GemmType::Normal || ...);
if (can_multicast) cluster_m = 2;
...
return Layout{swap_ab, block_m, best_bn, block_k_bytes, cluster_m, cluster_n};
```

讲解（针对本测试）：
- **`block_m = 128` 是写死的**，因为 MXF4 张量核指令的 M 维就是 128（1 个 CTA）或 256（2 个 CTA 协同）。
- **`block_n` 靠搜索 + 评分选**：先让 `waves`（"波数"= 所有输出 tile 要分几轮铺满全部 SM）最小，再用
  `est_stages² × bn` 这个综合分挑——既偏好更深的软件流水线（stage 多 → 更能掩盖访存延迟），也偏好更大的
  `bn`（每个 tile 干更多活）。
- **本测试 `m=128 < 512`**：`can_multicast=false`，所以 `cluster_m=cluster_n=1`（**单 CTA，无多播**）。
  当 `m=4096` 那组用例时才会触发 `cluster_m=2` 的 B-multicast。
- **`swap_ab=false`**：只有稀疏 MoE（MGroupedContiguous 且每组 M 很小）才置 true，dense 路径不涉及。

> ⚙️ **硬件细节：TMEM 列预算（`is_legal` 在卡什么）**
> Blackwell 每个 SM 的 TMEM 是 **128 lane × 512 列**。累加器（`block_n` 列）+ SFA 列 + SFB 列必须塞进
> 512 列。`is_legal(bn)` 就是检查"至少 1 个 epilogue stage 放得下"，`fits_2_epi(bn)` 检查"能否放下 2
> 个（双缓冲）"。这是 `block_n` 上限只能到 256 的根本原因。

### 4.4 `recompute_stages_for_fp4`：smem 容量再校正

源码（`...hpp:94-128`）。SM100 每 SM 的 smem 上限是 **232448 字节（≈227KB）**。共享内存里要放 A/B 的
若干 stage、SFA/SFB、CD 缓冲、各种 barrier。通用启发式对 FP4 的 SF 占用**低估了 2 倍**，所以这里按内核
实际布局重新算每个 stage 占多少字节，反推 `num_stages`：

```cpp
const int per_stage = config.storage_config.load_block_m * config.layout.block_k    // A 一个 stage
                    + config.storage_config.load_block_n * config.layout.block_k    // B 一个 stage
                    + sf_block_m * sf_packed_k_per_stage * 4                         // SFA
                    + sf_block_n * sf_packed_k_per_stage * 4;                        // SFB
...
const int max_stages_smem = (smem_capacity - fixed_extras) / per_stage;             // smem 能放几个 stage
const int num_k_blocks = ceil_div(k_fp4, config.layout.block_k * 2);                // K 方向总块数上限
const int new_num_stages = std::min({12, max_stages_smem, num_k_blocks});           // 取三者最小，封顶 12
```

讲解：`num_stages` 是软件流水线的"深度"——一边 TMA 在搬第 s+N 块，一边 MMA 在算第 s 块，stage 越多越能
掩盖延迟，但越吃 smem。这里在 smem 容量、K 总块数、硬上限 12 之间取最小。这个 `new_stages` 会作为模板
参数 `kNumStages` 烤进内核。

### 4.5 `SM100FP4Gemm1D1DRuntime::generate_impl`：把内核"实例化成源码"

源码（`...hpp:38-77`）。这是 JIT 的心脏——用 `fmt::format` 把一堆配置值填进一个内核实例化模板：

```cpp
static std::string generate_impl(const Args& args) {
    return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp4_gemm_1d1d.cuh>     // 引内核定义
using namespace deep_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp4_gemm_1d1d_impl<   // 取内核模板的具体实例地址
        {}, {},              // kMajorA, kMajorB
        {}, {}, {},          // SHAPE_M, SHAPE_N, SHAPE_K(int32)
        {}, {}, {},          // BLOCK_M, BLOCK_N, BLOCK_K(int32)
        {},                  // kNumGroups
        {}, {}, {},          // swizzle A/B/CD
        {},                  // kNumStages
        {}, {},              // 非 epilogue 线程数, epilogue 线程数
        {}, {},              // cluster size, cluster_n>1（是否多播 on A）
        {},                  // kNumSMs
        {},                  // kSwapAB
        {}, {}, {}           // GemmType, with_accumulation, cd_dtype
    >);
}};
)",
    to_string(args.gemm_desc.major_a), to_string(args.gemm_desc.major_b),
    get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),     // m 不在 "nk" → 填 0（运行期）
    get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),     // n 在 "nk" → 填 2112
    get_compiled_dim(args.gemm_desc.k / 8, 'k', args.gemm_desc.compiled_dims), // SHAPE_K 用 int32 单位 896
    args.gemm_config.layout.block_m, args.gemm_config.layout.block_n,
    args.gemm_config.layout.block_k / 4,                                        // BLOCK_K 用 int32 单位
    ...);
}
```

讲解几处关键：
- `R"( ... )"` 是 C++ **原始字符串字面量**，里面的 `{}` 是 `fmt::format` 的占位符（`{{`/`}}` 是转义的
  花括号，因为内核里真有 `{}`）。
- `get_compiled_dim(m, 'm', "nk")`：因为 `compiled_dims="nk"` **不含 'm'**，返回 0 → 模板里 `SHAPE_M=0`
  表示"M 在运行期才知道"；而 `n`、`k` 在 "nk" 里，直接把 `2112`、`896` 当**编译期常量**填进去。第 5 章会
  看到内核里 `shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;` 正是配合这个约定。
- `SHAPE_K` 和 `BLOCK_K` 都换算成 **int32 单位**（`k/8`、`block_k/4`），呼应 4.1(b) 的重解释。

`launch_impl`（`...hpp:79-88`）则在编译完成后，把 `m, n, shape_k_int32` 和 5 个 TMA 描述符作为运行期
实参喂给 cubin 入口。至此，控制权交给 GPU 内核——进入第 5、6 章。

### 4.6 本章产出：内核的全部模板参数（本测试取值）

| 模板参数 | 含义 | 本测试取值 |
|----------|------|-----------|
| `kMajorA, kMajorB` | A/B 主序 | `K, K` |
| `SHAPE_M / SHAPE_N / SHAPE_K` | 编译期形状（0=运行期） | `0 / 2112 / 896`(int32) |
| `BLOCK_M / BLOCK_N / BLOCK_K` | tile 形状 | `128 / 启发式 / 32`(int32) |
| `kNumGroups` | 分组数 | `1` |
| `kSwizzle{A,B,CD}Mode` | 三个 swizzle 模式 | 由 storage_config 定 |
| `kNumStages` | 流水线深度 | `recompute` 算出（≤12） |
| `kNumNonEpilogue / kNumEpilogueThreads` | 两类线程数 | 由 launch_config 定 |
| `kNumMulticast, kIsMulticastOnA` | 多播 | `1, false`（m=128 不多播） |
| `kNumSMs` | 用多少 SM | GPU 实际 SM 数 |
| `kSwapAB` | 是否交换 A/B | `false` |
| `kGemmType, kWithAccumulation, cd_dtype` | 类型/累加/输出 | `Normal, false, float` |

---

## 第 5 章 CUDA 内核总体结构

内核文件：`deep_gemm/include/deep_gemm/impls/sm100_fp4_gemm_1d1d.cuh`。本章先看"骨架"
（签名、编译期常量、smem/TMEM 切分、屏障初始化、warp 分工），第 6 章再钻进主循环。

### 5.1 内核签名与模板参数

源码（`...cuh:24-42`）：

```cpp
template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,        // 编译期形状（0=运行期给）
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,        // tile（BLOCK_K 是 int32 单位）
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages,                                         // 流水线深度
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          bool kSwapAB,
          GemmType kGemmType, bool kWithAccumulation, typename cd_dtype_t>
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)   // 限定每块线程数
sm100_fp4_gemm_1d1d_impl(int* grouped_layout,
                         uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,        // 运行期形状
                         const __grid_constant__ cute::TmaDescriptor tensor_map_a,    // 5 个 TMA 描述符
                         const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_d) {
```

讲解：
- 这些模板参数就是第 4.6 节表里那些值，由 JIT 在编译期填死，于是编译器能针对这个具体 tile/形状生成最优代码。
- `__global__` 表示这是 GPU 内核入口（CPU 可启动）。
- `__launch_bounds__(总线程数, 1)`：告诉编译器每个块最多这么多线程、每个 SM 最少驻留 1 个块，帮助寄存器
  分配。总线程 = 非 epilogue 线程 + epilogue 线程。

> ⚙️ **硬件细节：`__grid_constant__ TmaDescriptor` 是什么？**
> TMA 描述符是个 128 字节的硬件结构，必须放在一种特殊的常量内存里、且整个 grid 共享同一份。
> `__grid_constant__` 修饰的内核参数正好满足这要求——它把 host 构造好的 `CUtensorMap` 以只读常量形式
> 提供给所有线程块。内核里 `cute::prefetch_tma_descriptor(&tensor_map_a)`（`...cuh:125-131`）会预取它
> 到缓存，减少首次 TMA 的延迟。

### 5.2 编译期常量推导（FP4 / SF 相关）

源码（`...cuh:50-75`），挑关键的讲：

```cpp
constexpr uint32_t LAYOUT_AD_M = 128;                       // A/D 在 TMEM 的 M 布局粒度
constexpr uint32_t kNumMWaves = BLOCK_M / LAYOUT_AD_M;      // 128/128 = 1
constexpr uint32_t kNumSFAStagesPerLoad = 1, kNumSFBStagesPerLoad = 1;
constexpr uint32_t FP4_ELEMS_PER_INT32 = 8;                 // 8 个 FP4 = 1 个 int32
constexpr uint32_t MXF4_VS = 32;                            // 块缩放向量大小 = 32
constexpr uint32_t BLOCK_K_FP4 = BLOCK_K * FP4_ELEMS_PER_INT32;   // 32*8 = 256 个 FP4/K 块
constexpr uint32_t UMMA_K_FP4 = 64;                         // 一条 MMA 指令吃 64 个 FP4（K 子步）
constexpr uint32_t SF_K_PER_STAGE = BLOCK_K_FP4 / MXF4_VS;       // 256/32 = 8 个 SF/K 块
constexpr uint32_t SF_PACKED_K_PER_STAGE = SF_K_PER_STAGE / 4;   // 8/4 = 2（4 个 SF 打包成 1 int32）
```

这些数字（本测试取值已注在右侧）是后面所有循环边界的来源，强烈建议记住这几条换算链：
- **1 个 K 块（一个 stage 的 K）= `BLOCK_K`(32) 个 int32 = `BLOCK_K_FP4`(256) 个 FP4 = `SF_K_PER_STAGE`(8) 个缩放因子。**
- **1 条 MMA 指令 = `UMMA_K_FP4`(64) 个 FP4 = 8 个 int32 = 2 个缩放因子**（因为 64/32=2）。
- 所以一个 stage 内要发 `BLOCK_K_FP4 / UMMA_K_FP4 = 256/64 = 4` 条 MMA（对应第 6 章 `NUM_K_ITERS_PER_STAGE=4`）。

### 5.3 共享内存切分

源码（`...cuh:83,133-169`）。内核动态 smem 是一整块 `smem_buffer`，手动切成各区域：

```cpp
extern __shared__ __align__(1024) uint8_t smem_buffer[];     // 1024 字节对齐（swizzle-128B 要求）
...
cd_dtype_t* smem_cd[kNumTMAStoreStages];     // 输出 D 的暂存（2 个 store stage）
uint32_t* smem_sfa[kNumStages];              // 各 stage 的 SFA
uint32_t* smem_sfb[kNumStages];              // 各 stage 的 SFB
uint32_t* smem_a_packed[kNumStages];         // 各 stage 的 A（int32 packed FP4）
uint32_t* smem_b_packed[kNumStages];         // 各 stage 的 B
```

布局顺序（按地址从低到高）：`[CD 暂存] [A×kNumStages] [B×kNumStages] [SFA×kNumStages] [SFB×kNumStages]
[各种 barrier] [TMEM 指针]`。每个区域的指针都是从 `smem_buffer` 基址加偏移算出来的（`...cuh:140-169`）。

> ⚙️ **硬件细节：为什么 `__align__(1024)`？**
> TMA 的 128-byte swizzle 模式要求 smem 起始地址按 1024 字节对齐，这样 swizzle 的异或位运算才能正确
> 落在 bank 边界上。CD 区域大小也强制是 1024 的倍数（`...cuh:112` 的 `DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0)`）。

### 5.4 TMEM 列分配

源码（`...cuh:116-122`）：

```cpp
constexpr uint32_t kNumSFATmemCols = (SF_BLOCK_M / 32) * SF_PACKED_K_PER_STAGE;   // SFA 占的 TMEM 列
constexpr uint32_t kNumSFBTmemCols = (SF_BLOCK_N / 32) * SF_PACKED_K_PER_STAGE;   // SFB 占的列
constexpr uint32_t kNumEpilogueStages = (2 * kNumMWaves * BLOCK_N + kNumSFATmemCols + kNumSFBTmemCols) > 512 ? 1 : 2;
constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * kNumMWaves * BLOCK_N; // 累加器占的列
constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;            // SFA 起始列
constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;      // SFB 起始列
```

讲解：TMEM（共 512 列）被切成三段——**累加器**、**SFA**、**SFB**。`kNumEpilogueStages` 是个关键决策：
若"2 份累加器 + SF"放得下 512 列就用 **2**（双缓冲：一份在被 epilogue 读出、另一份已在被下一个 tile 的
MMA 写入，重叠起来），否则只能 **1**。这正是第 4.3 节 host 端 `fits_2_epi` 想预测的事。

> ⚙️ **硬件细节：TMEM（Tensor Memory）到底是什么？**
> 这是 Blackwell（SM100）**新增的一块片上内存**，专门给第 5 代张量核当**累加器**用，独立于寄存器和
> 共享内存。布局是 **128 个 lane（对应 128 行）× 512 列**，按"列地址 + lane"寻址。MMA 指令
> （`tcgen05.mma`）把乘加结果**直接累加写进 TMEM**，不经过寄存器；epilogue 再用 `tcgen05.ld`
> （`SM100_TMEM_LOAD`）把它读回寄存器。block-scaled MMA 还要求**缩放因子也先放进 TMEM**（这就是
> UTCCP 的活，见第 6 章）。TMEM 需要显式 `tcgen05.alloc` 申请、`tcgen05.dealloc` 释放。

### 5.5 屏障初始化与 TMEM 申请

源码（`...cuh:159-190`）：

```cpp
auto full_barriers         = ...;   // “数据已就绪”屏障：生产者(TMA)→消费者(MMA)
auto empty_barriers        = ...;   // “缓冲已空闲”屏障：消费者→生产者（可以覆盖了）
auto with_sf_full_barriers = ...;   // “SF 已转置完成”屏障：warp2 → warp1
auto tmem_full_barriers    = ...;   // “累加器算完”屏障：MMA(warp1) → epilogue
auto tmem_empty_barriers   = ...;   // “累加器已读走”屏障：epilogue → MMA

if (threadIdx.x == 0) {                                  // 0 号线程负责初始化所有 mbarrier
    for (uint32_t i = 0; i < kNumStages; ++ i) {
        full_barriers[i]->init(1);                       // 期望 1 次 arrive
        empty_barriers[i]->init(1);
        with_sf_full_barriers[i]->init(kNumMulticast * 32);  // 期望一个 warp(32 线程)都到
    }
    ...
    cutlass::arch::fence_view_async_shared();
    cutlass::arch::fence_barrier_init();
} else if (threadIdx.x >= 32 and threadIdx.x < 64) {     // 1 号 warp 负责申请 TMEM
    Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
}
kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();   // 全块/全 cluster 同步
```

> ⚙️ **硬件细节：mbarrier 与"相位（phase）"**
> mbarrier 是放在 smem 里的异步屏障对象。它内部维护一个**期望到达计数**和一个 **1-bit 相位**。
> 生产者干完活调 `arrive`（或 TMA 完成时自动 `arrive`），消费者调 `wait(phase)` 阻塞到计数满足；
> 一轮满足后相位翻转（0↔1）。这样同一个 barrier 能被流水线**循环复用**：第 0、1、2... 轮分别用
> 相位 0、1、0...。代码里满天飞的 `phase ^ 1`、`phase ^= 1` 就是在手动追踪当前该等哪个相位（第 6 章详见）。
> DeepGEMM 用 **5 类 barrier** 把"TMA 加载 / SF 转置 / MMA / epilogue"这几个角色串成一条精密流水线。

### 5.6 warp 分工总览（warp specialization）

源码（`...cuh:224-228, 229/298/464/533`）的注释和分支结构：

```cpp
// Dispatch warps into different roles:
//   warp 0   : TMA load producer            （把 A/B/SFA/SFB 从显存搬进 smem）
//   warp 1   : MMA consumer + UTCCP SF copy  （把 SF 拷进 TMEM，发 MMA 指令）
//   warp 2   : SF SMEM warp transpose        （把 SF 在 smem 里转置成 UTCCP 要的样子）
//   warp 3+  : Epilogue                      （把 TMEM 累加器读出、写回显存 D）
if (warp_idx == 0) { ... }                    // 生产者
else if (warp_idx == 1 and is_leader_cta) { ... }   // MMA + UTCCP
else if (warp_idx == 2) { ... }               // SF 转置
else if (warp_idx >= kNumNonEpilogueThreads / 32) { ... }   // epilogue
```

> ⚙️ **硬件细节：什么是 warp specialization、为什么这么写？**
> 传统 GEMM 里所有 warp 干一样的活（SIMT）。而 Hopper/Blackwell 的高性能内核改用 **warp 专业化**：
> 不同 warp 扮演不同角色，像流水线工位一样并行——**warp 0 一直在搬数据，warp 1 一直在算，warp 3+ 一直在
> 写回**，彼此用 mbarrier 交接。好处是 TMA（异步搬运）、tcgen05（异步 MMA）这些**异步硬件单元**能被持续
> 喂饱，访存延迟被计算掩盖。这就是为什么内核体是一个大 `if (warp_idx == ...)` 分派，而不是一段顺序代码。
> 其中 MMA 必须由 **leader CTA 的单个 warp** 发起（`warp_idx == 1 and is_leader_cta`），因为 tcgen05
> MMA 是"单线程发起、异步执行"的指令。

掌握了这张分工图，第 6 章就是分别走进这 4 个工位看它们各自的循环。

---

## 第 6 章 内核主循环——逐 warp 深入

这是全文的硬核章节。4 个工位（warp 角色）通过 5 类 mbarrier 协同，跑同一个"块调度 + K 循环"骨架。
我们先讲共用的骨架（6.1），再分别走进 4 个工位（6.2~6.5）。

### 6.1 块调度器与 K 循环骨架

源码（`...cuh:195-217`）：

```cpp
uint32_t m_block_idx, n_block_idx;
auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast,
                                  kIsMulticastOnA, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);
struct DivisibleK {}; struct NotDivisibleK {};       // 两个空标签类型，用于编译期分支
uint32_t phase = 0;                                   // mbarrier 相位追踪
auto launch_k_iterations = [&](const auto& func) {
    const uint32_t current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
    const uint32_t num_iterations = ceil_div(current_shape_k, kNumStages * BLOCK_K);   // K 要分几大轮
    const uint32_t num_last_stages = ceil_div(current_shape_k, BLOCK_K) % kNumStages;  // 最后一轮剩几个 stage
    if (num_last_stages == 0) {                       // K 正好整除：每轮都是满 kNumStages
        for (uint32_t k_iter = 0; k_iter < num_iterations; ++ k_iter, phase ^= 1)
            func(k_iter, DivisibleK{}, k_iter == num_iterations - 1, num_last_stages);
    } else {                                          // 最后一轮不满，单独处理
        for (uint32_t k_iter = 0; k_iter < num_iterations - 1; ++ k_iter, phase ^= 1)
            func(k_iter, DivisibleK{}, false, num_last_stages);
        func(num_iterations - 1, NotDivisibleK{}, true, num_last_stages), phase ^= 1;
    }
};
```

讲解：
- **`scheduler`**：负责把整个 `(M/BLOCK_M) × (N/BLOCK_N)` 的输出 tile 网格分配给各个 SM。每个 warp 都
  反复调 `scheduler.get_next_block(m_block_idx, n_block_idx)` 领取下一个要算的输出 tile，直到没有为止
  （这就是各工位最外层的 `while` 循环）。
- **`launch_k_iterations`**：把 K 维（896 个 int32）按"每轮 `kNumStages` 个 stage、每 stage `BLOCK_K`
  个 int32"切成 `num_iterations` 大轮，对每一轮回调 `func`。`phase ^= 1` 在每大轮翻转相位。
- **`DivisibleK{}` / `NotDivisibleK{}`** 这两个空结构体作为标签传进 `func`，配合 `if constexpr` 在编译期
  分出"满 stage"和"尾 stage"两套代码，零运行期开销。

> 💡 **C++ 语法：`[&](const auto& func) { ... }` 是泛型 lambda + 引用捕获**
> `[&]` 表示按引用捕获外部所有变量（如 `phase`、`shape_k`）。`const auto& func` 让这个 lambda 能接收
> 任意可调用对象。这种"把 K 循环骨架抽成高阶函数、各工位传入自己的 `func`"的写法，让 4 个工位复用同一套
> 循环结构而各自填充不同的循环体。

### 6.2 warp 0：TMA 加载生产者

源码（`...cuh:229-297`，节选核心）：

```cpp
if (warp_idx == 0) {
    while (scheduler.get_next_block(m_block_idx, n_block_idx)) {        // 领一个输出 tile
        launch_k_iterations([&](uint32_t k_iter, auto type, bool is_last_iter, uint32_t num_last_stages) {
            constexpr bool kHasDivisibleStages = cute::is_same_v<decltype(type), DivisibleK>;
            const uint32_t kNumInnerStages = kHasDivisibleStages ? kNumStages : num_last_stages;
            for (uint32_t s = 0; s < kNumInnerStages; ++ s) {          // 遍历本轮的各 stage
                empty_barriers[s]->wait(phase ^ 1);                    // 等“这个 smem 缓冲已被 MMA 用完”
                ...                                                    // 算 m_idx / n_idx / k 索引
                if (cute::elect_one_sync()) {                          // 选一个线程发 TMA
                    if constexpr (kMajorA == cute::UMMA::Major::K)
                        tma_copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode>(&tensor_map_a, full_barriers[s],
                                                                       smem_a_packed[s], k_a_idx, m_idx);
                    if constexpr (kMajorB == cute::UMMA::Major::K)
                        tma_copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode>(&tensor_map_b, full_barriers[s],
                                                                       smem_b_packed[s], k_b_idx, n_idx);
                }
                auto num_arrival_bytes = SMEM_A_PACKED_SIZE_PER_STAGE + SMEM_B_PACKED_SIZE_PER_STAGE;
                // —— 同时发 SFA / SFB 的 TMA ——
                if (sfa_tma_stage == 0 and cute::elect_one_sync()) {
                    for (uint32_t pk = 0; pk < SF_PACKED_K_PER_STAGE; ++ pk)
                        tma_copy<BLOCK_M, 1, 0>(&tensor_map_sfa, full_barriers[s], smem_sfa[s] + pk * SF_BLOCK_M, ...);
                    num_arrival_bytes += BLOCK_M * SF_PACKED_K_PER_STAGE * sizeof(uint32_t);
                }
                if (sfb_tma_stage == 0 and cute::elect_one_sync()) { ... 同理搬 SFB ... }
                if (cute::elect_one_sync())
                    full_barriers[s]->arrive_and_expect_tx(num_arrival_bytes);   // 声明本 stage 预计到达字节数
            }
        });
    }
}
```

讲解（这是理解 TMA 流水线的关键）：
- **`empty_barriers[s]->wait(phase ^ 1)`**：在往 `smem_a_packed[s]` 这个缓冲写新数据前，必须确认
  上一轮用这个缓冲的 MMA 已经读完（否则会覆盖正在用的数据）。`empty_barrier` 由 MMA 工位 arrive。
  这就是软件流水线的"反压"。
- **`cute::elect_one_sync()`**：从 warp 里选出**唯一一个线程**来发 TMA。因为 TMA 是"一个线程发起、引擎
  异步搬一大块"，不需要 32 个线程都发。
- **`tma_copy<BLOCK_K, LOAD_BLOCK_M, ...>`**（定义在 `common/tma_utils.cuh:14-88`）：内部调
  `cute::SM90_TMA_LOAD_2D::copy(...)`，让 TMA 引擎按 `tensor_map_a` 描述的布局，把 A 的一个
  `LOAD_BLOCK_M × BLOCK_K`（int32）tile 从显存异步搬进 `smem_a_packed[s]`，并 swizzle。
- **`arrive_and_expect_tx(num_arrival_bytes)`**：TMA 是异步的，怎么知道"搬完了"？答案是
  **事务字节计数（transaction count）**。这里告诉 `full_barriers[s]`："本 stage 总共会有
  `num_arrival_bytes` 字节通过 TMA 到达"。TMA 引擎每搬完一部分就自动给该 barrier 记账，记满了 barrier
  才放行等它的 MMA 工位。这是 Hopper/Blackwell TMA 同步的标准范式。

> ⚙️ **硬件细节：TMA（Tensor Memory Accelerator）一图流**
> TMA 是 Hopper 引入、Blackwell 增强的**异步多维拷贝引擎**。用法三步：① host 建好 `CUtensorMap` 描述源
> 张量；② 内核里单线程用 tile 坐标发起拷贝，绑定一个 mbarrier；③ 引擎在后台把整块数据（可选 swizzle）
> 搬进 smem，搬完通过事务计数让 mbarrier 放行。它把"循环 + 逐元素 load + 边界判断"全部下放到硬件，
> 解放了线程去干计算，是 warp 专业化流水线的基石。本内核里 A/B/SFA/SFB 四路数据全靠它搬。

### 6.3 warp 2：SF 在 smem 里 warp 转置

为什么需要这个工位？因为 MMA 的 block-scale 要求 SF 在 TMEM 里是**特定的转置布局**，而 TMA 搬进 smem
的 SF 是另一种排布。warp 2 专门在 smem 里把 SF 重排，喂给 warp 1 的 UTCCP。源码（`...cuh:464-532`，核心）：

```cpp
} else if (warp_idx == 2) {
    auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
        uint32_t values[4];
        for (uint32_t i = 0; i < 4; ++ i)
            values[i] = ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);   // 按 XOR 模式读
        __syncwarp();
        for (uint32_t i = 0; i < 4; ++ i)
            st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);      // 转置后写回
    };
    auto fill_sfb_missing_k_groups = [&](uint32_t* smem_ptr) {
        if constexpr (BLOCK_N < kNumUTCCPAlignedElems) {        // BLOCK_N < 128 时
            for (uint32_t pos = lane_idx; pos < kNumUTCCPAlignedElems; pos += 32)
                if (pos >= BLOCK_N) st_shared(smem_ptr + pos, 0u);   // 把 padding 区清零
            __syncwarp();
        }
    };
    while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
        launch_k_iterations([&](uint32_t k_iter, auto type, bool is_last_iter, uint32_t num_last_stages) {
            for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                full_barriers[s]->wait(phase);                  // 等 warp0 把 SF 搬进 smem
                // 对 SFA、SFB 各做 warp 转置
                ... utccp_required_smem_warp_transpose(smem_sfa[s] + ...);
                ... fill_sfb_missing_k_groups(...); utccp_required_smem_warp_transpose(smem_sfb[s] + ...);
                cutlass::arch::fence_view_async_shared();
                with_sf_full_barriers[s]->arrive(0u);           // 通知 warp1：SF 已转置好
            }
        });
    }
}
```

讲解：
- **`ld_shared` / `st_shared`**：从/向 smem 读写一个 32-bit 字（带 swizzle 地址）。`lane_idx` 是 warp 内
  线程号（0~31）。
- **`i ^ (lane_idx >> 3)`** 这个**异或 swizzle**：让 32 个线程读/写 smem 时落在不同 bank，避免
  **bank 冲突**（同一 bank 被多线程同时访问会串行化）。`lane_idx >> 3` 把 32 个线程按每 8 个分 4 组，
  组号参与异或，是 UTCCP 4×32 布局要求的特定重排。
- **`fill_sfb_missing_k_groups`**：当 `BLOCK_N < 128`（UTCCP 对齐粒度）时，SFB 在 smem 里有一段是
  padding。转置的 XOR 模式会读到全部 128 个位置，**若 padding 区是未初始化的脏值，会污染有效数据**，
  所以先清零。这是个容易忽略但很关键的正确性细节（源码注释专门强调了）。
- 转置完 `fence_view_async_shared()` 保证写对其它 warp 可见，再 `with_sf_full_barriers[s]->arrive()`
  通知 warp 1 可以做 UTCCP 了。

### 6.4 warp 1：MMA 消费者 + UTCCP（tcgen05 的心脏）★

这是整个内核最核心的工位：把 SF 拷进 TMEM（UTCCP），然后发 `tcgen05.mma.kind::mxf4` 指令做块缩放矩阵乘。
只有 **leader CTA 的 warp 1** 执行。

**(a) 准备 MMA 类型与指令描述符（`...cuh:298-355`）：**

```cpp
} else if (warp_idx == 1 and is_leader_cta) {
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * (kIsMulticastOnA ? 1 : kNumMulticast);  // 128（本测试）
    constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N * (...);                  // = BLOCK_N
    constexpr uint32_t UMMA_K_INT32 = UMMA_K_FP4 / FP4_ELEMS_PER_INT32;              // 64/8 = 8
    constexpr uint32_t NUM_K_ITERS_PER_STAGE = BLOCK_K / UMMA_K_INT32;               // 32/8 = 4
    constexpr uint32_t NUM_N_ITERS = (kSwapAB ? BLOCK_M : BLOCK_N) / UMMA_N;         // 1

    auto instr_desc_mxf4 = cute::UMMA::make_instr_desc_block_scaled<                  // MMA 指令描述符
        cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue8m0_t,  // A=E2M1,B=E2M1,累加=fp32,SF=UE8M0
        UMMA_M, UMMA_N, kMajorA, kMajorB>();
    using cute_mma_mxf4_t = cute::conditional_t<kNumMulticast == 1,
        cute::SM100_MMA_MXF4_SS<cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,  // 单 CTA MMA
                                cutlass::float_ue8m0_t, UMMA_M, UMMA_N, MXF4_VS, kMajorA, kMajorB>,
        cute::SM100_MMA_MXF4_2x1SM_SS<...>>;                                          // 2 CTA 协同 MMA
    using cute_utccp_t = cute::conditional_t<kNumMulticast == 1,
        cute::SM100_UTCCP_4x32dp128bit_1cta, cute::SM100_UTCCP_4x32dp128bit_2cta>;
    auto sf_desc = make_sf_desc(nullptr);                                            // SF 的 smem 描述符
    auto a_desc_base = make_umma_desc<kMajorA, BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a_packed[0], 0, 0);
    auto b_desc_base = make_umma_desc<kMajorB, BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b_packed[0], 0, 0);
    const auto runtime_desc_sf0 = make_runtime_instr_desc_with_sf_id(instr_desc_mxf4, 0, 0);             // sf_id=0
    const auto runtime_desc_sf2 = make_runtime_instr_desc_with_sf_id(instr_desc_mxf4, 2, 2);             // sf_id=2
```

讲解关键概念：
- **`make_instr_desc_block_scaled<float_e2m1_t, float_e2m1_t, float, float_ue8m0_t, ...>`**：构造一个
  64-bit 的"指令描述符"，告诉张量核：A、B 都是 **E2M1（FP4）**，累加用 **float**，缩放因子是
  **UE8M0**，以及 M/N/K 维度和主序。这是 tcgen05 block-scaled MMA 的元数据。
- **`SM100_MMA_MXF4_SS`**（cute 版）= 单 CTA 的 MXF4 MMA；`SM100_MMA_MXF4_2x1SM_SS` = 2 个 CTA 协同
  （M=256）。本测试 `kNumMulticast=1`，用前者。`_SS` 表示两个操作数都来自 **S**hared memory（用 smem
  描述符寻址）。
- **`make_umma_desc`**（`mma/sm100.cuh:94-133`）：构造 **UMMA 共享内存描述符**——一个 64-bit 值，编码了
  操作数在 smem 的起始地址、swizzle 布局、stride。张量核靠它直接从 smem 取操作数，不经过寄存器。
- **`sf_id` / `runtime_desc_sf0` 与 `sf2`**：一条 MMA 吃 64 个 FP4 = **2 个**缩放因子组（因为 VS=32）。
  `sf_id` 指明这条指令该用 TMEM 里 SF 的哪一组：偶数 K 子步用 `sf0`（id=0），奇数用 `sf2`（id=2）。

**(b) 等累加器空闲、循环体（`...cuh:357-461`，核心）：**

```cpp
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    dispatch_accum_stage_idx(scheduler.current_iter % kNumEpilogueStages, [&](uint32_t accum_stage_idx) {
        auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
        tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);   // 等 epilogue 把上轮累加器读走
        tcgen05_after_thread_sync();
        ...
        launch_k_iterations([&](uint32_t k_iter, auto type, bool is_last_iter, uint32_t num_last_stages) {
            for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                with_sf_full_barriers[s]->wait(phase);          // 等 warp2 把 SF 转置好
                tcgen05_after_thread_sync();
                // —— ① UTCCP：把 SF 从 smem 拷进 TMEM ——
                if (sfa_copy_stage == 0 and cute::elect_one_sync()) {
                    for (pk ...) for (i ...) {
                        replace_smem_desc_addr(sf_desc, smem_sfa[s] + ...);
                        cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + ...);   // SFA → TMEM
                    }
                    for (pk ...) for (i ...) {
                        replace_smem_desc_addr(sf_desc, smem_sfb[s] + ...);
                        cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + ...);   // SFB → TMEM
                    }
                }
                __syncwarp();
                // —— ② MMA：发 tcgen05.mma 指令 ——
                for (uint32_t k = 0; k < NUM_K_ITERS_PER_STAGE; ++k) {           // 4 条 MMA / stage
                    uint32_t packed_group = k / kUmmaStepsPerPacked;
                    const auto& runtime_desc_k = (k % 2 == 0) ? runtime_desc_sf0 : runtime_desc_sf2;
                    for (uint32_t n = 0; n < NUM_N_ITERS; ++n) {
                        auto b_desc = b_desc_base;
                        b_desc.lo = advance_umma_desc_lo<kMajorB, BLOCK_N, kSwizzleBMode, uint32_t>(
                            b_desc_stage_lo, n * UMMA_N * BLOCK_K, k * UMMA_K_INT32);
                        for (uint32_t w = 0; w < kNumMWaves; ++w) {              // 1 次
                            auto a_desc = a_desc_base;
                            a_desc.lo = advance_umma_desc_lo<kMajorA, BLOCK_M, kSwizzleAMode, uint32_t>(
                                a_desc_stage_lo, w * LAYOUT_AD_M * BLOCK_K, k * UMMA_K_INT32);
                            uint32_t tmem_col = accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N + n * UMMA_N;
                            cute_mma_mxf4_t::fma(a_desc, b_desc, tmem_col,        // ★ 发射 MMA
                                                 k_iter > 0 or s > 0 or k > 0,    // scale_c：首条清零、其后累加
                                                 runtime_desc_k,
                                                 tmem_sfa_base + w * (...),       // SFA 在 TMEM 的列
                                                 tmem_sfb_k);                     // SFB 在 TMEM 的列
                        }
                    }
                }
                empty_barrier_arrive(s, is_last_iter and s == kNumInnerStages - 1);  // 通知 warp0：缓冲可重用
            }
        });
    });
}
```

讲解（逐个硬件点）：

- **① UTCCP（`cute_utccp_t::copy`）**：把转置好的 SF 从 smem 拷进 TMEM 的指定列
  （`kTmemStartColOfSFA/B`，见 5.4）。`SM100_UTCCP_4x32dp128bit` 是一条专门的 tcgen05 拷贝指令，
  一次搬 4×32 的 128-bit 块。**为什么 SF 非得进 TMEM？** 因为 block-scaled MMA 指令的语法是
  `[...], [tmem_sfa], [tmem_sfb]`——缩放因子操作数是用 **TMEM 地址**给的（见下方 PTX）。注释强调 UTCCP
  必须和 MMA 在同一个 warp（warp 1）串行执行，因为 SF 的 TMEM 列在各 stage 间复用。

- **② `advance_umma_desc_lo`**（`mma/sm100.cuh:88-92`）：在不重建整个描述符的前提下，**只更新 smem
  地址低位**，把 A/B 描述符推进到当前 stage `s`、第 `k` 个 K 子步、第 `n` 个 N 切片对应的 smem 位置。
  这是热路径上的轻量寻址。

- **③ `cute_mma_mxf4_t::fma(a_desc, b_desc, tmem_col, scale_c, runtime_desc, tmem_sfa, tmem_sfb)`**：
  发射一条块缩放 MMA。它最终落到的 PTX 就是（`ptx/tcgen05.cuh:118-140`）：

```cpp
"tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], p;"
//                                                        累加器  A   B  指令描述  SFA   SFB  谓词
```

  - `[%0]` = `tmem_c`：累加器在 TMEM 的列地址——结果**原地累加**到这里。
  - `%1, %2` = A、B 的 **smem 描述符**（`a_desc`、`b_desc`）。
  - `%3` = 指令描述符高 32 位（含 `sf_id`、M/N/K、类型）。
  - `[%5], [%6]` = SFA、SFB 在 **TMEM** 的地址。
  - `p` = 谓词，由 `scale_c` 决定：`setp.ne.b32 p, scale_c, 0`。**第一条 MMA 时 `scale_c=0` → p=false →
    累加器被覆写（清零起算）；之后 `scale_c≠0` → p=true → 在原有累加值上继续加**。代码里那个布尔
    `k_iter > 0 or s > 0 or k > 0` 正是"是不是第一条 MMA"的判断。
  - `.block_scale.block32` 表示**每 32 个元素一个缩放因子**（即 VS=32），硬件在做点积时自动用 SFA/SFB
    把对应块的部分和缩放回真实尺度。

> ⚙️ **硬件细节：tcgen05 MMA 如何"同时吃 FP4 操作数 + UE8M0 缩放"？**
> 第 5 代张量核（tcgen05）原生支持 MXFP4：一条 `mma.kind::mxf4.block_scale` 指令在硬件内部做的是——
> 把 A、B 的 FP4 码各自**乘上所属 32 元素块的 UE8M0 缩放因子**（即 `× 2^e`，纯指数移位，几乎零成本），
> 再做 64 长度的 K 点积，结果**累加进 TMEM 的 fp32 累加器**。整条指令是**异步**的：warp 1 单线程发射后
> 立即返回，张量核在后台算。所以才需要 `tcgen05_after_thread_sync()` / `umma_arrive` 这套 fence 与
> barrier 来界定"何时累加器算完、可以被 epilogue 读"。`cta_group::1` 是单 CTA；`::2` 版本让 cluster
> 里两个 CTA 协同算 M=256 的更大 MMA（对应 m≥512 的多播路径）。

- **④ `empty_barrier_arrive`**（`...cuh:363-375`）：MMA 发完后，用 `cutlass::arch::umma_arrive` 让
  `empty_barriers[s]` arrive——告诉 warp 0"这个 smem 缓冲我用完了，可以搬下一块覆盖了"。注意它用的是
  **`umma_arrive` 而非普通 arrive**：因为 MMA 异步，必须等张量核真正读完 smem 操作数后才能让 barrier
  放行，`umma_arrive` 会把这个 arrive 排在 MMA 之后。最后一条 K 迭代时还会顺带 arrive
  `tmem_full_barriers`，通知 epilogue"累加器算完了"。

### 6.5 warp 3+：epilogue（TMEM → smem → 写回 D）

源码（非 swap 路径 `...cuh:647-719`，核心）：

```cpp
} else if (warp_idx >= kNumNonEpilogueThreads / 32) {
    ...
    while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
        dispatch_accum_stage_idx(scheduler.current_iter % kNumEpilogueStages, [&](uint32_t accum_stage_idx) {
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            if (epilogue_thread_idx == 0) cute::tma_store_wait<0>();          // 等上一次 TMA 写回完成
            cutlass::arch::NamedBarrier(kNumEpilogueThreads).sync();
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);       // 等 MMA 把累加器算完
            tcgen05_after_thread_sync();
            for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                for (uint32_t s = 0; s < BLOCK_N / STORE_BLOCK_N; ++ s) {
                    ... // 算 swizzle 后的 smem 地址 / 全局 m_idx, n_idx
                    uint32_t tmem_addr = accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N + s * STORE_BLOCK_N + ...;
                    cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr, values[0], values[1], values[2], values[3]); // 读 TMEM
                    cutlass::arch::fence_view_async_tmem_load();
                    st_shared(smem_ptr, values[0], values[1], values[2], values[3]);    // 写 smem（swizzle）
                    if (最后一个分块) { tcgen05_before_thread_sync(); tmem_empty_barriers[accum_stage_idx]->arrive(0u); }
                    __syncwarp();
                    cute::tma_store_fence();
                    cutlass::arch::NamedBarrier(kNumEpilogueThreads).sync();
                    if (epilogue_thread_idx == 0) {
                        using cute_tma_t = cute::conditional_t<kWithAccumulation,
                            cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;       // 累加 or 覆写
                        cute_tma_t::copy(&tensor_map_d, smem_cd[tma_stage_idx], n_idx, m_idx);   // TMA 写回 D
                        cute::tma_store_arrive();
                    }
                }
            }
        });
    }
    ...
    if (epilogue_warp_idx == 1) Allocator().free(0, kNumTmemCols);    // 释放 TMEM
}
```

讲解：
- **`SM100_TMEM_LOAD_32dp32b4x::copy`**：用 tcgen05 的 load 指令把累加器从 **TMEM 读回寄存器**
  （`values[0..3]`）。`32dp32b4x` = 32 行（datapath）× 32-bit × 4 列一次。
- **写回前先进 smem 再 TMA store**：寄存器里的结果先 `st_shared` 写进 `smem_cd`（带 swizzle 消 bank 冲突），
  再由 0 号线程发 **TMA store** 把整块从 smem 异步写回显存 D。`tma_store_fence` / `NamedBarrier` 保证
  smem 写对 TMA 可见、且全 epilogue warp 同步。
- **`SM90_TMA_STORE_2D` vs `SM90_TMA_REDUCE_ADD_2D`**：本测试 `kWithAccumulation=false`，用前者**直接覆写**
  D；若要累加（`D += A@Bᵀ`），用后者让 TMA 在写回时做**原子加**到 D 上（这正是第 3 章 `early_return`
  把 C 预拷进 D 后、这里再加结果的配合）。
- **`tmem_empty_barriers->arrive`**：累加器读完后通知 warp 1"这份 TMEM 累加器空了，可以算下一个 tile 了"。
  配合 5.4 的双缓冲（`kNumEpilogueStages=2`），epilogue 读这份的同时 MMA 能写另一份。
- 最后 `Allocator().free` 释放 TMEM（对应 5.5 的 allocate）。

> ⚙️ **硬件细节：为什么 epilogue 要"TMEM→寄存器→smem→显存"绕一圈？**
> 累加器在 TMEM，最终要落到显存 D，但**不能从 TMEM 直接 TMA 到显存**。所以路径是：tcgen05.ld 把
> TMEM 读进寄存器 → 在寄存器里按输出布局重排 → st.shared 写进 smem（顺便 swizzle）→ TMA store 从 smem
> 异步搬到显存。中间过 smem 是为了用 TMA 的高效批量写回、并把"按 TMEM 布局"转成"按 D 的行优先布局"。
> swap-AB 路径（`...cuh:543-646`，本测试不走）原理类似，只是 TMEM 里存的是 Dᵀ，按 16 行 M 切片写回以
> 跳过 padding 行。

### 6.6 一个输出 tile 的完整时序回顾

把 4 个工位串起来，算一个输出 tile（`128 × BLOCK_N`）的过程是：

1. **warp 0** 沿 K 把 A/B/SFA/SFB 一个 stage 一个 stage 地 TMA 进 smem，用 `full_barrier` 报到。
2. **warp 2** 等到 `full_barrier`，把该 stage 的 SF 在 smem 里 warp 转置，用 `with_sf_full_barrier` 报到。
3. **warp 1** 等到 `with_sf_full_barrier`，先 UTCCP 把 SF 拷进 TMEM，再发 4 条 `tcgen05.mma.mxf4`
   把这 stage 的部分积累加进 TMEM；用 `empty_barrier` 放行 warp 0 复用缓冲。K 全部算完后用
   `tmem_full_barrier` 通知 epilogue。
4. **warp 3+** 等到 `tmem_full_barrier`，把 TMEM 累加器读出 → smem → TMA 写回 D；用 `tmem_empty_barrier`
   放行 warp 1 复用累加器。

整个过程 4 个工位**重叠流水**：warp 0 已经在搬下一个 stage，warp 1 还在算这个 stage，warp 3+ 在写上一个
tile——这就是这类内核能逼近硬件峰值的原因。

---

## 第 7 章 回到测试——校验与基准

内核跑完，`d` 里就是结果。`test_gemm()` 最后做两件事：**校验精度**和**测性能**。

### 7.1 `calc_diff`：精度校验

源码（`deep_gemm/testing/numeric.py:5-11`）：

```python
def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()                  # 都升到 fp64，避免统计时再引入误差
    denominator = (x * x + y * y).sum()            # 分母：Σ(x²+y²)
    if denominator == 0:                           # 两者全 0 → 视为完全一致
        return 0.0
    sim = 2 * (x * y).sum() / denominator          # 相似度：2·Σ(xy) / Σ(x²+y²)
    return 1 - sim                                 # 返回“1 − 相似度”，越小越准
```

**作用**：衡量内核输出 `d` 与全精度参考 `ref_d` 的接近程度。**入参**：两个同形状张量（这里 `(128,2112)`）。
**返回**：一个标量 `float`，0 表示完全一致。

讲解：这个 `sim = 2Σxy / Σ(x²+y²)` 不是普通余弦相似度，但很接近。
- 普通余弦相似度是 `Σxy / (‖x‖·‖y‖)`。这里用的是 `2Σxy / (‖x‖²+‖y‖²)`。
- 当 `x ≈ y` 时，分母 `‖x‖²+‖y‖² ≈ 2‖x‖²`，分子 `2Σxy ≈ 2‖x‖²`，于是 `sim → 1`、`diff → 0`。
- 它对**幅度和方向都敏感**（既惩罚成比例缩放、也惩罚方向偏差），比纯余弦更严格，适合量化误差评估。
- `test_gemm()` 里 `assert diff < 0.02`（FP4×FP4 的 `max_diff`，见 1.4）——这就是判定内核"算对了"的标准。

### 7.2 `bench_kineto`：性能基准

源码（`deep_gemm/testing/bench.py:79-146`，节选核心）：

```python
def bench_kineto(fn, kernel_names, num_tests: int = 30, suppress_kineto_output: bool = False, ...):
    ...
    fn()                                            # 先跑一次：触发 JIT 编译 + 预热
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():                                # 可选地屏蔽 profiler 的输出
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)   # 1 轮预热 + 1 轮采样
        profiler = torch.profiler.profile(activities=[ProfilerActivity.CUDA], schedule=schedule, acc_events=True)
        with profiler:
            for i in range(2):                      # 两轮（对应 warmup=1, active=1）
                for _ in range(num_tests):          # 每轮跑 num_tests=30 次
                    if flush_l2:
                        torch.empty(flush_l2_size, dtype=torch.int, device='cuda').zero_()  # 8GB memset 清 L2
                    fn()                            # 跑一次待测内核
                torch.cuda.synchronize()
                profiler.step()
    # —— 解析 profiler 表格，挑出名字含 kernel_names 的那一行，取平均 GPU 时间 ——
    prof_lines = profiler.key_averages().table(sort_by='cuda_time_total', ...).split('\n')
    ...
    for name in kernel_names:                       # name = 'sm100_fp4_gemm'
        for line in prof_lines:
            if name in line:                        # 子串匹配
                time_str = line.split()[-2]; num_str = line.split()[-1]
                ... total_time += float(...) / scale * int(num_str); total_num += int(num_str)
        kernel_times.append(total_time / total_num if total_num > 0 else 0)
    return tuple(kernel_times) if is_tuple else kernel_times[0]   # 返回平均每次耗时（秒）
```

**作用**：用 PyTorch 的 Kineto profiler 实测某个内核的平均 GPU 执行时间。**入参**：`fn`（无参可调用，
即第 1 章那个 lambda）、`kernel_names`（要统计的内核名子串，这里 `'sm100_fp4_gemm'`）。
**返回**：平均每次调用的 GPU 时间（**秒**）。

讲解几个要点：
- **`fn()` 先调一次**：第一次会触发第 4 章的 JIT 编译（很慢），且预热缓存。所以正式计时前先"空跑"一次，
  避免把编译时间算进去。
- **`kernel_names='sm100_fp4_gemm'` 子串过滤**：profiler 会记录这段时间内所有 CUDA 内核的耗时；这里只
  把**名字里含 `sm100_fp4_gemm` 的那一行**挑出来——也就是我们 JIT 出来的那个内核（第 4.1 节
  `compiler->build("sm100_fp4_gemm_1d1d", code)` 决定了内核名）。这样就排除了 L2 flush 的 memset 等
  无关内核。`assert sum([name in line ...]) <= 1` 确保只匹配到一行，避免歧义。
- **L2 flush（8GB memset）**：每次测量前把 8GB 显存清零，**冲掉 L2 缓存**，保证每次内核都从"冷缓存"
  开始，测出的是稳定的、不被上一次残留缓存美化的性能。
- **`suppress_kineto_output=True`**：用 `suppress_stdout_stderr`（`bench.py:44-77`，通过 `os.dup2`
  重定向文件描述符）把 profiler 自己的打印吞掉，让测试输出干净。
- **schedule(warmup=1, active=1)**：profiler 跑两轮，第一轮 warmup 不计、第二轮 active 才统计，进一步
  排除冷启动抖动。

### 7.3 `count_bytes` 与性能公式

源码（`deep_gemm/testing/numeric.py:14-21`）：

```python
def count_bytes(*tensors):
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):           # 是元组/列表 → 递归（处理 (packed, sf) 对）
            total += count_bytes(*t)
        elif t is not None:                        # 跳过 None
            total += t.numel() * t.element_size()  # 元素个数 × 每元素字节数
    return total
```

**作用**：递归累加若干张量的总字节数。**入参**：任意多个张量/元组/None。**返回**：总字节数 `int`。

讲解：`a`、`b` 是 `(packed_fp4, sf)` 元组，`count_bytes(a, b, d)` 通过 `isinstance(t, (tuple,list))`
递归把元组里两个张量的字节都算上。`numel()` 是元素总数、`element_size()` 是每元素字节数（int8=1, fp32=4）。

> 💡 **Python 语法：`def count_bytes(*tensors)` 与 `count_bytes(*t)`**
> 形参 `*tensors` 是**可变位置参数**：调用 `count_bytes(a, b, d)` 时，`tensors` 收集成元组 `(a,b,d)`。
> 调用处 `count_bytes(*t)` 里的 `*t` 是**解包**：把元组 `t` 的元素**展开**成多个独立实参再传进去。
> 一个 `*` 在定义处是"收集"、在调用处是"展开"，方向相反。

回到 `test_gemm()` 的打印行（`tests/test_fp4.py:46-48`）：

```python
print(f' > Perf (...): '
      f'{t * 1e6:6.1f} us | {2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
      f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
```

- **耗时**：`t * 1e6` 把秒转微秒。
- **TFLOPS**：矩阵乘的浮点运算数约为 `2·m·n·k`（每个输出元素是 k 次乘加 = 2k flops，共 m·n 个）。
  `2mnk / t` 是每秒浮点运算数，`/ 1e12` 转成 TFLOPS。
- **GB/s**：`count_bytes(a,b,d)` 是这次 GEMM 读写的总字节，`/ 1e9 / t` 转成 GB/s，反映**带宽利用率**。
  FP4 的卖点之一就是字节少、这个数能很高。

至此，`test_gemm()` 的一个用例（造数据 → 调内核 → 校验 → 测速 → 打印）就完整闭环了。

---

## 附录

### A. 名词速查表

| 缩写 | 全称 | 一句话 |
|------|------|--------|
| FP4 / E2M1 | 4-bit float (2-exp, 1-mantissa) | 8 个幅值 `{0,.5,1,1.5,2,3,4,6}` + 符号，占 4 bit |
| MXFP4 | Microscaling FP4 | 每 32 个 FP4 共享一个缩放因子（OCP 标准） |
| UE8M0 | Unsigned E8M0 | 纯指数缩放因子，本质是 `2^e` |
| VS | Vector Size | 块缩放的块大小，本文 = 32 |
| SF / SFA / SFB | Scale Factor | 缩放因子；A 的 / B 的 |
| TMA | Tensor Memory Accelerator | 异步多维 gmem↔smem 拷贝引擎（描述符驱动） |
| TMEM | Tensor Memory | Blackwell 给张量核当累加器的片上内存（128×512） |
| tcgen05 | 5th-gen Tensor Core 指令 | 异步 MMA，结果写 TMEM；`kind::mxf4` 支持 FP4 块缩放 |
| UMMA | "U" + MMA | CUTLASS 对 tcgen05 MMA 的封装，smem 描述符喂操作数 |
| UTCCP | — | 把 SF 从 smem 拷进 TMEM 的 tcgen05 拷贝指令 |
| mbarrier | memory barrier | smem 异步屏障，带相位，用于生产者-消费者同步 |
| swizzle | — | smem 数据按异或重排，消 bank 冲突 / 配合 TMA-MMA 布局 |
| CTA / cluster | Cooperative Thread Array / Cluster | 线程块 / 一组协同 MMA、共享 smem 的 CTA |
| swap_ab | — | 交换 A/B 角色（稀疏 MoE 用，dense 不用） |

### B. 关键换算链（本测试 `m=128,n=2112,k=7168`）

```
K 维：  7168 FP4  =  896 int32(=k/8)  =  224 个 SF(=k/32)
1 个 K 块(stage)：  BLOCK_K=32 int32  =  256 FP4  =  8 个 SF
1 条 MMA：          UMMA_K_FP4=64 FP4  =  8 int32  =  2 个 SF
一个 stage 发的 MMA 条数： 256/64 = 4  (= NUM_K_ITERS_PER_STAGE)
打包：  2 个 FP4 / int8 字节（Python）；8 个 FP4 / int32（内核）；4 个 SF / int32（C++）
```

### C. 端到端数据流图

```
  bf16 A (128,7168)                          bf16 B (2112,7168)
        │ per_token_cast_to_fp4(gran_k=32)         │
        ▼                                          ▼
  packed int8 (128,3584) + SFA f32 (128,224)   packed int8 (2112,3584) + SFB f32 (2112,224)
        │                                          │           ┌─ deep_gemm.fp8_fp4_gemm_nt
        │   ┌──────────────────────────────────────┘           │   (= _C, pybind → C++)
        ▼   ▼                                                   ▼
  C++ transform_sf_into_required_layout:  SFA/SFB  f32 ──► packed int32 UE8M0, MN-major, TMA 对齐
        │                                                       │
        ▼  sm100_fp4_gemm_1d1d (host): pick tile / stages / 5×TMA desc / JIT(NVCC)
        ▼
  CUDA kernel sm100_fp4_gemm_1d1d_impl:
        warp0 TMA(A,B,SFA,SFB)→smem ─full→ warp2 SF 转置 ─sf_full→ warp1 UTCCP(SF→TMEM)+tcgen05.mma.mxf4→TMEM
                                                                          │ tmem_full
                                                                          ▼
                                            warp3+ epilogue: TMEM→reg→smem→TMA store→ D (128,2112) f32
        │
        ▼
  calc_diff(d, ref_d) < 0.02 ?     bench_kineto('sm100_fp4_gemm') → us / TFLOPS / GB/s
```

### D. 路径速查（文件 → 职责）

| 文件 | 职责 |
|------|------|
| `tests/test_fp4.py` | 测试入口 `test_gemm()` |
| `tests/generators.py` | `QuantConfig` / `generate_normal` / `cast_fp8_fp4_with_major` |
| `deep_gemm/utils/math.py` | `per_token_cast_to_fp4` / E2M1 / UE8M0 量化 |
| `deep_gemm/__init__.py` | 把 `_C` 里的 `fp8_fp4_gemm_nt` 暴露成 Python 名 |
| `csrc/apis/gemm.hpp` | C++ `fp8_fp4_gemm_nt`：检查 / `early_return` / 分发 |
| `csrc/apis/layout.hpp` | `transform_sf_into_required_layout`：SF 布局变换 |
| `csrc/jit_kernels/impls/sm100_fp4_gemm_1d1d.hpp` | host 装配 + JIT 代码生成 |
| `deep_gemm/include/deep_gemm/impls/sm100_fp4_gemm_1d1d.cuh` | CUDA 内核本体 |
| `deep_gemm/include/deep_gemm/mma/sm100.cuh` | UMMA / SF 描述符辅助 |
| `deep_gemm/include/deep_gemm/ptx/tcgen05.cuh` | `tcgen05.mma.kind::mxf4` 等 PTX 封装 |
| `deep_gemm/include/deep_gemm/common/tma_utils.cuh` | `tma_copy` 等 TMA 封装 |
| `deep_gemm/testing/{numeric,bench}.py` | `calc_diff` / `count_bytes` / `bench_kineto` |

