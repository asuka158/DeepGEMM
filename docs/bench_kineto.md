# `test_fp4_my.py` 与 `bench_kineto` 讲解

本文分两部分：

- **第一部分**：概览 `tests/my_test/test_fp4_my.py` 这个基准脚本的结构与数据流。
- **第二部分**：详细讲 `bench_kineto`——它到底**怎么测 GPU kernel 性能**。

约定：调用链只追到 Python 层的 `deep_gemm.fp8_fp4_gemm_nt`（pybind 入口）为止，再往下的 C++/CUDA 与计时逻辑无关，不展开；Python 的特殊语法在第一次出现处**行内讲解**。代码引用都带 `文件:行号`。

---

# 第一部分：`test_fp4_my.py` 脚本概览

## 1. 脚本在干什么

对 `shape.txt` 里每个 `(m, n, k)`：生成 bf16 输入并量化成 MXFP4 → 跑一次 FP4×FP4 稠密 GEMM → 和 fp32 参考结果对精度 → 用 `bench_kineto` 计时 → 把结果写进 CSV。

脚本被重构成了一组小函数，数据流如下：

```
main()                                         # 总调度
  └─ 对每个 shape 调用 bench_shape(m,n,k,recipe...)
        ├─ generate_normal(...)      → a, b, c, d, ref_d   # 造输入（来自 tests/generators.py）
        ├─ run_gemm(a,b,c,d,...)     → 写入 d              # 跑一次 GEMM
        ├─ check_correctness(d,ref_d)→ (diff, ok)          # 对精度
        └─ benchmark(m,n,k,...)      → (time_us,tflops,gbps)
              └─ bench_kineto(lambda: run_gemm(...), 'sm100_fp4_gemm')   ← 第二部分主角
  └─ write_csv(path, rows)                                 # 落盘
```

## 2. 模块级准备

`test_fp4_my.py:26-47`：

- `sys.path.insert(0, _p)`（L32-34）：把仓库根目录和 `tests/` 插到模块搜索路径最前面，这样 `import deep_gemm` 和 `from generators import ...` 不管在哪个目录运行都能找到。
- 导入：`deep_gemm`、以及 `from deep_gemm.testing import bench_kineto, calc_diff, count_bytes`（L37）。
- 三个常量（决定本次测的量化格式与精度）：
  - `FP4_FP4 = QuantConfig((32, 32, True, True))`（L44）：两个操作数都用 packed-FP4，scale 块大小 32（VS=32 的 UE8M0）。
  - `OUT_DTYPE = torch.float`（L46）：输出 `d` 必须是 fp32（FP4×FP4 kernel 不支持 bf16 写回）。
  - `KERNEL_TYPE = KernelType.Kernel1D1D`（L47）：用 1D1D 调度的 kernel。

## 3. 逐函数速览

**`read_shapes(path)`**（L50-59）— 读 `shape.txt`，返回 `[(m,n,k), ...]`。

```python
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line or line.lower().startswith('m'):   # 跳过空行/表头
            continue
        m, n, k = (int(x) for x in line.split())
        shapes.append((m, n, k))
```
- `with open(path) as f:`：上下文管理器，块结束自动关文件（详见第二部分第 5 节）。
- `(int(x) for x in line.split())` 是**生成器表达式**：把一行按空白切开、逐个转 int。`m, n, k = ...` 是**解包赋值**，把三个值一次性分给三个变量。

**`run_gemm(a, b, c, d, recipe, recipe_a, recipe_b)`**（L62-65）— 跑一次 GEMM，结果原地写进 `d`。

```python
deep_gemm.fp8_fp4_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=False,
                          recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
```
- `deep_gemm.fp8_fp4_gemm_nt` 是**统一的 FP4/FP8 GEMM API 入口**（来自编译扩展 `_C` 的 pybind 函数，**调用链到此为止**）。本例两个操作数都是 packed-FP4，所以它内部会分发到纯 FP4×FP4 的 `sm100_fp4_gemm_1d1d`（其 kernel 名含 `sm100_fp4_gemm`，正好对应第二部分计时时的名字过滤）。
- 入参里 `a`、`b` 各是一个 `(数据, scale)` **二元组**；`d` 是 `(m,n)` 的 fp32 输出缓冲；`c=None`（不做累加）。

**`check_correctness(d, ref_d)`**（L68-70）— 返回 `(diff, ok)`。
```python
diff = calc_diff(d, ref_d)
return diff, diff < FP4_FP4.max_diff()
```
`calc_diff` 算 `d` 与 fp32 参考 `ref_d` 的差异度（越小越准）；`ok` 是「是否小于阈值」的布尔。

**`benchmark(m, n, k, a, b, c, d, recipe, recipe_a, recipe_b)`**（L73-79）— **本脚本通往第二部分的桥**。
```python
t = bench_kineto(
    lambda: run_gemm(a, b, c, d, recipe, recipe_a, recipe_b),
    'sm100_fp4_gemm', suppress_kineto_output=True)
tflops = 2 * m * n * k / t / 1e12
gbps = count_bytes(a, b, d) / 1e9 / t
return t * 1e6, tflops, gbps
```
- 把「跑一次 GEMM」包成 `lambda`（匿名函数，加 `()` 才执行）交给 `bench_kineto`，它返回**单次平均耗时 `t`（秒）**。
- 用 `t` 换算 `tflops`（`2*m*n*k` 是 GEMM 的浮点运算次数）和 `gbps`（`count_bytes` 统计 `a,b,d` 读写字节数）；`t*1e6` 把秒转成微秒。

**`write_csv(path, rows)`**（L82-89）— 把所有结果写成 CSV。`with open(path, 'w', newline='')` 开文件，`csv.writer` 写表头和每行；`f"{r['time_us']:.3f}"` 是 **f-string** 的格式化，`:.3f` 表示保留 3 位小数。

**`bench_shape(m, n, k, recipe, recipe_a, recipe_b)`**（L92-103）— 单个 shape 的完整流程，把上面几步串起来：`generate_normal` 造输入 → `run_gemm` → `check_correctness` → `benchmark`，最后用 `dict(m=m, n=n, ...)` 把这一行结果打包成字典返回。

**`main()`**（L106-130）— 固定随机种子 → `read_shapes` → `FP4_FP4.get_recipes()` 拿量化配方 → `for idx, (m, n, k) in enumerate(shapes)` 逐个 `bench_shape` 并打印 → `write_csv` 落盘。
- `enumerate(shapes)` 同时给出下标 `idx` 和元素 `(m,n,k)`。
- `sum(r['correct'] for r in rows)`：对布尔求和（`True=1`）数出通过的 shape 数。

## 4. 输入张量一览（`generate_normal` 的产物）

| 变量 | shape | dtype | 含义 |
|------|-------|-------|------|
| `a` | 二元组 `(data, scale)` | — | A 操作数：`data` 是 `(m, k)` packed-FP4，`scale` 是 UE8M0 块缩放 |
| `b` | 二元组 `(data, scale)` | — | B 操作数：`data` 是 `(n, k)` packed-FP4 |
| `c` | — | — | `None`（`accumulate=False`，不做 `D += C`） |
| `d` | `(m, n)` | `torch.float`(fp32) | 输出缓冲，kernel 把结果写这里 |
| `ref_d` | `(m, n)` | fp32 | 高精度参考，用于 `check_correctness` |

> 为什么 `a/b` 是二元组：FP4 是块缩放量化，除压缩数据外还要带一组缩放因子，所以打包成 `(数据, scale)`。为什么 `d` 是 fp32：FP4×FP4 这条 kernel 没有 bf16 epilogue。

---

# 第二部分：`bench_kineto` 详解

源码：`deep_gemm/testing/bench.py:79-146`。

## 1. 概览：它怎么测性能

一句话：**用 PyTorch 自带的 Kineto profiler（`torch.profiler`）抓 GPU 上每个 CUDA kernel 的真实执行时间，把待测函数重复跑很多次，从汇总表里读出目标 kernel 的「平均单次耗时」，以「秒」返回。**

为什么不用 `time.time()` 掐表？因为 GPU kernel 是**异步**的：Python 调用只是把 kernel「提交」给 GPU 就立刻返回，真正计算还在 GPU 上跑。CPU 计时会把提交开销也算进去，量不准。Kineto 直接从 GPU 侧拿每个 kernel 的起止时间戳，测的是**纯 GPU 执行时间**。

下面按代码顺序拆成 6 步。

## 2. 签名与参数

```python
def bench_kineto(fn, kernel_names, num_tests: int = 30,
                 suppress_kineto_output: bool = False,
                 trace_path: str = None, flush_l2: bool = True,
                 with_multiple_kernels: bool = False,
                 barrier: Optional[Callable] = None):
```

行内语法：`num_tests: int = 30` 里，`: int` 是**类型注解**（只是提示，运行时不强制），`= 30` 是**默认值**；`Optional[Callable]` 表示「一个可调用对象，或 `None`」。

| 参数 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `fn` | 可调用 | 必填 | 待测函数，内部反复 `fn()` 调用。本例是那个跑 GEMM 的 `lambda`。 |
| `kernel_names` | `str` 或 `tuple` | 必填 | 要统计的 kernel 名（**子串匹配**）。本例传 `'sm100_fp4_gemm'`。 |
| `num_tests` | `int` | `30` | 每个测量窗口里 `fn()` 调用次数，越多越稳。 |
| `suppress_kineto_output` | `bool` | `False` | 是否把 profiler 期间的输出重定向到黑洞。 |
| `trace_path` | `str` | `None` | 给路径则导出 Chrome trace 文件。 |
| `flush_l2` | `bool` | `True` | 每次 `fn()` 前是否冲刷 L2 缓存，使每次计时条件一致。 |
| `with_multiple_kernels` | `bool` | `False` | 是否允许一个名字匹配多行 kernel。 |
| `barrier` | `Callable`/`None` | `None` | 多进程基准用的同步栅栏，单卡用不到。 |

## 3. 第 1 步：校验与工具避让

```python
assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
is_tuple = isinstance(kernel_names, tuple)

if int(os.environ.get('DG_USE_NVIDIA_TOOLS', 0)):
    return (1, ) * len(kernel_names) if is_tuple else 1
```
- `assert 条件`：不满足就抛错中断；`isinstance(x, T)` 判断类型。这里要求 `kernel_names` 是字符串或元组。
- `is_tuple` 记下「传的是不是元组」，决定最后返回单个数还是元组。
- `os.environ.get('DG_USE_NVIDIA_TOOLS', 0)`：取环境变量，取不到给默认 `0`（用 `.get` 而非 `[...]` 可避免 KeyError）。
- `return (1, ) * len(...) if is_tuple else 1` 是**三元表达式**（`A if 条件 else B`）：设了该环境变量（说明你正用 Nsight / Compute Sanitizer 等，会和 profiler 冲突）就跳过计时、返回占位值 1。`(1, )` 是单元素元组（逗号不能省），`元组 * n` 是元组重复。

## 4. 第 2 步：预热与 L2 配置

```python
flush_l2_size = int(8e9 // 4)   # 约 8GB 能放多少个 4 字节 int
fn()                            # 先空跑一次
```
- `8e9` 是科学计数法（`8×10⁹`，float）；`//` 是**整除**（向下取整）。
- 进 profiler **之前**先空跑一次 `fn()`：DeepGEMM 的 kernel 是**运行时即时编译（JIT）并缓存**的，第一次调用要现场编译、autotune——这些是一次性高开销。先做掉，profiler 测的才是稳定执行时间（这是**第一层预热**）。

## 5. 第 3 步：输出抑制（上下文管理器）

```python
suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
with suppress():
    ...
```
第一行用三元表达式**选一个类**（注意 `suppress` 此刻是类本身），下一行 `suppress()` 才实例化并用 `with` 启用。

**`with 对象:` 是上下文管理器**：进块前自动调对象的 `__enter__`，出块时自动调 `__exit__`——**即使块内报错，收尾也一定执行**。一个对象能被 `with` 用，就因为它实现了这两个方法。`empty_suppress`（L36-41）是「什么都不做」的占位版。

真正干活的 `suppress_stdout_stderr`（L44-76）用**文件描述符（fd）重定向**屏蔽输出：
- `os.devnull` 是系统「黑洞」（`/dev/null`），写进去即丢弃；`fileno()` 取流底层 fd（stdout 固定是 1）。
- `os.dup(fd)` 复制 fd 用来**备份**原 stdout/stderr；`os.dup2(src, dst)` 把 `dst` 改指向 `src`——把 fd 1 指向黑洞，于是所有 stdout 输出被丢弃。`__exit__` 再用备份 fd 还原。
- **为什么动 fd 而不是 `sys.stdout = ...`**：Kineto/CUDA 底层很多输出是 C/C++ 直接写 fd 的，绕过了 Python 的 `sys.stdout`，只有在 fd 层重定向才能挡住。

效果：profiler 工作期间终端干净，出块即恢复。

## 6. 第 4 步：配置 Kineto profiler

```python
schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
profiler = torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule, acc_events=True)
with profiler:
    ...
```
- `schedule(...)` 定义每一「步」profiler 的状态，每调一次 `profiler.step()` 推进一步：`wait=0` 不空等、`warmup=1` 预热 1 步（**数据丢弃**，让 GPU 升频/缓存就绪）、`active=1` 激活 1 步（**真正记录**）、`repeat=1` 只做一轮。配合第 7 节的两次循环：第一遍 warmup（丢），第二遍 active（采）。
- `profile(...)`：`activities=[...CUDA]` 只抓 GPU 活动（不抓 CPU，省开销）；`acc_events=True` 跨周期累计事件，保证汇总完整。`with profiler:` 进块启动 profiling、出块停止整理。

## 7. 第 5 步：测量主循环（核心）

```python
with profiler:
    for i in range(2):
        for _ in range(num_tests):
            if flush_l2:
                torch.empty(flush_l2_size, dtype=torch.int, device='cuda').zero_()
            if barrier is not None:
                torch.cuda._sleep(int(2e7))  # ~10ms
                barrier()
            fn()
        torch.cuda.synchronize()
        profiler.step()
```
- `for i in range(2)`：**两个窗口**。结合第 6 节的 schedule，第一遍是 warmup（数据丢弃，**第二层预热**，抹平时钟/缓存抖动），第二遍是 active（真正采集）。
- `for _ in range(num_tests)`：把 `fn()` 跑 `num_tests`（默认 30）次。`_` 是「循环但不在乎计数」的惯用变量名。在 active 窗口里目标 kernel 因此执行 30 次。
- 每次迭代做三件事：
  1. **冲刷 L2**：`torch.empty(...).zero_()` 在 GPU 上开约 8GB 并原地清零（方法名末尾 `_` 是 PyTorch 约定，表示**原地修改**）。这块大写操作把 L2 挤掉，让**每次** `fn()` 都从「缓存冷」的相同条件出发，避免上一次的数据残留让下一次偏快。
  2. **可选栅栏**：`if barrier is not None`（判空用 `is`/`is not`）——多进程基准才用，单卡跳过。
  3. **`fn()`**：真正跑一次被测 GEMM。它一路走到 `run_gemm` → `deep_gemm.fp8_fp4_gemm_nt`（pybind），见下一节。
- 窗口收尾：`torch.cuda.synchronize()` 让 CPU **阻塞等 GPU 把这一窗口所有 kernel 跑完**（因为 kernel 异步，必须等它真正结束时间戳才落定）；`profiler.step()` 推进 schedule 状态机（结束 warmup→进入 active；再 step 结束 active→采集完成）。

## 8. `fn()` 通向哪里（止于 pybind）

每次 `fn()` 的去向：
```
fn （lambda） → run_gemm(...) → deep_gemm.fp8_fp4_gemm_nt(...)   # 来自编译扩展 _C 的 pybind 函数
```
`deep_gemm.fp8_fp4_gemm_nt` 跨过 Python/C++ 边界后，再往下是 C++ 派发 → JIT 编译 → CUDA kernel 启动。但**那部分与 `bench_kineto` 的计时逻辑无关**，本文不展开——对计时而言，`fn()` 就是「让 GPU 跑一次目标 kernel」，profiler 在背后记录它的 GPU 耗时。

## 9. 第 6 步：解析表并算平均

退出 profiler 后，从汇总表里抽耗时。

```python
prof_lines = profiler.key_averages().table(sort_by='cuda_time_total', max_name_column_width=100).split('\n')
kernel_names = (kernel_names, ) if isinstance(kernel_names, str) else kernel_names
if not with_multiple_kernels:
    for name in kernel_names:
        assert sum([name in line for line in prof_lines]) <= 1, f'Errors ... {prof_lines}'
```
- `key_averages()` 把事件**按名字聚合**（同名 kernel 多次调用合并成一行，带总/平均时间、调用次数）；`.table(...)` 渲染成文本表；`.split('\n')` 按行切成列表 `prof_lines`。
- 若 `kernel_names` 是单字符串就 `(kernel_names, )` 包成单元素元组，方便后面统一 `for name in ...`。
- 唯一性断言：`[name in line for line in prof_lines]` 是**列表推导式**，对每行算布尔「是否含该名字」，得到形如 `[False, True, ...]` 的列表；`sum([...])` 对布尔求和（`True=1`）即匹配行数；`assert ... <= 1` 要求最多匹配一行。`f'...{prof_lines}'` 是 **f-string**（`{}` 内嵌变量值）。

```python
units = {'ms': 1e3, 'us': 1e6}
kernel_times = []
for name in kernel_names:
    total_time = 0
    total_num = 0
    for line in prof_lines:
        if name in line:
            time_str = line.split()[-2]
            num_str = line.split()[-1]
            for unit, scale in units.items():
                if unit in time_str:
                    total_time += float(time_str.replace(unit, '')) / scale * int(num_str)
                    total_num += int(num_str)
                    break
    kernel_times.append(total_time / total_num if total_num > 0 else 0)

return tuple(kernel_times) if is_tuple else kernel_times[0]
```
- `units` 是字典：键是单位、值是「换算成秒要除的倍数」（ms÷1e3、us÷1e6）。
- 命中目标名字的行：`line.split()` 按空白切成列；**负索引** `[-1]` 是最后一列（**调用次数**），`[-2]` 是倒数第二列（**平均单次耗时**，带单位如 `123.456us`）。
  > 列名随 PyTorch 版本略有差异，但「最后一列=次数、倒数第二列=带单位的平均时间」结构稳定。
- `for unit, scale in units.items()`（`.items()` 同时取键和值）找出是哪种单位：`time_str.replace(unit, '')` 去掉单位后缀、`float(...)` 转数、`/scale` 换算成秒、`*int(num_str)` 乘次数得该行总耗时，累加；`break` 命中一种单位即停。
- `total_time / total_num`（用三元表达式防除零）= **该 kernel 单次调用平均耗时（秒）**。先 `*次数` 再 `/总次数`，是为了在同名命中多行时做按次数加权平均（通常只命中一行，结果就是那一行的平均值）。
- 返回：传元组就返回耗时元组，否则返回**单个浮点数（秒）**。本例返回单个 `t`，供第一部分 `benchmark` 换算 TFLOPS/GB·s。
