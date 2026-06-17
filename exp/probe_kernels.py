import sys, torch
sys.path.insert(0,'/root/workspace/gb_gemm_benchmark/DeepGEMM'); sys.path.insert(0,'/root/workspace/gb_gemm_benchmark/DeepGEMM/tests')
import deep_gemm
from generators import KernelType, MajorTypeAB, QuantConfig, generate_normal
from torch.profiler import profile, ProfilerActivity
FP4=QuantConfig((32,32,True,True))
m=n=k=4096
a,b,c,d,_=generate_normal(m,n,k,MajorTypeAB.KMajor,MajorTypeAB.KMajor,accumulate=False,out_dtype=torch.float,kernel_type=KernelType.Kernel1D1D,use_ue8m0=True,quant_config=FP4)
recipe,ra,rb=FP4.get_recipes()
def gemm(): deep_gemm.fp8_fp4_gemm_nt(a,b,d,c=c,disable_ue8m0_cast=False,recipe=recipe,recipe_a=ra,recipe_b=rb)
for _ in range(3): gemm()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(3): gemm()   # 3 次调用 → 每个 kernel 应出现 3 次
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by='cuda_time_total', max_name_column_width=160, row_limit=30))
