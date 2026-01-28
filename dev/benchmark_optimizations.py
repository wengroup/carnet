import torch
import time
from carnet.core.unit_vector import Polyadics as PolyadicsOpt
from carnet.core.unit_vector_3 import Polyadics as PolyadicsRef
from carnet.module.product import TensorProduct as TPOpt
from carnet.module.product1 import TensorProduct as TPRef
from carnet.module.product2 import TensorProduct as TPOpt1

def benchmark_speed(func, *args, n_runs=100, **kwargs):
    # Warmup
    for _ in range(10):
        func(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(n_runs):
        func(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / n_runs

def get_memory(model):
    mem = 0
    for p in model.parameters():
        mem += p.element_size() * p.nelement()
    for b in model.buffers():
        mem += b.element_size() * b.nelement()
    return mem / 1024 / 1024  # MB

def run_benchmarks(L=2, B=10000, F=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmarks on {device}")
    
    a = torch.randn(B, 3).to(device)
    a /= a.norm(dim=-1, keepdim=True)
    
    # --- Polyadics Benchmark ---
    print(f"\n--- Polyadics Benchmark (L={L}, B={B}) ---")
    poly_ref = PolyadicsRef(L).to(device)
    poly_opt = PolyadicsOpt(L).to(device)
    
    t_ref = benchmark_speed(poly_ref, a)
    t_opt = benchmark_speed(poly_opt, a)
    
    m_ref = get_memory(poly_ref)
    m_opt = get_memory(poly_opt)
    
    print(f"Ref (unit_vector_3): {t_ref*1000:.4f} ms, Mem: {m_ref:.4f} MB")
    print(f"Opt (unit_vector):   {t_opt*1000:.4f} ms, Mem: {m_opt:.4f} MB")
    print(f"Speedup: {t_ref/t_opt:.2f}x")

    # --- TensorProduct Benchmark ---
    print(f"\n--- TensorProduct Benchmark (L1=L2=L3={L}, B={B}, F={F}) ---")
    T = (3**(L+1)-1)//2
    x = torch.randn(B, F, T).to(device)
    y = torch.randn(B, F, T).to(device)
    y_atomic = torch.randn(B, T).to(device)
    
    # Standard TP
    tp_ref = TPRef(F, L, L, L).to(device)
    tp_opt1 = TPOpt1(F, L, L, L).to(device)
    tp_opt = TPOpt(F, L, L, L).to(device)
    
    t_tp_ref = benchmark_speed(tp_ref, x, y)
    t_tp_opt1 = benchmark_speed(tp_opt1, x, y)
    t_tp_opt = benchmark_speed(tp_opt, x, y)
    
    m_tp_ref = get_memory(tp_ref)
    m_tp_opt1 = get_memory(tp_opt1)
    m_tp_opt = get_memory(tp_opt)
    
    print(f"Ref (product1): {t_tp_ref*1000:.4f} ms, Mem: {m_tp_ref:.4f} MB")
    print(f"Opt1 (product2): {t_tp_opt1*1000:.4f} ms, Mem: {m_tp_opt1:.4f} MB")
    print(f"Opt (product):  {t_tp_opt*1000:.4f} ms, Mem: {m_tp_opt:.4f} MB")
    print(f"Speedup Opt vs Ref: {t_tp_ref/t_tp_opt:.2f}x")

    # Atomic Moment TP
    tp_ref_am = TPRef(F, L, L, L, for_atomic_moment=True).to(device)
    tp_opt1_am = TPOpt1(F, L, L, L, for_atomic_moment=True).to(device)
    tp_opt_am = TPOpt(F, L, L, L, for_atomic_moment=True).to(device)
    
    # Weights R
    R = {}
    for l3 in tp_opt_am.L3:
        for p in tp_opt_am.paths[l3]:
            R[str(p)] = torch.randn(B, F).to(device)
            
    t_am_ref = benchmark_speed(tp_ref_am, x, y_atomic, R=R)
    t_am_opt1 = benchmark_speed(tp_opt1_am, x, y_atomic, R=R)
    t_am_opt = benchmark_speed(tp_opt_am, x, y_atomic, R=R)
    
    print(f"\nAtomic TP Ref (product1): {t_am_ref*1000:.4f} ms")
    print(f"Atomic TP Opt1 (product2): {t_am_opt1*1000:.4f} ms")
    print(f"Atomic TP Opt (product):  {t_am_opt*1000:.4f} ms")
    print(f"Speedup Opt vs Ref: {t_am_ref/t_am_opt:.2f}x")

if __name__ == "__main__":
    # Benchmark parameters:
    # L: Maximum rank of the natural tensors
    # B: Batch size (number of points/configurations)
    # F: Feature (channel) dimension
    L = 2
    B = 10000
    F = 128
    run_benchmarks(L=L, B=B, F=F)
