import torch
import time

def verify_gpu():
    print("=" * 50)
    print("GPU Verification Test")
    print("=" * 50)
    
    # Basic GPU Info
    print("\n1. System Information:")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**2:.0f}MB")
    
    # Performance Test
    print("\n2. Performance Test:")
    size = 4000
    print(f"Testing {size}x{size} matrix multiplication")
    
    # CPU Test
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    
    start = time.time()
    c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.time() - start
    
    # GPU Test
    if torch.cuda.is_available():
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()
        
        # Warmup
        c_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        start = time.time()
        c_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"\nResults:")
        print(f"CPU Time: {cpu_time:.2f} seconds")
        print(f"GPU Time: {gpu_time:.2f} seconds")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")

if __name__ == "__main__":
    verify_gpu()