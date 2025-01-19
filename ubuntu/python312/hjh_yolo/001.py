import torch
import platform

def check_environment():
    """Check Python, PyTorch, CUDA environments and log info."""
    print("=== Environment Check ===")
    print(f"Python version : {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Platform       : {platform.platform()}")

    # CUDA 관련 체크
    if torch.cuda.is_available():
        print("\n[CUDA INFO]")
        print(f"CUDA is available with {torch.cuda.device_count()} GPU(s).")
        for i in range(torch.cuda.device_count()):
            print(f" - Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("\n[CUDA INFO]")
        print("CUDA is NOT available. Check GPU drivers or CUDA installation.")

if __name__ == "__main__":
    check_environment()
