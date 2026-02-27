import platform
import sys

def main() -> None:
    print("Python:", sys.version.replace("\n", " "))
    print("Executable:", sys.executable)
    print("Platform:", platform.platform())

    try:
        import numpy as np
        print("NumPy:", np.__version__)
    except Exception as e:
        print("NumPy: not installed:", repr(e))

    try:
        import torch
        print("Torch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
    except Exception:
        print("Torch: not installed")

if __name__ == "__main__":
    main()