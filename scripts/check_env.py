"""Step 1.2 — Verify Taichi + GPU detection.

Run:
    python3 scripts/check_env.py
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import taichi as ti
    except ImportError:
        print("[FAIL] taichi not installed. Run: pip install -r requirements.txt")
        return 1

    print(f"Taichi version: {ti.__version__}")

    # Try CUDA first; fall back to CPU.
    try:
        ti.init(arch=ti.cuda)
        backend = "CUDA"
    except Exception as e_cuda:
        print(f"[INFO] CUDA init failed ({e_cuda}); falling back to CPU.")
        ti.init(arch=ti.cpu)
        backend = "CPU"
    print(f"Backend: {backend}")

    @ti.kernel
    def hello():
        for i in range(4):
            print("hello from kernel", i)

    hello()
    ti.sync()
    print("[OK] Taichi kernel executed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
