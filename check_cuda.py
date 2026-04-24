import torch


def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("CUDA is NOT available — exiting.")
        return

    print(f"CUDA version (built against): {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            f"  [{i}] {props.name} | "
            f"compute {props.major}.{props.minor} | "
            f"{props.total_memory / 1024**3:.1f} GiB"
        )

    device = torch.device("cuda:0")
    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    c = a @ b
    torch.cuda.synchronize()
    print(f"Matmul OK on {device}: output shape {tuple(c.shape)}, dtype {c.dtype}")


if __name__ == "__main__":
    main()
