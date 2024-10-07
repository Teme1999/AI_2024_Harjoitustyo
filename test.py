import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("No GPU available, using CPU.")