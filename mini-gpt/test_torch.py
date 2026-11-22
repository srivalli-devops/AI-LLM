
import torch
import platform

print("Python version:", platform.python_version())
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Create a tensor
x = torch.tensor([10, 20, 30, 40])

print("Original tensor:", x)
print("Multiplied tensor:", x * 2)

# Create a simple neural network
model = torch.nn.Linear(4, 2)

output = model(x.float())

print("Neural network output:", output)

print("Mini GPT environment is ready!")
