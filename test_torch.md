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

============================================================

This script checks your PyTorch environment, creates a basic tensor, and passes it through a single-layer neural network.

## 1. Imports & System Checks

• import torch: Loads the main PyTorch library so you can build tensors and neural networks.
• import platform: Loads Python's built-in module for inspecting system and environment details.
• print("Python version:", platform.python_version()): Displays the exact version of Python running the script (e.g., 3.11.5).
• print("PyTorch version:", torch.__version__): Displays the installed PyTorch library version.
• print("CUDA available:", torch.cuda.is_available()): Checks if a compatible NVIDIA GPU is available (True) to speed up math operations, or if it will fall back to the CPU (False).

## 2. Tensor Creation & Math

• # Create a tensor: A comment ignored by Python, used to label the next section of code.
• x = torch.tensor([10, 20, 30, 40]): Creates a 1D PyTorch tensor (similar to a list or array) holding four integer values.
• print("Original tensor:", x): Outputs the tensor: tensor([10, 20, 30, 40]).
• print("Multiplied tensor:", x * 2): Multiplies every element in the tensor by 2, outputting tensor([20, 40, 60, 80]).

## 3. Neural Network Forward Pass

• # Create a simple neural network: A comment introducing the neural network section.
• model = torch.nn.Linear(4, 2): Creates a single fully connected neural network layer that takes 4 input features and outputs 2 values, initializing random weights and biases internally.
• output = model(x.float()): Converts the integer tensor x into floating-point decimals (required by PyTorch layers) and passes it through the model to compute the output.
• print("Neural network output:", output): Prints the 2 resulting numbers along with a grad_fn tag, which shows PyTorch is tracking operations for future training.


============================================================

<img width="732" height="163" alt="image" src="https://github.com/user-attachments/assets/52ff8a3a-8e2d-46d9-84c4-613c0033625b" />
