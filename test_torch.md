import torch

import platform

print("Python version:", platform.python_version())

print("PyTorch version:", torch.__version__)

print("CUDA available:", torch.cuda.is_available())

x = torch.tensor([10, 20, 30, 40])

print("Original tensor:", x)

print("Multiplied tensor:", x * 2)

model = torch.nn.Linear(4, 2)

output = model(x.float())

print("Neural network output:", output)

print("Mini GPT environment is ready!")

print("Weights (2x4 matrix):\n", model.weight)

print("Biases (2 values):\n", model.bias)

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

Output:

<img width="749" height="291" alt="image" src="https://github.com/user-attachments/assets/7877bbc0-10c7-4f0e-87e5-babec8aabf4a" />

============================================================

<img width="858" height="454" alt="image" src="https://github.com/user-attachments/assets/17fe111c-b6ec-4f11-a61a-18153ec1f82e" />

============================================================

In AI and math books, this "multiply-and-add" pattern is called a Dot Product (or Matrix Multiplication when doing multiple rows at once). Every single GPT model—from a tiny Mini GPT to massive AI systems—is built out of millions of these exact multiply-and-add steps stacked together.

<img width="900" height="470" alt="image" src="https://github.com/user-attachments/assets/4b14db83-9fdc-42e7-8e16-60d5df6b8513" />

============================================================

What if the case is torch.nn.Linear(4, 6)

<img width="803" height="444" alt="image" src="https://github.com/user-attachments/assets/9b3a2802-101a-40cf-b9ca-f28428dfc275" />



