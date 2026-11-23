## Step 1 : 
Confirmed environment in VM

| Component | Your environment |
|---|---|
| Host OS | Windows 11 |
| Virtualization | Oracle VirtualBox |
| Guest OS | Ubuntu 20.04.6 LTS |
| Python | 3.8.10 |
| CPU | Intel i7-11800H |
| CPU cores visible | 6 |
| RAM visible | 21 GB |
| GPU | No dedicated GPU |

## Step 2:
Python version: Let's upgrade our project environment

Python 3.8 is old for a new AI development project.

Python 3.11 is recommended for our Mini GPT project. It has broad compatibility with current PyTorch releases.
We will keep Ubuntu's system Python untouched and install a separate Python version for our project.

## Step 3:
Install Python 3.11 using uv

We'll use uv, a Python version and package manager. It allows us to install Python 3.11 without changing Ubuntu's system Python.
First install prerequisites:

**sudo apt update**

**sudo apt install -y curl ca-certificates build-essential**

<img width="832" height="274" alt="image" src="https://github.com/user-attachments/assets/00101441-a1ce-4046-a118-fb03162abc3e" />

Install uv:

**curl -LsSf https://astral.sh/uv/install.sh | sh**

Load uv into your current terminal:

**source $HOME/.local/bin/env**

Verify:

**uv --version**

Install Python 3.11:

**uv python install 3.11**

<img width="751" height="304" alt="image" src="https://github.com/user-attachments/assets/4ea4a39e-bf58-463b-a66b-eefdb8d03caa" />

## 4. Create our project virtual environment

Create a Python 3.11 virtual environment in the existing project:

**uv venv --python 3.11 venv**

Activate it:

**source venv/bin/activate**

Check:

**python --version**

**which python**

System Python remains Python 3.8.10. Only our project uses Python 3.11.

<img width="569" height="163" alt="image" src="https://github.com/user-attachments/assets/942f1ee6-d2e1-410b-9a03-1abe5b851b1e" />


## 5. Install PyTorch CPU

With the virtual environment activated:

**python -m ensurepip --upgrade**

**python -m pip install --upgrade pip**

<img width="730" height="151" alt="image" src="https://github.com/user-attachments/assets/9677abbc-fa67-486e-ad97-a2e50e4ed0dd" />


Download CPU-compatible PyTorch:

**python -m pip install torch --index-url https://download.pytorch.org/whl/cpu**

<img width="1358" height="670" alt="image" src="https://github.com/user-attachments/assets/428db3f9-aabd-44dc-9df0-3f64205403a4" />

**python -m pip install numpy matplotlib**

<img width="1359" height="755" alt="image" src="https://github.com/user-attachments/assets/ba02b844-6ab0-44da-871e-cd607364cb20" />


=================================

To use onnxruntime

<img width="1362" height="525" alt="image" src="https://github.com/user-attachments/assets/8ad5b37c-39d3-4024-a170-85c52974ea19" />
