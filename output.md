## Clone the repo

<img width="868" height="254" alt="image" src="https://github.com/user-attachments/assets/c97b2d3b-2ad0-427f-aa2d-861b2b55fed9" />

## Train the model

<img width="1360" height="789" alt="image" src="https://github.com/user-attachments/assets/daabd6e3-453e-4e3d-8d66-d6f62d5f1ca3" />

<img width="1363" height="385" alt="image" src="https://github.com/user-attachments/assets/1e469e07-b266-43fb-b5a9-3c4eac1888e8" />

## Run the script

<img width="1368" height="645" alt="image" src="https://github.com/user-attachments/assets/22077b8f-ba38-49b4-b13b-1bf09e80d631" />

<img width="1353" height="456" alt="image" src="https://github.com/user-attachments/assets/c20ab158-99d6-480f-bf32-99989b810ae0" />

## Data flow

<img width="1303" height="774" alt="image" src="https://github.com/user-attachments/assets/2d898805-57bb-4a5e-bb66-1963877276ea" />

## Observations to note


What New Insights You Will See When You Run This

1. In embeddings.py: You will watch the exact math of adding Token Identity + Position 0 to create a single context vector.
2. In attention.py: You will see the before-and-after effect of the Causal Mask—raw scores in the upper triangle turning into -inf, and Softmax turning those -inf entries into 0.00 while making each row add up to 1.00.
3. In transformer.py: You will see exactly how many neurons ReLU switches off (zeros out) inside the 16 -> 64 -> 16 expansion layer.
4. In train.py: You will print lm_head.weight before, its computed gradient, and the updated weight after optimizer.step().
5. In generate.py: You will see the model's Top 3 candidate letters and confidence percentages at each step as it builds "the cat sleeps...".

## Running this LLM with an agent

Your terminal will show your MiniGPT streaming text and pausing mid-sentence to execute real Linux/file actions on your VM:

<img width="1358" height="831" alt="image" src="https://github.com/user-attachments/assets/8822e4ff-d8cf-4434-8425-9dd26598dd4f" />

## Pictorial view

<img width="1397" height="764" alt="image" src="https://github.com/user-attachments/assets/3cd163b0-ec55-4456-b389-bbf3197dbbbe" />
