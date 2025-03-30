# Assignment 1: Reinforcement Learning Algorithms

## Team Members

- **Amit Jain** (Me24s003)
- **Sridhar Ramachandran** (Me20b172)

## Project Overview

This repository contains implementations of two reinforcement learning algorithms, **SARSA with Epsilon-Greedy Policy** and **Q-Learning with Softmax Policy**, applied to two OpenAI Gym environments:

- **CartPole-v1**
- **MountainCar-v0**

The objective of the project is to experiment with different hyperparameters and understand how these algorithms perform on different environments.

## Repository Structure

The repository contains four Jupyter Notebook (`.ipynb`) files, each implementing a specific algorithm for a specific environment:

1. **SARSA\_CartpoleV1.ipynb** - Implements SARSA with Epsilon-Greedy Policy for CartPole-v1.
2. **SARSA\_MountainCarV0.ipynb** - Implements SARSA with Epsilon-Greedy Policy for MountainCar-v0.
3. **Qlearning\_CartpoleV1.ipynb** - Implements Q-Learning with Softmax Policy for CartPole-v1.
4. **Qlearning\_MountainCarV0.ipynb** - Implements Q-Learning with Softmax Policy for MountainCar-v0.

Each notebook contains the necessary code for training the respective algorithm and tuning hyperparameters.

## Dependencies

To run the code, the following Python libraries are required:

```python
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
import seaborn as sns
from IPython.display import clear_output
from collections import deque
```

Ensure you have all dependencies installed using:

```bash
pip install numpy matplotlib gymnasium tqdm seaborn
```

## How to Run the Code

Follow these steps to execute the notebooks:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt  # If a requirements file is added
   ```

   OR manually install using:

   ```bash
   pip install numpy matplotlib gymnasium tqdm seaborn
   ```

3. Open Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Navigate to any of the four `.ipynb` files and run the cells.

5. Alternatively, you can run the notebooks on **Google Colab**:
   - Open [Google Colab](https://colab.research.google.com/)
   - Upload the notebook file
   - Run the cells directly in Colab's environment

## Notes

- The **SARSA algorithm** is implemented with an **Epsilon-Greedy Policy**.
- The **Q-Learning algorithm** is implemented with a **Softmax Policy**.
- Each notebook includes code for **tuning hyperparameters** for the respective algorithm and environment.
- The models are trained and tested using **OpenAI Gymnasium** environments.

## Results

The code is designed to solve both environments using respective algorithms. Hyperparameters can be adjusted within the notebooks to experiment with different configurations and improve learning performance.




