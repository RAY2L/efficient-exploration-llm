# Efficient Exploration for Large Language Models

This project implements efficient exploration techniques for tuning large language models using human feedback, as described in the paper "Efficient Exploration for LLMs" by Vikranth Dwaracherla et al.

## Project Setup

To set up the project environment, please run the following commands:

```bash
pip install --upgrade transformers
pip install openai autoawq
```

These commands will:
1. Upgrade the `transformers` library to the latest version
2. Install the `openai` library for API access
3. Install `autoawq` for quantization support

Make sure you have pip installed and updated before running these commands.

## Running the Project

After installing the required packages, run the main script using:

```bash
python main.py
```

Execute this command in your terminal from the project's root directory.

## Project Structure

The main components of this project are:

1. `RewardModel`: A neural network that estimates the reward for a given input.
2. `EpistemicNeuralNetwork`: An ensemble of reward models to capture uncertainty.
3. `GPT4PreferenceSimulator`: A class that simulates human preferences using the GPT-4 API.
4. `Agent`: The main agent class that implements various exploration algorithms.

## Exploration Algorithms

The project implements several exploration algorithms:

- Passive Exploration
- Boltzmann Exploration
- Infomax
- Double Thompson Sampling

## Configuration

Before running the project, make sure to set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Note

This project uses the GPT-4 API to simulate human preferences, which may incur costs. Please be aware of your API usage and any associated charges.

## References

Dwaracherla, V., Asghari, S. M., Hao, B., & Van Roy, B. (2024). Efficient Exploration for LLMs. arXiv preprint arXiv:2402.00396v2.