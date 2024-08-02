import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
from openai import OpenAI
import os

os.environ['HF_TOKEN'] = ''
os.environ['OPENAI_API_KEY'] = ''

class RewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.mlp(x)

class EpistemicNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_particles=10):
        super().__init__()
        self.particles = nn.ModuleList([RewardModel(input_dim, hidden_dim) for _ in range(num_particles)])

    def forward(self, x, z):
        return self.particles[z](x)


class GPT4PreferenceSimulator:
    def __init__(self):
        self.model = "gpt-4"
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_preference(self, prompt, response1, response2):
        messages = [
            {"role": "system", "content": "You are an AI assistant tasked with evaluating the quality of responses to given prompts. You will be presented with a prompt and two responses. Your job is to determine which response is better. Start your response with 'Response 1' or 'Response 2' to indicate the better response, then provide a brief explanation for your choice on a new line."},
            {"role": "user", "content": f"Prompt: {prompt}\n\nResponse 1: {response1}\n\nResponse 2: {response2}\n\nWhich response is better? Please provide your answer in the format described."}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=150
            )

            explanation = response.choices[0].message.content
            lines = explanation.split('\n', 1)
            
            if lines[0].strip().lower() == "response 1":
                preference = 0
            elif lines[0].strip().lower() == "response 2":
                preference = 1
            else:
                # If GPT-4 doesn't clearly indicate a preference, choose randomly
                preference = random.choice([0, 1])
                explanation = "Error: Unclear preference. " + explanation

            if len(lines) > 1:
                explanation = lines[1].strip()
            else:
                explanation = "No explanation provided."

            return preference, explanation

        except Exception as e:
            print(f"Error in GPT-4 API call: {e}")
            return random.choice([0, 1]), "Error in preference simulation"

class Agent:
    def __init__(self, model_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", num_responses=100, exploration_algo="double_ts"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.num_responses = num_responses
        self.exploration_algo = exploration_algo

        embedding_dim = self.model.config.hidden_size
        self.reward_model = EpistemicNeuralNetwork(embedding_dim).to(self.device).to(torch.float16)

    def generate_responses(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=self.num_responses
            )

        responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return responses

    def get_embeddings(self, prompt: str, response: str) -> torch.Tensor:
        combined_input = f"{prompt} {response}"
        inputs = self.tokenizer(combined_input, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden_state = outputs.hidden_states[-1]
        embedding = last_hidden_state.mean(dim=1).to(torch.float16)
        return embedding

    def select_responses(self, prompt, responses):
        if self.exploration_algo == "passive":
            return random.sample(responses, 2)
        
        embeddings = torch.stack([self.get_embeddings(prompt, r) for r in responses])
        
        if self.exploration_algo == "boltzmann":
            rewards = self.reward_model(embeddings, z=0).squeeze()
            probs = torch.softmax(rewards / 0.1, dim=0)  # temperature=0.1
            indices = torch.multinomial(probs, 2, replacement=False)
            return [responses[i] for i in indices]
        elif self.exploration_algo == "infomax":
            M = 30  # number of epistemic indices for infomax
            rewards = torch.stack([self.reward_model(embeddings, z=i) for i in range(M)])
            pref_probs = torch.sigmoid(rewards[:, :, None] - rewards[:, None, :])
            uncertainty = pref_probs.var(dim=0)
            i, j = np.unravel_index(uncertainty.argmax().item(), uncertainty.shape)
            return [responses[i], responses[j]]
        elif self.exploration_algo == "double_ts":
            z1, z2 = random.sample(range(10), 2)  # sample two different epistemic indices
            rewards1 = self.reward_model(embeddings, z=z1).squeeze()
            rewards2 = self.reward_model(embeddings, z=z2).squeeze()
            i, j = rewards1.argmax().item(), rewards2.argmax().item()
            if i == j:
                j = random.choice([k for k in range(len(responses)) if k != i])
            return [responses[i], responses[j]]

    def update_reward_model(self, prompt, response1, response2, preference):
        embedding1 = self.get_embeddings(prompt, response1)
        embedding2 = self.get_embeddings(prompt, response2)
        
        loss = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=1e-4)
        
        for _ in range(10):  # number of SGD steps
            z = random.randint(0, 9)  # random epistemic index
            reward1 = self.reward_model(embedding1.unsqueeze(0), z)
            reward2 = self.reward_model(embedding2.unsqueeze(0), z)
            
            # Ensure rewards are scalar values
            reward1 = reward1.squeeze()
            reward2 = reward2.squeeze()
            
            # Calculate the difference and ensure it's a scalar
            reward_diff = (reward1 - reward2).squeeze()
            
            # Create a scalar target
            target = torch.tensor(1.0, dtype=torch.float16).to(self.device)
            
            if preference == 0:
                l = loss(reward_diff, target)
            else:
                l = loss(-reward_diff, target)
            
            optimizer.zero_grad()
            l.backward()
        optimizer.step()

def run_experiment(agent, simulator, num_epochs=100, queries_per_epoch=32):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for query in range(queries_per_epoch):
            prompt = f"Generate a response to the following scenario: {random.choice(['A day at the beach', 'A futuristic city', 'An alien encounter', 'A historical event'])}"
            responses = agent.generate_responses(prompt)
            selected_responses = agent.select_responses(prompt, responses)
            
            preference, explanation = simulator.get_preference(prompt, selected_responses[0], selected_responses[1])
            
            print(f"Query {query + 1}/{queries_per_epoch}")
            print(f"Prompt: {prompt}")
            print(f"Response 1: {selected_responses[0]}")
            print(f"Response 2: {selected_responses[1]}")
            print(preference)
            print(f"Preference: {'Response 1' if preference == 0 else 'Response 2'}")
            print(f"Explanation: {explanation}")
            print()

            agent.update_reward_model(prompt, selected_responses[0], selected_responses[1], preference)
        
        # Here you would typically evaluate the agent's performance
        # For simplicity, we're not implementing this part

if __name__ == "__main__":
    agent = Agent(exploration_algo="double_ts")
    simulator = GPT4PreferenceSimulator()
    run_experiment(agent, simulator)