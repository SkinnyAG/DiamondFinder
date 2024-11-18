from minecraft_agent_env import MinecraftAgentEnv
from q_network import QNetwork
from replay_buffer import ReplayBuffer
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch

# Hyperparameters
num_episodes = 256
learning_rate = 0.001
update_target_freq = 2
batch_size = 2048
gamma = 0.9995
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.9885

legal_diag = [0] * num_episodes
illegal_diag = [0] * num_episodes

legal_forward = [0] * num_episodes
illegal_forward = [0] * num_episodes

legal_mining = [0] * num_episodes
illegal_mining = [0] * num_episodes

rotations = [0] * num_episodes
useless_rotations = [0] * num_episodes

#legal_tilt = [0] * num_episodes
#illegal_tilt = [0] * num_episodes
ores_mined = [0] * num_episodes

all_rewards = [0] * num_episodes
all_loss = [0] * num_episodes

env = MinecraftAgentEnv()
observation_space = env.observation_space
action_size = env.action_space.n
coordinate_size = len(observation_space[0].nvec)
surrounding_size = len(observation_space[1].nvec)
#tilt_size = 1
#direction_size = 1

observation_size = coordinate_size + surrounding_size #+ direction_size

q_network = QNetwork(observation_size, action_size)

target_network = QNetwork(observation_size, action_size)

replay_buffer = ReplayBuffer()

optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
epsilon = epsilon_start

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    total_loss = 0

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = torch.argmax(q_network(state_tensor), dim=1).item()

        next_state, reward, done, result = env.step(action)

        if result == "/disconnect":
            print("Received /disconnect. Ending training...")
            done = True
            break 
        match result:
            case "successful-forward":
                legal_forward[episode] += 1
            case "illegal-forward":
                illegal_forward[episode] += 1
            case "successful-diag":
                legal_diag[episode] += 1
            case "illegal-diag":
                illegal_diag[episode] += 1
            case action_result if "_ore" in action_result:
                ores_mined[episode] += 1
            case action_result if "successful-mine-" in action_result:
                legal_mining[episode] += 1
            case action_result if "illegal-mine-" in action_result:
                illegal_mining[episode] += 1
            case "rotation":
                rotations[episode] += 1
            case "useless-rotation":
                useless_rotations[episode] += 1

        replay_buffer.add((state, action, reward, next_state, done))

        if replay_buffer.size() >= batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states_tensor = torch.FloatTensor(np.array(states))
            actions_tensor = torch.LongTensor(np.array(actions))
            rewards_tensor = torch.FloatTensor(np.array(rewards))
            next_states_tensor = torch.FloatTensor(np.array(next_states))
            dones_tensor = torch.FloatTensor(np.array(dones))

            with torch.no_grad():
                best_actions = torch.argmax(q_network(next_states_tensor), dim=1)

                max_next_q_values = target_network(next_states_tensor).gather(1, best_actions.unsqueeze(1)).squeeze()

                targets = rewards_tensor + (1 - dones_tensor) * gamma * max_next_q_values

            
            current_q_values = q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze()

            loss = torch.nn.HuberLoss()(current_q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        state = next_state
        total_reward += reward
    
    all_rewards[episode] = total_reward
    all_loss[episode] = total_loss

    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    if episode % update_target_freq == 0:
        target_network.load_state_dict(q_network.state_dict())

    if result == "/disconnect":
        break

    print(f"Episode: {episode+1}, Ores Mined: {ores_mined[episode]}, Total Reward: {total_reward}, Epsilon: {epsilon}, Total Loss: {total_loss}")

torch.save(q_network.state_dict(), 'qnet2.pth')
torch.save(target_network.state_dict(), 'tnet2.pth')
env.close()

episodes = list(range(1, num_episodes + 1))

fig, axes = plt.subplots(3, 2, figsize=(12, 12))  # Adjust the grid layout as needed
fig.tight_layout(pad=5.0)

# Scatter plots for each list of values (example of legal_diag vs. illegal_diag)
axes[0, 0].scatter(episodes, legal_diag, color='green', label='Legal Diagonal', alpha=0.6)
axes[0, 0].scatter(episodes, illegal_diag, color='red', label='Illegal Diagonal', alpha=0.6)
axes[0, 0].set_title("Legal vs Illegal Diagonal Movement")
axes[0, 0].set_xlabel('Episodes')
axes[0, 0].set_ylabel('Count')
axes[0, 0].legend()

axes[0, 1].scatter(episodes, legal_forward, color='blue', label='Legal Forward', alpha=0.6)
axes[0, 1].scatter(episodes, illegal_forward, color='orange', label='Illegal Forward', alpha=0.6)
axes[0, 1].set_title("Legal vs Illegal Forward Movement")
axes[0, 1].set_xlabel('Episodes')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend()

axes[1, 0].scatter(episodes, legal_mining, color='purple', label='Legal Mining', alpha=0.6)
axes[1, 0].scatter(episodes, illegal_mining, color='brown', label='Illegal Mining', alpha=0.6)
axes[1, 0].set_title("Legal vs Illegal Mining Actions")
axes[1, 0].set_xlabel('Episodes')
axes[1, 0].set_ylabel('Count')
axes[1, 0].legend()



axes[1, 1].scatter(episodes, rotations, color='yellow', label='Rotations', alpha=0.6)
axes[1, 1].set_title("Rotations")
axes[1, 1].set_xlabel('Episodes')
axes[1, 1].set_ylabel('Count')

axes[2, 0].scatter(episodes, ores_mined, color='cyan', label="Ores mined", alpha=0.6)
axes[2, 0].set_title("Ores mined")
axes[2, 0].set_xlabel('Episodes')
axes[2, 0].set_ylabel('Count')
"""
axes[2, 0].scatter(episodes, legal_tilt, color='cyan', label='Legal Tilt', alpha=0.6)
axes[2, 0].scatter(episodes, illegal_tilt, color='magenta', label='Illegal Tilt', alpha=0.6)
axes[2, 0].set_title("Legal vs Illegal Tilt Actions")
axes[2, 0].set_xlabel('Episodes')
axes[2, 0].set_ylabel('Count')
axes[2, 0].legend()
"""

axes[2, 1].scatter(episodes, all_rewards, color='green', label="Rewards", alpha=0.6)
axes[2, 1].scatter(episodes, all_loss, color='red', label='Loss', alpha=0.6)
axes[2, 1].set_title("Rewards & Loss")
axes[2, 1].set_xlabel('Episodes')
axes[2, 1].set_ylabel('Total')
axes[2, 1].legend()

#plt.show()
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plot_filename = os.path.join(plots_dir, f"plot_{current_date}.png")

plt.savefig(plot_filename)