import gymnasium as gym
from gymnasium import spaces
from mappings.block_mappings import BLOCK_MAPPINGS
from mappings.reward_mappings import REWARD_MAPPINGS
import numpy as np
import socket
import json
import time
from mss import mss
from PIL import Image
from ultralytics import YOLO
import torch

actions = ["turn-right", "turn-left", "move-forward", "mine", "mine-lower", "mine-below-lower", "mine-above-upper", "mine-down", "mine-above", "forward-up", "forward-down"]
block_directions = ["targetBlock", "down", "up",
                    "underForward", "underBehind", "underRight", "underLeft",
                    "lowerForward", "lowerBehind", "lowerRight", "lowerLeft",
                    "upperForward", "upperBehind", "upperRight", "upperLeft",
                    "aboveForward", "aboveBehind", "aboveRight", "aboveLeft"]
"""block_directions = ["targetBlock", "down", "up",
                "underEast", "underWest", "underNorth", "underSouth",
                  "lowerEast", "lowerWest","lowerNorth", "lowerSouth",
                    "upperEast", "upperWest", "upperNorth", "upperSouth",
                        "aboveEast", "aboveWest", "aboveNorth", "aboveSouth"]"""
#directions = ["NORTH", "SOUTH", "EAST", "WEST"]
#tilt_values = [-90,-45,0,45,90]

start_coords = (64, 64)

mon = {'left': 160, 'top': 160, 'width': 700, 'height': 700}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_model = YOLO("best.pt").to(device)


class MinecraftAgentEnv(gym.Env):
    def __init__(self, host="localhost", port=5000, max_steps=2048):
        super(MinecraftAgentEnv, self).__init__()

        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self._start_server()
        self.longest_distance = 0

        # Shifted coordinate space
        #x_min, y_min, z_min = -512, -60, -512
        #x_max, y_max, z_max = 512, 1, 512
        self.coordinate_space = spaces.MultiDiscrete([1025, 62, 1025])
        #x_max, y_max, z_max = 128, 62, 128
        #self.coordinate_space = spaces.MultiDiscrete([x_max + 1, y_max + 1, z_max + 1])

        #self.tilt_space = spaces.Discrete(len(tilt_values))

        #self.direction_space = spaces.Discrete(len(directions))

        # Surrounding block space (18 blocks surrounding the player + 1 target block)
        #num_block_types = len(BLOCK_MAPPINGS)
        num_block_types = 4
        self.surrounding_block_space = spaces.MultiDiscrete([num_block_types] * len(block_directions))

        self.observed_ores = spaces.MultiBinary(1) # 5 observable ore types

        # Observation space as a combination of coordinates and surrounding blocks
        self.observation_space = spaces.Tuple((
            self.coordinate_space,
            self.surrounding_block_space,
            self.observed_ores,
            #self.tilt_space,
            #self.direction_space
        ))

        # Action space: 11 discrete actions (turn-left, turn-right, forward, forward-up, forward-down, mine, mine-lower, mine-below-lower, mine-above-upper, mine-down, mine-up)
        self.action_space = spaces.Discrete(11)

        self.max_steps = max_steps
        self.step_count = 0

    def _start_server(self):
        # Sets up the server socket, waiting for the plugin to connect using /connect
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Server listening on {self.host}:{self.port}")

        self.client_socket, addr = self.server_socket.accept()
        print(f"Client connected: {addr}")

    

    def reset(self):
        # Consider whether best approach is for model to issue reset, 
        # or if plugin should keep track of the amount of actions performed and calls reset after a given amount.
        self.client_socket.sendall(b"RESET\n")
        initial_state, result = self._receive_state()
    
        self.step_count = 0
        return initial_state, {}
    
    def step(self, action):
        # Sends action to plugin
        #print(f"list of actions: {actions}")
        #print(f"Actions size: {len(actions)}")
        #print(f"Action index: {action}")
        action_str = actions[action]
        #time.sleep(10)
        self.client_socket.sendall(f"{action_str}\n".encode('utf-8'))

        state, result = self._receive_state()
        if result == "/disconnect":
            return None, 0, True, "/disconnect"
        #print(f"Result: {result}")

        reward = self._calculate_reward(result, [state[0], state[2]])
        #print(f"Reward: {reward}")

        self.step_count += 1

        done = self.step_count >= self.max_steps

        return state, reward, done, result

    def _receive_state(self):
        try:
            state_data = self.client_socket.recv(1024).decode('utf-8').strip()
        except Exception as e:
            print(f"Error receiving data: {e}")
            state_data = "/disconnect"
        if state_data == "/disconnect":
            return None, "/disconnect"

        try:
            state_json = json.loads(state_data)

            x,y,z = state_json.get("x", 0), state_json.get("y", 0), state_json.get("z", 0)
            x_shifted = x + 512
            y_shifted = y + 60
            z_shifted = z + 512

            encoded_coordinates = np.array([x_shifted, y_shifted, z_shifted], dtype=np.float32)

            # print(encoded_coordinates)
            #raw_tilt = state_json.get("tilt", 0)
            #tilt_index = tilt_values.index(raw_tilt)

            #raw_direction = state_json.get("direction", "unknown")
            #print(f"Direction: {raw_direction}")
            #direction_index = directions.index(raw_direction)

            action_result = state_json.get("actionResult", "unknown")
            # print(action_result)

            surrounding_blocks_dict = state_json.get("surroundingBlocks", {})

            with mss() as sct:
                screenshot = sct.grab(mon)
                img = Image.frombytes('RGB', (screenshot.width, screenshot.height), screenshot.rgb)
                image_predictions = image_model.predict(img, device=device, verbose=False)

                ores = np.zeros(5, dtype=np.float32)
                for detection in image_predictions:
                    boxes = detection.boxes
                    for box in boxes:
                        ores[int(box.cls[0])] = 1
                    

            surrounding_blocks = [
                BLOCK_MAPPINGS.get(surrounding_blocks_dict.get(direction, "UNKNOWN"), 0) 
                for direction in block_directions
            ]
            
            # If the ores array contains 1 or more ones, set ore_observed to 1
            ore_observed = 1 if np.sum(ores) > 0 else 0
            print(ore_observed)

            state = np.concatenate([encoded_coordinates, surrounding_blocks, [ore_observed]])
            #print(f"State: {state}")
            return state, action_result
        except json.JSONDecodeError:
            print("Failed to decode json from plugin")
            return self.observation_space.sample()
        
    def close(self):
        if self.client_socket:
            self.client_socket.close()
            print("Client disconnected")
        if self.server_socket:
            self.server_socket.close()
            print("Server closed")

    def _calculate_reward(self, result, coordinates):
        # Calculate the length of the vector from the starting coordinates to the current coordinates
        # and give rewards based on distance
        # distance = np.linalg.norm(np.array(coordinates) - np.array(start_coords))
        # distance_reward = 100 if distance > self.longest_distance else 0
        # if distance > self.longest_distance:
        #     self.longest_distance = distance

        
        return REWARD_MAPPINGS.get(result, REWARD_MAPPINGS.get("unknown", 0))
