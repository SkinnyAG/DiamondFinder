# Machine Learning: Minecraft Diamond Finder
This is the main repository for the Minecraft diamond finder model. 
See also the repository containing the required plugin and Image Model

- [Ore-analyzer](https://github.com/MindChirp/ore-analyzer)
- [DiamondFinder Plugin](https://github.com/SkinnyAG/DiamondFinderPlugin)

## Model Architecture
The model consists of two main parts: 
- A DQN (Deep Q-Network) responsible for determining an optimal mining pattern and player movement.
- An Image Proccessing model built on YOLO (You only look once) informing the DQN when a ore (resource block) is on the screen.

The environment definition is inspired by Gymnasium's AI lab.

## Environment Communication
The model communicates with the player agent inside the Minecraft environment through a socket connection. The model and agent exchange the current state and actions to perform over this connection. 
The Bukkit API gathers information about the environment while programatically controlling the player character.
