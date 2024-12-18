# Machine Learning: Minecraft Diamond Finder
This is the main repository for the Minecraft diamond finder model. 
See also the repository containing the required plugin and Image Model

- [Ore-analyzer](https://github.com/MindChirp/ore-analyzer)
- [DiamondFinder Plugin](https://github.com/SkinnyAG/DiamondFinderPlugin)

## Video demonstrations
- [Snaking Pattern](https://www.youtube.com/watch?v=AWCI2RaHlGQ)
- [Branching Pattern within a rectangular area](https://www.youtube.com/watch?v=gfClWMDjIlc)
- [Common Mining Pattern from 'blind' agent](https://www.youtube.com/watch?v=VQ95LVMJ90w)
- [Ore Recognition using YOLO](https://youtu.be/zuUPoIzV7co?si=IA6jFS3aWXohjD9q)

Necessary packages

`$ pip install torch numpy matplotlib gymnasium`

## Model Architecture
The model consists of two main parts: 
- A DQN (Deep Q-Network) responsible for determining an optimal mining pattern and player movement.
- An Image Proccessing model built on YOLO (You only look once) informing the DQN when an ore (resource block) is on the screen.

The environment definition is inspired by Gymnasium's AI lab.

## Environment Communication
The model communicates with the player agent inside the Minecraft environment through a socket connection. The model and agent exchange the current state and actions to perform over this connection. 
The plugin's Bukkit API gathers information about the environment while programatically controlling the player character. More details in the plugin repository!
