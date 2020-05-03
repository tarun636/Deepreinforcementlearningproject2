
# Udacity Deep Reinforcement Learning Nanodegree

## Project 2: Continuous Control
Introduction
This project repository contains Nishi Sood’s work for the Udacity's Deep Reinforcement Learning Nanodegree `Project 2: Continuous Control`. For this project, I have trained an agent that could control a double-jointed arm towards specific locations.


## Project's goal
 - In this environment, a double-jointed arm can move to target locations. 
 - A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 
 - The goal of the agent is to maintain its position at the target location for as many time steps as possible.
 - The environment solved in this project is a variant of the Reacher Environment agent by Unity.
 - The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
 

## Project's Environment
 - The environment is based on Unity ML-agents. The project environment provided by Udacity is similar to the Reacher environment on the Unity ML-Agents GitHub page.
 - Set-up: Double-jointed arm which can move to target locations.
 - Goal: The agents must move its hand to the goal location, and keep it there.
 - Agents: The environment contains 20 agents linked to a single Brain.
     - The provided Udacity agent versions are Single Agent or 20-Agents
   - Agent Reward Function (independent):
     - `+0.1` Each step agent's hand is in goal location.
 - Brains: One Brain with the following observation/action space.
 - Vector Observation space: 26 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigid bodies.
 - Vector Action space: (Continuous) Size of 4, corresponding to torque applicable to two joints.
   - Visual Observations: None.
   - Reset Parameters: Two, corresponding to goal size, and goal movement speed.
   - The task is episodic, with 1000 timesteps per episode. In order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.
   - Benchmark Mean Reward: 30

## Solving the Environment
Depending on the chosen environment for the implementation, there are 2 possibilities:
 - Option 1: Solve the First Version
The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.
 - Option 2: Solve the Second Version
The barrier for solving the second version of the environment is slightly different, to consider the presence of many agents. In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically:
After each episode, the rewards that each agent received (without discounting) are added up , to get a score for each agent. This yields 20 (potentially different) scores. The average of these 20 scores is then used.
This yields an average score for each episode (where the average is over all 20 agents).
The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.
In the implementation I have chosen to solve the Second version of the environment (Multi Agent) using the off-policy DDPG algorithm. 


## Project Environment Setup/ Configuring Dependencies:
### Step 1: Clone the DRLND Repository and installing dependencies
#### 1. Set Up for the  python environment to run the code in this repository
#### Environment
conda create --name drlnd python=3.6
source activate drlnd

#### 2. Install of OpenAI gym packages
#### OpenAI gym
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
pip install -e .[classic_control]
conda install swig
pip install -e .[box2d]

#### 3.  Other Dependencies
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .

Trouble shooting:
If you get the error for installing torch library:
`On Windows`:
    - 1.Open deep-reinforcement-learning\python\requirements.txt
    - 2.Remove the line torch==0.4.0
`On Anaconda`:
    - 3.conda install pytorch=0.4.0 -c pytorch
    - 4.cd deep-reinforcement-learning/python
        - pip install .
        - conda install pytorch=0.4.0 -c pytorch

#### 4. Create an IPython kernel for the drlnd environment
#### Kernel
python -m ipykernel install --user --name drlnd --display-name "drlnd"
#### 5. Before running code in a notebook, change the kernel to match the drlnd environment by using the drop-down Kernel menu.

### Step 2: Download the Unity Environment
1.Download the environment from one of the links below. You need only select the environment that matches your operating system:
Version 1: One (1) Agent
 - Linux: click here
 - Mac OSX: click here
 - Windows (32-bit): click here
 - Windows (64-bit): click here
Version 2: Twenty (20) Agents
 - Linux: click here
 - Mac OSX: click here
 - Windows (32-bit): click here
 - Windows (64-bit): click here
2.Then, place the file in the p2_continuous-control/  folder in the DRLND GitHub repository, and unzip (or decompress) the file.

### Step 3: Explore the Environment

After you have followed the instructions above, open Continuous_Control.ipynb (located in the p2_continuous-control/  folder in the DRLND GitHub repository) and follow the instructions to learn how to use the Python API to control the agent.
 
Note: Don’t forget to change the kernel to the Virtual Env ”dlrnd”
Output of the Unity agent after running “Continuous_Control_EnvExploration.ipynb”
 
Description
 - `Continuous_Control_EnvExploration.ipynb`:  Notebook used to verify the Environment Setup and initial working of Unity Agent
 - `Continuous_Control_Solution.ipynb`: Notebook used to control and train the final version of the agent
 - `ddpg_agent.py`: Create an Agent class that interacts with and learns from the environment
 - `model.py`: Actor and Critic classes
 - `checkpoint_actor.pth` : saved model for the actor
 - `checkpoint_critic.pth` : saved model for the critic
 - `learning_curves.jpg` : Learning Curve Displaying the trained agent
 - `Project2_Report.pdf`: The submission includes a file in the root of the GitHub repository that provides a technical description of the implementation.
 - `README.md` : The README describes the project environment details (i.e., the state and action spaces, and when the environment is considered solved). It has instructions for installing dependencies or downloading needed files and process to run the code in the repository, to train the agent.
 

 ## Steps to Run the Project
 - 1.Download the environment and unzip it into the directory of the project. Rename the folder to Reacher_Windows_x86_64_20Agents and ensure that it contains Reacher.exe
 - 2.Use jupyter to run the Continuous_Control_Solution.ipynb notebook: jupyter notebook Continuous_Control_Solution.ipynb
 - 3.Run the cells in order in the notebook Continuous_Control_Solution.ipynb to train an agent that solves our required task of moving the double-jointed arm.
 - 4.They will initialize the environment and train until it reaches the goal condition of +30
 - 5.A graph of the scores during training will be displayed after training.
 

## Results
Plot showing the score per episode over all the episodes.
Environment solved in 150 episodes with Average Score: 30.11

![learning_curves](learning_curves.jpg)

## Ideas for future work
- We could have used different Optimization algorithm to check the difference in the Agent trained and its performance.
- There could be better results by making use of prioritized experience replay with the existing learning algorithm.
- Distributed Distributional Deterministic Policy Gradients (D4PG) has achieved state of the art results on continuous control problems. Also, PPO, A3C can be used in multi agent training environment.
- It would be interesting to see how the agent performs on this environment in future implementations.
- In future implementations,  I can try testing the agent with difference hyperparameter values, like different Sigma values, faster and smaller learning rates, tweaking the neural network, to choose the final model

