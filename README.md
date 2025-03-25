# Radar Jamming Using Deep Reinforcement Learning

This repository contains all code for the project Radar Jamming Using Deep Reinforcement Learning. This includes:
* Implementation of the paper: An Intelligent Strategy Decision Method for Collaborative Jamming Based on Hierarchical Multi-agent Reinforcement Learning.
* Multi-agent ParallelEnv for the jamming task created using PettingZoo and Gymnasium.
* Code for Prioritized Experience Replay of single transitions, and Sequence Replay Buffer for sequences of transitions.
* Solving the task using DDQN (Double Deep Q-Network), getting a scaled reward of 0.6 (from -1 to 1), from a rolling average of 100 episodes.
* Exploring other approaches to solve the problem - converting the problem into a POMDP (Partially Observable Markov Decision Process), and training multiple agents together using a shared DRQN, with separate replay buffers for agents.
