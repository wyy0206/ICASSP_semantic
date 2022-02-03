# Performance Optimization for Wireless Semantic Communications over Energy Harvesting Networks

This is the simplified code for the paper ''Meta-Reinforcement Learning for Reliable Communication in THz/VLC Wireless VR Networks''. This repo is heavily inspired by the fantastic implementation  [mohammadasghari/dqn-multi-agent-rl](https://github.com/mohammadasghari/dqn-multi-agent-rl).

## Usage
You can use the [`semantic_mutiagent.py`](semantic_mutiagent.py) script in order to train the proposed cooperative DQN model.
```
python semantic_mutiagent.py
```
You can also use the [`semantic_mutiagent.py`](semantic_mutiagent.py) script in order to test the trained model.
```
python semantic_mutiagent.py --test --name
```
This script was tested with:
Python 3.6
numpy 1.14.0
tensorflow 2.0.0.