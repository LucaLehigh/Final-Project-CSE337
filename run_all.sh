#!/usr/bin/env bash
set -e  # stop if any command fails

echo "1) Running linear Q-learning"
python QLearning.py

echo "2) Recording Q-learning policy (record)"
python record.py

echo "3) Plotting Q-learning training"
python plot_training.py

echo "4) Running DQN WITHOUT env randomization"
python train_dqn.py --episodes 6000

echo "5) Plotting DQN training (no randomization)"
python plot_dqn_training.py

echo "6) Running DQN WITH env randomization"
python train_dqn.py --episodes 6000 --randomize-env

echo "7) Plotting DQN training (with randomization)"
python plot_dqn_training.py

echo "All runs complete ✅"
