# Proximal Policy Optimization with Eligibility Trace Exploration

This repository introduces a novel extension to the Proximal Policy Optimization (PPO) algorithm by incorporating eligibility traces to enhance exploration and learning efficiency. The experiments are conducted exclusively in the LunarLander environment from the `Box2D` suite, with custom reward strategies and position detection. The goal is to analyze the impact of eligibility traces on policy learning in a controlled, high-feedback environment.

## Overview

Proximal Policy Optimization (PPO) is a widely used reinforcement learning algorithm known for its robustness and stability. By combining PPO with eligibility traces, we aim to improve temporal credit assignment and exploration in continuous control settings. This modification enhances the agent’s ability to learn from sparse or delayed feedback, particularly in scenarios requiring fine-grained control, such as the LunarLander environment.

## LunarLander Environment and Custom Success Metrics

The experiments in this repository focus on the LunarLander environment with a modified reward function and success detection criteria. The success of a landing is determined by a custom function that evaluates factors such as the lander's upright angle, position on the landing pad, velocities, and leg contact with the ground.

### Custom Success Criteria

A successful landing requires:
- The lander is upright, with an angle below a threshold.
- The lander is positioned within a specified range on the landing pad.
- Horizontal and vertical velocities are below defined limits.
- Both legs of the lander are in contact with the ground.
- The episode ends with the lander above ground level.

This approach ensures that the agent not only achieves terminal states but also learns stable and controlled landings.

## Installation

Setting up the repository is straightforward using the provided Makefile. To install dependencies and prepare the environment, run:

```bash
make install
```

This command installs system dependencies, sets up a Python virtual environment, and installs the required Python packages.

## Running the Algorithm

To execute the PPO algorithm with eligibility trace enhancements, use the following command:

```bash
make sb3
```

This command runs the training script, logging outputs to `.log/sb3.log` for later review.

## Key Features

This repository focuses on advancing PPO with eligibility traces and is equipped with:
- **Custom LunarLander Environment**: Features additional reward strategies and precise success detection metrics for improved agent evaluation.
- **Eligibility Trace-Enhanced PPO**: Builds on the Stable-Baselines3 PPO implementation to integrate eligibility traces, bridging the gap between immediate and delayed feedback.
- **Simplified Experimentation**: A Makefile simplifies running, logging, and managing experiments.
- **Focused Scope**: Purpose-built for controlled experiments in the LunarLander environment, ensuring clear insights into the impact of eligibility traces.

## Further Work

To comprehensively evaluate the effectiveness of eligibility trace-enhanced PPO, future research should include:
- **Baseline Comparisons**: Assess performance against standard PPO, as implemented in Stable-Baselines3, to quantify the gains from incorporating eligibility traces.
- **Comparative Analysis**: Benchmark against state-of-the-art reinforcement learning algorithms such as:
  - **Soft Actor-Critic (SAC)**: Known for its entropy-driven exploration and stability in continuous control.
  - **Truncated Quantile Critics (TQC)**: Excelling in performance consistency through distributional reinforcement learning techniques.
  - **Proximal Policy Optimization (Standard)**: To measure the direct impact of eligibility traces.
  - **Advanced On-Policy Algorithms**: Including A2C and other advanced techniques.
- **Environment Generalization**: Extend experiments beyond LunarLander to more complex continuous control environments such as Ant, Humanoid, or HalfCheetah to validate scalability and robustness.
- **Hyperparameter Sensitivity**: Analyze the role of eligibility trace decay rates, learning rates, and other critical parameters in shaping performance.
- **Ablation Studies**: Isolate the impact of eligibility traces by systematically varying their inclusion or modifying the custom success metrics.

## Why Use This Repository?

This repository offers a focused and practical implementation for exploring the impact of eligibility traces on PPO within a well-defined environment. It combines rigorous experimentation with simplified workflows, making it a valuable resource for researchers and practitioners. By including baseline comparisons and planning extensions to state-of-the-art algorithms, it paves the way for a deeper understanding of how eligibility traces influence reinforcement learning in continuous control domains.
