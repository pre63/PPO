import gymnasium as gym
from stable_baselines3 import PPO
from Models.PPOTrace import PPOTrace
from stable_baselines3.common.vec_env import SubprocVecEnv
from Environments.LunarLander import make
import numpy as np


def create_env(rank, seed=0):
  """
  Utility function to create a single environment instance with a unique seed.
  """
  def _init():
    env = make()
    env.seed(seed + rank)
    return env
  return _init


def evaluate_model(model, eval_timesteps=20000):
  """
  Evaluates a model's performance over a set number of timesteps.
  """
  env = make()
  t = 0
  successes = []
  while t < eval_timesteps:
    state, _ = env.reset()
    done = False
    while not done:
      action, _ = model.predict(state)
      state, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated
      t += 1
    successes.append(int(info.get("success", False)))
  success_rate = sum(successes) / len(successes)
  return success_rate


def replay(env, model):

  env = make(render_mode='human')
  state, _ = env.reset()
  done = False

  while not done:
    action, _ = model.predict(state)
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()


if __name__ == "__main__":
  # Number of CPU cores to use
  n_cpu = 8  # Adjust based on the number of available cores

  # Create vectorized environments
  env = SubprocVecEnv([create_env(i) for i in range(n_cpu)])

  # Training parameters
  total_timesteps = 10000000
  eval_timesteps = total_timesteps // 5
  params_trident = {
      "trace_lambda": [0.95, 0.9, 0.5],
      "trace_gamma": [0.99, 0.95, 0.9],
      "learning_rate": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
  }
  permutations = [(l, g, lr) for l in params_trident["trace_lambda"]
                  for g in params_trident["trace_gamma"]
                  for lr in params_trident["learning_rate"]]

  results = []

  for l, g, lr in permutations:
    # Train PPO
    ppo_model = PPO('MlpPolicy', env, verbose=1, n_steps=2048 // n_cpu, learning_rate=lr)
    ppo_model.learn(total_timesteps=total_timesteps)

    # Train PPOTrace
    trace_model = PPOTrace('MlpPolicy', env, verbose=1, n_steps=2048 // n_cpu, trace_lambda=l, trace_gamma=g, learning_rate=lr)
    trace_model.learn(total_timesteps=total_timesteps)

    # Evaluate PPO
    ppo_success_rate = evaluate_model(ppo_model, eval_timesteps)
    print(f"PPO: {ppo_success_rate} - alpha: {lr}")

    # Evaluate PPOTrace
    trace_success_rate = evaluate_model(trace_model, eval_timesteps)
    print(f"PPOTrace: {trace_success_rate} - lambda: {l}, gamma: {g}, alpha: {lr}")

    # Store results
    results.append({
        "trace_lambda": l,
        "trace_gamma": g,
        "alpha": lr,
        "ppo_success_rate": ppo_success_rate,
        "trace_success_rate": trace_success_rate,
        "trace": trace_model,
        "ppo": ppo_model
    })

  # Log summary of results
  print("\n--- Summary of Results ---\n")
  for result in results:
    print(
        f"Lambda: {result['trace_lambda']}, Gamma: {result['trace_gamma']}, Alpha: {result['alpha']}, "
        f"PPO:      {result['ppo_success_rate']:.2f}, "
        f"PPOTrace: {result['trace_success_rate']:.2f}"
    )

  # Render final comparison for the best PPOTrace model
  best_result = max(results, key=lambda x: x['trace_success_rate'])

  replay(env, best_result['trace'])
  replay(env, best_result['ppo'])

  print(f"Best PPOTrace configuration: Lambda: {best_result['trace_lambda']}, Gamma: {best_result['trace_gamma']}, LR: {best_result['alpha']}, Success Rate: {best_result['trace_success_rate']:.2f}")
