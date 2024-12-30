import numpy as np

import torch
from torch.nn import functional as F

from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance


class PPOTrace(PPO):
  def __init__(self, *args, trace_lambda: float = 0.95, trace_gamma=0.99, **kwargs):
    super().__init__(*args, **kwargs)
    self.trace_lambda = trace_lambda
    self.trace_gamma = trace_gamma

  def _setup_model(self) -> None:
    super()._setup_model()
    self.eligibility_traces = {
        name: torch.zeros_like(param, requires_grad=False)
        for name, param in self.policy.named_parameters()
        if param.requires_grad
    }

  def _update_eligibility_traces(self, log_prob: torch.Tensor) -> None:
    log_prob_sum = log_prob.sum()
    grads = torch.autograd.grad(
        log_prob_sum,
        self.policy.parameters(),
        retain_graph=True,
        allow_unused=True
    )
    for (name, param), grad in zip(self.policy.named_parameters(), grads):
      if grad is not None:
        # In case some parameters never show up in grad,
        # or if they appear after initialization for any reason:
        if name not in self.eligibility_traces:
          self.eligibility_traces[name] = torch.zeros_like(param, requires_grad=False)
        self.eligibility_traces[name] = (
            self.trace_gamma * self.trace_lambda * self.eligibility_traces[name] + grad
        )

  def train(self) -> None:
    self.policy.set_training_mode(True)
    self._update_learning_rate(self.policy.optimizer)
    clip_range = self.clip_range(self._current_progress_remaining)
    entropy_losses = []
    pg_losses, value_losses = [], []
    clip_fractions = []

    for epoch in range(self.n_epochs):
      approx_kl_divs = []
      for rollout_data in self.rollout_buffer.get(self.batch_size):
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
          actions = rollout_data.actions.long().flatten()

        values, log_prob, entropy = self.policy.evaluate_actions(
            rollout_data.observations,
            actions
        )
        values = values.flatten()

        advantages = rollout_data.advantages
        if self.normalize_advantage and len(advantages) > 1:
          advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(log_prob - rollout_data.old_log_prob)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        pg_losses.append(policy_loss.item())

        clip_fraction = torch.mean(
            (torch.abs(ratio - 1) > clip_range).float()
        ).item()
        clip_fractions.append(clip_fraction)

        value_loss = F.mse_loss(rollout_data.returns, values)
        value_losses.append(value_loss.item())

        if entropy is None:
          entropy_loss = -torch.mean(-log_prob)
        else:
          entropy_loss = -torch.mean(entropy)
        entropy_losses.append(entropy_loss.item())

        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        self.policy.optimizer.zero_grad()
        # First compute standard PPO gradients
        loss.backward(retain_graph=True)

        # Update the traces with the newly computed log_prob
        self._update_eligibility_traces(log_prob)

        # Add the eligibility trace contribution to each parameter's gradient
        for name, param in self.policy.named_parameters():
          if param.grad is not None and name in self.eligibility_traces:
            param.grad += self.trace_lambda * self.eligibility_traces[name]

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

      self._n_updates += 1

    explained_var = explained_variance(
        self.rollout_buffer.values.flatten(),
        self.rollout_buffer.returns.flatten()
    )
    self.logger.record("train/entropy_loss", np.mean(entropy_losses))
    self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
    self.logger.record("train/value_loss", np.mean(value_losses))
    self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
    self.logger.record("train/clip_fraction", np.mean(clip_fractions))
    self.logger.record("train/loss", loss.item())
    self.logger.record("train/explained_variance", explained_var)
    self.logger.record("traces/lambda", self.trace_lambda)
    self.logger.record("traces/log_std", self.eligibility_traces['log_std'].mean())

  def train(self) -> None:
    """
    Update policy using the currently gathered rollout buffer.
    """
    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)
    # Update optimizer learning rate
    self._update_learning_rate(self.policy.optimizer)
    # Compute current clip range
    clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
    # Optional: clip range for the value function
    if self.clip_range_vf is not None:
      clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

    entropy_losses = []
    pg_losses, value_losses = [], []
    clip_fractions = []

    continue_training = True
    # train for n_epochs epochs
    for epoch in range(self.n_epochs):
      approx_kl_divs = []
      # Do a complete pass on the rollout buffer
      for rollout_data in self.rollout_buffer.get(self.batch_size):
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
          # Convert discrete action from float to long
          actions = rollout_data.actions.long().flatten()

        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        values = values.flatten()
        # Normalize advantage
        advantages = rollout_data.advantages
        # Normalization does not make sense if mini batchsize == 1, see GH issue #325
        if self.normalize_advantage and len(advantages) > 1:
          advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        ratio = torch.exp(log_prob - rollout_data.old_log_prob)

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        # Logging
        pg_losses.append(policy_loss.item())
        clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
        clip_fractions.append(clip_fraction)

        if self.clip_range_vf is None:
          # No clipping
          values_pred = values
        else:
          # Clip the difference between old and new value
          # NOTE: this depends on the reward scaling
          values_pred = rollout_data.old_values + torch.clamp(
              values - rollout_data.old_values, -clip_range_vf, clip_range_vf
          )
        # Value loss using the TD(gae_lambda) target
        value_loss = F.mse_loss(rollout_data.returns, values_pred)
        value_losses.append(value_loss.item())

        # Entropy loss favor exploration
        if entropy is None:
          # Approximate entropy when no analytical form
          entropy_loss = -torch.mean(-log_prob)
        else:
          entropy_loss = -torch.mean(entropy)

        entropy_losses.append(entropy_loss.item())

        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
          log_ratio = log_prob - rollout_data.old_log_prob
          approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
          approx_kl_divs.append(approx_kl_div)

        if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
          continue_training = False
          if self.verbose >= 1:
            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
          break

        # Optimization step
        self.policy.optimizer.zero_grad()
        loss.backward(retain_graph=True)

        # Update the traces with the newly computed log_prob
        self._update_eligibility_traces(log_prob)

        # Add the eligibility trace contribution to each parameter's gradient
        for name, param in self.policy.named_parameters():
          if param.grad is not None and name in self.eligibility_traces:
            param.grad += self.trace_lambda * self.eligibility_traces[name]

        # Clip grad norm
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

      self._n_updates += 1
      if not continue_training:
        break

    explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

    # Logs
    self.logger.record("train/entropy_loss", np.mean(entropy_losses))
    self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
    self.logger.record("train/value_loss", np.mean(value_losses))
    self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
    self.logger.record("train/clip_fraction", np.mean(clip_fractions))
    self.logger.record("train/loss", loss.item())
    self.logger.record("train/explained_variance", explained_var)

    self.logger.record("traces/lambda", self.trace_lambda)
    self.logger.record("traces/gamma", self.trace_gamma)
    self.logger.record("traces/log_std", self.eligibility_traces['log_std'].mean().item())

    if hasattr(self.policy, "log_std"):
      self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    self.logger.record("train/clip_range", clip_range)
    if self.clip_range_vf is not None:
      self.logger.record("train/clip_range_vf", clip_range_vf)
