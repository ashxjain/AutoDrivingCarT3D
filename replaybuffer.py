import numpy as np

class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states_a, batch_states_b, batch_next_states_a, batch_next_states_b, batch_actions, batch_rewards, batch_dones = [], [], [], [], [], [], []
    for i in ind:
      state_a, state_b, next_state_a, next_state_b, action, reward, done = self.storage[i]
      batch_states_a.append(np.array(state_a, copy=False))
      batch_states_b.append(np.array(state_b, copy=False))
      batch_next_states_a.append(np.array(next_state_a, copy=False))
      batch_next_states_b.append(np.array(next_state_b, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states_a), np.array(batch_states_b), np.array(batch_next_states_a), np.array(batch_next_states_b), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)
