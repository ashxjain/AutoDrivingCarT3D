import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d_block(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Dropout2d(0.1)
    )

def transition_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(0.1)
    )

# Creating the architecture of the Neural Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # size after below block = input_size - 2
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0),
            nn.ReLU()
        )
        # size after below block = input_size - 2 - 4
        self.convblock2 = conv2d_block(in_channels=16, out_channels=16)
        # size after below block = (input_size - 2 - 4) / 2
        self.transitionblock = transition_block(in_channels=16, out_channels=10)
        # size after below block = ((input_size - 2 - 4) / 2) - 4
        self.convblock3 = conv2d_block(in_channels=10, out_channels=16)
        # size after below block = ((input_size - 2 - 4) / 2) - 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )
        conv_output_size_h = ((state_dim[0] - 2 - 4) / 2) - 4
        conv_output_size_w = ((state_dim[1] - 2 - 4) / 2) - 4
        linear_input_size = conv_output_size_w * conv_output_size_h * 10
        self.fc = nn.Linear(linear_input_size, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.transitionblock(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.fc(x.view(x.size(0), -1))
        x = self.max_action * torch.tanh(x)
        return x

# Creating the architecture of the Neural Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        # size after below block = input_size - 2
        self.c1_convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0),
            nn.ReLU()
        )
        # size after below block = input_size - 2 - 4
        self.c1_convblock2 = conv2d_block(in_channels=16, out_channels=16)
        # size after below block = (input_size - 2 - 4) / 2
        self.c1_transitionblock = transition_block(in_channels=16, out_channels=10)
        # size after below block = ((input_size - 2 - 4) / 2) - 4
        self.c1_convblock3 = conv2d_block(in_channels=10, out_channels=16)
        # size after below block = ((input_size - 2 - 4) / 2) - 4
        self.c1_convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )
        conv_output_size_h = ((state_dim[0] + action_dim - 2 - 4) / 2) - 4
        conv_output_size_w = ((state_dim[1] - 2 - 4) / 2) - 4
        linear_input_size = conv_output_size_w * conv_output_size_h * 10
        self.c1_fc = nn.Linear(linear_input_size, 1)

        # Defining the second Critic neural network
        # size after below block = input_size - 2
        self.c2_convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0),
            nn.ReLU()
        )
        # size after below block = input_size - 2 - 4
        self.c2_convblock2 = conv2d_block(in_channels=16, out_channels=16)
        # size after below block = (input_size - 2 - 4) / 2
        self.c2_transitionblock = transition_block(in_channels=16, out_channels=10)
        # size after below block = ((input_size - 2 - 4) / 2) - 4
        self.c2_convblock3 = conv2d_block(in_channels=10, out_channels=16)
        # size after below block = ((input_size - 2 - 4) / 2) - 4
        self.c2_convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )
        conv_output_size_h = ((state_dim[0] + action_dim - 2 - 4) / 2) - 4
        conv_output_size_w = ((state_dim[1] - 2 - 4) / 2) - 4
        linear_input_size = conv_output_size_w * conv_output_size_h * 10
        self.c2_fc = nn.Linear(linear_input_size, 1)

    def forward(self, x, u):
      xu = torch.cat([x, u], 2)
      # Forward-Propagation on the first Critic Neural Network
      x1 = self.c1_convblock1(xu)
      x1 = self.c1_convblock2(x1)
      x1 = self.c1_transitionblock(x1)
      x1 = self.c1_convblock3(x1)
      x1 = self.c1_convblock4(x1)
      x1 = self.c1_fc(x1.view(x1.size(0), -1))
      # Forward-Propagation on the second Critic Neural Network
      x2 = self.c2_convblock1(xu)
      x2 = self.c2_convblock2(x2)
      x2 = self.c2_transitionblock(x2)
      x2 = self.c2_convblock3(x2)
      x2 = self.c2_convblock4(x2)
      x2 = self.c2_fc(x2.view(x2.size(0), -1))
      return x1, x2

    def Q1(self, x, u):
      xu = torch.cat([x, u], 2)
      x1 = self.c1_convblock1(xu)
      x1 = self.c1_convblock2(x1)
      x1 = self.c1_transitionblock(x1)
      x1 = self.c1_convblock3(x1)
      x1 = self.c1_convblock4(x1)
      x1 = self.c1_fc(x1.view(x1.size(0), -1))
      return x1

# Building the whole Training Process into a class
class T3D(object):
  
  def __init__(self, state_dim, action_dim, max_action):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.max_action = torch.Tensor(max_action).to(device)
    self.actor = Actor(state_dim, action_dim, self.max_action).to(device)
    self.actor_target = Actor(state_dim, action_dim, self.max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    
  def select_action(self, state):
    state = state.unsqueeze(0)
    return self.actor(state).cpu().data.numpy().flatten()

  def expand_action(self, action, state_dim):
    vel = torch.stack([action[:,0]]*state_dim)
    vel = vel.T # (bsize,s_size)
    vel = vel.unsqueeze(1).unsqueeze(1) # (bsize, 1, 1, s_size)

    ang = torch.stack([action[:,1]]*state_dim)
    ang = ang.T # (bsize,s_size)
    ang = ang.unsqueeze(1).unsqueeze(1) # (bsize, 1, 1, s_size)

    return torch.cat([vel,ang], 2)

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for it in range(iterations):
      
      # Step 4: We sample a batch of transitions (s, s', a, r) from the memory
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)
      
      # Step 5: From the next state s', the Actor target plays the next action a'
      next_action = self.actor_target(next_state).squeeze(1)

      # Step 6: We add Gaussian noise to this next action a' and we clamp it in a range of values supported by the environment
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action[:, 0] = (next_action[:, 0] + noise[:, 0]).clamp(0, self.max_action[0])
      next_action[:, 1] = (next_action[:, 1] + noise[:, 1]).clamp(-self.max_action[1], self.max_action[1])

      # Step 6.1: Because we want to append action of dim=[batch_size,] to state of dim=[batch_size,1,32,32]
      #           We expand action to this dim=[batch_size,1,1,32] and then concat it with state
      next_action = self.expand_action(next_action, batch_states.shape[-1])
      action = self.expand_action(action, batch_states.shape[-1])

      # Step 7: The two Critic targets take each the couple (s', a') as input and return two Q-values Qt1(s',a') and Qt2(s',a') as outputs
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + gamma * min(Qt1, Qt2), where gamma is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state, action)
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        q1_action = self.actor(state)
        q1_action = q1_action.squeeze(1)
        q1_action = self.expand_action(q1_action, state.shape[-1])
        actor_loss = -self.critic.Q1(state, q1_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
  
  # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename), map_location=torch.device('cpu')))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename), map_location=torch.device('cpu')))
