# %%
import torch
import numpy as np
from torch import nn
from torch.distributions import Beta

# %%
class PPO_net(nn.Module):
    def __init__(self, basic_config):
        super(PPO_net, self).__init__()
        self.ac_style = basic_config['AC_STYLE']
        self.device = basic_config["DEVICE"]
        if self.ac_style:
            self.policy = PPO_actor(basic_config)
            self.value = PPO_critic(basic_config)
        else:
            self.policy_mix = PPO_mix(basic_config)

    def load_model(self):
        self.load_state_dict(torch.load('models/ppo_latest.pt'))

    def save_model(self):
        torch.save(self.state_dict(), 'models/ppo_latest.pt')

    @torch.no_grad()
    def get_action(self, state):
        state1 = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
        (alpha, beta), _ = self.policy_mix.forward(state1)
        dist = Beta(alpha, beta)
        act = dist.sample()
        if act.dim() == 1 and (act.size()[0] == 1 or not act.size()):
            act_logprob = dist.log_prob(act)
        else:
            act_logprob = dist.log_prob(act).sum()
        act = act.squeeze().cpu().numpy()
        act_logprob = act_logprob.item()
        return act, act_logprob

    def get_value(self, state):
        (_, _), value = self.policy_mix.forward(state)
        return value

    def get_new_lp(self, state, action):
        (alpha, beta), _ = self.policy_mix.forward(state)
        dist = Beta(alpha, beta)
        act_logprob = dist.log_prob(action).sum(dim = 1, keepdim = True)  ## n_batch * 1
        return act_logprob


## PPO network architure for policy and value sharing the same network
class PPO_mix(nn.Module):
    def __init__(self, basic_config):
        super(PPO_mix, self).__init__()
        ## step 1: read config from basic config
        self.input_stack = basic_config["INPUT_SIZE"]
        self.init_weight = basic_config["INIT_WEIGHT"]

        ## step 2: define network architecture
        self.cnn_basic = nn.Sequential(
            nn.Conv2d(self.input_stack, 8, kernel_size=4, stride=2),  ## (1,4,96,96)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.value_fc = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.policy_fc = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU()
        )
        self.alpha_output = nn.Sequential(
            nn.Linear(100, 3),
            nn.Softplus()
        )
        self.beta_output = nn.Sequential(
            nn.Linear(100, 3),
            nn.Softplus()
        )

        ## step 3: weight initialization
        if self.init_weight:
            self.apply(self._weight_init)

    @staticmethod
    def _weight_init(m):  ## m is the Module
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, state):
        output = self.cnn_basic(state)
        value = self.value_fc(output)
        output = self.policy_fc(output)
        alpha_output = self.alpha_output(output) + 1
        beta_output = self.beta_output(output) + 1
        return (alpha_output, beta_output), value


## PPO network architure for actor ctic style method
class PPO_actor(nn.Module):
    def __init__(self, basic_config):
        super(PPO_actor, self).__init__()

    def forward(self):
        pass


class PPO_critic(nn.Module):
    def __init__(self, basic_config):
        super(PPO_critic, self).__init__()

    def forward(self):
        pass
