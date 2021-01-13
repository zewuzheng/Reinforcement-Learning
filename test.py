from PPO_net import PPO_net
from Env_wrapper import Environ_test
import argparse
import torch
import numpy as np


parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('-img_stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('-seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('-actor_critic', action='store_true', default = False, help='Use actor critic agent (default: False)')
args = parser.parse_args()

device = torch.device('cpu')
print("using device: ", device)

basic_config = {
    "ACTION_SIZE": (3,),
    "AC_STYLE": True,
    "DEVICE": device,
    "GAME": "CarRacing-v0",
    "GAME_SEED": args.seed,
    "INIT_WEIGHT": 'xavier',
    "IMG_STACK": args.img_stack,
    "STATE_SIZE":  (args.img_stack, 96, 96),
    "STORE": "models/ppo_latest.pt"
}


ppo = PPO_net(basic_config).double().to(device)
env = Environ_test(basic_config)
ppo.load_model()
average_reward = []
for _ in range(10):
    round_reward = 0
    state = env.reset()
    while True:
        action, _ = ppo.get_action(state)
        state_, reward, die, _ = env.step(action* np.array([2., 1., 1.]) + np.array([-1., 0., 0.]), 10)
        round_reward += reward
        env.render()
        state = state_
        if die:
            break
    average_reward.append(round_reward)
    print(f"The reward of current round is {round_reward}")
    print(f"The average reward is {np.mean(average_reward)}")
##
from PPO_utils import Replay_buffer
# action = torch.randn((4, 3,))
# dist = Beta(2, 2)
# print(dist.log_prob(action).sum(dim = 1,keepdim = True))


#%%
## elementwise product
# a = torch.randn(3)
# b = torch.randn(3)
# print(a*b)


#%%
# ppo = PPO_net(basic_config)
# print(ppo.state_dict()["policy_mix.cnn_basic.0.bias"])
# ppo.load_model()
# print(ppo.state_dict()["policy_mix.cnn_basic.0.bias"])

#%%
# a = torch.randn(1, requires_grad= True)
# b = torch.randn(1, requires_grad= True)
# c = a*b
# d = torch.clamp(c, -0.5, 0.5)
# c.retain_grad()
# print("a",a.is_leaf)
# print("b",b.is_leaf)
# print("c",c.is_leaf)
# pint("d",d.is_leaf)
# print(d)
# d.backward)
# print(c.grad)

#%%
# alpha = torch.tensor([[1.5, 2, 2], [1, 2, 3]]).float().to(device)
# beta = torch.tensor([[2, 3, 4], [2,2,2]]).float().to(device)

# alpha = torch.tensor(2).float().to(device)
# beta = torch.tensor(3).float().to(device)
# m = Beta(alpha, beta)
# act = m.sample()
# if not act.size():
#     print("no action size")
# if act.dim() == 1 and act.size()[0] == 1:
#     print('single action')
#     act_logprob = m.log_prob(act)
# else:
#     print('multi-action')
#     act_logprob = m.log_prob(act).sum()
# print(act_logprob)


#%%
# import gym
# import matplotlib.pyplot as plt
# import numpy as np
# env = gym.make('CarRacing-v0')
# state = env.reset()
# for _ in range(30):
#     action = [0.5,0.5,0]
#     state_,_,_,_ = env.step(action)
#
# gray = np.dot(state_[..., :], [0.299, 0.587, 0.114])
# print(np.mean(state_[63:83, 38:58, 1]))
# plt.imshow(state_)
# plt.show()

#%%
# import torch
# import torch.nn as nn
# a = PPO_net(basic_config)
# a.load_model('models/ppo_latest_ac.pt')
# print(a.state_dict())