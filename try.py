import ray
import argparse
import torch
import numpy as np

from PPO_utils import Replay_buffer
from pyvirtualdisplay import Display
from PPO_net import PPO_net
from Env_wrapper import Base_env

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('-gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('-plambda', type=float, default=0.90, metavar='G', help='lambda for GAE (default: 0.90)')
parser.add_argument('-img_stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('-seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('-render', action='store_true', default=False, help='render the environment (default: False)')
parser.add_argument('-vis', action='store_true', default=False, help='use visdom (default: False)')
parser.add_argument('-store', default='models/ppo_latest.pt', metavar='PATH',
                    help='store the newest param to path (default: models/ppo_latest)')
parser.add_argument('-weight', default='xavier', metavar='WEIGHT_INIT',
                    help='xavier orthogonal or normal (default: xavier)')
parser.add_argument('-clip', type=float, default=0.1, metavar='G', help='set the PPO clip value epsilon (default: 0.1)')
parser.add_argument('-load', action='store_true', default=False, help='Load recent model (default: False)')
parser.add_argument('-server', action='store_true', default=False,
                    help='Fix display problem in server (default: False)')
parser.add_argument('-actor_critic', action='store_true', default=False, help='Use actor critic agent (default: False)')
parser.add_argument('-val_norm', action='store_true', default=False, help='Value network norm (default: False)')
parser.add_argument('-adv_norm', action='store_true', default=False, help='Advantages norm (default: False)')
parser.add_argument('-use_cpu', action='store_true', default=False, help='Use cpu instead (default: False)')

args = parser.parse_args()
if args.server:
    virtual_display = Display(visible=0, size=(1, 1))
    virtual_display.start()

use_cuda = torch.cuda.is_available()
if not args.use_cpu and use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print("using device: ", device)
device = torch.device('cuda')

basic_config = {
    "ACTION_SIZE": (3,),
    "AC_STYLE": args.actor_critic,
    "ADV_NORM": args.adv_norm,
    "BATCH_SIZE": 128,
    "BUFFER_SIZE": 2048,
    "DEVICE": device,
    "EPSILON": args.clip,
    "ENV_RENDER": args.render,
    "ENV_PALL": 2,
    "GAMMA": args.gamma,
    "GAME": "CarRacing-v0",
    "GAME_SEED": args.seed,
    "INIT_WEIGHT": args.weight,
    "IMG_STACK": args.img_stack,
    "LR_RATE": 1e-3,
    "LOAD_MODEL": args.load,
    "LAMBDA": args.plambda,
    "MAX_TRAIN_STEP": 100000,
    "PPO_EP": 10,
    "STATE_SIZE": (args.img_stack, 96, 96),
    "STORE": args.store,
    "USE_VIS": args.vis,
    "VAL_NORM": args.val_norm
}
ray.init(num_cpus=8, num_gpus=1)
ppo_net = PPO_net(basic_config).double().to(torch.device('cuda'))
env = Base_env(basic_config)
buffer = Replay_buffer(basic_config)
state = env.reset()
print(type(state))
print(state.shape)
action, log_p = ppo_net.get_action(state)
print("action:", action, "log_p:", log_p)
trans = env.step(action)
for i in range(basic_config['ENV_PALL']):
    [state_, reward, done, die, reward_real] = list(trans[i])
    mask = 0 if (done or die) else 1
    buffer.add_sample(state[i], action if basic_config['ENV_PALL'] == 1 else action[i], reward, state_,
                      log_p if basic_config['ENV_PALL'] == 1 else log_p[i], mask, 1)

