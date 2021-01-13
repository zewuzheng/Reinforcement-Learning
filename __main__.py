## for rtx3080, cuda version should >= 11.0, pytorch version should >= 1.7,
## otherwise, gpu cannot be used for computation
import argparse
import torch
from PPO_training import PPO_train
from pyvirtualdisplay import Display
import ray

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('-gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('-plambda', type=float, default=0.90, metavar='G', help='lambda for GAE (default: 0.90)')
parser.add_argument('-img_stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('-seed', type=int, default=1, metavar='N', help='random seed (default: 0)')
parser.add_argument('-render', action='store_true', default = False, help='render the environment (default: False)')
parser.add_argument('-vis', action='store_true', default = False, help='use visdom (default: False)')
parser.add_argument('-store', default='models/ppo_latest.pt', metavar= 'PATH',help='store the newest param to path (default: models/ppo_latest)')
parser.add_argument('-weight', default='xavier', metavar= 'WEIGHT_INIT',help='xavier orthogonal or normal (default: xavier)')
parser.add_argument('-clip', type=float, default=0.1, metavar='G', help='set the PPO clip value epsilon (default: 0.1)')
parser.add_argument('-load', action='store_true', default = False, help='Load recent model (default: False)')
parser.add_argument('-server', action='store_true', default = False, help='Fix display problem in server (default: False)')
parser.add_argument('-actor_critic', action='store_true', default = False, help='Use actor critic agent (default: False)')
parser.add_argument('-val_norm', action='store_true', default = False, help='Value network norm (default: False)')
parser.add_argument('-adv_norm', action='store_true', default = False, help='Advantages norm (default: False)')
parser.add_argument('-use_cpu', action='store_true', default = False, help='Use cpu instead (default: False)')
parser.add_argument('-comment', default='', metavar= 'Comment',help='comment on your setting')


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

basic_config = {
    "ACTION_SIZE": (3,),
    "AC_STYLE": args.actor_critic,
    "ADV_NORM": args.adv_norm,
    "BATCH_SIZE": 128,
    "BUFFER_SIZE": 2048,
    "COMMENT": args.comment,
    "DEVICE": device,
    "EPSILON": args.clip,
    "ENV_RENDER": args.render,
    'ENV_PALL': 7,
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
    "STATE_SIZE":  (args.img_stack, 96, 96),
    "STORE": args.store,
    "USE_VIS": args.vis,
    "VAL_NORM": args.val_norm
}
if use_cuda:
    torch.cuda.manual_seed(basic_config["GAME_SEED"])

ray.init(num_cpus=8, num_gpus=1)
ppo_training = PPO_train(basic_config)
ppo_training.train()


