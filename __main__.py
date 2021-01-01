## for rtx3080, cuda version should >= 11.0, pytorch version should >= 1.7,
## otherwise, gpu cannot be used for computation
import argparse
import torch
from PPO_training import PPO_train
from pyvirtualdisplay import Display

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('-gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('-img_stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('-seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('-render', action='store_true', default = False, help='render the environment')
parser.add_argument('-vis', action='store_true', default = False, help='use visdom')
parser.add_argument('-store', default='models/ppo_latest.pt', metavar= 'PATH',help='store the newest param to path')
parser.add_argument('-clip', type=float, default=0.1, metavar='G', help='set the PPO clip value epsilon')
parser.add_argument('-load', action='store_true', default = False, help='Load recent model')
parser.add_argument('-server', action='store_true', default = False, help='Fix display problem in server')
parser.add_argument('-actor_critic', action='store_true', default = False, help='Use actor critic agent')

args = parser.parse_args()
if args.server:
    virtual_display = Display(visible=0, size=(1, 1))
    virtual_display.start()

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
print("using device: ", device)

basic_config = {
    "ACTION_SIZE": (3,),
    "ACTOR_LR": 0.0001,
    "AC_STYLE": args.actor_critic,
    "BATCH_SIZE": 128,
    "BUFFER_SIZE": 2000,
    "CRITIC_LR": 0.0001,
    "DEVICE": device,
    "EPSILON": args.clip,
    "ENV_RENDER": False,
    "GAMMA": args.gamma,
    "GAME": "CarRacing-v0",
    "GAME_SEED": args.seed,
    "INIT_WEIGHT": True,
    "IMG_STACK": args.img_stack,
    "LR_RATE": 1e-3,
    "MAX_TRAIN_STEP": 100000,
    "PPO_EP": 10,
    "STATE_SIZE": (4, 96, 96),
    "STORE": args.store,
    "USE_VIS": args.vis,
    "LOAD_MODEL": args.load
}
if cuda:
    torch.cuda.manual_seed(basic_config["GAME_SEED"])
ppo_training = PPO_train(basic_config)
ppo_training.train()


