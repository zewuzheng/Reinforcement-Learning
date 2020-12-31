## for rtx3080, cuda version should >= 11.0, pytorch version should >= 1.7,
## otherwise, gpu cannot be used for computation

import torch
from PPO_training import PPO_train

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
print("using device: ", device)
basic_config = {
    "ACTION_SIZE": (3,),
    "ACTOR_LR": 0.0001,
    "AC_STYLE": False,
    "BATCH_SIZE": 128,
    "BUFFER_SIZE": 2000,
    "CRITIC_LR": 0.0001,
    "DEVICE": device,
    "EPSILON": 0.1,
    "ENV_RENDER": False,
    "GAMMA": 0.9,
    "GAME": "CarRacing-v0",
    "GAME_SEED": 0,
    "INPUT_SIZE": 4,
    "INIT_WEIGHT": True,
    "LR_RATE": 1e-3,
    "MIN_BATCH_SIZE": 64,
    "MAX_TRAIN_STEP": 100000,
    "PPO_EP": 10,
    "STATE_SIZE": (4, 96, 96),
    "UPDATE_STEP": 15,
    "LOAD_MODEL": False
}
if cuda:
    torch.cuda.manual_seed(basic_config["GAME_SEED"])
ppo_training = PPO_train(basic_config)
ppo_training.train()


