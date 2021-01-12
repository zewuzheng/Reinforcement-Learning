import numpy as np
import visdom
import torch


class Replay_buffer(object):
    def __init__(self, basic_config):
        self.state_size = basic_config["STATE_SIZE"]
        self.action_size = basic_config["ACTION_SIZE"]
        self.buffer_size = basic_config["BUFFER_SIZE"]
        self.buffer_count = 0
        self.data_type = np.dtype([('s', np.float64, self.state_size), ('a', np.float64, self.action_size),
                                   ('r', np.float64), ('s_pi', np.float64, self.state_size),
                                   ('old_logp', np.float64), ('mask', np.int)])
        self.ind_buffer = {}
        self.buffer = [] #np.empty(0, dtype=self.data_type)

    def add_sample(self, s, a, r, s_pi, logp, mask, index):
        if self.buffer_count == 0:
            self.buffer = []

        if index not in self.ind_buffer:
            self.ind_buffer[index] = []
        self.ind_buffer[index].append((s, a, r, s_pi, logp, mask))
        self.buffer_count += 1
        if self.buffer_count == self.buffer_size:
            for key_ind in self.ind_buffer.keys():
                self.buffer = self.buffer + self.ind_buffer[key_ind]  #self.buffer + self.ind_buffer[key_ind])

            self.buffer = np.array(self.buffer, dtype=self.data_type)
            self.ind_buffer = {}

    def is_ready(self):
        if self.buffer_count == self.buffer_size:
            self.buffer_count = 0
            return True
        else:
            return False


class Plot_result():

    def __init__(self, env, title, xlabel=None, ylabel=None):
        self.vis = visdom.Visdom()
        self.update_flag = False
        self.env = env
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

    def __call__(
            self,
            xdata,
            ydata,
    ):
        if not self.update_flag:
            self.win = self.vis.line(
                X=np.array([xdata]),
                Y=np.array([ydata]),
                opts=dict(
                    xlabel=self.xlabel,
                    ylabel=self.ylabel,
                    title=self.title,
                ),
                env=self.env,
            )
            self.update_flag = True
        else:
            self.vis.line(
                X=np.array([xdata]),
                Y=np.array([ydata]),
                win=self.win,
                env=self.env,
                update='append',
            )

if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    basic_config = {
        "ACTION_SIZE": (3,),
        "ACTOR_LR": 0.0001,
        "AC_STYLE": False,
        "BATCH_SIZE": 128,
        "CRITIC_LR": 0.0001,
        "DEVICE": device,
        "EPSILON": 0.2,
        "ENV_RENDER": True,
        "GAMMA": 0.9,
        "GAME": "CartPole-v0",
        "INPUT_SIZE": 4,
        "INIT_WEIGHT": False,
        "LR_RATE": 1e-3,
        "MIN_BATCH_SIZE": 64,
        "MAX_TRAIN_STEP": 100000,
        "PPO_EP": 10,
        "STATE_SIZE": (4, 96, 96),
        "UPDATE_STEP": 15
    }
    test = Plot_result('test', 'test_x')
    y = [1, 2, 4 ,6 ,67,34,87,123,77,5,8,4,2,76]
    test.draw_lines(y, 'test_y', 'green')
