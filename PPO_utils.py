import numpy as np
import visdom
import ray
import torch
from Env_wrapper import Environ
from PPO_net import PPO_net
from collections import deque


class Replay_buffer(object):
    def __init__(self, basic_config):
        self.state_size = basic_config["STATE_SIZE"]
        self.action_size = basic_config["ACTION_SIZE"]
        self.buffer_size = basic_config["BUFFER_SIZE"]
        self.buffer_count = 0
        self.data_type = np.dtype([('s', np.float64, self.state_size), ('a', np.float64, self.action_size),
                                   ('r', np.float64), ('s_pi', np.float64, self.state_size),
                                   ('old_logp', np.float64), ('mask', np.int)])
        self.buffer = []  # np.empty(0, dtype=self.data_type)

    def add_sample(self, s, a, r, s_pi, logp, mask):
        self.buffer.append((s, a, r, s_pi, logp, mask))
        self.buffer_count += 1

    def clear(self):
        if type(self.buffer) == list:
            self.buffer.clear()
        else:
            self.buffer = list(self.buffer)
            self.buffer.clear()
        self.buffer_count = 0


@ray.remote(num_cpus=1)
class Para_process():
    """
    rollout one episode of game, and collect them into replay buffer
    input: the network  to be load

    output: replay buffer in list: [(s, a, r , s_, log_p, mask)], total score
    """

    def __init__(self, basic_config):
        self.seed = np.random.randint(10000)
        self.render = basic_config['ENV_RENDER']
        self.ppo = PPO_net(basic_config).double().to('cpu')
        self.ppo.device = torch.device('cpu')
        self.env = Environ(basic_config, self.seed)

    def collect_data(self, net_param):
        self.ppo.load_state_dict(net_param)
        buffer = []
        score = 0
        t = 0
        state = self.env.reset()
        while True:
            action, act_logp = self.ppo.get_action(state)
            state_, reward, done, die, reward_real = self.env.step(
                action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]), t)
            mask = 0 if (done or die) else 1
            buffer.append((state, action, reward, state_, act_logp, mask))
            score += reward_real
            state = state_
            if self.render:
                self.env.render()
            if done or die:
                break
        print("collect data done....")
        return buffer, score


class Data_collecter():
    """
    collect data para from Para_process, and add them to replay buffer, this is a asyn process
    """

    def __init__(self, actor, basic_config):
        self._idle_actors = deque(actor)
        self.buffer = Replay_buffer(basic_config)
        self.average_score = 0
        self.step = 0
        self._future_to_actor = {}

    def get_buffer(self, m_average_score, net_param):
        self.step = 0
        self.buffer.clear()
        self.average_score = m_average_score
        while self.buffer.buffer_count < self.buffer.buffer_size:
            while self._idle_actors:
                actors = self._idle_actors.popleft()
                future = actors.collect_data.remote(net_param)
                self._future_to_actor[future] = actors

            ready_idx, _ = ray.wait(list(self._future_to_actor.keys()))

            for future in ready_idx:
                self._idle_actors.append(self._future_to_actor[future])
                self._future_to_actor.pop(future)
                value = ray.get(future)
                self.average_score = self.average_score * 0.99 + value[1] * 0.01
                self.buffer.buffer = self.buffer.buffer + value[0]
                self.buffer.buffer_count += len(value[0])
                print(self.buffer.buffer_count)
                self.step += 1

        for future in self._future_to_actor.keys():
            self._idle_actors.append(self._future_to_actor[future])
            #ray.kill(future)
            #ray.kill(self._future_to_actor[future])
        self._future_to_actor = {}
        buffer_length = len(self.buffer.buffer)
        print("buffer size is:", buffer_length)

        return np.array(self.buffer.buffer,dtype=self.buffer.data_type), self.step, self.average_score, buffer_length


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
