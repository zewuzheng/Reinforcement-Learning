import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from Env_wrapper import Environ
from PPO_net import PPO_net
from PPO_utils import Replay_buffer
from PPO_utils import Plot_result


class PPO_train():
    def __init__(self, basic_config):
        self.PPO_epoch = basic_config["PPO_EP"]
        self.max_train_step = basic_config["MAX_TRAIN_STEP"]
        self.device = basic_config["DEVICE"]
        self.vis = basic_config["USE_VIS"]
        self.gamma = basic_config["GAMMA"]
        self.epsilon = basic_config["EPSILON"]
        self.batch_size = basic_config["BATCH_SIZE"]
        self.ac = basic_config['AC_STYLE']
        self.ppo_net = PPO_net(basic_config).double().to(self.device)
        if basic_config["LOAD_MODEL"]:
            self.ppo_net.load_model()
        self.replay_buffer = Replay_buffer(basic_config)
        self.optimizer = optim.Adam(self.ppo_net.parameters(), lr=basic_config["LR_RATE"])
        self.env = Environ(basic_config)
        self.render = basic_config["ENV_RENDER"]
        self.plot_result = Plot_result(basic_config["GAME"], "Agent_score", "episode", "score")
        self.plot_loss = Plot_result(basic_config["GAME"], "PPO_loss", "episode", "loss")
        self.total_loss = 0
        self.m_average_score = 0

    def update(self):
        ## buffer_size * shape
        s = torch.tensor(self.replay_buffer.buffer['s'], dtype=torch.double).to(
            self.device)  ## torch.double equals to torch.tensor.to(float64)
        a = torch.tensor(self.replay_buffer.buffer['a'], dtype=torch.double).to(self.device)
        r = torch.tensor(self.replay_buffer.buffer['r'], dtype=torch.double).to(self.device).view(-1, 1)
        s_pi = torch.tensor(self.replay_buffer.buffer['s_pi'], dtype=torch.double).to(self.device)
        old_logp = torch.tensor(self.replay_buffer.buffer['old_logp'], dtype=torch.double).to(self.device).view(-1, 1)

        with torch.no_grad():
            current_value = self.ppo_net.get_value(s)
            current_q = r + self.gamma * self.ppo_net.get_value(s_pi)
            advantages = current_q - current_value

        ## update PPO
        for _ in range(self.PPO_epoch):
            for ind in BatchSampler(SubsetRandomSampler(range(self.replay_buffer.buffer_size)), self.batch_size, False):
                if self.ac:
                    print("updating actor critic agent!!!!!!!!!!!!!!!!")
                else:
                    print("updating normal agent!!!!!!!!!!!!!!")
                new_logp = self.ppo_net.get_new_lp(s[ind], a[ind])
                ratio = torch.exp(new_logp - old_logp[ind])
                s1_loss = advantages[ind] * ratio  ## * is elementwise product for tensor
                ## policy network clipping
                s2_loss = advantages[ind] * torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
                policy_loss = torch.mean(-torch.min(s1_loss,
                                                    s2_loss))  ## why get mean in the end, because the comparation is related to each state action pair
                # v1_loss = (self.ppo_net.get_value(s[ind]) - current_q[ind]).pow(2)
                # v_ratio = self.ppo_net.get_value(s[ind]) / current_value[ind]
                # ## trick value function clipping
                # v2_loss = (torch.clamp(v_ratio, 1 - self.epsilon, 1 + self.epsilon) * current_value[ind] - current_q[ind]).pow(2)
                # value_loss = torch.mean(torch.min(v1_loss, v2_loss))
                value_loss = F.smooth_l1_loss(self.ppo_net.get_value(s[ind]), current_q[ind])
                total_loss = policy_loss + value_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                ### trick norm clipping
                # torch.nn.utils.clip_grad_norm_(self.ppo_net.parameters(), 0.5)
                ### fine-grained control for high speed states
                if self.m_average_score > 700:
                    self.optimizer = optim.Adam(self.ppo_net.parameters(), lr=0.0001)
                else:
                    self.optimizer = optim.Adam(self.ppo_net.parameters(), lr=0.001)
                self.optimizer.step()
                self.total_loss = total_loss.item()

    def train(self):
        score_all = []
        for step in range(self.max_train_step):
            state = self.env.reset()
            score = 0
            game_time = 0
            while True:
                print(f'Begin in {step} step')
                game_time += 1
                action, action_logp = self.ppo_net.get_action(state)  ## action in range 0 to 1
                state_, reward, done, die = self.env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]),
                                                          game_time)
                ### reward clipping
                ## reward_clip = np.clip(reward, -10, 10)
                self.replay_buffer.add_sample(state, action, reward, state_, action_logp)
                if self.render:
                    self.env.render()

                if self.replay_buffer.is_ready():
                    self.update()

                score += reward
                state = state_

                if done or die or game_time >= 10000:
                    break

            self.m_average_score = 0.99 * self.m_average_score + 0.01 * score
            if step % 10 == 0:
                self.ppo_net.save_model()
                score_all.append(score)
                if self.vis:
                    self.plot_result(step, self.m_average_score)
                    self.plot_loss(step, self.total_loss)

            if self.m_average_score > 880:
                print("Our agent is performing over threshold, ending training")
                break
