import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.optim as optim
import torch.nn.functional as F

from PPO_net import PPO_net
from PPO_utils import Replay_buffer,Para_process,Plot_result,Data_collecter



class PPO_train():
    def __init__(self, basic_config):
        self.bc = basic_config
        self.device = basic_config["DEVICE"]
        self.gamma = basic_config["GAMMA"]
        self.env_pall = basic_config['ENV_PALL']
        self.ppo_net = PPO_net(basic_config).to(self.device)
        if basic_config["LOAD_MODEL"]:
            self.ppo_net.load_model()
        self.replay_buffer = Replay_buffer(basic_config)
        self.optimizer = optim.Adam(self.ppo_net.parameters(), lr=basic_config["LR_RATE"])
        self.plot_result = Plot_result(basic_config["GAME"],
                                       f"Agent_score_gamma_{self.gamma}_lambda_{self.bc['LAMBDA']}_advnorm_"
                                       f"{self.bc['ADV_NORM']}_valnorm_{self.bc['VAL_NORM']}_epsilon_{self.bc['EPSILON']}_comment_{basic_config['COMMENT']}",
                                       "episode", "score")
        self.plot_loss = Plot_result(basic_config["GAME"], "PPO_loss", "episode", "loss")
        self.total_loss = 0

    def update(self, buffer_length):
        ## buffer_size * shape
        s = torch.tensor(self.replay_buffer.buffer['s'], dtype=torch.float).to(
            self.device)  ## torch.double equals to torch.tensor.to(float64)
        a = torch.tensor(self.replay_buffer.buffer['a'], dtype=torch.float).to(self.device)
        r = torch.tensor(self.replay_buffer.buffer['r'], dtype=torch.float).to(self.device).view(-1, 1)
        s_pi = torch.tensor(self.replay_buffer.buffer['s_pi'], dtype=torch.float).to(self.device)
        old_logp = torch.tensor(self.replay_buffer.buffer['old_logp'], dtype=torch.float).to(self.device).view(-1, 1)
        mask = torch.tensor(self.replay_buffer.buffer['mask'], dtype=torch.float).to(self.device).view(-1, 1)
        ## mask final state
        mask[-1] = 0
        pre_return = 0
        pre_advantage = 0
        returns = torch.zeros_like(r, dtype=torch.float).to(self.device)
        advantages = torch.zeros_like(r, dtype=torch.float).to(self.device)

        with torch.no_grad():
            current_value = self.ppo_net.get_value(s)
            current_q = r + self.gamma * self.ppo_net.get_value(s_pi)
            delta = current_q - current_value
            ## compute GAE advantage
            for i in reversed(range(buffer_length)):
                returns[i] = r[i] + self.gamma * pre_return * mask[i]
                advantages[i] = delta[i] + self.gamma * self.bc["LAMBDA"] * pre_advantage * mask[i]
                # print(delta[i],"  ",delta[i].size(), "                   adv", advantages[i],"_____", advantages[i].size())
                pre_advantage = advantages[i]
                pre_return = returns[i]
            ### advantages normalization......
            if self.bc["ADV_NORM"]:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # - advantages.mean())

        ## update PPO
        for _ in range(self.bc['PPO_EP']):
            for ind in BatchSampler(SubsetRandomSampler(range(buffer_length)), self.bc["BATCH_SIZE"], False):
                if self.bc['AC_STYLE']:
                    print("updating actor critic agent!!!!!!!!!!!!!!!!")
                else:
                    print("updating normal agent!!!!!!!!!!!!!!")
                new_logp = self.ppo_net.get_new_lp(s[ind], a[ind])
                ratio = torch.exp(new_logp - old_logp[ind])
                s1_loss = advantages[ind] * ratio  ## * is elementwise product for tensor
                ## policy network clipping
                s2_loss = advantages[ind] * torch.clamp(ratio, 1.0 - self.bc["EPSILON"], 1.0 + self.bc["EPSILON"])
                policy_loss = torch.mean(-torch.min(s1_loss,
                                                    s2_loss))  ## why get mean in the end, because the comparation is related to each state action pair

                if self.bc["VAL_NORM"]:
                    value_loss = torch.mean((self.ppo_net.get_value(s[ind]) - current_q[ind]).pow(2)) / returns[
                        ind].std()  # returns[ind]
                else:
                    #value_loss = torch.mean((self.ppo_net.get_value(s[ind]) - current_q[ind]).pow(2))  # returns[ind])
                    value_loss = F.smooth_l1_loss(self.ppo_net.get_value(s[ind]), current_q[ind])
                total_loss = policy_loss + value_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                ### trick norm clipping
                # torch.nn.utils.clip_grad_norm_(self.ppo_net.parameters(), 0.5)
                self.optimizer.step()
                self.total_loss = total_loss.item()

    def train(self):
        step = 0
        lr_index = 0
        m_average_score = 0
        para_collector = [Para_process.remote(self.bc) for _ in range(self.env_pall)]
        buffer_collector = Data_collecter(para_collector, self.bc)
        while step <= self.bc["MAX_TRAIN_STEP"]:
            self.replay_buffer.buffer, step_int, m_average_score, buffer_length = buffer_collector.get_buffer(m_average_score, self.ppo_net.get_weight())
            self.update(buffer_length)
            self.replay_buffer.clear()
            step += step_int

            self.plot_result(step, m_average_score)
            self.plot_loss(step, self.total_loss)
            #### learning rate adjusting ~~~~~~~~~~~~~~
            for p in self.optimizer.param_groups:
                if m_average_score < 250 and lr_index == 0:
                    p['lr'] = 0.001
                    lr_index = 1
                elif 250 <= m_average_score < 600 and lr_index == 1:
                    p['lr'] = 0.0005
                    lr_index = 2
                elif 600 <= m_average_score < 700 and lr_index == 2:
                    p['lr'] = 0.0002
                    lr_index = 3
                elif 700 <= m_average_score < 780 and lr_index == 3:
                    p['lr'] = 0.00008
                    lr_index = 4
                elif m_average_score >= 780 and lr_index == 4:
                    p['lr'] = 0.00003
                else:
                    pass
                print("current learning rate is, ", p['lr'])

            self.ppo_net.save_model()
            if m_average_score > 900:
                print("Our agent is performing over threshold, ending training")
                break
