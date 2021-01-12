import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from Env_wrapper import Base_env
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
        self.plambda = basic_config["LAMBDA"]
        self.adv_norm = basic_config["ADV_NORM"]
        self.val_norm = basic_config["VAL_NORM"]
        self.epsilon = basic_config["EPSILON"]
        self.batch_size = basic_config["BATCH_SIZE"]
        self.ac = basic_config['AC_STYLE']
        self.env_pall = basic_config['ENV_PALL']
        self.ppo_net = PPO_net(basic_config).double().to(self.device)
        if basic_config["LOAD_MODEL"]:
            self.ppo_net.load_model()
        self.replay_buffer = Replay_buffer(basic_config)
        self.optimizer = optim.Adam(self.ppo_net.parameters(), lr=basic_config["LR_RATE"])
        self.env = Base_env(basic_config)
        self.render = basic_config["ENV_RENDER"]
        self.plot_result = Plot_result(basic_config["GAME"],
                                       f"Agent_score_gamma_{self.gamma}_lambda_{self.plambda}_advnorm_"
                                       f"{self.adv_norm}_valnorm_{self.val_norm}_epsilon_{self.epsilon}",
                                       "episode", "score")
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
        mask = torch.tensor(self.replay_buffer.buffer['mask'], dtype=torch.double).to(self.device).view(-1, 1)
        ## mask final state
        mask[-1] = 0
        pre_return = 0
        pre_advantage = 0
        returns = torch.zeros_like(r, dtype=torch.double).to(self.device)
        advantages = torch.zeros_like(r, dtype=torch.double).to(self.device)

        with torch.no_grad():
            current_value = self.ppo_net.get_value(s)
            current_q = r + self.gamma * self.ppo_net.get_value(s_pi)
            delta = current_q - current_value
            ## compute GAE advantage
            for i in reversed(range(self.replay_buffer.buffer_size)):
                returns[i] = r[i] + self.gamma * pre_return * mask[i]
                advantages[i] = delta[i] + self.gamma * self.plambda * pre_advantage * mask[i]
                # print(delta[i],"  ",delta[i].size(), "                   adv", advantages[i],"_____", advantages[i].size())
                pre_advantage = advantages[i]
                pre_return = returns[i]
            ### advantages normalization......
            if self.adv_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # - advantages.mean())

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

                if self.val_norm:
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
        while step <= self.max_train_step:
            state = self.env.reset()
            score = 0
            lr_index = 0
            env_index = list(range(self.env_pall))
            for t in range(1000):
                print(f'Begin in {step} step')
                temp_state = []
                rm_index = []
                action, action_logp = self.ppo_net.get_action(state)  ## action in range 0 to 1
                trans = self.env.step(action, env_index, t)
                for i, j in zip(env_index, range(len(env_index))):
                    [state_, reward, done, die, reward_real] = list(trans[j])
                    mask = 0 if (done or die) else 1
                    if len(env_index) == 1:
                        self.replay_buffer.add_sample(state, action, reward, state_,
                                                      action_logp, mask, i)
                    else:
                        self.replay_buffer.add_sample(state[j], action if self.env_pall == 1 else action[j], reward,
                                                      state_,
                                                      action_logp if self.env_pall == 1 else action_logp[j], mask, i)

                    if self.replay_buffer.is_ready():
                        self.update()

                    if not (done or die):
                        temp_state.append(state_)
                    else:
                        rm_index.append(i)
                        step += 1

                        if step % 10 == 0:
                            self.ppo_net.save_model()
                            if self.vis:
                                self.plot_result(step, self.m_average_score)
                                self.plot_loss(step, self.total_loss)

                    score += reward_real

                if rm_index:
                    for ind in rm_index:
                        env_index.remove(ind)

                if self.render:
                    self.env.render()

                state = np.array(temp_state)

                if not env_index:
                    break

            score = score / self.env_pall
            self.m_average_score = (1 - 0.01 * self.env_pall) * self.m_average_score + 0.01 * self.env_pall * score

            #### learning rate adjusting ~~~~~~~~~~~~~~
            for p in self.optimizer.param_groups:
                if self.m_average_score < 250 and lr_index == 0:
                    p['lr'] = 0.0015
                    lr_index = 1
                elif 250 <= self.m_average_score < 600 and lr_index == 1:
                    p['lr'] = 0.0008
                    lr_index = 2
                elif 600 <= self.m_average_score < 740 and lr_index == 2:
                    p['lr'] = 0.0003
                    lr_index = 3
                elif 740 <= self.m_average_score < 800 and lr_index == 3:
                    p['lr'] = 0.0001
                    lr_index = 4
                elif self.m_average_score >= 800 and lr_index == 4:
                    p['lr'] = 0.00003
                else:
                    pass
                print("current learning rate is, ", p['lr'])

            if self.m_average_score > 900:
                print("Our agent is performing over threshold, ending training")
                break
