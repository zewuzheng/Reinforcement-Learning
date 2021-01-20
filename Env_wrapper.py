import gym
import numpy as np

class Environ():
    def __init__(self, basic_config, seed):
        self.env = gym.make(basic_config["GAME"])
        self.env.seed(seed)
        self.img_stack = basic_config["IMG_STACK"]
        self.reward_threshold = self.env.spec.reward_threshold

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * self.img_stack  # four frames for decision
        return np.array(self.stack)

    def step(self, action, game_timer):
        total_reward = 0
        reward_real = 0
        for i in range(8):
            img_rgb, reward, die, _ = self.env.step(action)
            reward_real += reward
            # don't penalize "die state"
            if die:
               reward += 100
            ## green penalty
            if np.mean(img_rgb[63:83, 38:58, 1]) > 165.0:  # 185.0: 63:83, 38:58  and game_timer > 10
                reward -= 0.03 #0.05  # reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            # done = True if (self.av_r(reward) <= -0.1 or (np.mean(img_rgb[63:83, 38:58, 1]) > 165.0 and game_timer > 20)) else False  ## -0.1
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == 4
        return np.array(self.stack), total_reward, done, die, reward_real  # done

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory



### for testing
class Environ_test():
    def __init__(self, basic_config):
        self.env = gym.make(basic_config["GAME"])
        self.env.seed(0)
        self.img_stack = basic_config["IMG_STACK"]

    def reset(self):
        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * self.img_stack  # four frames for decision
        return np.array(self.stack)

    def step(self, action, game_timer):
        total_reward = 0
        for i in range(8):
            img_rgb, reward, die, _ = self.env.step(action)
            total_reward += reward
            if die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == 4
        return np.array(self.stack), total_reward, die, _  # done

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

