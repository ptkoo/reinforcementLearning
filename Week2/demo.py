import gym
from gym.utils import play

env = play.play(gym.make('LunarLander-v2', render_mode='rgb_array').env, zoom=1,  keys_to_action={"s":0, "1":1,"2":2,"3":3}, noop=0)   