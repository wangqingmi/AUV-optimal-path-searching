import matplotlib
matplotlib.use("Agg")
import torch
from env import Auv_changeable
from policy import DQNAgent

env = Auv_changeable(max_fuels=500, image_save_path="assets/eval_q2")
agent = DQNAgent(False, env.observation_space.shape[0], env.action_space.n,
                 buffer_capacity=1000, batch_size=64)
agent.model.load_state_dict(torch.load("result/q2.pth", map_location="cpu"))
agent.model.eval()
agent.epsilon = 0.0

for episode in range(5):
    state = env.reset()
    done = False
    total = 0.0
    while not done:
        state, reward, done = env.step(agent.act(state))
        total += reward
    print(f"episode={episode + 1} reward={total:.1f} steps={env.current_time} success={total>0}")
