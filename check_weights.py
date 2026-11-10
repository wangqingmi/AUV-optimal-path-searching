import matplotlib
matplotlib.use("Agg")
import torch
from env import Auv_SimpleAction
from policy import DQNAgent

weights = [
    "result/q1.pth",
    "result/q11.pth",
    "result/Simple-5000-1000--model.pth",
    "result/Simple-10000-1000--model.pth",
    "result/Simple-10000-5000--model.pth",
]

for weight in weights:
    env = Auv_SimpleAction(max_fuels=500, image_save_path="assets/tmp_eval")
    agent = DQNAgent(False, env.observation_space.shape[0], env.action_space.n,
                     buffer_capacity=1000, batch_size=64)
    agent.model.load_state_dict(torch.load(weight, map_location="cpu"))
    agent.model.eval()
    agent.epsilon = 0.0
    state = env.reset()
    done = False
    total = 0.0
    while not done:
        state, reward, done = env.step(agent.act(state))
        total += reward
    dist = ((env.agent_x-env.goal_x)**2 + (env.agent_y-env.goal_y)**2 + (env.agent_z-env.goal_z)**2) ** 0.5
    print(f"{weight}: reward={total:.1f} steps={env.current_time} dist={dist:.3f} success={total>0}")
