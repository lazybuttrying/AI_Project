import torch
from model.model import ActorCritic
from environment.env import SudokuEnv

device = torch.device("cuda")
model = ActorCritic()
model.load_state_dict(
    torch.load("./asset/model_156_0.7222222222222361.pth",
               map_location=device))
model.to(device)
print(
    model.eval()
)

env = SudokuEnv({"device": device})
state = env.reset()

print("=== Game Start ===")
env.env.printBoard()

for i in range(200):
    policy, value = model(state.float())
    action = {
        v: torch.distributions.Categorical(
            logits=policy[v].view(-1)).sample()
        for v in ["x", "y", "v"]
    }
    # print(action)
    state, reward, _, _, _ = env.step(action)
    # env.render()

    print(reward)
    env.env.printBoard()
