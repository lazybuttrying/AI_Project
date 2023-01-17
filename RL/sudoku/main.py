import torch
from env import SudokuEnv
from model import ActorCritic
import gc

import wandb


EPOCH = 99999

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def preprocess():
    for i in range(torch.cuda.device_count()):
        print(f"# DEVICE {i}: {torch.cuda.get_device_name(i)}")
        print("- Memory Usage:")
        print(
            f"  Allocated: {round(torch.cuda.memory_allocated(i)/1024**3,1)} GB")
        print(
            f"  Cached:    {round(torch.cuda.memory_reserved(i)/1024**3,1)} GB\n")


def update_params(optim, values, log_probs, rewards, clc=0.1, gamma=0.95):
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
    log_probs = {v: torch.stack(log_probs[v]).flip(dims=(0,)).view(-1)
                 for v in ["x", "y", "v"]}
    values = torch.stack(values).flip(dims=(0,)).view(-1)

    Returns = []
    retrun_ = torch.Tensor([0])
    for r in rewards:
        retrun_ = r + gamma * retrun_
        Returns.append(retrun_)

    Returns = torch.stack(Returns).view(-1)

    actor_loss = {
        v: -log_prob * (Returns - values.detach())
        for v, log_prob in log_probs.items()
    }
    critic_loss = torch.pow(Returns - values, 2)
    loss = torch.sum(actor_loss["x"]) + torch.sum(actor_loss["y"]) + \
        torch.sum(actor_loss["v"]) + torch.sum(critic_loss)*clc
    loss.backward()
    optim.step()
    return loss, len(rewards)


if __name__ == "__main__":
    preprocess()

    # exit(0)

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(DEVICE)
    print("Current Device: ", torch.cuda.current_device())
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    env = SudokuEnv({})

    model = ActorCritic().to(DEVICE)

    wandb.init(project="sudoku", entity="koios")
    wandb.watch(model)

    for ep in range(EPOCH):
        state = env.reset()

        values = []
        rewards = []
        log_probs = {
            "x": [],
            "y": [],
            "v": []
        }

        done = False

        model.optimizer.zero_grad()
        while not done:

            policy, value = model(state.float().cuda())

            action = {
                v: torch.distributions.Categorical(
                    logits=policy[v].view(-1)).sample()
                for v in ["x", "y", "v"]
            }

            state, reward, truncated, done, info = env.step(action)

            values.append(value)
            rewards.append(reward)

            for v in ["x", "y", "v"]:
                log_probs[v].append(policy[v].view(-1)[action[v]])

            if truncated:
                done = True

        loss, length = update_params(
            model.optimizer, values, log_probs, rewards
        )
        del values, rewards, log_probs
        gc.collect()
        print(ep, loss, length)

        wandb.log({"loss": loss, "length": length}, step=ep)

        # env.env.printBoard()

        # preprocess()
