import torch
import pandas as pd
from io import StringIO
from environment.env import SudokuEnv
from model.model import ActorCritic
from model.a2c import run_episode, update_params
import gc
import random
import wandb


EPOCH = 99999
EPSILON = 0.99
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gpu_preprocess():
    for i in range(torch.cuda.device_count()):
        print(f"# DEVICE {i}: {torch.cuda.get_device_name(i)}")
        print("- Memory Usage:")
        print(
            f"  Allocated: {round(torch.cuda.memory_allocated(i)/1024**3,1)} GB")
        print(
            f"  Cached:    {round(torch.cuda.memory_reserved(i)/1024**3,1)} GB\n")

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(DEVICE)
    print("Current Device: ", torch.cuda.current_device())
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


if __name__ == "__main__":
    gpu_preprocess()

    env = SudokuEnv({"device": DEVICE})

    model = ActorCritic().to(DEVICE)

    wandb.init(project="sudoku", entity="koios")
    wandb.watch(model)

    state = env.reset()
    best = -500
    for ep in range(EPOCH):
        if best > 100:
            print("Well Done!")
            break

        log = {
            "total_loss": [],
            "actor_loss_x": [],
            "actor_loss_y": [],
            "actor_loss_v": [],
            "critic_loss": [],
            "length": [],
            "reward": [],
            "best": best
        }

        values, log_probs, rewards = run_episode(env, model,
                                                 {"epsilon": EPSILON})

        log["total_loss"], actor_loss, log["critic_loss"] = update_params(
            model.optimizer, values, log_probs, rewards,
            gamma=model.gamma
        )
        log["actor_loss_x"] = actor_loss["x"]
        log["actor_loss_y"] = actor_loss["y"]
        log["actor_loss_v"] = actor_loss["v"]

        log["length"] = len(rewards)
        log["reward"] = sum(rewards)

        # log["last_board"] = pd.read_csv(StringIO(env.env.printBoard(printing=False)), sep=",")
        # print(ep, loss, length, rewards[-1], log["reward""])

        del values, rewards, log_probs
        torch.cuda.empty_cache()
        gc.collect()

        if log["reward"] > log["best"]:
            torch.save(model.state_dict(),
                       f"./asset/model_{ep}_{log['best']}.pth")
            log["best"] = log["reward"]

        # print(f"Epoch: {ep}")
        # print(f"""Total: {log['total_loss'].item()},
        #         Critic:{log['critic_loss'].item()},
        #       Actor: ({log['actor_loss_x'].item()}, {log['actor_loss_y'].item()}, {log['actor_loss_v'].item()}),
        #       Reward: {log['reward']}, Best: {log['best']}""")
        # print()

        wandb.log(log, step=ep)
        best = log["best"]
        del log
        gc.collect()
