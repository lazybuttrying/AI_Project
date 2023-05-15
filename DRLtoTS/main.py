# %%
import pickle as pkl
from dataset import Dataset
from deepRL.environment import ForecastYieldEnv

from deepRL.policy_net import PG_Args, PolicyGradient

from log import LOGGER

EPOCHS = 100
loss_history = []

# with open('main_obj.p', 'rb') as file:
#     pg = pkl.load(file)
#     env = pkl.load(file)


pg = PolicyGradient(PG_Args(121, 0.01, 0.99))
env = ForecastYieldEnv(pg, Dataset())

with open('main_obj.p', 'wb') as file:
    pkl.dump(pg, file)
    pkl.dump(env, file)


# %%

if __name__ == '__main__':
    for i in range(EPOCHS):
        LOGGER.info(f"\n\n\nEpoch: {i}")
        losses = []
        past_state = env.reset()

        for j in range(pg.pg.len_episode):
            LOGGER.info(f"\n\tStep: {j} _ {pg.pg.len_episode}")
            observation, reward, _, info, done = env.step(past_state)

            # LOGGER.info(
            #     f"observation: {observation[0,0,:]}, past_state: {past_state[0,0,:]}")

            past_state = observation
            losses.append(info["loss"])
            LOGGER.info(f"info: {info}")
            if done:
                LOGGER.info(f"Done: {reward}")
                break
        loss_history += losses

    print([a.item() for a in loss_history])

# %%