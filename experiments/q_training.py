from os import mkdir
from os.path import exists, join

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

import config
from agents.bot import BaseAgent, QAgent
from columns_gym.envs import Environment
from experiments.db import RecoveryHistory
from models.nn import QNet


def epoch_history_collection(env: Environment, agent: BaseAgent, history: RecoveryHistory) -> None:
    history.gather_mode()

    with torch.no_grad():
        while not history.is_full():

            previous_env_state = env.step()  # first step with None action just returns environment state

            while True:
                agent_action = agent.take_action(previous_env_state)

                env_state, agents_reward = env.step(agent_action)

                history.put(previous_env_state, agent_action, agents_reward)

                previous_env_state = env_state


def epoch_training(estimator: QNet, optimizer, loss_func: eval, history: RecoveryHistory, batch_size: int,
                   training_epochs: int, device: torch.device) -> None:
    history.train_mode()

    train_loader = DataLoader(history, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    estimator.train()

    for i_epoch in range(1, training_epochs):
        batch_idx = 1
        loss = 0.0

        for batch_idx, (batch_step, batch_rewards) in enumerate(train_loader, start=1):
            batch_step = batch_step.to(device)
            batch_rewards = batch_rewards.to(device)

            optimizer.zero_grad()

            predicted_rewards = estimator(batch_step)

            loss = loss_func(predicted_rewards, batch_rewards)
            loss.backward()
            loss += loss.item()

            optimizer.step()

        loss /= batch_idx

        print(f'Epoch: {i_epoch} - Loss: {loss}')


def save_parameters(estimator: QNet, history_path: str, i_epoch: int) -> None:
    if not exists(history_path):
        mkdir(history_path)

    estimator.save_parameters(weights_path=join(history_path, f'weights__epoch_{i_epoch}'))


def main():
    seed = 2531
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained = True

    learning_rate = 5e-3
    batch_size = 256
    per_epoch_updating = 1
    max_game_history_size = 300
    recovery_epochs = 30_000
    training_epochs = 40

    estimator = QNet().to(device)

    env = Environment()  # TODO

    history = RecoveryHistory(max_game_history_size)

    if exists(config.weights_path) and pretrained:
        estimator.load_parameters(config.weights_path)

    optimizer = optim.Adam(estimator.parameters(), lr=learning_rate)

    loss_func = nn.MSELoss()  # for example

    agent = QAgent(estimator)

    try:
        for i_epoch in range(1, recovery_epochs + 1):
            epsilon = max(0.01, 0.2 - 0.01 * (i_epoch / 200))  # Linear annealing from 8% to 1%  0.08

            agent.update_epsilon(epsilon)
            history.reset()

            epoch_history_collection(env, agent, history)

            epoch_training(estimator, optimizer, loss_func, history, batch_size, training_epochs, device)

            if i_epoch % per_epoch_updating == 0:
                # print(f'Recovery epoch: {i_epoch} - Total reward: {env.total_reward}')

                agent.update_estimator(estimator)
                save_parameters(estimator, config.weights_history_path, i_epoch)

    except KeyboardInterrupt:
        print('Interrupted')

    finally:
        print('*Recovery closing*')
        # env.close()


if __name__ == '__main__':
    main()
