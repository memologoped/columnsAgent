from agents import bot
from columns_gym import envs


def main():
    file = open("examples\meaningful_text.txt", "r", encoding="utf-8")
    environment = envs.Environment(file)
    environment_state = envs.EnvironmentState(environment)
    agent = bot.RandomAgent()

    while not environment_state.is_over():
        environment_state = agent.take_action(environment_state)

    result = environment_state.show_res()
    print("Result: ", result)


if __name__ == '__main__':
    main()
