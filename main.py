from agents import bot
from columns_gym import envs


def main():
    with open("examples/meaningful_text.txt", mode="r", encoding="utf-8") as file:
        environment = envs.Environment(file)
    environment_state = envs.EnvironmentState(environment)

    agent = bot.RandomAgent()

    while not environment_state.is_over():
        environment_state = agent.take_action(environment_state)

    result = environment_state.show_res()
    print("Result: ", result)


if __name__ == '__main__':
    main()


# TODO check thar error
"""

Traceback (most recent call last):
  File "columnsAgent/main.py", line 20, in <module>
    main()
  File "columnsAgent/main.py", line 13, in main
    environment_state = agent.take_action(environment_state)
  File "columnsAgent/agents/bot.py", line 43, in take_action
    return self.choose_symbol(env_state)
  File "columnsAgent/agents/bot.py", line 67, in choose_symbol
    top_2 = list(np.random.choice(sym_col[last_col + 1],
  File "mtrand.pyx", line 954, in numpy.random.mtrand.RandomState.choice
ValueError: Cannot take a larger sample than population when 'replace=False'

"""