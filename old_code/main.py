from columns import Environment
import agent

if __name__ == '__main__':
    file = open("meaningful_text.txt", "r", encoding="utf-8")
    while True:
        env = Environment(file)
        print(env.sym_col)
        bot = agent.RandomBot(env)
        result = bot.create_sentence()
        print("Possible text: ", result, "\n" * 3)
