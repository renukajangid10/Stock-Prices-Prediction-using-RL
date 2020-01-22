from agent.agent import Agent
from functions import *
import sys


from keras.callbacks import TensorBoard, EarlyStopping

try:
        if len(sys.argv) != 5:
                print ("Usage: python train.py [stock] [window] [episodes]")
                exit()

        stock_name, window_size, episode_count, initial_money = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

        agent = Agent(window_size)
        data = getStockDataVec(stock_name)
        l = len(data) - 1
        batch_size = 32

        for e in range(episode_count + 1):
                print ("Episode " + str(e) + "/" + str(episode_count))
                state = getState(data, 0, window_size + 1)
                starting_money = initial_money
                total_profit = 0
                agent.inventory = []

                for t in range(l):
                        action = agent.act(state)

                        # sit
                        next_state = getState(data, t + 1, window_size + 1)
                        reward = 0

                        if action == 1 and starting_money >= data[t]: # buy
                                agent.inventory.append(data[t])
                                starting_money -= data[t]
                                print ("Buy: " + formatPrice(data[t]))

                        elif action == 2 and len(agent.inventory) > 0: # sell
                                bought_price = agent.inventory.pop(0)
                                reward = max(data[t] - bought_price, 0)
                                total_profit += data[t] - bought_price
                                starting_money += data[t]
                                print ("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

                        done = True if t == l - 1 else False
                        invest = ((starting_money - initial_money) / initial_money)
                        agent.memory.append((state, action, reward, next_state, done))
                        state = next_state

                        if done:
                                print ("--------------------------------")
                                print ("Total Profit: " + formatPrice(total_profit))
                                print ("--------------------------------")

                        if len(agent.memory) > batch_size:
                                agent.expReplay(batch_size)

                if e % 10 == 0:
                        agent.model.save("models/model_ep" + str(e))
except Exception as e:
        print("Error occured: {0}".format(e))
finally:
        exit()
