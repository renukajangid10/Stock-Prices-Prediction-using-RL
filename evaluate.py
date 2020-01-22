import keras
from keras.models import load_model
import seaborn as sns


from agent.agent import Agent
from functions import *
import sys
import numpy as np

import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

try:
        if len(sys.argv) != 3:
                print ("Usage: python evaluate.py [stock] [model]")
                exit()

        stock_name, model_name= sys.argv[1], sys.argv[2]
        model = load_model("models/" + model_name)
        window_size = model.layers[0].input.shape.as_list()[1]

        agent = Agent(window_size, True, model_name)
        data = getStockDataVec(stock_name)
        l = len(data) - 1
        batch_size = 32
        sit_count=0

        state = getState(data, 0, window_size + 1)
        initial_money = 10000
        starting_money = initial_money
        total_profit = 0
        agent.inventory = []
        sell_count=0

        #Setup our plot
        #fig, ax = plt.subplots()

        #timeseries_iter = 0
        #plt_data = []
        states_sell = []
        states_buy = []

        for t in range(l):
                action = agent.act(state)
                if action==2:
                        sell_count+=1
                        if sell_count>=40:
                                action=1
                                sell_count=0
                if sit_count>=10:
                        reward = -500
                        action=1
                        sit_count=0

                # sit
                next_state = getState(data, t + 1, window_size + 1)
                reward = 0
                if action==0:
                        sit_count+=1

                if action == 1 and starting_money >= data[t]: # buy
                        agent.inventory.append(data[t])
                        initial_money -= data[t]
                        #plt_data.append((timeseries_iter, data[t], 'Buy'))
                        states_buy.append(t)
                        print ("Buy : " ,formatPrice(data[t]))

                elif action == 2 and len(agent.inventory) > 2: # sell
                        bought_price = agent.inventory.pop(0)
                        sell_count = 0
                        reward = max(data[t] - bought_price, 0)
                        initial_money += data[t]
                        total_profit += data[t] - bought_price
                        #plt_data.append((timeseries_iter, data[t], 'Sell'))
                        states_sell.append(t)
                        print ("Sell: ",formatPrice(data[t]))
                        print("Profit: ", formatPrice(data[t] - bought_price))

                
                #timeseries_iter += 1
                done = True if t == l - 1 else False
                invest = (total_profit / starting_money) * 100
                total_gains = initial_money - starting_money
                agent.memory.append((state, action, reward, next_state, done))
                state = next_state

                if done:
                        print ("--------------------------------")
                        print (stock_name + " Total Profit: " + formatPrice(total_profit))
                        print ("--------------------------------")
                        print ("Profit Percentage : ", invest)
                
                if len(agent.memory) > batch_size:
                                agent.expReplay(batch_size) 

        #plt_data = np.array(plt_data)
        #ax.plot(plt_data[:, 0], plt_data[:, 1])
        #Display our plots
        #plt.show()
        fig = plt.figure(figsize = (15,5))
        plt.plot(data, color='g', lw=2.)
        plt.plot(data, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
        plt.plot(data, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
        plt.title('total profit %f, Profit Percentage %f%%'%(total_profit, invest))
        plt.legend()
        plt.show()
                                
except Exception as e:
        print("Error is: " + e)
finally:
        exit()
