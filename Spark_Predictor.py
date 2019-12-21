_Author_ = "******"

# Predictor for Spark

from keras.models import model_from_json
import keras
import pandas as pd
from Initializer import Initialize
from keras import backend as K
import time
from sklearn.externals import joblib
import os
import numpy as np
init_object = Initialize()
#import pysftp
import requests
import json
# Define Thresholds
he = 10.0
ae = 10.0
le = 6.0
hd = 3000
ad = 3000
ld = 2500
global_energy_forecast = 10.0
global_data_forecast = 2500


f = open("pattern_spark.txt","w")
f.close()
def load_model():
    # Load the model
    K.clear_session()
    json_file_co = open(init_object.model_path + 'model1_master.json', 'r')
    loaded_model_co_json = json_file_co.read()
    json_file_co.close()
    loaded_model_co = model_from_json(loaded_model_co_json)
    # load weights into new model
    loaded_model_co.load_weights(init_object.model_path + "model1_master.h5")
    print("Loaded model from disk")
    return loaded_model_co

# Load three scalars

# Perform the online Q-Learning Here
class Env():
    # Create the environment with states and actions to perform reinforcement learning
    def __init__(self):
        self.rows = 9 # The MXM matrix for state action space
        self.columns = 3
        self.posX = 0  # The initial X and Y coordinate
        self.posY = 0
        self.endX = self.rows-1
        self.endY = self.columns - 1
        self.actions = [0,1,2]    # Move left,right,up and down
        self.stateCount = self.rows * self.columns
        self.actionCount = len(self.actions)
        #self.adaptaion_file = "adaptation.txt"
        self.file = open("pattern_spark_rl.txt","a")
        self.patterns = ["co", "su", "sc"]
        self.input_json_path = "$PATH/jsons/"
        self.output_json_path = "$PATH/jsons/output.json"
        self.pattern_map = {"co": "30", "su": "10", "sc": "20"}


    def reset(self):
        # To reset the board back to the original state
        self.posX = 0
        self.posY = 0
        self.done = False
        return 0,0,False

    def step(self,action,energy_forecast,traffic_forecast):
        # Get the energy and traffic forecast and give the corresponding action based on the forecast
        print("step")
        if action == 0:
            # Pattern SU
            pattern = "su"
            start_nw = time.time()
            #adaptation_file = open(self.adaptaion_file,"w")
            #adaptation_file.write("ada 10")
            #adaptation_file.close()
            #srv.put(remote_file)
            request_json = {}
            request_json["pattern"] = self.pattern_map[pattern]
            request_json_obj = json.dumps(request_json)
            response = requests.post(init_object.request_url, request_json_obj)
            end_nw = time.time()
            print ("network time ", str(end_nw-start_nw))

            self.posX = 0
            self.file.write("su\n")

        elif action == 1:
            # Pattern SC
            pattern = "sc"
            start_nw = time.time()
            #adaptation_file = open(self.adaptaion_file,"w")
            #adaptation_file.write("ada 20")
            #adaptation_file.close()
            #srv.put(remote_file)
            request_json = {}
            request_json["pattern"] = self.pattern_map[pattern]
            request_json_obj = json.dumps(request_json)
            response = requests.post(init_object.request_url, request_json_obj)
            end_nw = time.time()
            print("network time ", str(end_nw - start_nw))
            self.posX = 1
            self.file.write("sc\n")

        elif action == 2:
            # Pattern CO
            pattern =  "co"
            start_nw = time.time()
            #adaptation_file = open(self.adaptaion_file,"w")
            #adaptation_file.write("ada 30")
            #adaptation_file.close()
            request_json = {}
            request_json["pattern"] = self.pattern_map[pattern]
            request_json_obj = json.dumps(request_json)
            response = requests.post(init_object.request_url, request_json_obj)
            #srv.put(remote_file)
            end_nw = time.time()
            print("network time ", str(end_nw - start_nw))
            self.posX = 2
            self.file.write("co\n")

        reward = None
        if (energy_forecast >= he) and traffic_forecast >=hd:
            self.posY = 0
            # reward = ((he-total_energy) +  (hd-traffic_forecast))/100
            reward = -10

        elif energy_forecast >= he and (traffic_forecast>=ld and traffic_forecast<ad):
            self.posY = 1
            # reward = ((he-total_energy) + (traffic_forecast-ad) + ad)/100 # make the reward high
            reward = -9

        elif energy_forecast >= he and (traffic_forecast<=ld):
            self.posY = 2
            # reward = ((he-total_energy) + (traffic_forecast-ld) + ld)/100  # Make the reward very high for data traffic
            reward = -8

        elif (energy_forecast >= le and energy_forecast <ae) and (traffic_forecast>=hd):
            self.posY = 3
            # reward = ((total_energy-ae) + (hd-traffic_forecast))/100  # Add the benefit for energy
            reward = -4 # Add the benefit for energy

        elif (energy_forecast >=le and energy_forecast <ae) and (traffic_forecast>=ld and traffic_forecast<ad):
            self.posY = 4
            # reward = ((total_energy-ae) + (traffic_forecast-ad) )/100 # Both are average
            reward = -3
        elif (energy_forecast >= le and energy_forecast<ae) and (traffic_forecast<=ld):
            self.posY = 5
            # reward = ((total_energy-ae) + (traffic_forecast-ld) + ld)/100
            reward = -2

        elif (energy_forecast<=le) and (traffic_forecast >= hd):
            self.posY = 6
            # reward = ((total_energy - le) + (hd - traffic_forecast)  + le)/100# Add the benefit for energy
            reward = 0
        elif energy_forecast<= le and (traffic_forecast >= ld and traffic_forecast < ad):
            self.posY = 7
            #reward = ((total_energy - le) + (traffic_forecast - ad) + le) / 100  # Both are average
            reward = 8
        elif energy_forecast<= le and (traffic_forecast<= ld):
            self.posY = 8
            #reward = ((total_energy - le) + (traffic_forecast - ld) + le + ld) / 100
            reward = 1

        print("Reward ", reward)

        done = self.posX == self.endX

        # 9 will represent the state 3*X + y  = 9
        next_state = self.columns * self.posY + self.posX  # Send X and Y as numbers as multiples than sending as pair
        # if done:
        #    reward = 2 # This will be basically the result of the simulation in real case

        posX = self.posX
        posy = self.posY
        return next_state, reward, done, posX, posy


    def random_action(self):
        # Return a random action from the set of available actions
        return np.random.choice(self.actions)


env = Env() # Initalize the environment
class Spark_Predictor():
    loaded_model = None
    scalar = None
    normal = 1
    high = 0
    low = 0
    current_pattern = 10

    def __init__(self):
        self.loaded_model = load_model()
        self.scalar_energy = joblib.load("scaler_co.save")
        self.scalar_traffic = joblib.load("scaler.save")
        self.main_energy_list  = []
        self.main_data_traffic_list  =[]

        self.qTable = np.random.rand(env.stateCount, env.actionCount).tolist()
        #self.qTable = np.load("Qtable.txt.npy")   # Load the pre-trained table
        #self.qTable= self.qTable.tolist()
        print(self.qTable)
        self.state = 2
        self.gamma = 0.02  # Learning rate
        self.epsilon = 0.08
        self.decay = 0.2
        self.current_pattern = 2


    def predictor(self,analyze,energy_forecast=0,data_traffic_forecast=0):
        global global_energy_forecast
        global global_data_forecast
        print ("reinforce")


        action = 0
        #if np.random.uniform() < self.epsilon:
        #    action = env.random_action()
        #else:
        action = self.qTable[self.state].index(max(self.qTable[self.state]))

        print("Action " + str(action))

        if (data_traffic_forecast<0):
            data_traffic_forecast = -1*data_traffic_forecast


        if energy_forecast==0 and global_energy_forecast >0:
            energy_forecast = global_energy_forecast
        elif energy_forecast > 0:
            global_energy_forecast = energy_forecast

        if data_traffic_forecast == 0 and global_data_forecast>0:
            data_traffic_forecast = global_data_forecast
        elif data_traffic_forecast > 0:
            global_data_forecast = data_traffic_forecast



        print ("received Forecasts")
        print (energy_forecast)
        print (data_traffic_forecast)

        if energy_forecast>0 and data_traffic_forecast >0:
            next_state, reward, done, posX, posY = env.step(action,energy_forecast,data_traffic_forecast)
            # Alpha of 0.2 and 1-alpha = 0.8

            self.qTable[self.state][action] = (0.8)*self.qTable[self.state][action]+ reward + self.gamma * max(self.qTable[next_state])
            np.save("Qtable_rl.txt",self.qTable) # Save the Q-Table and this can be later used for further improvements
            # print (next_state)
            self.state = next_state
            self.current_pattern = action
