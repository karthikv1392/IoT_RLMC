_Author_ = "******"

# Predictor for Spark

from keras.models import model_from_json
import keras
import pandas as pd
from Initializer import Initialize
from keras import backend as K
import time
from sklearn.externals import joblib
import random
import os
import numpy as np
import json
import requests

init_object = Initialize()
# import pysftp

# Define Thresholds
he = 10.0
ae = 10.0
le = 6.0
hd = 3000
ad = 3000
ld = 2500
global_energy_forecast = 10.0
global_data_forecast = 2500

request_url = init_object.request_url  ## The reequest url for the web service where the adpatation pattern needs to be updated

class Spark_Predictor():
    loaded_model = None
    scalar = None
    normal = 1
    high = 0
    low = 0
    current_pattern = 10

    def __init__(self):
        self.main_energy_list = []
        self.main_data_traffic_list = []
        self.patterns = ["co","su","sc"]
        self.input_json_path = "./jsons/"
        self.output_json_path  = "./jsons/output.json"
        self.pattern_map = {"co" :"30","su":"10","sc" : "20"}

    def predictor(self, analyze, energy_forecast=0, data_traffic_forecast=0):
        rl_mc_results = open("mc_working_results.txt", 'a')
        global global_energy_forecast
        global global_data_forecast
        print("reinforce")
        action = 0
        # if np.random.uniform() < self.epsilon:
        #    action = env.random_action()
        # else:

        # if self.current_pattern == 2 and (data_traffic_forecast > 0 or data_traffic_forecast<0):
        #    data_traffic_forecast = data_traffic_forecast + 350

        if (data_traffic_forecast < 0):
            data_traffic_forecast = -1 * data_traffic_forecast

        print("received Forecasts")
        print(energy_forecast)
        print(data_traffic_forecast)
        request_json = {}
        request_json["pattern"] = "co" # Default
        if energy_forecast > 0 and data_traffic_forecast > 0:
            traffic_pattern_select = {}
            energy_pattern_select = {}
            min_energy_pattern = ""
            min_energy = 10000
            max_score = 0
            for pattern in self.patterns:
                input_file = "$PATH$/jsons/archlearner_spark_output_" + pattern + ".json"  ## Get the corresponding file to be sent to model checker
                analyze.run(input_file)
                analyze.exportToJSON(self.output_json_path)  ## This will generate the output json
                output_json = {}
                energy_consumption_mc = 0.0
                traffic_consumption_mc = 0.0
                with open(self.output_json_path, "r") as jsonfile:
                    output_json = json.load(jsonfile)
                    energy_consumption_mc= output_json["energy_total"]
                
                    energy_pattern_select[pattern] = energy_consumption_mc
                    traffic_consumption_mc = output_json["traffic_total"]
                    traffic_consumption_mc = traffic_consumption_mc*2 # model checker considers only send meesages so just to make it consistent with overall approach
                    #traffic_consumption_mc = traffic_consumption_mc + random.randint(-50, 150)
                    traffic_pattern_select[pattern] = traffic_consumption_mc

                    print("model checker")

                    total_score = (he - energy_consumption_mc)*1000 + (hd - data_traffic_forecast) *100
                    #total_score = (he - energy_consumption_mc)*100 + (hd - data_traffic_forecast)

                    print(energy_consumption_mc, traffic_consumption_mc,total_score)
                    if total_score > max_score:
                        max_score = total_score
                        request_json["pattern"] = self.pattern_map[pattern]

        
            print ("Pattern Selected ", request_json["pattern"])
            request_json_obj = json.dumps(request_json)
            response = requests.post(init_object.request_url, request_json_obj)
