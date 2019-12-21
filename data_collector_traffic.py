_Author_ = "*******"

# Data traffic path

from Initializer import Initialize
from datetime import datetime
from datetime import timedelta
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
init_object = Initialize()


class DataLoader():
    # class to load the csv data to json
    # This will also support loading real-time simulation data from kafka queue for prediction

    data_traffic_path = ""
    data_traffic_file = ""
    def __init__(self):
        #self.data_traffic_path = init_object.data_traffic_path
        self.data_traffic_path = init_object.data_traffic_path
        self.data_traffic_file = init_object.data_traffic_file


    def load_data_to_csv(self):
        # loads the log file to csv file
        file =  open(self.data_traffic_path + self.data_traffic_file,"r")
        time_data_frame  = {}
        traffic_count  = 0
        traffic_induvidual_count = 0 # For each time instance
        prev_time  = 0.0 # Keep a check on the time
        df_dict = {}
        line_count = 0
        for line in file.readlines():
            if "Time" in line:
                if line_count>0:
                    #if prev_time in df_dict:
                    #    df_dict[prev_time]=df_dict[prev_time] + traffic_induvidual_count
                    #else:
                    if prev_time not in df_dict:
                        df_dict[prev_time] = traffic_induvidual_count
                        traffic_induvidual_count = 0
                    current_time = float(line.split(":")[1].split(" ")[1])

                    #print (current_time)
                    prev_time = current_time


                line_count += 1
            if ("finished sending") in line:
                # Count the traffic
                #if ("S11") or ("S47") or("S48") or ("") not in line:
                # The database
                traffic_induvidual_count += 1
                #print (line)
                traffic_count += 1

            if ("is receiving") in line:
                #if ("S11") not in line:
                # Count the traffic
                traffic_induvidual_count += 1
                # print (line)
                traffic_count += 1

        #print (traffic_count)
        #print (sum(df_dict.values()))
        df_dict[prev_time]=traffic_induvidual_count
        max_time =prev_time  # The last time value inserted becomes the maximum time
        start_time = datetime.now() - timedelta(seconds=max_time)
        #print (traffic_count)
        print (start_time)
        new_df_dict = {}
        check_Sum = 0
        dataframe_dict = {}
        dataframe_dict["timestamp"] = []
        dataframe_dict["traffic"] = []
        for key in df_dict.keys():
            check_Sum =  check_Sum + df_dict[key]
            #print (key)
            milliseconds = float(key *1000)
            #print (milliseconds)
            time_value = start_time + timedelta(milliseconds=milliseconds)
            #print (time_value)
            dataframe_dict["timestamp"].append(time_value)
            #print(time_value)
            #print (time_value)
            timestamp = int(time_value.timestamp()*1000)
            #print (timestamp)
            if time_value in new_df_dict:
                new_df_dict[time_value] = new_df_dict[time_value] +  df_dict[key]
                #dataframe_dict["traffic"].append(new_df_dict[time_value] +  df_dict[key])
                dataframe_dict["traffic"].append(df_dict[key])

            else:
                new_df_dict[time_value] = df_dict[key]
                dataframe_dict["traffic"].append(df_dict[key])

        #print (check_Sum)
        #print (traffic_count)

        #print(len(df_dict.keys()))

        #print (len(new_df_dict.keys()))
        #print (sum(new_df_dict.values()))


        processed_dataframe = pd.DataFrame(dataframe_dict)
        #print (sum(processed_dataframe["traffic"].tolist()))
        processed_dataframe.index = processed_dataframe["timestamp"]
        # aggregate the dataframe now for one minute intervals
        aggregate_df = processed_dataframe.resample('1T').sum()
        #aggregate_df.to_csv(init_object.data_path+ "aggregated_traffic_" + self.data_traffic_file.split("_")[1] + "_su"+ ".csv",index =True)
        aggregate_df.to_csv(init_object.data_path+ "aggregated_traffic_CO_5Dec" +".csv",index =True)
        #aggregate_df.to_csv(init_object.data_path+ "aggregated_traffic_ada_testNov" +".csv",index =True)

        #print (aggregate_df)
        #print (init_object.data_path+ "aggregated_traffic_" + self.data_traffic_file.split("_")[1] + ".csv")
        print ("Data Generation Complete")
        list_vals = aggregate_df["traffic"].tolist()
        print(sum(list_vals))
        plt.plot(aggregate_df["traffic"])
        plt.savefig("trafficplot.png")

def make_combine_plots():
    data_sc = pd.read_csv(init_object.data_path+ "aggregated_traffic_SC_5Dec.csv")
    data_su = pd.read_csv(init_object.data_path+ "aggregated_traffic_SU_5Dec.csv")
    data_co = pd.read_csv(init_object.data_path+ "aggregated_traffic_CO_5Dec.csv")
    data_ada = pd.read_csv(init_object.data_path+ "aggregated_traffic_ada_29Nov.csv")
    data_rlmc = pd.read_csv(init_object.data_path+ "aggregated_traffic_rlmc_4Dec.csv")
    data_mc = pd.read_csv(init_object.data_path+ "aggregated_traffic_mc_5Dec.csv")
    data_rl = pd.read_csv(init_object.data_path+ "aggregated_traffic_rl_5Dec.csv")



    ## Aggregate plots
    list_co = data_co.head(1441)["traffic"].tolist()
    co_agg_traffic = []
    co_traffic_sum = 0
    count = 0
    for data in list_co:
        co_traffic_sum= co_traffic_sum+ data
        if count % 10 == 0 and count != 0:
            co_agg_traffic.append(co_traffic_sum)
            co_traffic_sum= 0
        count+=1
    #############################################
    list_sc = data_sc.head(1441)["traffic"].tolist()
    sc_agg_traffic = []
    sc_traffic_sum = 0
    count = 0
    for data in list_sc:
        sc_traffic_sum = sc_traffic_sum + data
        if count % 10 == 0 and count != 0:
            sc_agg_traffic.append(sc_traffic_sum)
            sc_traffic_sum = 0

        count += 1

    #################################################
    list_su = data_su.head(1441)["traffic"].tolist()
    su_agg_traffic = []
    su_traffic_sum = 0
    count = 0
    for data in list_su:
        su_traffic_sum = su_traffic_sum + data
        if count % 10 == 0 and count != 0:
            su_agg_traffic.append(su_traffic_sum)
            su_traffic_sum = 0
        count += 1
    ################################################
    list_ada = data_ada.head(1441)["traffic"].tolist()
    ada_agg_traffic = []
    ada_traffic_sum = 0
    count = 0
    for data in list_ada:
        ada_traffic_sum = ada_traffic_sum + data
        if count % 10 == 0 and count != 0:
            ada_agg_traffic.append(ada_traffic_sum)
            ada_traffic_sum = 0
        count += 1
    ################################################

    list_rlmc = data_rlmc.head(1441)["traffic"].tolist()
    rlmc_agg_traffic = []
    rlmc_traffic_sum = 0
    count = 0
    for data in list_rlmc:
        rlmc_traffic_sum = rlmc_traffic_sum + data
        if count % 10 == 0 and count != 0:
            rlmc_agg_traffic.append(rlmc_traffic_sum)
            rlmc_traffic_sum = 0
        count += 1

    ################################################

    list_mc = data_mc.head(1441)["traffic"].tolist()
    mc_agg_traffic = []
    mc_traffic_sum = 0
    count = 0
    for data in list_mc:
        mc_traffic_sum = mc_traffic_sum + data
        if count % 10 == 0 and count != 0:
            mc_agg_traffic.append(mc_traffic_sum)
            mc_traffic_sum = 0
        count += 1

    ################################################

    list_rl = data_rl.head(1441)["traffic"].tolist()
    rl_agg_traffic = []
    rl_traffic_sum = 0
    count = 0
    for data in list_rl:
        rl_traffic_sum = rl_traffic_sum + data
        if count % 10 == 0 and count != 0:
            rl_agg_traffic.append(rl_traffic_sum)
            rl_traffic_sum = 0
        count += 1


    plt.plot(co_agg_traffic, label="CO")
    plt.plot(sc_agg_traffic, label="SC")
    plt.plot(su_agg_traffic,label = "SU")
    #plt.plot(ada_agg_traffic,label = "Random")
    plt.plot(rlmc_agg_traffic, label="RLMC",linewidth=2.0)
    plt.plot(mc_agg_traffic, label="MC")
    plt.plot(rl_agg_traffic, label="RL")


    plt.xlabel("Time (in minutes) aggregated over 10 minutes ")
    #print (sum(ada_agg_traffic))
    plt.ylabel("# of messages")
    plt.legend()
    plt.grid(True)
    plt.axhline(y=3000, color='green', linestyle='--', linewidth=1.5)
    plt.legend(loc="upper left")
    plt.text(x=24, y=3000.0, s="Traffic Limit", fontsize=7,
             bbox=dict(facecolor='whitesmoke', boxstyle="round, pad=0.4"))
    plt.savefig("./plots/trafficplot_overall_24hours_aggregated_Dec6.png",dpi=300)


    print (sum(co_agg_traffic))
    print(sum(sc_agg_traffic))
    print(sum(su_agg_traffic))
    print(sum(rlmc_agg_traffic))
    print(sum(mc_agg_traffic))
    print(sum(rl_agg_traffic))


    #plt.plot(data_co.groupby(data_co.index // 10).sum(),label="co")
    #plt.plot(data_su.groupby(data_sc.index // 10).sum(),label="su")
    #plt.plot(data_sc.groupby(data_su.index // 10).sum(),label="sc")

    '''
    plt.plot(data_co["traffic"], label="CO")
    plt.plot(data_sc["traffic"], label="SC")
    plt.plot(data_su["traffic"].head(241),label = "SU")
    #plt.plot(data_ada.head(241)["traffic"],label = "Random")
    plt.plot(data_rlmc.head(241)["traffic"],label = "RLMC",linewidth=3)
    plt.plot(data_mc.head(241)["traffic"],label = "MC")
    plt.plot(data_rl.head(241)["traffic"],label = "RL")
    plt.axhline(y=300)
    plt.xlabel("Time in Minutes")
    plt.ylabel("# of messages")
    plt.legend()
    plt.grid(True)
    plt.axhline(y=300, color='green', linestyle='--', linewidth=1.5)
    plt.legend(loc="upper left")
    plt.text(x=240, y=300.0, s="Traffic Limit", fontsize=7,
             bbox=dict(facecolor='whitesmoke', boxstyle="round, pad=0.4"))
    plt.savefig("./plots/trafficplot_overall_4hours_minutes_Dec2.png",dpi=300)
    '''

if __name__ == '__main__':

    data_loader_object = DataLoader()
    data_loader_object.load_data_to_csv()
    #make_combine_plots()

