_Author_ = "*************************************"

# Class to fetch data from the sources mentioned
from Custom_Logger import logger
from Initializer import Initialize
from SQL_Util import SQLUtils
from datetime import datetime
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd



init_object = Initialize()



class Data_Extractor():

    # Member functions decide on the data fetched and processed

    def __init__(self):
        # initialize any source operations
        self.data_path = init_object.data_path
        self.sql_object = SQLUtils()
        self.start_time = datetime.now() - timedelta(days=1)
        self.sensor_list = ["S34","S33","S24","S25","S41","S42","S1","S2","S18","S20"]
        self.table_name = "sensorData"
        self.mode_directory = "mode"

    def extract_sensor_data(self):
        # Extracts the sensor data and stores it in MySQL Database
        file = open(self.data_path + "log.txt", "r")
        logger.info("inside the extract sensor data function")
        prev_time = 0.0  # Keep a check on the time
        line_count = 0
        df_dict = {}
        current_time = 0
        try:
            for line in file.readlines():
                if "Time" in line:
                    if line_count > 0:
                        df_dict[prev_time] = 0
                    current_time = float(line.split(":")[1].split(" ")[1])
                    # print (current_time)

                if ("is writing the message") in line:
                    if any(sensor in line for sensor in self.sensor_list):
                        time_val = self.start_time + timedelta(seconds=float(current_time))
                        sensor_id = line.split(" ")[0]
                        value_text = line.split(" ")[6]
                        print (value_text)
                        value = value_text.split("#")[-1].strip("\"")
                        print (value)
                        query = "insert into " + self.table_name +"values(\"" + sensor_id + "\"," + "\"" + str(time_val) + "\"," + str(
                            float(value)) + ");"
                        logger.info(query)
                        self.sql_object.insert(query)

            logger.info("Data Extraction to Database completed successfully")

        except Exception as e:
            logger.error(e)


    def extract_sensor_data_tocsv(self,sensorId):
        # Takes the MySQL data and stores as a csv file for building ML models
        # Specify the sensorId for which the model needs to be generated
        logger.info("Extracting data from database to csv")

        extract_query = "select * from" + " " + self.table_name + " " +"where sensorId=\"" + sensorId + "\"" +  " " +\
                        "order by" + " " + "timeval;"
        query_result = self.sql_object.query_table(extract_query)

        df_dict = {}     ## Dictionary to create the dataframe from the query results
        df_dict["values"] = []
        timestamps = []
        start_timestamp = next(iter(query_result))
        compare_timestamp  = start_timestamp
        df_dict["values"].append(query_result[start_timestamp])   # Insert the first value anyway
        timestamps.append(start_timestamp)
        for key in query_result.keys():
            # Keep only 60 seconds different time values
            compare_timestamp = key
            if int((compare_timestamp - start_timestamp).total_seconds()/60) == 1:
                timestamps.append(key)
                df_dict["values"].append(query_result[key])
                start_timestamp = compare_timestamp


        data_frame = pd.DataFrame(df_dict)
        data_frame.index = timestamps
        data_frame.to_csv(self.data_path + self.mode_directory+"/" + sensorId + ".csv",index=True,index_label="timestamp")


        return query_result


    def data_plot_generatpor(self,query_result_dict):

        # Function to plot the data obtained
        values = []
        for data in query_result.keys():
            values.append(query_result[data])
        plt.plot(values)
        #plt.show()
        plt.savefig("mode_plot.png")




if __name__ == '__main__':
    data_extract_obj = Data_Extractor()
    #data_extract_obj.extract_sensor_data()
    for sensor_id in ["S34","S33","S24","S25","S41","S42","S1","S2","S18","S20"]:
        query_result = data_extract_obj.extract_sensor_data_tocsv(sensor_id)
    #data_extract_obj.data_plot_generatpor(query_result)
