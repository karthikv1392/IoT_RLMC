_Author_ = "*****************"


# Producer module to keep sending the sensor data from cupCarbon to Streamer

import subprocess

import csv
import sys
import time
from kafka import KafkaConsumer, KafkaProducer
from Initializer import Initialize


from datetime import datetime,timedelta
from Custom_Logger import logger
from SQL_Util import  SQLUtils

init_object = Initialize()


class kafka_producer():
    def publish_message(self,producer_instance, topic_name, key, value):
        try:
            key_bytes = bytearray(key,'utf8')
            value_bytes = bytearray(value,'utf8')
            producer_instance.send(topic_name, key=key_bytes, value=value_bytes)
            producer_instance.flush()
            print('Message published successfully.')
        except Exception as ex:
            print('Exception in publishing message')
            print(str(ex))

    def connect_kafka_producer(self):
        _producer = None
        try:
            _producer = KafkaProducer(bootstrap_servers=['localhost:9092'], api_version=(0, 10))
            #_producer = KafkaProducer(bootstrap_servers=['192.168.1.41:9092'], api_version=(0, 10))
        except Exception as ex:
            print('Exception while connecting Kafka')
            print(str(ex))
        finally:
            return _producer


producer_object = kafka_producer()


class Data_Streamer():

    def __init__(self):
        self.start_time = datetime.now() - timedelta(days=20)
        self.sensor_list = ["S34", "S33", "S24", "S25", "S41", "S42", "S1", "S2", "S18", "S20"]
        self.table_name = "StreamingSensorData"
        self.sql_object = SQLUtils()


    def stream_data(self):
        # Stream data to Kafka
        producer_instance = producer_object.connect_kafka_producer()
        sensor_id_seconds = {}
        sensor_id_value = {}
        for id in self.sensor_list:
            sensor_id_seconds[id] = 0
            sensor_id_value[id] = 0
        seconds  = 0
        with open(init_object.data_traffic_path+ init_object.data_traffic_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')

            count = 0
            total_seconds = 0
            while (True):

                #print (start_timestamp)
                for row in csv.reader(iter(csv_file.readline, '')):
                    if len(row) > 0:
                        line = row[0].strip("\n")
                        #print (line)
                        # print (line_data.split(";"))

                        if "Time" in line:
                            if ":" in line and " " in line:
                                time_list = line.split(":")[1].split(" ")
                                if len(time_list) > 1:
                                    if "." in time_list[1]:
                                        try:
                                           current_time = float(line.split(":")[1].split(" ")[1])
                                           #start_timestamp = self.start_time + timedelta(seconds=float(current_time))
                                        except:
                                            pass

                            # print (current_time)

                    if ("is writing the message") in line:
                        if any(sensor in line for sensor in self.sensor_list):
                            #res = list(filter(lambda x: x in line, self.sensor_list))
                            if count == 0:
                                start_timestamp = self.start_time + timedelta(seconds=float(current_time))
                                #print (start_timestamp)
                                #print (line)
                                sensor_id = line.split(" ")[0]
                                print (sensor_id)
                                if sensor_id in self.sensor_list:
                                    #print(line)
                                    # It can happen that S2 is matched in the list by any() above instead of S20 since both have S2
                                    value_text = line.split(" ")[6]
                                    value = value_text.split("#")[-1].strip("\"")
                                    #seconds = sensor_id_seconds[sensor_id]
                                    sensor_id_seconds[sensor_id] += 0
                                    sensor_id_value[sensor_id] = value
                                    line_data = sensor_id + ";" +str(seconds) + ";" + str(value)
                                    query = "insert into " + self.table_name + " values(\"" + sensor_id + "\"," + "\"" + str(
                                        start_timestamp) + "\"," + str(
                                        float(value)) + ");"
                                    logger.info(query)
                                    print(query)
                                    # self.sql_object.insert(query)
                                    producer_object.publish_message(producer_instance, "sensorData", "data", line_data)
                                    logger.info("published message ", line_data)
                                    #print (sensor_id_seconds[sensor_id])
                                    if int(sensor_id_seconds[sensor_id]) >= 60:
                                    #if sensor_id[seconds] >= 600:
                                        # Reset the value for each 10 values
                                        sensor_id_seconds[sensor_id] = 0

                                    #print(line_data)
                                    #sensor_id_seconds[sensor_id] = sensor_id_seconds[sensor_id]  + 60
                            elif count>0:
                                #print (current_time)
                                #print (current_time)
                                time_val = self.start_time + timedelta(milliseconds=float(current_time))
                                #print (time_val)
                                #print (start_timestamp, time_val)
                                #print (int((time_val - start_timestamp).total_seconds()/60), count)
                                #if (int((time_val - start_timestamp).total_seconds()/60)) == count:
                                #if (time_val - start_timestamp).total_seconds() % 60 == 0:
                                    #print (time_val, start_timestamp)
                                    #print (line)

                                sensor_id = line.split(" ")[0]
                                if sensor_id in self.sensor_list:
                                    #print (line)
                                    # It can happen that S2 is matched in the list by any() above instead of S20 since both have S2
                                    value_text = line.split(" ")[6]
                                    value = value_text.split("#")[-1].strip("\"")
                                    #seconds = sensor_id_seconds[sensor_id]

                                    seconds = int((time_val - start_timestamp).total_seconds())

                                    #sensor_id_seconds[sensor_id]+=seconds # Add the seconds\
                                    #if sensor_id=="S1":
                                        #print (seconds)
                                        #print (sensor_id,sensor_id_seconds[sensor_id])
                                        #print (line)
                                    if value!="A" and value!="N":
                                        sensor_id_value[sensor_id] = value
                                        #print(value)
                                        mod_value = int(current_time / 60)

                                        if mod_value > sensor_id_seconds[sensor_id]:
                                            #print(sensor_id, mod_value,sensor_id_seconds[sensor_id] )
                                            sensor_id_seconds[sensor_id] = mod_value
                                            sensor_id_value[sensor_id] = value
                                            line_data = sensor_id + ";" + str(sensor_id_seconds[sensor_id]) + ";" + str(sensor_id_value[sensor_id]) ## To easily get the sensorId
                                            #line_data = sensor_id + ";" + str(total_seconds) + ";" + str(
                                            #    sensor_id_value[sensor_id])  ## To easily get the sensorId
                                            print (line_data)
                                            producer_object.publish_message(producer_instance, "sensorData", "data",
                                                                            line_data)
                                            logger.info("published message ", line_data)
                                            #seconds = seconds + 60
                                            #sensor_id_seconds[sensor_id] = sensor_id_seconds[sensor_id] + 60
                                            #if seconds>600:
                                                # Reset the value for each 10 values
                                                #seconds = 0

                                            #print (sensor_id_seconds)

                            count+=1

if __name__ == '__main__':
    data_producer_object = Data_Streamer()
    data_producer_object.stream_data()
