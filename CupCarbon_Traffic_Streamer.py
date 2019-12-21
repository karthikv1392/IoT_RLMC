_Author_ = "***************"

import subprocess

import csv
import sys
import time
from kafka import KafkaConsumer, KafkaProducer

from Initializer import Initialize

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
            #_producer = KafkaProducer(bootstrap_servers=['localhost:9092'], api_version=(0, 10))
            _producer = KafkaProducer(bootstrap_servers=['localhost:9092'], api_version=(0, 10))
            #_producer = KafkaProducer(bootstrap_servers=['192.168.1.41:9092'], api_version=(0, 10))
        except Exception as ex:
            print('Exception while connecting Kafka')
            print(str(ex))
        finally:
            return _producer

producer_object = kafka_producer()



def stream_csv_file():
    # read and stream csv files
    producer_instance = producer_object.connect_kafka_producer()
    with open(init_object.data_traffic_path + init_object.data_traffic_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        traffic_count = 0
        prev_mod_value = 0
        while (True):
            for row in csv.reader(iter(csv_file.readline,'')):
                if len(row)>0:
                    line= row[0].strip("\n")
                    #print (line_data)
                    #print (line_data.split(";"))

                    if "Time" in line:
                        if ":" in line and " " in line:
                            time_list = line.split(":")[1].split(" ")
                            if len(time_list)>1:
                                    if "." in time_list[1]:
                                        try:
                                            current_time = float(time_list[1])
                                            mod_value = int(current_time/60)
                                            if current_time > 0 and mod_value > prev_mod_value:
                                                #if current_time >0 and int(current_time%60==0):
                                            #    mod_value = int(current_time)/60
                                            #    if mod_value > prev_mod_value:
                                                line_data = str(int(current_time)) + ";" + str(traffic_count)
                                                print (line_data)
                                                # Every 60 seconds send the data to kafka
                                                producer_object.publish_message(producer_instance, "sensor", "data", line_data)
                                                prev_mod_value = mod_value
                                                traffic_count = 0
                                                # print (current_time)
                                                prev_time = current_time
                                        except:
                                            pass

                        #print (current_time)

                    if ("finished sending") in line:
                        # Count the traffic
                        # if ("S11") or ("S47") or("S48") or ("") not in line:
                        # The database
                        # print (line)
                        traffic_count += 1

                    if ("receiving") in line:
                        # if ("S11") not in line:
                        # Count the traffic
                        # print (line)
                        traffic_count += 1

                    #time.sleep(1)
                    #producer_object.publish_message(producer_instance,"traffic","data",line_data)



                #time.sleep(1)
if __name__ == '__main__':
    stream_csv_file()
