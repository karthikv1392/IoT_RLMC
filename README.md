This is the repository of the ICSA submission 70

## Project Content Description
1.  *CupCarbon-master_4.0* contains the modified source code of cupcarbon. The application can be started by running *cupcarbon.java*. Inside the source folder, go to senscript_functions/Functions.java, add the path for "adaptation.txt" in line 253
2.  XYZ_Adapter contains the cupCarbon project of the case study mentioned in the paper. It can be opened by opening the XYZ_Adapter.cup filefrom the 
    *open project* option available in the cupCarbon UI. Further details can be found in [www.cupcarbon.com](url)
    The natevents folder contains the sensor datasets used which was generated using a Poisson Distibution
3.  The data folder contains the different datasets used for experimentation and evaluations
        1.  *adap_energy_data.csv* and *adap_traffic.csv* contains the aggregated energy and traffic data simulated for a period of 10 days for building the machine learning models
        2.  *aggregated_traffic_CO_5Dec.csv, aggregated_traffic_SU_5Dec.csv, aggregated_traffic_SC_5Dec.csv, aggregated_traffic_rlmc_4Dec.csv, aggregated_traffic_mc_5Dec.csv, aggregated_traffic_rl_5Dec.csv, * represents the real-time simulation aggregated traffic data for the period of 1 day for approaches co, su, sc, rlmc, mc and rl respectively
        4.  *aggregate_energy_CO_5Dec.csv, aggregate_energy_SU_5Dec.csv, aggregate_energy_SC_5Dec.csv, aggregate_energy_rlmc_4Dec.csv, aggregate_energy_mc_5Dec.csv, aggregate_energy_rl_5Dec.csv* represents the real-time simulation aggregated energy data for the period of 1 day for approaches co, su, sc, rlmc, mc and rl respectively
        5.  *aggregated_energy_rl_day.csv* and *aggregated_traffic_rl_day.csv* contains the traffic and energy data generated by the reinforcement learning algorithm by using the approach
4.  *model* folder contains the machine learning models developed using Keras for energy, data traffi consumption and sensor behavior forecasts. model1_master.h5 and model1_master.json represents
    the LSTM models for energy forecasting and *model1_master_traffic.h5* and *model1_master_traffic.json* represents the data traffic forecsating models. *model_mode_<sensorId>.h5* and *model_mode_<sensorId>.json* contains the model used for forecasting energy and data traffic for each sensor in the case study
5.  *CupCarbon_Controller.py*, *CupCarbon_Traffic_Streamer.py*, *Sensor_Data_Producer.py* streams the real-time energy, data traffic and sensor data logs respectively during CupCarbon project execution to Kafka
6.  *Initializer.py* is a class file and is responsible for initalizing the preliminary configuirations
7.  *LSTM_Learner_Vector_Out.py*, *LSTM_Traffic_Predictor.py* *Mode_Learner.py* generates the energy, data traffic models and sensor behaviour forecast models respectively
8.  *Streaming_Consumer.py* is responsible for consuming the energy, data traffic logs and sensor data logs to predict the data in real-time and further invoke the adaptation approaches if need be
9.  *Spark_Predictor.py* is a class file. It is responsible for performing adpatations using the RL approach
10. *Spark_Predictor_RLMC.py* is a class file. It is reponsible for performing adpatations using the RLMC approach
11. *Spark_Predictor_MC.py* is a class file. It is responsible for performing adpatations using the MC approach
12. *data_collector.py* and *data_collector_traffic.py* are part of the feature extractor and they perform aggregationg of the raw energy as well as the data traffic logs
    data logs obtained from cupCarbon so as to aid the model building process
13. *ArchLearner_Services.py* contains the tornado service, it is responsible for communicating the adaptation to CupCarbon usiong text file "adaptation.txt"
14. *SQL_Util.py* contains the utility functions for accessing the MySQL database
15. *Sensor_Data_Extractor.py* contains the code for loading data from CupCarbon logs to MySQL and further extract as datasets for building sensor behaviour forecast models
12. *settings.conf* contains the inital configurations required for all programs and this is inturn read and processed by Initalizer.py
13. *Model_Checker* directory contains the code for probablistic model checking using PRISM Model Checker and *ModelChecker.jar* is the executable jar for the same
14. *libs* folder contains the libaries needed by PRISM. If you are using Mac, kindly use the same libs. For Linux or windows, the corresponding libararies can be downloaded from [https://www.prismmodelchecker.org/] (PRISM web page)
14. *run.sh* contains the script for executing the ModelChecker.jar


## Installation Requirements
1. Install Apache Kafka  - [https://kafka.apache.org/quickstart](url)
2. Install kafka-python -[https://pypi.org/project/kafka-python/](url)
3. Install Keras in Python -[https://keras.io](url)
4. Install Jpype - [https://jpype.readthedocs.io/en/latest/install.html]

## Instructions
1. Run the model checker using the script ./run.sh
2. Select the approach to use, for example: If RL is needed, in the Streaming_Consumer.py, change the statement "from Spark_Predictor_RLMC import Spark_Predictor" to "from Spark_Predictor_RL import Spark_Predictor" and so on
2. Set the necessary configuration as per your directory choice in *settings.conf*
3. Run the CupCarbon from the modified source code (First import the CupCarbon-Master_4.0 as an eclipse project)
4. Open the *XYZ_Adapter* project from the *open project* option in cupCarbon UI
5. Set the simulation parameters in cupCabron to one day (86400 s), mark the result field for 60s and run the Simulation
6. Immidiately run the programs *CupCarbon_Controller.py*, *CupCarbon_Traffic_Streamer.py* and *Sensor_Data_Producer.py*
7. Start web service by running: python Web_Services.py 
8. Start the Streaming_Consumer.py by running the following command inside the project folder : python Streaming_Consumer.py
9. If there are no errors, the program keeps running till the simulation stops and we will be able to see the patterns being changed in the CupCarbon UI, or in the command line where Web_Services.py is being executed.