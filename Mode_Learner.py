_Author_ = "**************************"

# This class is responsible for the training, testing and prediciton of modes

from Initializer import Initialize

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from keras.layers import Bidirectional
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from keras.layers import RepeatVector
from keras.layers import TimeDistributed

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Flatten
from keras.layers import LSTM
from math import sqrt
import matplotlib
matplotlib.use('Agg')
from keras.layers import Dropout
from matplotlib import pyplot
from numpy import array
from sklearn.externals import joblib
import numpy as np
from Initializer import Initialize
import time
from keras.models import model_from_json
import math

from Custom_Logger import logger



init_object = Initialize()

class Mode_Learner():
    def __init__(self,sensorId):

        # The sensor for which the learning needs to be performed
        self.data_path = init_object.data_path
        self.mode_folder = "mode_new"
        self.sensorId = sensorId



    def parser(self, x):
        # date-time parsing function for loading the dataset
        logger.info("Inside date parser")
        return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


    def read_data(self, filepath):
        logger.info("reading the dataset")
        # Load the dataset into the memory and start the learning
        # Takes as input the path of the csv which contains the descriptions of energy consumed

               # Datasets reside in the mode folder
        aggregated_df = read_csv(filepath, header=0, parse_dates=[0], index_col=0,
                                 squeeze=True, date_parser=self.parser,nrows=600)

        aggregated_series = aggregated_df.values  # Convert the dataframe to a 2D array and pass back to the calling function


        return aggregated_series, aggregated_df


    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        # Convert the dataset into form such that the data set is shifted for different timesteps until the number of forecasts
        # convert time series into supervised learning problem
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # Concatenate all columns into one dataframe
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)

        return agg


    def difference(self, dataset, interval=1):
        # create a differenced series
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)

        return Series(diff)

        # transform series into train and test sets for supervised learning

    def prepare_data(self, series, prev_steps, num_forecasts):
        # Normalize the data values so that they all are in the same range
        # Convert the problem into a supervised Learning problem
        # rescale values to -1, 1

        series = series.values
        series = series.reshape(len(series), 1)
        scaler = MinMaxScaler(feature_range=(0, 1))  # Energy value is always posetive
        #scaler = joblib.load(init_object.model_path + "scaler_mode" + self.sensorId+ ".save")
        scaled_values = scaler.fit_transform(series)

        joblib.dump(scaler, init_object.model_path + "scaler_mode" + self.sensorId + ".save")
        # transform into supervised learning problem X, y
        # prev_steps indicates the number of time steps that needs to be taken into consideration for a decision
        # n_seq denotes the number of values to be predicite
        scaled_values = scaled_values.reshape(len(scaled_values), 1)
        supervised = self.series_to_supervised(scaled_values, prev_steps, num_forecasts)
        supervised_values = supervised.values
        # Save the scalar used for learning the hyperparameters


        return supervised_values, scaler


    def generate_train_test_data(self,series,propotion_value,num_obs,num_features):
        # Takes the series and converts it into a training and testing set based on the percentatge of division as
        # indicated by the propotion_value parameter
        # num_features indicate the number of features to be predicted
        # num_obs indicate the total number of observations
        test_count = int(len(series) * propotion_value)+1  # Integer value for representing the count
        train_data = series[:test_count, :]
        test_data = series[test_count:, :]

        # split into input and outputs
       # print (train_data[:, -5:-1])

        # No of observations is 60 and number of features is 12
        print ("observations :", num_obs)
        train_X, train_y = train_data[:, :num_obs], train_data[:, num_obs:]
        test_X, test_y = test_data[:, :num_obs], test_data[:, num_obs:]
        #test_X, test_y = test_data[:, :-num_features], test_data[:, -num_features: -1]

        print (train_X.shape)
        print (train_y.shape)
        return train_X,train_y, test_X,test_y



    def reshape_test_train_dataset(self,train_X,test_X,lag,num_features):
        # Reshape the training and testing set as required by the Keras LSTM Network
        # reshape input to be 3D [samples, timesteps, features]
        #print(train_X.shape, len(train_X), train_y.shape)
        #num_features = 12
        train_X = train_X.reshape((train_X.shape[0], lag, num_features))
        test_X = test_X.reshape((test_X.shape[0], lag, num_features))
        return train_X,test_X


    def create_model(self,train_X,num_features):
        # fit an LSTM network to training data
        # reshape training into [samples, timesteps, features]
        # Design the LSTM Network
        model = Sequential()

        print (train_X.shape)
        model.add(LSTM(5, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))

        #model.add(Bidirectional(LSTM(100,activation="tanh",return_sequences=True), input_shape=(train_X.shape[1], train_X.shape[2])))
        #model.add(RepeatVector(10))
        #model.add(Bidirectional(LSTM(100, activation='relu', return_sequences=True)))
        #model.add(TimeDistributed(Dense(10)))
        #print (train_X.shape[1])
        #print(train_X.shape[2])
        model.add(LSTM(units=50))
        #model.add(Dropout(0.1))
        model.add(Dense(num_features))  # Output shall depend on the number of features that needs to be predicted
        #model.add(Activation('relu'))
        model.compile(loss='mae', optimizer='adam')
        print (model.summary())
        #time.sleep(3)
        return model




    def forecast_lstm(self,model, X, n_batch):
        # make one forecast with an LSTM,
        # reshape input pattern to [samples, timesteps, features]
        X = X.reshape(1, 1, len(X))
        # make forecast
        forecast = model.predict(X, batch_size=n_batch)
        # convert to array
        return [x for x in forecast[0, :]]



    def make_forecasts(self,model, n_batch, train, test, n_lag, n_seq):
        # evaluate the persistence model
        forecasts = list()
        for i in range(len(test)):
            X, y = test[i, 0:n_lag], test[i, n_lag:]
            # make forecast
            forecast = self.forecast_lstm(model, X, n_batch)
            # store the forecast
            forecasts.append(forecast)
        return forecasts



    def inverse_difference(self,last_ob, forecast):
        # invert differenced forecast
        # invert differenced value
        # invert differenced forecast
        # invert first forecast
        inverted = list()
        inverted.append(forecast[0] + last_ob)
        # propagate difference forecast using inverted first value
        for i in range(1, len(forecast)):
            inverted.append(forecast[i] + inverted[i - 1])
        #print(inverted)
        return inverted


    # inverse data transform on forecasts
    def inverse_transform(self,series, forecasts, scaler, n_test):
        inverted = list()
        for i in range(len(forecasts)):
            # create array from forecast
            forecast = array(forecasts[i])
            forecast = forecast.reshape(1, len(forecast))
            # invert scaling
            inv_scale = scaler.inverse_transform(forecast)
            inv_scale = inv_scale[0, :]
            #print (inv_scale)
            # invert differencing
            #print ("length " ,len(series))
            index = len(series) - n_test + i - 1
            #print (index)
            last_ob = series.values[index]
            #print (last_ob)
            inv_diff = self.inverse_difference(last_ob, inv_scale)
            # store
            inverted.append(inv_diff)
        return inverted


    # evaluate the RMSE for each forecast time step
    def evaluate_forecasts(self,test, forecasts, n_lag, n_seq):
        for i in range(n_seq):
            actual = [row[i] for row in test]
            predicted = [forecast[i] for forecast in forecasts]
            rmse = sqrt(mean_squared_error(actual, predicted))
            print('t+%d RMSE: %f' % ((i + 1), rmse))


    # plot the forecasts in the context of the original dataset
    def plot_forecasts(self,series, forecasts, n_test):
        # plot the entire dataset in blue
        pyplot.plot(series.values)
        # plot the forecasts in red

        for i in range(len(forecasts)):
            off_s = len(series) - n_test + i - 1
            off_e = off_s + len(forecasts[i]) + 1
            xaxis = [x for x in range(off_s, off_e)]
            yaxis = [series.values[off_s]] + forecasts[i]
            pyplot.plot(xaxis, yaxis, color='red')
        # show the plot
        #pyplot.show()
        pyplot.savefig("rmse_plot_" + self.sensorId + ".png")




if __name__ == '__main__':
    sensor_list = ["S34", "S33", "S24", "S25", "S41", "S42", "S1", "S2", "S18", "S20"]
    for sensor_id in sensor_list:
        if sensor_id not in ["S101"]:
            learner = Mode_Learner(sensor_id)
            prev_steps = 10
            num_forecasts = 10
            number_obs = prev_steps * 1  # This will give the total number of observations, we have only one feature

            aggregated_series, aggregated_df = learner.read_data(init_object.data_path + learner.mode_folder + "/" + learner.sensorId+".csv")
            #print(aggregated_df.std(axis=0))
            #print(max(aggregated_df) - min(aggregated_df))

            # Perform differentiation of the data set to ensure the consistency
            diff_series = learner.difference(aggregated_series, 1)
            # print (diff_values)
            #print (diff_series.shape)
            # supervised_series = learner.series_to_supervised(diff_values,1)
            #print (aggregated_series.shape)
            values, scaler = learner.prepare_data(diff_series, prev_steps, num_forecasts)
            #values, scaler = learner.prepare_data(aggregated_df, prev_steps, num_forecasts)
            train_X, train_y, test_X, test_y = learner.generate_train_test_data(values, num_features=num_forecasts,
                                                                                propotion_value=init_object.propotion_value,
                                                                                num_obs=number_obs)

            lag = prev_steps
            train_X, test_X = learner.reshape_test_train_dataset(train_X, test_X, lag,
                                                                 num_features=1)  # It is a univariate data

            model = learner.create_model(train_X,
                                         num_forecasts)  # num_forecasts is the number of features here as its a univariate

            model_json = model.to_json()

            history = model.fit(train_X, train_y, epochs=init_object.epochs, batch_size=init_object.batch_size,
                                validation_data=(test_X, test_y), verbose=2,
                                shuffle=False)

            with open(init_object.model_path + "model_mode_"+ learner.sensorId+  ".json", "w") as json_file:
                json_file.write(model_json)
                # serialize weights to HDF5
            model.save_weights(init_object.model_path + "model_mode_" + learner.sensorId + ".h5")
            print("Saved model to disk")


            # Perform prediction
            print("Test X", test_X.shape)
            forecasts = model.predict(test_X)
            test_X = test_X.reshape((test_X.shape[0], number_obs))

            # Perform inverse transformation
            n_test = int(len(values) * (1 - init_object.propotion_value))

            # print (actual)

            actual = [row[lag:] for row in values[-n_test:]]
            # print ("actual length" ,len(actual))
            # 2881,2882,2883,2884,2885,2886
            actual = learner.inverse_transform(aggregated_df, actual, scaler, n_test + 9)

            #print (actual)
            #print(n_test)
            #print(type(aggregated_df))
            forecasts = learner.inverse_transform(aggregated_df, forecasts, scaler, n_test + 9)
            #print (forecasts)
            # forecasts = learner.inverse_transform(   aggregated_df, forecasts, scaler, 2881)

            # print("lengths :", len(forecasts))
            try:
                learner.evaluate_forecasts(actual, forecasts, 1, 10)
            except Exception as e:
                pass




