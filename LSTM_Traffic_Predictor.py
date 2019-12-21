_Author_ = "******"

# Program to predict data traffic, here the data traffic is considered as a uni-variate data

# LSTM Learner to predict forecast Energy Consumption for the next 10 seconds
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
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


pattern = "CO"
init_object = Initialize()

class LSTM_Learner():
    # Class to perform Model Creation and Learning
    # Consists of different functions for normalization and data arrangement

    # date-time parsing function for loading the dataset
    def parser(self, x):
        return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    # Load the dataset into the memory and start the learning
    def read_data(self,filepath):
        # Takes as input the path of the csv which contains the descriptions of energy consumed

        aggregated_df = read_csv(filepath, header=0, parse_dates=[0], index_col=0,
                                 squeeze=True, date_parser=self.parser)

        aggregated_series =  aggregated_df.values # Convert the dataframe to a 2D array and pass back to the calling function

        return aggregated_series,aggregated_df

    # convert time series into supervised learning problem
    def series_to_supervised(self,data, n_in=1, n_out=1, dropnan=True):
        # Convert the dataset into form such that the data set is shifted for different timesteps until the number of forecasts
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
        #print(agg)
        return agg


    # create a differenced series
    def difference(self,dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return Series(diff)


    # transform series into train and test sets for supervised learning
    def prepare_data(self,series,prev_steps, num_forecasts):
        # Normalize the data values so that they all are in the same range
        # Convert the problem into a supervised Learning problem
        # rescale values to -1, 1

        series = series.values
        series = series.reshape(len(series),1)
        scaler = MinMaxScaler(feature_range=(-1, 1)) # Energy value is always posetive
        scaled_values = scaler.fit_transform(series)

        joblib.dump(scaler, "scaler_traffic_" +pattern+".save")

        # transform into supervised learning problem X, y
        # prev_steps indicates the number of time steps that needs to be taken into consideration for a decision
        # n_seq denotes the number of values to be predicite
        scaled_values = scaled_values.reshape(len(scaled_values), 1)
        supervised = self.series_to_supervised(scaled_values, prev_steps, num_forecasts)
        supervised_values = supervised.values
        return supervised_values,scaler



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




    # fit an LSTM network to training data
    def create_model(self,train_X,num_features):
        # reshape training into [samples, timesteps, features]
        # Design the LSTM Network
        model = Sequential()

        print (train_X.shape)
        model.add(LSTM(1, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
        #print (train_X.shape[1])
        #print(train_X.shape[2])
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(num_features))  # Output shall depend on the number of features that needs to be predicted
        #model.compile(loss='mae', optimizer='adam')
        model.compile(loss='mae', optimizer='adam')
        print (model.summary())
        #time.sleep(3)
        return model




    # make one forecast with an LSTM,
    def forecast_lstm(self,model, X, n_batch):
        # reshape input pattern to [samples, timesteps, features]
        X = X.reshape(1, 1, len(X))
        # make forecast
        forecast = model.predict(X, batch_size=n_batch)
        # convert to array
        return [x for x in forecast[0, :]]


    # evaluate the persistence model
    def make_forecasts(self,model, n_batch, train, test, n_lag, n_seq):
        forecasts = list()
        for i in range(len(test)):
            X, y = test[i, 0:n_lag], test[i, n_lag:]
            # make forecast
            forecast = self.forecast_lstm(model, X, n_batch)
            # store the forecast
            forecasts.append(forecast)
        return forecasts


    # invert differenced forecast
    # invert differenced value
    # invert differenced forecast
    def inverse_difference(self,last_ob, forecast):
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
        pyplot.savefig("rmse_traffic.png")



if __name__ == '__main__':

    learner = LSTM_Learner()
    prev_steps = 10
    num_forecasts = 10
    number_obs = prev_steps * 1 # This will give the total number of observations, we have only one feature


    # load dataset
    #aggregated_series,aggregated_df = learner.read_data(init_object.data_path + "aggregated_traffic_ada.txt.csv")
    aggregated_series,aggregated_df = learner.read_data(init_object.data_path + "aggregated_traffic_ada.csv")
    print (aggregated_df.std(axis=0))
    print  (max(aggregated_df)-min(aggregated_df))

    # Perform differentiation of the data set to ensure the consistency
    diff_series  = learner.difference(aggregated_series,1)
    #print (diff_values)

    #supervised_series = learner.series_to_supervised(diff_values,1)


    values,scaler = learner.prepare_data(diff_series,prev_steps,num_forecasts)
    print(values.shape)
    train_X, train_y, test_X, test_y = learner.generate_train_test_data(values,num_features=num_forecasts,propotion_value=init_object.propotion_value,num_obs=number_obs)



    lag = prev_steps
    train_X, test_X = learner.reshape_test_train_dataset(train_X, test_X, lag, num_features=1) # It is a univariate data


    model = learner.create_model(train_X, num_forecasts) # num_forecasts is the number of features here as its a univariate

    model_json = model.to_json()


    history = model.fit(train_X, train_y, epochs=init_object.epochs, batch_size=init_object.batch_size,
                        validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)

    #with open(init_object.model_path + "model_traffic_" +pattern +".json", "w") as json_file:
    with open(init_object.model_path + "model_traffic_" + ".json", "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
    model.save_weights(init_object.model_path + "model_traffic_" + ".h5")
    print("Saved model to disk")

    '''

    #json_file = open(init_object.model_path + 'model_traffic_ada.json', 'r')
    json_file = open(init_object.model_path + 'model1_master_traffic.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    #model.load_weights(init_object.model_path + "model_traffic_ada.h5")
    model.load_weights(init_object.model_path + "model1_master_traffic.h5")
    print("Loaded model from disk")
    '''
    # Perform prediction
    print ("Test X" , test_X.shape)
    forecasts = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], number_obs))


    # Perform inverse transformation
    n_test  = int(len(values) * (1-init_object.propotion_value))



    #print (actual)

    actual = [row[lag:] for row in values[-n_test:]]
    #print ("actual length" ,len(actual))
    #2881,2882,2883,2884,2885,2886
    actual = learner.inverse_transform(aggregated_df,actual,scaler,n_test+10)

    print(n_test)
    forecasts = learner.inverse_transform(aggregated_df, forecasts, scaler, n_test+10)

    #forecasts = learner.inverse_transform(aggregated_df, forecasts, scaler, 2881)

    #print("lengths :", len(forecasts))

    learner.evaluate_forecasts(actual,forecasts,1,10)
    #pyplot.plot(actual[10],label ="actual")
    #pyplot.plot(forecasts[10],label="forecasts")
    #pyplot.savefig("rmse_traffic.png")

    actual_sum = []
    forecast_sum = []

    # Sum up the forecasts for each and put it in a list

    for datapoints in actual:
        #print (datapoints)
        #actual_sum.append(sum(datapoints))
        #print (datapoints)
        actual_sum.append(datapoints[9])


    for datapoints in forecasts:
        forecast_sum.append(datapoints[9])

    rmse = sqrt(mean_squared_error(actual_sum, forecast_sum))

    print ("rmse: " , rmse)

    #pyplot.plot(actual_sum,label = "actual")
    #pyplot.plot(forecast_sum,label = "forecast")
    pyplot.plot(actual_sum, label="actual")
    pyplot.plot(forecast_sum,label = "forecast")
    #pyplot.axis([0, 100, 200, 300])
    pyplot.xlabel("Time (Minutes)")
    pyplot.ylabel("# of Messages Exchanged")
    pyplot.legend(loc='lower right')
    pyplot.savefig("rmse_traffic.png",transparent=True,dpi=300,quality=95)
