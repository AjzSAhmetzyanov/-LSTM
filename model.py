import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import time as t
import datetime as dt
import traceback as tb
# %matplotlib inline
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Activation

from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from scikeras.wrappers import KerasRegressor
from functools import partial
import tensorflow
import concurrent.futures


data = pd.read_excel("Файл с хроникой.xlsx")
data = data.sort_values(['alpha', 'dd'], ascending=[True, True])
need_alphas = pd.read_excel("data/Справочник.xlsx")
print("file readed")
alphas = need_alphas["Показатель"]
alphas = vse_alphas

bad_alphas = []

class Forecast:
    def __init__(self, periods=30, temp='', alpha=''):
        self.periods = periods
        self.data = temp
        self.alpha = alpha
        self.model = Sequential()
        self.testX = 0
        self.scaler = MinMaxScaler()
        self.sequence_length = 10
        self.units = 50
        self.activation = 'sigmoid'
        self.batch_size = 64
        self.epochs = 100
        self.optimizer = 'adam'

    @staticmethod
    def create_model(X_train, layers, activation):
        model= Sequential()
        for i, nodes in enumerate(layers):
            if i==0:
                model.add(Dense(nodes, input_dim=X_train.shape[1]))
                model.add(Activation(activation))
            else:
                model.add(Dense(nodes))
                model.add(Activation(activation))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    @staticmethod
    def prepare_data(data, time_steps=1):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i+time_steps)])
            y.append(data[i+time_steps])
        return np.array(X), np.array(y)

    @staticmethod
    def get_best_params(X_train, y_train, epochs=None, batch_size=None, activations=None):
        model= KerasRegressor(build_fn=partial(Forecast.create_model, X_train, layers=20, activation='relu'), verbose=0)

        layers=[[20],[40,20], [45, 30, 15]]
        activations = activations
        param_grid = dict(layers=layers, activation=activations, batch_size=batch_size, epochs=epochs)

        tscv = TimeSeriesSplit(n_splits=3)

        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, error_score='raise')
        grid_result = grid.fit(X_train, y_train)

        best_params_ = grid_result.best_params_
        return best_params_

    @staticmethod
    def get_best_params_for_data(data, time_steps=1, units=None, epochs=None, batch_sizes=None, activations=None):
        scaler = MinMaxScaler(feature_range=(0,1))
        data = data.reshape(-1,1)
        data_normalized = scaler.fit_transform(data)

        X,y = Forecast.prepare_data(data_normalized, time_steps)

        train_size = int(len(X) * 0.8)
        test_size = len(X) - train_size

        X_train, X_test = X[0:train_size], X[train_size:len(X)]
        y_train, y_test = y[0:train_size], y[train_size:len(y)]

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        best_params_ = Forecast.get_best_params(X_train, y_train, epochs, batch_sizes, activations)
        print(best_params_)
        return best_params_

    def create_dataset(self, dataset, sequence_length):
        sequences = []
        target = []
        for i in range(len(dataset) - sequence_length + 1):
            seq = dataset['scaled_data'].values[i:i+sequence_length]
            label = dataset['scaled_data'].values[i+sequence_length - 1]
            sequences.append(seq)
            target.append(label)
        return np.array(sequences), np.array(target)

    def moving_test_window_preds(self, n_future_preds=30):
        ''' n_future_preds - Represents the number of future predictions we want to make
                             This coincides with the number of windows that we will move forward
                             on the test data
        '''
        preds_moving = []                                    # Use this to store the prediction made on each test window
        moving_test_window = [self.testX[0,:].tolist()]          # Creating the first test window
        moving_test_window = np.array(moving_test_window)    # Making it an numpy array

        for i in range(n_future_preds):
            preds_one_step = self.model.predict(moving_test_window) # Note that this is already a scaled prediction so no need to rescale this
            preds_moving.append(preds_one_step[0,0]) # get the value from the numpy 2D array and append to predictions
            preds_one_step = preds_one_step.reshape(1,1) # Reshaping the prediction to 3D array for concatenation with moving test window
            moving_test_window = np.concatenate((moving_test_window[:,1:], preds_one_step), axis=1) # This is the new moving test window, where the first element from the window has been removed and the prediction  has been appended to the end

        preds_moving = np.array(preds_moving).reshape(-1,1)
        preds_moving = self.scaler.inverse_transform(preds_moving)
        preds_moving = preds_moving.astype(int)
        preds_moving = [0 if i < 0 else i for i in preds_moving]

        return preds_moving

    def start(self):
        another = np.array(self.data[self.data['alpha'] == self.alpha].set_index('dd')['value'].astype(float))

        params = self.get_best_params_for_data(another, time_steps=10, units=[50,100],epochs=[50,100], batch_sizes=[32, 64],
                                          activations=['tanh', 'relu', 'sigmoid'])

        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        self.activations = params['activation']
        # self.optimizer = params['optimizer']

        result = pd.DataFrame({'alpha':"", 'dd':0,"Прогноз по дням": 0, 'Сумма прогноза за месяц': 0}, index=[0])

        self.model.add(LSTM(self.units, input_shape=(self.sequence_length,1), activation=self.activation)) #, return_sequences=True, kernel_regularizer=regularizers.l2(0.01), recurrent_dropout=0.2, recurrent_initializer='glorot_uniform'
        self.model.add(Dense(1))
        self.model.compile(loss='mean_absolute_error', optimizer=self.optimizer) #Adagrad

        test_date = dt.datetime.strptime("1-10-2023", "%d-%m-%Y")
        workdays = pd.date_range(test_date, periods=self.periods)
        self.periods = len(workdays)

        try:
            temp = self.data
            temp['value'] = temp['value'].astype(float)
            temp['dd'] = pd.to_datetime(temp['dd'])
            temp.set_index('dd', inplace=True)
            temp['scaled_data'] = self.scaler.fit_transform(temp['value'].values.reshape(-1,1))

            # temp_ = temp_.dropna()

            X,y = self.create_dataset(temp, self.sequence_length)

            trainX, self.testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            self.model.fit(trainX, trainY, epochs=self.epochs, batch_size=self.batch_size, validation_data=(self.testX, testY),callbacks=[early_stopping], verbose=0)

            n_forecast = len(self.testX)
            forecast = self.model.predict(self.testX)

            forecast = self.scaler.inverse_transform(forecast)

            testY = testY.reshape(len(testY),1)
            testY = self.scaler.inverse_transform(testY)

            if len(forecast) > len(testY):
                forecast = forecast[:len(testY)]
            else:
                testY = testY[:len(forecast)]

            preds_moving = self.moving_test_window_preds(self.periods)
            sum_month = sum(preds_moving)[0]

            for j in range(0, self.periods):

                result.loc[len(result.index)] = [self.alpha, workdays[j], preds_moving[j][0], sum_month]

        except Exception as e:
            print(tb.format_exc())
            bad_alphas.append(self.alpha)

        result = result.drop(0, axis=0)
        return result

def process_alpha(alpha):
    periods = 30
    try:
        temp = data[data['alpha'] == alpha].sort_values(['alpha', 'dd'], ascending=[True, True])

        forecast = Forecast(periods, temp, alpha)

        result = forecast.start()

        return alpha, result

    except Exception as e:
        print(f"{str(e)} fsefesf")
        return alpha, None

def main():
    result_df = pd.DataFrame()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_alpha, alpha): alpha for alpha in alphas[:1]}

        for future in concurrent.futures.as_completed(futures):
            try:
                result_alpha = future.result()

                if result_alpha is not None:
                    alpha_result = pd.DataFrame(result_alpha[1])
                    alpha_result['alpha'] = result_alpha[0]
                    result_df = pd.concat([result_df, alpha_result])

            except Exception as e:
                print(f"{e} dawdawd")
    return result_df

if __name__ == "__main__":
    before = t.time()
    result = main()
    after = t.time()
    print(f"{(after-before)/60} min")
    print(result)
    result.to_excel("Result_october.xlsx")
