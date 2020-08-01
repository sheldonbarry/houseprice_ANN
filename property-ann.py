import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# King County, Washington State, USA house data
# historic data of houses sold between May 2014 to May 2015
# https://github.com/dbendet/coursera_machine_learning/blob/master/kc_house_data.csv
data = pd.read_csv("kc_house_data.csv")

# check data
# print(data.columns.values)

# drop data with zero bedrooms and bathrooms and bedrooms outlier
data = data.drop(data[data.bedrooms == 0].index)
data = data.drop(data[data.bedrooms == 33].index)
data = data.drop(data[data.bathrooms == 0].index)
# drop columns that we won't be using
data = data.drop(['id', 'date', 'zipcode'], axis=1)

# scale data to values between 0 and 1
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data) # numpy array
# convert numpy array to dataframe with data columns names
data_scaled = pd.DataFrame(data_scaled, columns=data.columns.values)

# dependent variable
y_scaled = data_scaled[['price']].values

# extract independent variables (all columns except price)
X_scaled = data_scaled.drop(['price'], axis=1).values

# randomly split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.2)

x_len = len(X_scaled[0])

model = Sequential([
	# dense layer with 32 neurons, relu activation and x_len features
	Dense(32, activation='relu', input_shape=(x_len,)),
	# dense layer with 16 neurons and relu activation (constrain to 0 to 1)
	Dense(16, activation='relu'),
	# dense output layer with one neuron
	Dense(1, activation='linear'),						
	])

# build the model
model.compile(optimizer='adam',
	loss='mean_squared_error')

# train model
hist = model.fit(X_train, y_train,
	batch_size=32, epochs=50,
	validation_data=(X_test, y_test))

# make some predictions using the model
y_pred = model.predict(X_test)

# evaluate model based on the predictions
print ('Model evaluation:')
print ('R squared:\t {}'.format(r2_score(y_test, y_pred)))
# calculate RMSE based on unscaled value
rmse = sqrt(mean_squared_error(y_test, y_pred)) * (data['price'].max() - data['price'].min())
print ('RMSE:\t\t {}'.format(rmse))

# visualise loss during training
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training loss', 'Validation loss'], loc='upper right')
plt.tick_params(labelsize=8)
plt.tight_layout()
plt.show()
