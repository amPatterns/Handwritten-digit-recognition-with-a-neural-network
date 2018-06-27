import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils


# baseline model with 2 hidden layers
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(300, input_dim=784, init='normal', activation='sigmoid'))
    model.add(Dense(10, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
estimator = KerasClassifier(build_fn=baseline_model, epochs=50, batch_size=1000, verbose=2)

dataframet = pandas.read_csv("images_test.csv", header=None)
Xt = dataframet.values
Yt=pandas.read_csv("labels_test.csv", header=None)
dummy_yt = np_utils.to_categorical(Yt)

dataframe = pandas.read_csv("images_train.csv", header=None)
X = dataframe.values
Y=pandas.read_csv("labels_train.csv", header=None)
dummy_y = np_utils.to_categorical(Y)

# fit our model 
estimator.fit(X,dummy_y)
