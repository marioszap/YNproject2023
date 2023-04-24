import pandas as pd
import keras
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from scipy.stats import zscore
import matplotlib.pyplot as plt


csvDf = pd.read_csv('dataset-HAR-PUC-Rio.csv', sep=';')
classes = csvDf.pop('class')

falseFloatCols = ['how_tall_in_meters', 'body_mass_index']

for i in falseFloatCols:
    csvDf[i] = csvDf[i].str.replace(',', '.').astype(float)
#line 122076: z4 = -14420-11-2011 04:50:23.713 needs to be set to -144 manually

encoder = LabelEncoder()
nonNumericCols = csvDf.select_dtypes('object').columns
#gender: 1=female, 0=male | user: debora=0, jose-carlos=1, katia=2, wallace=3
for i in nonNumericCols:
    csvDf[i] = encoder.fit_transform(csvDf[i])

csvDf_centered = pd.DataFrame()
csvDf_normalized = pd.DataFrame()
csvDf_z = pd.DataFrame()

#csvDf = csvDf.drop(['user', 'gender', 'age', 'body_mass_index'], axis=1)
for i in csvDf.columns:
    csvDf_centered[i] = csvDf[i]-csvDf[i].mean()
    csvDf_normalized[i] = (csvDf[i]-csvDf[i].min()) / (csvDf[i].max()-csvDf[i].min())
csvDf_z = zscore(csvDf)

rescaledData = csvDf_z.to_numpy()

X = rescaledData
Y = encoder.fit_transform(classes)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mseList = []
accList = []
CElossList = []

loss = 'sparse_categorical_crossentropy'

num_epochs = 500
epochs = list(range(num_epochs))
ax = [None] * 5
coor = [(2,6), (0,0), (0,2), (0,4), (1,1), (1,3)]

for i, (train, test) in enumerate(kfold.split(X, classes)):
    model = Sequential()

    model.add(Dense(28, input_shape=(18,), activation="relu"))#, activity_regularizer=regularizers.l2(0.1)))
    #uncomment to use deep neural network
    #model.add(Dense(23, activation='relu'))#, activity_regularizer=regularizers.l2(0.1)))
    #model.add(Dense(10, activation='relu'))
    model.add(Dense(5, activation="softmax"))#, activity_regularizer=regularizers.l2(0.1)))

    keras.optimizers.SGD(learning_rate=0.001, momentum=0.2, decay=0.0, nesterov=False)
    model.compile(loss=loss, optimizer='sgd', metrics=['accuracy', 'MeanSquaredError'])
    es = EarlyStopping(monitor = 'val_loss', verbose=0, patience=50, mode='min')
    mc = ModelCheckpoint('best_m.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=0)
    history = model.fit(X[train], Y[train], epochs=num_epochs, batch_size=500, verbose=2, validation_data=(X[test], Y[test]), callbacks=[es, mc])
    scores = model.evaluate(X[test], Y[test], verbose=2)

    mseList.append(scores[2])
    accList.append(scores[1])
    CElossList.append(scores[0])
    #print("Fold :", i+1, " RMSE:", scores[1], '\n')
    ax[i] = plt.subplot2grid(coor[0], coor[i+1], colspan=2)
    ax[i].set_title(f'Fold {i+1}')
    ax[i].set_xlabel('Epochs')

    ax[i].plot(epochs, history.history['accuracy'], label = 'Accuracy')
    ax[i].plot(epochs, history.history['val_accuracy'], label = 'Eval Accuracy')
    ax[i].plot(epochs, history.history['loss'], label = 'Loss')
    ax[i].plot(epochs, history.history['val_loss'], label = 'Eval Loss')
    ax[i].set_ylim(ymax=1)
    ax[i].legend()
print('CE: ',np.mean(CElossList), 'ACC: ',np.mean(accList), 'MSE: ',np.mean(mseList))
plt.subplots_adjust(hspace=0.3, wspace=0.6)
wm = plt.get_current_fig_manager()
wm.window.state('zoomed')
plt.show()