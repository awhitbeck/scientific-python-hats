from MLJEC_MCTruth_Model import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i", "--inFile", dest="input_filename",
                  help="file to load data from",default="new_ak5.pkl")
parser.add_option("-m", "--modelFile", dest="model_filename",
                  help="file to load model from",default="model_eta_dense_pt_dense_updatedJuly14")
parser.add_option("-o", "--outFile", dest="output_filename",
                  help="file to save mode to (leave off extension)",default="model_ak5_eta_dense_pt_dense")

(options, args) = parser.parse_args()
os.environ['KERAS_BACKEND'] = 'tensorflow'

print "LOADING DATA"
df = pd.read_pickle(options.input_filename)

print "COMPUTING JES_TRUTH"
df['jes_truth'] = df.apply(lambda row: row['jet_pt']/row['gen_pt'] if row['jet_gen_dr']<0.25 else -1.,axis=1)
print "SELECTING GOOD JETS"
df = df.loc[(df['jes_truth']>0.)&(df['jes_truth']<2.)&(df['ak5pfcand_ijet']<2)]

print "TRANSFORMING ETA/PT DISTRIBUTIONS"
scaler = StandardScaler()
df['jet_eta_scaled'] = df['jet_eta'] / 2.5
df['jet_pt_exp'] = map(np.log,df['jet_pt'])
df['jet_pt_scaled'] = scaler.fit_transform(df['jet_pt_exp'].reshape(-1, 1))
print df.head()

print "BUILDING TEST/TRAIN DATASET"
df = df.sample(frac=1)
df_train,df_test = np.array_split(df,2)
#print df_train.head()
print df_test.head()

print "REFORMATTING"
jet_image_train = np.array(map(lambda x : x[0] , df_train['jet_image']))
inputs_train = [jet_image_train.reshape([-1,30,30,1]),np.array(df_train['jet_pt_scaled']),np.array(df_train['jet_eta_scaled'])]
jet_image_test = np.array(map(lambda x : x[0] , df_test['jet_image']))
inputs_test = [jet_image_test.reshape([-1,30,30,1]),np.array(df_test['jet_pt_scaled']),np.array(df_test['jet_eta_scaled'])]

print "LOADING MODEL"
model = loadModel(options.model_filename)
model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['accuracy','mse','msle'])

from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='val_loss', patience=20)

history=model.fit(inputs_train, np.array(df_train['jes_truth']), validation_data=(inputs_test, np.array(df_test['jes_truth'])), 
                    nb_epoch=100, batch_size=1024, verbose=1)#, callbacks=[early_stopping])

val_loss = np.asarray(history.history['val_loss'])
loss = np.asarray(history.history['loss'])
plt.plot(np.log(val_loss), label='test')
plt.plot(np.log(loss), label='train')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Loss')
fig = plt.gcf()
plt.draw()
fig.savefig('ak5_training_summary.png',dpi=100)

model_json = model.to_json()
with open(options.output_filename+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(options.output_filename+".h5")
print("Saved model to disk")

