
# coding: utf-8

# In[66]:

from MLJEC_MCTruth_Model import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from plotterUtils import *


# In[2]:

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES']='2'


# In[3]:

model = loadModel('model_eta_dense_pt_dense_updatedJuly14')


# In[64]:

model.summary()


# In[4]:

df = pd.read_pickle('new.pkl')


# In[6]:

scaler = StandardScaler()
df['jet_eta_ak7_scaled'] = df['jet_eta_ak7'] / 2.5
df['jet_pt_ak7_exp'] = map(np.log,df['jet_pt_ak7'])
df['jet_pt_ak7_scaled'] = scaler.fit_transform(df['jet_pt_ak7_exp'].reshape(-1, 1))
df.head()


# In[7]:

plt.hist(df['jet_pt_ak7_scaled'],bins=20)
plt.show()
plt.hist(df['jet_eta_ak7_scaled'],bins=20)
plt.show()


# In[8]:

jet_image = np.array(map(lambda x : x[0] , df['jet_image']))


# In[9]:

inputs = [jet_image.reshape([-1,30,30,1]),np.array(df['jet_pt_ak7_scaled']),np.array(df['jet_eta_ak7_scaled'])]


# In[10]:

model.compile(optimizer='adam', loss='mse', metrics=['accuracy','mse','msle'])


# In[11]:

from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='val_loss', patience=20)


# In[12]:

model.fit(inputs, np.array(df['jet_jes_ak7']), validation_data=(inputs, np.array(df['jet_jes_ak7'])), 
                    nb_epoch=50, batch_size=1024, verbose=1, callbacks=[early_stopping])


# In[14]:

model_json = model.to_json()
with open("model_eta_dense_pt_dense_updatedSept19.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_eta_dense_pt_dense_updatedSept19.h5")
print("Saved model to disk")


# In[15]:

df['pred'] = model.predict(inputs)
df['residual'] = df['pred']-df['jet_jes_ak7']


# In[34]:

plt.hist(df['pred'],bins=np.arange(0.5,1.2,.01),histtype='step')
plt.hist(df['jet_jes_ak7'],bins=np.arange(0.5,1.2,.01),histtype='step')
plt.show()


# In[58]:

plt.hist2d(df['jet_pt_ak7'],df['residual'],weights=df['mcweight'],bins=50)#,norm=LogNorm())
plt.grid()
plt.colorbar()
plt.show()


# In[60]:

plt.hist2d(df['jet_pt_ak7'],df['residual'],weights=df['mcweight'],bins=50,norm=LogNorm())
plt.grid()
plt.colorbar()
plt.show()


# In[61]:

plt.hist2d(df['jet_eta_ak7'],df['residual'],weights=df['mcweight'],bins=50)#,norm=LogNorm())
plt.grid()
plt.colorbar()
plt.show()


# In[62]:

plt.hist2d(df['jet_eta_ak7'],df['residual'],weights=df['mcweight'],bins=50,norm=LogNorm())
plt.grid()
plt.colorbar()
plt.show()


# In[50]:

# df['photon_filter'] = df.apply(lambda row : map(lambda x : abs(x)==22,row['ak7pfcand_id']),axis=1)
# df['electron_filter'] = df.apply(lambda row : map(lambda x : abs(x)==11,row['ak7pfcand_id']),axis=1)
# df['muon_filter'] = df.apply(lambda row : map(lambda x : abs(x)==13,row['ak7pfcand_id']),axis=1)
# df['charHad_filter'] = df.apply(lambda row : map(lambda x : abs(x)==211,row['ak7pfcand_id']),axis=1)
# df['neutHad_filter'] = df.apply(lambda row : map(lambda x : abs(x)==130,row['ak7pfcand_id']),axis=1)

# df['jet_photon_mult_ak7'] = df.apply(lambda row : sum(row['photon_filter']),axis=1)
# df['jet_electron_mult_ak7'] = df.apply(lambda row : sum(row['electron_filter']),axis=1)
# df['jet_muon_mult_ak7'] = df.apply(lambda row : sum(row['muon_filter']),axis=1)
# df['jet_charHad_mult_ak7'] = df.apply(lambda row : sum(row['charHad_filter']),axis=1)
# df['jet_neutHad_mult_ak7'] = df.apply(lambda row : sum(row['neutHad_filter']),axis=1)

# df['jet_photon_frac_ak7'] = df.apply(lambda row : sum(np.multiply(row['photon_filter'],row['ak7pfcand_pt']))/row['jet_pt_ak7'],axis=1)
# df['jet_electron_frac_ak7'] = df.apply(lambda row : sum(np.multiply(row['electron_filter'],row['ak7pfcand_pt']))/row['jet_pt_ak7'],axis=1)
# df['jet_muon_frac_ak7'] = df.apply(lambda row : sum(np.multiply(row['muon_filter'],row['ak7pfcand_pt']))/row['jet_pt_ak7'],axis=1)
# df['jet_charHad_frac_ak7'] = df.apply(lambda row : sum(np.multiply(row['charHad_filter'],row['ak7pfcand_pt']))/row['jet_pt_ak7'],axis=1)
# df['jet_neutHad_frac_ak7'] = df.apply(lambda row : sum(np.multiply(row['neutHad_filter'],row['ak7pfcand_pt']))/row['jet_pt_ak7'],axis=1)


# In[68]:

#df.to_pickle("new_withCNN.pkl")


# In[63]:

for v in plot_var : 
    plt.hist2d(df[v],df['residual'],weights=df['mcweight'],bins=50,norm=LogNorm())
    plt.grid()
    plt.ylabel('residual')
    plt.xlabel(v)
    plt.colorbar()
    plt.show()


# In[72]:

def residual_profile(plot_var,low_bin,high_bin,nbins):
    print "plot_var:",plot_var
    print "low_bin:",low_bin
    print "high_bin:",high_bin
    print "nbins:",nbins
    df[plot_var+'_bins'] = pd.cut(df[plot_var],np.arange(low_bin,high_bin+0.00001,(high_bin-low_bin)/float(nbins)),labels=range(nbins))
    means = df.groupby([plot_var+'_bins']).mean()['residual'].values
    counts = df.groupby([plot_var+'_bins']).count()['residual'].values
    sqrt_counts = map(sqrt,counts)
    errs  = df.groupby([plot_var+'_bins']).std()['residual'].values
    errs = errs/sqrt_counts
    bin_center=np.arange(low_bin,high_bin+0.00001,(high_bin-low_bin)/float(nbins))
    bin_center = bin_center[:-1]
    bin_center = map(lambda x : x+(high_bin-low_bin)/float(nbins)/2.,bin_center)
    plt.errorbar(x=bin_center,y=means,yerr=errs,fmt='o',color='m')
    plt.xlabel(plot_var)
    plt.ylabel("<residual>")
    plt.ylim(-0.02,0.02)
    plt.show()


# In[82]:

plot_vars=['jet_pt_ak7','jet_eta_ak7','jet_electron_frac_ak7','jet_muon_frac_ak7','jet_photon_frac_ak7','jet_neutHad_frac_ak7','jet_charHad_frac_ak7','jet_electron_mult_ak7','jet_muon_mult_ak7','jet_photon_mult_ak7','jet_neutHad_mult_ak7','jet_charHad_mult_ak7']
var_binning = [(100,1000,90),(-3,3,60),(0,1.5,50),(0,1.5,50),(0,1.5,50),(0,1.5,50),(0,1.5,50),(0,100,100),(0,100,100),(0,100,100),(0,100,100),(0,100,100)]
for v,b in zip(plot_vars,var_binning) :
    residual_profile(v,*b)


# In[ ]:



