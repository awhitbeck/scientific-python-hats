import matplotlib.pyplot as plt

# now let's prepare some jet images
print params['TT'].dtype.names

df_dict_jet = {}
df_dict_jet['TT'] = pd.DataFrame(params['TT'],columns=['run', 'lumi', 'event', 'met', 'sumet', 'rho', 'pthat', 'mcweight', 'njet_ak7', 'jet_pt_ak7', 'jet_eta_ak7', 'jet_phi_ak7', 'jet_E_ak7', 'jet_msd_ak7', 'jet_area_ak7', 'jet_jes_ak7', 'jet_tau21_ak7', 'jet_isW_ak7', 'jet_ncand_ak7','ak7pfcand_ijet'])
df_dict_jet['TT'] = df_dict_jet['TT'].drop_duplicates()
df_dict_jet['TT'] =  df_dict_jet['TT'][(df_dict_jet['TT'].jet_pt_ak7 > 200) & (df_dict_jet['TT'].jet_pt_ak7 < 500) &  (df_dict_jet['TT'].jet_isW_ak7==1)]

df_dict_cand = {}
df_dict_cand['TT'] = pd.DataFrame(params['TT'],columns=['event', 'jet_pt_ak7', 'jet_isW_ak7', 'ak7pfcand_pt', 'ak7pfcand_eta', 'ak7pfcand_phi', 'ak7pfcand_id', 'ak7pfcand_charge', 'ak7pfcand_ijet'])
df_dict_cand['TT'] =  df_dict_cand['TT'][(df_dict_cand['TT'].jet_pt_ak7 > 200) & (df_dict_cand['TT'].jet_pt_ak7 < 500) &  (df_dict_cand['TT'].jet_isW_ak7==1)]


for QCDbin in ['QCD120','QCD170','QCD300','QCD470']:
    df_dict_jet[QCDbin] = pd.DataFrame(params[QCDbin],columns=['run', 'lumi', 'event', 'met', 'sumet', 'rho', 'pthat', 'mcweight', 'njet_ak7', 'jet_pt_ak7', 'jet_eta_ak7', 'jet_phi_ak7', 'jet_E_ak7', 'jet_msd_ak7', 'jet_area_ak7', 'jet_jes_ak7', 'jet_tau21_ak7', 'jet_isW_ak7', 'jet_ncand_ak7','ak7pfcand_ijet'])
    df_dict_jet[QCDbin] = df_dict_jet[QCDbin].drop_duplicates()
    df_dict_jet[QCDbin] =  df_dict_jet[QCDbin][(df_dict_jet[QCDbin].jet_pt_ak7 > 200) & (df_dict_jet[QCDbin].jet_pt_ak7 < 500) &  (df_dict_jet[QCDbin].jet_isW_ak7==0)]
    # take every 20th jet just to make the training faster and have a sample roughly the size of W jets
    df_dict_jet[QCDbin] = df_dict_jet[QCDbin].iloc[::20, :]
    
    df_dict_cand[QCDbin] = pd.DataFrame(params[QCDbin],columns=['event', 'jet_pt_ak7', 'jet_isW_ak7', 'ak7pfcand_pt', 'ak7pfcand_eta', 'ak7pfcand_phi', 'ak7pfcand_id', 'ak7pfcand_charge', 'ak7pfcand_ijet'])
    df_dict_cand[QCDbin] =  df_dict_cand[QCDbin][(df_dict_cand[QCDbin].jet_pt_ak7 > 200) & (df_dict_cand[QCDbin].jet_pt_ak7 < 500) &  (df_dict_cand[QCDbin].jet_isW_ak7==0)]
    
df_dict_jet['QCD'] = pd.concat([df_dict_jet['QCD120'],df_dict_jet['QCD170'],df_dict_jet['QCD300'],df_dict_jet['QCD470']])
df_dict_cand['QCD'] = pd.concat([df_dict_cand['QCD120'],df_dict_cand['QCD170'],df_dict_cand['QCD300'],df_dict_cand['QCD470']])

print len(df_dict_jet['QCD'])



nx = 30 # size of image in eta
ny = 30 # size of image in phi
xbins = np.linspace(-1.4,1.4,nx+1)
ybins = np.linspace(-1.4,1.4,ny+1)
jet_images = {}
# 4D tensor
# 1st dim is jet index
# 2nd dim is pt value (or rgb, etc.)
# 3rd dim is eta bin
# 4th dim is phi bin


# In[8]:

def plotJet(process='TT',njets_to_plot=-1):
    list_x = []
    list_y = []
    list_w = []
    njets = 0

    if K.image_dim_ordering()=='tf':
        jet_images[process] = np.zeros((len(df_dict_jet[process]), nx, ny, 1))
    else:        
        jet_images[process] = np.zeros((len(df_dict_jet[process]), 1, nx, ny))
    
    njets_ = []
    if njets_to_plot == -1:
        njets_ = range(0,len(df_dict_jet[process]))
    else:
        njets_ = range(0,njets_to_plot)
    
    for i in njets_:
        njets+=1
        # get the ith jet
        df_dict_cand_i = {}
        df_dict_cand_i[process] = df_dict_cand[process][(df_dict_cand[process]['ak7pfcand_ijet'] == df_dict_jet[process]['ak7pfcand_ijet'].iloc[i]) & (df_dict_cand[process]['event'] == df_dict_jet[process]['event'].iloc[i]) ]
        # relative eta
        x = df_dict_cand_i[process]['ak7pfcand_eta']-df_dict_cand_i[process]['ak7pfcand_eta'].iloc[0]
        # relative phi
        y = df_dict_cand_i[process]['ak7pfcand_phi']-df_dict_cand_i[process]['ak7pfcand_phi'].iloc[0]
        weights = df_dict_cand_i[process]['ak7pfcand_pt'] # pt of candidate is the weight
        x,y = rotate_and_reflect(x,y,weights)
        list_x.append(x)
        list_y.append(y)
        list_w.append(weights)
        hist, xedges, yedges = np.histogram2d(x, y,weights=weights, bins=(xbins,ybins))
        for ix in range(0,nx):
            for iy in range(0,ny):
                if K.image_dim_ordering()=='tf':
                    jet_images[process][i,ix,iy,0] = hist[ix,iy]
                else:
                    jet_images[process][i,0,ix,iy] = hist[ix,iy]
    all_x = np.concatenate(list_x)
    all_y = np.concatenate(list_y)
    all_w = np.concatenate(list_w)
    all_w = 1.*all_w/njets # to get average
    plt.figure('W') 
    plt.hist2d(all_x, all_y, weights=all_w, bins=(xbins,ybins), norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.xlabel('eta')
    plt.ylabel('phi')
    plt.show()


plotJet(process='TT',njets_to_plot=1)
plotJet('QCD',-1)


#Plot the ROC curves for the training above
for cv, color in zip(range(0,2), colors):
    nbatches = 100
    jetImageGenerator2 = JetImageGenerator()
    gen = jetImageGenerator2.generator(test=True)
    X_predict = []
    y_predict = []
    for i in range(nbatches):
        if i%10==0:
            print "Jet",i
        Xp, yp = gen.next()
        X_predict += [Xp]
        y_predict += [yp]
    X_predict = np.concatenate(X_predict)
    y_predict = np.concatenate(y_predict)
    y_score = conv_model.predict(X_predict)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_predict, y_score)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    i += 1
    
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')
mean_tpr /= kfold.get_n_splits(X, encoded_Y)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
plt.xlim([0, 1.0])
plt.ylim([0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[ ]:

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    #x *= 255
    if K.image_dim_ordering()=='tf':
        x = x.transpose((1, 2, 0))
        x = x.transpose((1, 0, 2))
    #x = np.clip(x, 0, 255).astype('uint8')
    return x

def plot_layer(conv_model):
    # dimensions of the generated pictures for each filter.
    img_width = nx
    img_height = ny

    layer_dict = dict([(layer.name, layer) for layer in conv_model.layers])
    input_img = conv_model.input

    pics = {}
    num_filters = {}
    for key, layer in layer_dict.iteritems():
        if 'convolution2d' in key:
            num_filters[key] = 8
    layer_name = num_filters.keys()[1]
    print layer_name

    kept_filters = []

    # can be any integer from 0 to 7, as there are 8 filters in that layer
    for filter_index in range(0,num_filters[layer_name]):
        # build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
        if K.image_dim_ordering()=='tf':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
    
        if K.image_dim_ordering()=='tf':
            input_img_data = np.random.random((1, img_width, img_height, 1))
        else:
            input_img_data = np.random.random((1, 1, img_width, img_height))
        
        input_img_data = (input_img_data - 0.5) * 20 + 128

        # we run gradient ascent for 100 steps
        for i in range(100):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

        # decode the resulting input image
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    
    plt.figure(figsize=(15,15))
    i = 0
    j = 0
    for img, loss  in kept_filters:
        i+=1
        plt.subplot(3,3,i)
        plt.imshow(img[0])
        plt.colorbar()
    plt.show()
