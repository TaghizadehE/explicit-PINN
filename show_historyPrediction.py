import numpy as np
import tensorflow.keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import config
# from train import load_model
from datasets import prepare_learning_data

def showHistory(history, model):
#     %matplotlib inline
    # -------------------- Plot model and history of training ----------------------
    # plot_model(model, to_file='regression-model.png', show_shapes=True, show_layer_names=True)
    # print(history.history.keys())
    
    #  "MSE"
    fs = 28
    plt.figure(figsize=(12,8))
    plt.subplot()
    plt.plot(history.history['mse'], linewidth=4)
    plt.plot(history.history['val_mse'], linewidth=4)
#         plt.title('model MSE')
    plt.ylabel('mean squared error', fontsize=fs)
    plt.xlabel('epoch', fontsize=fs)
    plt.legend(['train', 'validation'], loc='upper right', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xticks(np.arange(0.0, config.EPOCH+1.0, step=400))
    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(0, (1.03)*ymax, 'B', fontsize=fs+8)
    plt.tight_layout()
    plt.savefig('figs/MSE_E'+str(config.EPOCH)+'.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print('min training mse',             np.array(history.history['mse']).min())
    print('min validation mse',           np.array(history.history['val_mse']).min())
    print('training mse at last epoch',   np.array(history.history['mse'])[-1])
    print('validation mse at last epoch', np.array(history.history['val_mse'])[-1])


    # "Loss"
    plt.figure(figsize=(12,8))
    plt.subplot()
    plt.plot(history.history['loss'], linewidth=4)
    plt.plot(history.history['val_loss'], linewidth=4,) # alpha=0.7)
    # plt.title('model loss')
#         plt.ylabel('loss (mean_absolute_percentage_error)')
    plt.ylabel('loss (%)', fontsize=fs)
    plt.xlabel('epoch',    fontsize=fs)
    plt.legend(['train', 'validation'], loc='upper right',fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xticks(np.arange(0.0, config.EPOCH+1.0, step=400))
    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(0, (1.03)*ymax, 'A', fontsize=fs+8)
    plt.tight_layout()
    plt.savefig('figs/loss_E'+str(config.EPOCH)+'.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print('min training loss',             np.array(history.history['loss']).min())
    print('min validation loss',           np.array(history.history['val_loss']).min())
    print('training loss at last epoch',   np.array(history.history['loss'])[-1])
    print('validation loss at last epoch', np.array(history.history['val_loss'])[-1])
    
    
    
def showPredictionXY(Y_train, predictions_train, Y_test, predictions_test):
    fs = 28
    Nstep = 1 #round(self.nb_realization/500)
    plt.figure(figsize=(12,8))
    plt.subplot()
    plt.plot(Y_train[::Nstep], predictions_train[::Nstep], 'or', markersize=fs-16, fillstyle='none', label='train')
#         plt.plot(Y_test, predictions_test,   ',g', markersize=fs-0, fillstyle='none', label='test', alpha=0.2)
    plt.plot(Y_test[::Nstep], predictions_test[::Nstep],   '.g', markersize=fs-12, label='test', alpha=0.5)
    plt.plot([0, 0.5, 1], [0, 0.5, 1], 'k', linewidth=2.5, label='y=x')
#         plt.plot([np.amin(Y_train), 0.5, np.amax(Y_train)], [np.amin(Y_train), 0.5, np.amax(Y_train)], 'k')
    plt.xlabel('$\eta$', fontsize=fs)
    plt.ylabel('predicted $\eta$', fontsize=fs)
    plt.legend(loc='upper left', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(0, (1.05)*ymax, 'A', fontsize=fs+8)
    plt.savefig('figs/predicted_eta_E'+str(config.EPOCH)+'.png', dpi=300, bbox_inches='tight')
    plt.savefig('figs/predicted_eta_E'+str(config.EPOCH)+'.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    
    # histogram
    plt.subplot()
    plt.hist(np.concatenate((Y_train,Y_test), axis=0) )
    plt.xlabel('$\eta$')
    plt.ylabel('# occurance')
    plt.savefig('figs/hist_eta_E'+str(config.EPOCH)+'.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    
def prepare_PredictionSurf():
    load_model = tensorflow.keras.models.load_model(config.SAVE_MODEL_NAME)
    features_norm, values, features = prepare_learning_data()
    predictions_train_test = load_model.predict(features_norm).flatten()

    # feature sequence: (0)epsilon_beta, (1)phi2, (2)Pe, (3)Dr, (4)c_f_avg_norm, (5)d_f_c_avg_norm,  
    # target: (6)eta_d_f
    fsx = 4
    fsy = 2

    if fsx == 0:
        x  = features[:,0]
        labelx = '$\epsilon_\beta$'
    elif fsx == 1:
        x  = features[:,1]
        labelx = '$\phi^2$'
    elif fsx == 2:
        x  = features[:,2]
        labelx = '$Pe$'
    elif fsx == 3:
        x  = features[:,3]
        labelx = '$D_r$'
    elif fsx == 4:
        x  = features[:,4]
        labelx = r'$\langle c_{\beta}\rangle^\beta$'
    elif fsx == 5:
        x  = features[:,5]
        labelx = r'$\frac{\partial}{\partial x} \langle c_{\beta}\rangle^\beta$'
    else:
        print('sth wrong!')



    if fsy == 0:
        y  = features[:,0]
        labely = '$\epsilon_\beta$'            
    elif fsy == 1:
        y  = features[:,1]
        labely = '$\phi^2$'
    elif fsy == 2:
        y  = features[:,2]
        labely = '$Pe$'
    elif fsy == 3:
        y  = features[:,3] 
        labely = '$D_r$'
    elif fsy == 4:
        y  = features[:,4]
        labely = r'$\langle c_{\beta}\rangle^\beta$'
    elif fs1 == 5:
        y  = features[:,5]
        labely = r'$\frac{\partial}{\partial x} \langle c_{\beta}\rangle^\beta$'
    else:
        print('sth wrong!')


    print('the selected features for plotting are:', labelx, 'and', labely)

    z1 = values
    z2 = predictions_train_test
    
    return x, labelx, y, labely, z1, z2
 

def showPredictionSurf():
    print('entering 2D plot')
#         %matplotlib qt
#         %matplotlib inline
    x, labelx, y, labely, z1, z2 = prepare_PredictionSurf() 
    
    fs1=28
    fs2=10

    # choose 2D
#         fig3, (ax1, ax2, ax3,) = plt.subplots(1, 3)
    fig3 = plt.figure(figsize=(12,8))
#         plt.subplots_adjust(wspace=0.30,)        
    plt.subplots_adjust(right=1.4,)

    ax1 = fig3.add_subplot(131)
    sc1 = ax1.scatter(x, y, c=z1, alpha=0.9, vmin=0)
    ax1.set_title('$\eta$', fontsize=fs1)
    ax1.set_xlabel(labelx, fontsize=fs1)
    ax1.set_ylabel(labely, fontsize=fs1)
    plt.xticks(fontsize=fs1-fs2)
    plt.yticks(fontsize=fs1-fs2)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(0, (1.05)*ymax, 'B', fontsize=fs1+8)

    ax2 = fig3.add_subplot(132)
    sc2 = ax2.scatter(x, y, c=z2, alpha=0.9)
#         ax2.axes.get_yaxis().set_visible(False)
    plt.xticks(fontsize=fs1-fs2)
    plt.yticks(fontsize=fs1-fs2)
    ax2.set_yticklabels([])
    ax2.set_title('predicted $\eta$', fontsize=fs1)
#        ax2.set_xlabel(self.labelx, fontsize=fs)
#         ax2.set_ylabel(labely)

    cax, kw = mpl.colorbar.make_axes([ax1, ax2])
    cb12 = fig3.colorbar(sc2, cax=cax, **kw,)
    cb12.outline.set_visible(False)
    cb12.ax.tick_params(labelsize=fs1-fs2)
#         cb12.set_label('$\eta$', fontsize=fs1, rotation=0)

    ax3 = fig3.add_subplot(133)
#         plt.subplots_adjust(right=2.0,)
#         plt.subplots_adjust(left=1, bottom=None, right=None, top=None, wspace=None, hspace=None)
    sc3 = ax3.scatter(x, y, c=np.absolute((z2-z1)/z1*100), cmap='plasma_r', vmin=0, vmax=8)
    plt.xticks(fontsize=fs1-fs2)
    plt.yticks(fontsize=fs1-fs2)
#         ax3.set_yticklabels([])
    ax3.set_xlabel(labelx, fontsize=fs1)
    ax3.set_ylabel(labely, fontsize=fs1, labelpad=0)
    cax, kw = mpl.colorbar.make_axes([ax3])
    cb3 = fig3.colorbar(sc3, cax=cax, **kw)
    cb3.outline.set_visible(False)
    cb3.ax.tick_params(labelsize=fs1-fs2)
    ax3.set_title('err. (%)', fontsize=fs1)
#         cb3.set_label('err (%)', fontsize=fs1, rotation=90)
#         plt.tight_layout()
#         plt.show()
    plt.savefig('figs/2D_predicted_eta_'+str(config.EPOCH)+'.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figs/2D_predicted_eta_'+str(config.EPOCH)+'.png', dpi=300, bbox_inches='tight')
   