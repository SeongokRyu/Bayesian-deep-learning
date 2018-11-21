import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
#from tensorflow.python.layers.core import Dense
from tensorflow.keras.layers import Dense
from ConcreteDropout import ConcreteDropout
plt.switch_backend('agg')


hidden_dim = int(sys.argv[1])    # 512
N_train = int(sys.argv[2])       # 10, 50, 100, 500, 1000, 5000, 10000
N_test = int(sys.argv[3])        # 5000
noise = float(sys.argv[4])         # 0.1, 0.5, 1.0
length = float(sys.argv[5])        # 0.001
num_epochs = int(sys.argv[6])            # 100
batch_size = 100

filename = str(hidden_dim)+'_'+str(N_train)+'_'+str(N_test)+'_'+str(noise)+'_'+str(length)+'_cdo'
print("hidden_dim:", str(hidden_dim), "\t N_train:", str(N_train), "\t N_test:", str(N_test), "\t noise:", str(noise), "\t length", str(length))

def gen_data(N):
    X = np.random.rand(N,1).flatten()
    e = np.random.normal(0.0,noise,N).flatten()
    Y = 2.*X + 8. + e

    X = X.reshape((N,1))
    Y = Y.reshape((N,1))

    return X, Y

def load_data_train(N):
    data_ = np.load('./data/synthetic_'+str(noise)+'.npy')
    X = data_[0][:N]
    Y = data_[1][:N]
    X = X.reshape((N,1))
    Y = Y.reshape((N,1))
    return X, Y

def load_data_test(N):
    data_ = np.load('./data/synthetic_'+str(noise)+'.npy')
    X = data_[0][-N:]
    Y = data_[1][-N:]
    X = X.reshape((N,1))
    Y = Y.reshape((N,1))

    return X, Y

def heteroskedastic_loss(y_truth, y_mean, y_logvar):
    y_truth = tf.reshape(y_truth, [-1])
    y_mean = tf.reshape(y_mean, [-1])
    y_logvar = tf.reshape(y_logvar, [-1])
    reg_losses = tf.reduce_sum(tf.losses.get_regularization_losses())
    pred_losses = tf.reduce_mean(0.5*tf.exp(-y_logvar)*(y_truth - y_mean)**2 + y_logvar*0.5)
    loss = pred_losses+reg_losses
    return loss, pred_losses, reg_losses

def mle_loss(y_truth, y_mean):
    loss = tf.reduce_sum((y_truth-y_mean)**2., -1)    
    return loss

def lr_step_decay(epoch):
    initial_lrate=0.1
    drop=0.5
    epochs_drop=100.0

def fit_model():

    # 1. load training and test dataset
    X_train, Y_train = load_data_train(N_train)
    X_test, Y_test = load_data_test(N_test)

    # 2. Configuration
    wd = length**2 / N_train
    dd = 2. / N_train

    # 3. Model construction
    ## Hello world!
    ## DO NOT rm * !!
    tf.reset_default_graph()
    sess = tf.Session()

    inp = tf.placeholder(shape=(None,1), dtype=tf.float32)
    out = tf.placeholder(shape=(None,1), dtype=tf.float32)
    h = ConcreteDropout( Dense(hidden_dim, activation=tf.nn.relu),
                         weight_regularizer=wd, 
                         dropout_regularizer=dd, 
                         trainable=True )(inp, training=True)
    h = ConcreteDropout( Dense(hidden_dim, activation=tf.nn.relu),
                         weight_regularizer=wd, 
                         dropout_regularizer=dd, 
                         trainable=True )(h, training=True)
    h = ConcreteDropout( Dense(hidden_dim, activation=tf.nn.relu),
                         weight_regularizer=wd, 
                         dropout_regularizer=dd, 
                         trainable=True )(h, training=True)
    """
    y_mean = ConcreteDropout( Dense(1, activation=None),
                              weight_regularizer=wd, 
                              dropout_regularizer=dd, 
                              trainable=True )(h, training=True)
    y_logvar = ConcreteDropout( Dense(1, activation=None),
                                weight_regularizer=wd, 
                                dropout_regularizer=dd, 
                                trainable=True )(h, training=True)
    """
    y_mean = Dense(1, activation=None)(h)
    y_logvar = Dense(1, activation=None)(h)

    loss, pred_loss, reg_loss = heteroskedastic_loss(out, y_mean, y_logvar)
    lr = tf.Variable(0.0, trainable=False)
    opt = tf.train.AdamOptimizer(lr).minimize(loss)

    sess.run(tf.global_variables_initializer())

    # 4. Training
    _losses = []
    _pred_losses = []
    _reg_losses = []
    count = 0
    init_lr = 0.01
    sess.run(tf.assign(lr, init_lr))
    for epoch in range(num_epochs):
        old_batch = 0
        num_batch = int( np.ceil(X_train.shape[0] / batch_size) )
        for batch in range(num_batch):
            count += 1
            if (count % 100000 == 0):
                init_lr *= 0.5
                sess.run(tf.assign(lr, init_lr))
            batch = batch+1
            Xb = X_train[old_batch:batch_size*batch]
            Yb = Y_train[old_batch:batch_size*batch]
            _, _loss, _pred_loss, _reg_loss = sess.run([opt, loss, pred_loss, reg_loss], feed_dict={inp:Xb, out:Yb})
            old_batch = batch_size*batch
        _losses.append(_loss)
        _pred_losses.append(_pred_loss)
        _reg_losses.append(_reg_loss)
        print("Epoch:", str(epoch), "Loss:", str(_loss), "Pred loss:", str(_pred_loss), "KL condition:", str(_reg_loss))
    
    fig = plt.figure()
    x_ = list(range(num_epochs))
    plt.plot(x_, _losses, label='Total loss')
    plt.plot(x_, _pred_losses, label='Pred loss')
    plt.plot(x_, _reg_losses, label='Reg loss')
    plt.legend()
    fig.savefig('./figures/'+filename+'_loss.png')

    # 5. Test function
    def model_predict(X_val):
        _means = None
        _vars = None
        old_batch = 0
        num_batch = int( np.ceil(X_val.shape[0] / batch_size) )
        for batch in range(num_batch):
            batch = batch+1
            Xb = X_val[old_batch:batch_size*batch]
    
            _mean, _var = sess.run([y_mean, y_logvar], feed_dict={inp:Xb})
            old_batch = batch_size*batch
            _mean = _mean.flatten()
            _var = _var.flatten()
            _means = _mean if _means is None else np.concatenate([_means, _mean], axis=0)
            _vars = _var if _vars is None else np.concatenate([_vars, _var], axis=0)
        return _means, _vars

    return X_train, Y_train, X_test, Y_test, model_predict, sess

def plot_inference(X1, Y1, X2, Y_pred):
    fig = plt.figure()
    plt.title('N_training: ' + str(N_train) + ',  N_test: ' + str(N_test) + ',  noise: ' + str(noise))
    # 1) Training samples
    plt.scatter(X1, Y1, c='r', s=8)

    # 2) Truth line
    _x = np.arange(-2.5, 2.5)
    _y = 2.*_x + 8.
    plt.plot(_x, _y, c='black')

    # 3) Predictive mean
    pred_mean = np.mean(Y_pred, axis=0)
    plt.scatter(X2, pred_mean, c='b', s=8)
    plt.tight_layout()
    plt.savefig('./figures/Results_'+str(N_train)+'_'+str(N_test)+'_'+str(noise)+'.png')

    return

def uncertainties(mean, var):
    ale_unc = np.mean(var, axis=0)
    epi_unc = np.var(mean, axis=0)
    tot_unc = ale_unc + epi_unc

    return ale_unc, epi_unc, tot_unc

# 1. model construction & training
X_train, Y_train, X_test, Y_test, model_predict, sess = fit_model()

# 2. Monte Carlo inference
MC_sampling = 100
MC_mean = []
MC_var = []
for k in range(MC_sampling):
    mean_k, var_k = model_predict(X_test)
    MC_mean.append(mean_k)
    MC_var.append(var_k)

MC_mean = np.asarray(MC_mean)
MC_var = np.exp(np.asarray(MC_var))
# 3. Plot results
plot_inference(X_train, Y_train, X_test, MC_mean)

# 4. Uncertainty analysis
ale_unc, epi_unc, tot_unc = uncertainties(MC_mean, MC_var)

# 5. Summary
Y_truth = 2.*X_test + 8.
ps = np.asarray([sess.run(layer_p) for layer_p in tf.get_collection('LAYER_P')])
rmse = np.sqrt(np.mean((Y_truth.flatten() - np.mean(MC_mean, axis=0))**2))
summary = [rmse, np.mean(ale_unc), np.mean(epi_unc), np.mean(tot_unc), ps[0], ps[1], ps[2]]
#summary = [rmse, np.mean(ale_unc), np.mean(epi_unc), np.mean(tot_unc), ps[0], ps[1], ps[2], ps[3], ps[4]]
print ("RMSE, Aleatoric, Epistemic, Predictive uncertanties, Dropout probability #1/#2/#3/#4")
print (summary)

# 6. Save quantities
np.save('./statistics/Summary_'+filename+'.npy', np.asarray(summary))
np.save('./statistics/Aleatoric_'+filename+'.npy', np.asarray(ale_unc))
np.save('./statistics/Epistemic_'+filename+'.npy', np.asarray(epi_unc))
np.save('./statistics/Total_'+filename+'.npy', np.asarray(tot_unc))
