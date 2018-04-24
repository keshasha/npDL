from myfunctions import *
import numpy as np

RANDOM_LEARNING = False

class Affine:
    trainable = True
    def __init__(self, shape, initializer="He"):
        self.shape = shape
        self.n_input = shape[0]
        self.n_output = shape[1]
        
        self.initialize_weight(loc=0, scale=0.01, initializer=initializer)
        # Momentum
        self.v_W = 0
        self.v_b = 0
        
        # AdaGrad
        self.h_W = 0
        self.h_b = 0
        
        # AdamGrad
        self.adam_v_W = 0
        self.adam_m_W = 0
        
        self.adam_v_b= 0
        self.adam_m_b = 0
        
        self.iter=0
        
    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
    
    def initialize_weight(self, loc=0, scale=0.01, initializer="He"):
        if initializer.lower()=="he":
            self.W = np.random.randn(self.n_input, self.n_output)/np.sqrt(self.n_input/2)
        elif initializer.lower()=="xavier":
            self.W = np.random.rand(self.n_input, self.n_output)/np.sqrt(self.n_input/1)
        else:
            self.W = np.random.normal(loc=loc, scale=scale, size=[self.n_input, self.n_output])
        self.b = np.zeros(shape=[1, self.n_output])
    
class Relu:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x <=0)
        out = x.copy()
        out[self.mask] = 0
        return out
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    
class Sigmoid:
    def __init__(self):
        self.out=None
    def forward(self, x):
        self.out = 1/(1+np.exp(-x))
        return self.out
    def backward(self, dout):
        dx = dout*(1.0-self.out)*self.out
        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t)/batch_size
        return dx

class Batch_normalization:
    trainable = True
    def __init__(self):
        self.xhat=None
        self.gamma=1
        self.beta = 0
        self.xmu=None
        self.ivar=None
        self.sqrtvar=None
        self.var=None
        self.eps=1e-7
        
        self.dgamma = 0
        self.dbeta = 0
        
    def forward(self, x):
        N, D = x.shape
        mu = 1./N * np.sum(x, axis = 0)
        xmu = x - mu
        sq = xmu ** 2
        var = 1./N * np.sum(sq, axis = 0)
        sqrtvar = np.sqrt(var + self.eps)
        ivar = 1./sqrtvar
        xhat = xmu * ivar
        gammax = self.gamma * xhat
        out = gammax + self.beta
        self.xhat,self.xmu,self.ivar,self.sqrtvar,self.var = (xhat,xmu,ivar,sqrtvar,var)
        return out
    
    def backward(self, dout):
        xhat,xmu,ivar,sqrtvar,var = self.xhat,self.xmu,self.ivar,self.sqrtvar,self.var

        N,D = dout.shape

        self.dbeta = np.sum(dout, axis=0)
        dgammax = dout

        self.dgamma = np.sum(dgammax*xhat, axis=0)
        dxhat = dgammax * self.gamma

        divar = np.sum(dxhat*xmu, axis=0)
        dxmu1 = dxhat * ivar

        dsqrtvar = -1. /(sqrtvar**2) * divar
        dvar = 0.5 * 1. /np.sqrt(var+self.eps) * dsqrtvar
        dsq = 1. /N * np.ones((N,D)) * dvar
        dxmu2 = 2 * xmu * dsq
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
        dx2 = 1. /N * np.ones((N,D)) * dmu
        dx = dx1 + dx2
        return dx

    
class Convolution:
    trainable = True
    # input_shape = [input_c, input_h, input_w], filter_shape = [filter_h,filter_w], filter_n=filter_n
    def __init__(self, input_shape, filter_shape, filter_n, stride=1, pad=0, initializer="He"):
        
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.filter_n = filter_n
        self.pad = pad
        self.stride = stride
        
        self.out_shape = [
            self.filter_n, self.input_shape[0],
            int((self.input_shape[1]+2*self.pad-self.filter_shape[0])/self.stride+1),
            int((self.input_shape[2]+2*self.pad-self.filter_shape[1])/self.stride+1)
            ]
        
        self.W = None
        self.b = None
        self.initialize_weight(initializer=initializer)

        self.x = None
        self.dW = None
        self.db= None
        
        # Momentum
        self.v_W = 0
        self.v_b = 0
        
        # AdaGrad
        self.h_W = 0
        self.h_b = 0
        
        # AdamGrad
        self.adam_v_W = 0
        self.adam_m_W = 0
        
        self.adam_v_b = 0
        self.adam_m_b = 0
        
        self.iter=0
    
    def initialize_weight(self, loc=0, scale=0.01, initializer="He"): 
        if initializer.lower() == "he":
            self.W = np.random.randn(self.filter_n,self.input_shape[0],self.filter_shape[0],self.filter_shape[1])/np.sqrt(self.input_shape[0]*np.prod(self.filter_shape)/2)
        elif initializer.lower() == "xavier":
            self.W = np.random.rand(self.filter_n,self.input_shape[0],self.filter_shape[0],self.filter_shape[1])/np.sqrt(self.input_shape[0]*np.prod(self.filter_shape)/1)
        else:
            self.W = np.random.normal(loc=loc, scale=scale, size=[self.filter_n,self.input_shape[0],self.filter_shape[0],self.filter_shape[1]])
        self.b = np.zeros([1,self.filter_n])
    
    def forward(self, x):
        filter_h=self.filter_shape[0]
        filter_w=self.filter_shape[1]
        x_c = im2col(x, filter_h, filter_w, stride=self.stride, pad=self.pad)
        W_c = self.W.reshape(self.filter_n,-1)
        out_c = np.dot(x_c, W_c.T)+self.b
        
        self.x = x
        out = out_c.reshape(x.shape[0], self.out_shape[2], self.out_shape[3],-1 ).transpose(0,3,1,2)
        self.x_c = x_c
        return out
    
    def backward(self, dout):
        FN, C, FH, FW = self.filter_n, self.input_shape[0], self.filter_shape[0], self.filter_shape[1]
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)
        self.db = np.sum(dout, axis=0)
        
        self.dW = np.dot(self.x_c.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        W_c = self.W.reshape(self.filter_n,-1)
        dcol = np.dot(dout, W_c)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
    
class Pooling:
    def __init__(self, filter_shape, stride=1, pad=0):
        self.filter_shape = filter_shape
        self.stride = stride
        self.pad = pad
        
        self.x_shape = None
        self.arg_max = None
    
    def forward(self, x):
        filter_h=self.filter_shape[0]
        filter_w=self.filter_shape[1]
        
        out_shape = [
            int(1+(x.shape[2]-filter_h)/self.stride),
            int(1+(x.shape[3]-filter_w)/self.stride),
            ]
        out_c = im2col(x, filter_h, filter_w, self.stride, self.pad)
        out_c = out_c.reshape(-1, filter_h*filter_w)
        arg_max = np.argmax(out_c, axis=1)
        out = np.max(out_c, axis=1)
        out = out.reshape(x.shape[0], out_shape[0], out_shape[1], x.shape[1]).transpose(0,3,1,2)
        
        self.x_shape = x.shape
        self.arg_max = arg_max
        return out
    
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        filter_h = self.filter_shape[0]
        filter_w = self.filter_shape[1]
        filter_size = filter_h * filter_w
        dmax = np.zeros((dout.size, filter_size))

        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (filter_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x_shape, filter_h, filter_w, self.stride, self.pad)
        
        return dx

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

class Flatten:
    def __init__(self):
        self.shape = None
    
    def forward(self, x):
        self.shape = x.shape
        out = x.reshape(x.shape[0], -1)
        return out
        
    def backward(self, dout):
        dx = dout.reshape(self.shape)
        return dx
    
class Network:
    layers = None
    loss_layer = None
    
    def __init__(self):
        self.layers=[]
        self.beta1 = 0.9
        self.beta2= 0.999
    
    def add(self, layer):
        self.layers.append(layer)
        self.n_layers = len(self.layers)
        
    def set_loss(self, layer):
        self.loss_layer = layer
    
    def predict(self, x, train_flg=False):
        out = self.layers[0].forward(x)
        for layer, i in zip(self.layers, range(self.n_layers)):
            if i==0: continue
            if isinstance(layer, Dropout):
                out = layer.forward(out, train_flg)
            else:
                out = layer.forward(out)
        return out
    
    def train(self, x, y, learning_rate=0.01, momentum_rate=0, adaGrad_rate = 0, RMSProp_rate=0, optimizer="adam"):
        return self.fit(x, y, learning_rate, momentum_rate, adaGrad_rate, RMSProp_rate, optimizer)
        
    def fit(self, x, y, learning_rate=0.01, momentum_rate=0, adaGrad_rate=0, RMSProp_rate=0, optimizer="adam"):
        out = self.predict(x, train_flg=True)
        loss = self.loss_layer.forward(out, y)

        dout = 1
        dout = self.loss_layer.backward(dout)
        for layer in self.layers[::-1]:
            dout = layer.backward(dout)
            if hasattr(layer, 'trainable'):
                if layer.trainable is True:
                    if isinstance(layer, Affine) or isinstance(layer, Convolution):
                        if optimizer=="adam":
                            layer.iter+=1
                            lr_t = learning_rate * np.sqrt(1.0-self.beta2**layer.iter)/(1.0-self.beta1**layer.iter)
                            layer.adam_m_W+=(1-self.beta1)*(layer.dW-layer.adam_m_W)
                            layer.adam_v_W+=(1-self.beta2)*(layer.dW**2-layer.adam_v_W)
                            layer.W -= lr_t*layer.adam_m_W/(np.sqrt(layer.adam_v_W)+1e-7)

                            layer.adam_m_b+=(1-self.beta1)*(layer.db-layer.adam_m_b)
                            layer.adam_v_b+=(1-self.beta2)*(layer.db**2-layer.adam_v_b)
                            layer.b -= lr_t*layer.adam_m_b/(np.sqrt(layer.adam_v_b)+1e-7)
                            
                        elif optimizer=="alex":
                            layer.v_W = 0.9*layer.v_W - 0.0005*learning_rate*layer.W - learning_rate*layer.dW
                            layer.W += layer.v_W
                            
                            layer.v_b = 0.9*layer.v_b - 0.0005*learning_rate*layer.b - learning_rate*layer.db
                            layer.b += layer.v_b
                            
                        elif optimizer=='none':
                            if RANDOM_LEARNING is True:
                                dW_r = np.abs(np.random.normal(loc= 1.0, scale = 0.1,
                                                             size = layer.dW.shape))
                                db_r = np.abs(np.random.normal(loc= 1.0, scale = 1,
                                                             size = layer.db.shape))
                                layer.W -= learning_rate*np.multiply(layer.dW, dW_r)
                                layer.b -= learning_rate*np.multiply(layer.db, db_r)
                            else:
                                layer.W -= learning_rate*layer.dW, dW_r
                                layer.b -= learning_rate*layer.db, db_r
                            
                        else :
                            layer.v_W = momentum_rate*layer.v_W - learning_rate*layer.dW
                            layer.h_W = RMSProp_rate*layer.h_W+(1-RMSProp_rate)*np.power(layer.dW, 2)
                            layer.W += layer.v_W - learning_rate/(np.sqrt(layer.h_W)+1e-7)*layer.dW

                            layer.v_b = momentum_rate*layer.v_b - learning_rate*layer.db
                            layer.h_b = RMSProp_rate*layer.h_b+(1-RMSProp_rate)*np.power(layer.db, 2)
                            layer.b += layer.v_b - learning_rate/(np.sqrt(layer.h_b)+1e-7)*layer.db
                        

                        
                    elif isinstance(layer, Batch_normalization):
                        layer.gamma += learning_rate*layer.dgamma
                        layer.beta += learning_rate*layer.dbeta
                    
        return loss
    
    def accuracy(self, x, t, top=1):
        y = self.predict(x)
        t_=np.argmax(t, axis=1)
        y_=np.argsort(y, axis=1)
        accuracy = 0
        for i in range(top):
            accuracy += np.sum(t_==y_[:,-i-1])
        accuracy /= float(x.shape[0])
        return accuracy
    
    def initialize_weight(self, initializer='Xavier'):
        for layer in self.layers:
            if hasattr(layer, 'trainable'):
                if initializer.lower() == 'xavier':
                    loc=0
                    scale=1/np.sqrt(layer.n_input)
                elif initializer.lower() == 'he':
                    loc=0
                    scale=2/np.sqrt(layer.n_input)
                layer.initialize_weight(loc=loc, scale=scale)

    def set_save_mode(self):
        for layer in self.layers:
            if hasattr(layer, 'x'):
                layer.x = None
            if hasattr(layer, 'dW'):
                layer.dW = None
            if hasattr(layer, 'db'):
                db = None
            if hasattr(layer, 'adam_v_W'):
                adam_v_W = 0
            if hasattr(layer, 'adam_v_b'):
                adam_v_b = 0
            if hasattr(layer, 'iter'):
                iter = 0