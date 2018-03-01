import numpy as np
import matplotlib.pyplot as plt

def softmax(a):
    c = np.max(a, axis=1)
    c = np.expand_dims(c, 1)
    return np.exp(a-c)/ np.expand_dims(np.sum(np.exp(a-c), axis=1), 1)

def mean_squared_error(y, t):
    batch_size = y.shape[0]
    return np.sum((y-t)**2)/2/batch_size

def cross_entropy_error(y, t):
    batch_size = y.shape[0]
    delta = 1e-7
    return -(np.sum(t*np.log(y+delta))/batch_size)

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = int((H + 2*pad - filter_h)/stride + 1)
    out_w = int((W + 2*pad - filter_w)/stride + 1)

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = int((H + 2*pad - filter_h)/stride + 1)
    out_w = int((W + 2*pad - filter_w)/stride + 1)
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

def filter_show(filters, nx=8, margin=3, scale=10, color=True):
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))
    
    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    
    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
    
def one_hot_encoder(y, class_n):
    one_hot_targets = np.eye(class_n)[y]
    return one_hot_targets