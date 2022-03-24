import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

'''以下是读取数据并预处理步骤'''
#载入数据
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#预览图片例子
index =5 
# plt.imshow(train_set_x_orig[index])
# print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

#训练集数量
m_train = train_set_x_orig.shape[0]
#测试集数量
m_test = test_set_x_orig.shape[0]
#单个样本像素值
num_px = train_set_x_orig.shape[1]
#输出以上内容
# print ("Number of training examples: m_train = " + str(m_train))
# print ("Number of testing examples: m_test = " + str(m_test))
# print ("Height/Width of each image: num_px = " + str(num_px))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_set_x shape: " + str(train_set_x_orig.shape))
# print ("train_set_y shape: " + str(train_set_y.shape))
# print ("test_set_x shape: " + str(test_set_x_orig.shape))
# print ("test_set_y shape: " + str(test_set_y.shape))

#reshape 训练集和测试集样本
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T 
#将训练集，重新排成m行，m为样本数,之后转置
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T 
#将训练集，重新排成m行，m为样本数,之后转置
#输出以上信息
# print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
# print ("train_set_y shape: " + str(train_set_y.shape))
# print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
# print ("test_set_y shape: " + str(test_set_y.shape))
# print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

#标准化数据集 归一化
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

'''以下是算法部分'''

#定义sigmoid函数
def sigmoid(x):
    '''用于计算基于Numpy的sigmoid函数'''
    ans = 1/(1+np.exp(-x))
    return ans

# print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))

#参数初始化
def initialize_with_zeros(dim):
    '''
    对W的参数进行初始化
    dim--W的规模
    返回w，b表示初始化后的结果
    '''
    w = np.zeros((dim,1))
    b =0
    return w,b

#前向传播和反向传播
def propagate(w, b, X, Y):
    '''
    w-权重矩阵  b-偏置  X-数据  Y-标签
    
    返回 cost-损失函数  dw--w的偏导数  db-b的偏导数
    
    '''
    m = X.shape[1] #样本数
    
    ##前向传播##
    A = sigmoid(np.dot(w.T, X)+b)
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))   
    
    ##反向传播##
    dw = 1/m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)
    
    ##使用assert做判断，保证没有出错
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
grads, cost = propagate(w, b, X, Y)
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))
# print ("cost = " + str(cost))


#梯度下降更新参数
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    
    costs = []
    
    for i in range(num_iterations):
        #计算成本和梯度
        grads, cost = propagate(w,b,X,Y)
        
        #提取dw 和 db
        dw = grads["dw"]
        db = grads["db"]
        
        #更新参数值
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        #每迭代100次，记录cost的值
        if i % 100 == 0:
            costs.append(cost)
        
        #每迭代100次，打印一次
        if print_cost and i % 100 ==0 :
            print ("Cost after iteration %i: %f" %(i, cost))
        
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

# print ("w = " + str(params["w"]))
# print ("b = " + str(params["b"]))
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))
# print(costs)

#预测
def predict(w,b,X):
    '''
    使用逻辑回归预测标签是0还是1
    返回预测值（矩阵)
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1,m)) #预测值，共有m个
    w = w.reshape(X.shape[0],1) #将w矩阵重新排列
    
    A = sigmoid(np.dot(w.T, X)+b)
    
    for i in range(A.shape[1]):
        if A[0,i] <= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

# print ("predictions = " + str(predict(w, b, X)))

'''以下是训练部分'''

#开始训练模型

def model (X_train, Y_train, X_test, Y_test, num_iterations = 2000, \
    learning_rate = 0.5, print_cost = False):
    
    #初始化参数
    w, b = initialize_with_zeros(X_train.shape[0])
    
    #梯度下降
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations,\
        learning_rate, print_cost)
    
    #提取w和b
    w = parameters["w"]
    b = parameters["b"]
    
    #预测值
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    
    #打印训练集和测试集的准确率
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, \
    learning_rate = 0.001, print_cost = True)

#被错误分类的例子
# index = 1
# plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))

#打印损失函数曲线
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()