import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#读取数据(手动更改你的数据路径)
file_path1 = 'D:\code\py\PRML\homework1\TrainingData.csv'
file_path2 = 'D:\code\py\PRML\homework1\TestData.csv'
train_data = pd.read_csv(file_path1)
test_data = pd.read_csv(file_path2)
x_train = train_data.iloc[:, 0].values  
y_train = train_data.iloc[:, 1].values  
x_test = test_data.iloc[:, 0].values   
y_test = test_data.iloc[:, 1].values   


def gradient_descent(x, y, lr, epoch = 100):
    """
    梯度下降法求解线性回归参数
    """
    # 初始化参数
    n=len(y)
    theta0, theta1 = 0,0
    loss_history = []
    
    # 迭代更新参数
    for _ in range(epoch):
        # 计算梯度
        y_pred = theta0 + theta1 * x
        loss = 1/(2*n) * np.sum((y_pred - y)**2)
        loss_history.append(loss)
        dtheta0 = 1/n * np.sum(y_pred - y)
        dtheta1 = 1/n * np.sum((y_pred - y) * x)
        # 更新参数
        theta0 = theta0 - lr* dtheta0
        theta1 = theta1 - lr* dtheta1

    return theta0, theta1, loss_history

theta0, theta1, loss_history = gradient_descent(x_train, y_train, lr = 0.005, epoch = 2000)
print(f"Theta0: {theta0}, Theta1: {theta1}")

x_line = np.linspace(0, 10, 200)
y_line = theta0 + theta1* x_line

# 使用测试集计算预测值
y_test_pred = theta0 + theta1 * x_test
loss_test = 1/(2*len(y_test)) * np.sum((y_test_pred - y_test)**2)
print('Test loss:', loss_test)

# 使用训练集计算最终的损失
y_train_pred = theta0 + theta1 * x_train
loss_train = 1/(2*len(y_train)) * np.sum((y_train_pred - y_train)**2)
print('Train loss:', loss_train)

plt.figure(figsize=(16, 6))

# plot1训练数据和拟合直线
plt.subplot(1, 3, 1)
plt.scatter(x_train, y_train, color='blue', label='Data points')
plt.plot(x_line, y_line, color='red', label='Fitted line')
plt.title('Linear Regression via Gradient Descent')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.annotate(f'Train loss: {loss_train:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, color='red', ha='right', va='top')

# plot2测试数据和拟合直线
plt.subplot(1, 3, 2)
plt.scatter(x_test, y_test, color='blue', label='Test Data')
plt.plot(x_test, y_test_pred, color='red', label='Predicted Line')
plt.title('Test Data and Predicted Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.annotate(f'Test loss: {loss_test:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, color='red', ha='right', va='top')

# plot3损失函数迭代
plt.subplot(1, 3, 3)
plt.plot(loss_history, color='green')
plt.title('Loss Function Minimization')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')

plt.tight_layout()
plt.savefig('D:\code\py\PRML\homework1\gd.png')
plt.show()
        
