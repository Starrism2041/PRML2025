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

def newtown(x, y, lr, epoch = 100, reg_lambda = 1e-3):
    """
    牛顿法求解线性回归参数
    """
    n = len(y)
    theta0, theta1 = np.random.randn(2) * 0.01
    loss_history = []

    for _ in range(epoch):
        y_pred = theta0 + theta1* x
        loss = 1/ (2*n)* np.sum((y_pred - y)**2)
        loss_history.append(loss)

        theta0_grad = np.mean(y_pred - y)
        theta1_grad = np.mean((y_pred - y)*x)
        hessian00 = 1
        hessian01 = np.mean(x)
        hessian10 = np.mean(x)
        hessian11 = np.mean(x**2) + reg_lambda
        hessian = np.array([[hessian00, hessian01],[hessian10, hessian11]])
        gradient = np.array([theta0_grad, theta1_grad])

        theta_update = np.linalg.inv(hessian).dot(gradient)
        theta0 = theta0 - lr* theta_update[0]
        theta1 = theta1 - lr* theta_update[1]

    return theta0, theta1, loss_history
    
#训练模型
theta0, theta1, loss_history = newtown(x_train, y_train, lr=0.01, epoch = 1000)

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

#plot1训练数据和拟合直线
plt.subplot(1, 3, 1)
plt.scatter(x_train, y_train, color='blue', label='Data points')
plt.plot(x_line, y_line, color='red', label='Fitted line')
plt.title('Linear Regression via Newton Method')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.annotate(f'Train loss: {loss_train:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, color='red', ha='right', va='top')

#plot2测试数据和拟合直线
plt.subplot(1, 3, 2)
plt.scatter(x_test, y_test, color='blue', label='Test Data')
plt.plot(x_test, y_test_pred, color='red', label='Predicted Line')
plt.title('Test Data and Predicted Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.annotate(f'Test loss: {loss_test:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, color='red', ha='right', va='top')

#plot3损失函数迭代
plt.subplot(1, 3, 3)
plt.plot(loss_history, color='green')
plt.title('Loss Function Minimization')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')

plt.tight_layout()
plt.savefig('D:\code\py\PRML\homework1\\newtown.png')
plt.show()