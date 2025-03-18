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

# 添加截距项
x_train = np.vstack((np.ones(len(x_train)), x_train)).T
x_test = np.vstack((np.ones(len(x_test)), x_test)).T

def ols(x, y):
    XTX_inv = np.linalg.inv(x.T @ x)
    XTY = x.T @ y
    theta = XTX_inv @ XTY
    return theta

theta = ols(x_train, y_train)
theta0 = theta[0]
theta1 = theta[1]

print(f"Theta0: {theta0}, Theta1: {theta1}")
y_train_pred = x_train@ theta
loss_train = 1/(2*len(y_train))* np.sum((y_train_pred - y_train)**2)
print("train_loss:",loss_train)

y_test_pred = x_test@ theta
loss_test = 1/(2*len(y_test))* np.sum((y_test_pred-y_test)**2)
print("test_loss:",loss_test)

# 绘制结果
plt.figure(figsize=(12, 4))

# plot1 训练数据和拟合直线
plt.subplot(1, 2, 1)
plt.scatter(x_train[:, 1], y_train, color='blue', label='Data points')
plt.plot(x_train[:, 1], y_train_pred, color='red', label='Fitted line')
plt.title('Linear Regression via OLS')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.annotate(f'Train loss: {loss_train:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, color='red', ha='right', va='top')

# plot2 测试数据和拟合直线
plt.subplot(1, 2, 2)
plt.scatter(x_test[:, 1], y_test, color='blue', label='Test Data')
plt.plot(x_test[:, 1], y_test_pred, color='red', label='Predicted Line')
plt.title('Test Data and Predicted Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.annotate(f'Test loss: {loss_test:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, color='red', ha='right', va='top')

plt.tight_layout()
plt.savefig('D:\code\py\PRML\homework1\ols.png')
plt.show()

