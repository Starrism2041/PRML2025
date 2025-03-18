import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrix  # 用于生成B-样条基函数

# # 读取数据
# df = pd.read_csv("D:\code\py\PRML\homework1\TrainingData.csv")
# x = df["x"]
# y = df["y_complex"].values


#读取数据(手动更改你的数据路径)
file_path1 = 'D:\code\py\PRML\homework1\TrainingData.csv'
file_path2 = 'D:\code\py\PRML\homework1\TestData.csv'
train_data = pd.read_csv(file_path1)
test_data = pd.read_csv(file_path2)
x_train = train_data.iloc[:, 0].values  
y_train = train_data.iloc[:, 1].values  
x_test = test_data.iloc[:, 0].values   
y_test = test_data.iloc[:, 1].values   

# 设置节点
knots = [2, 3, 4, 5, 6, 8]

# 生成B-样条基函数（自然样条边界条件）
formula = "bs(x, knots=knots, degree=3, include_intercept=True)"
B = dmatrix(formula, {"x": x_train, "knots": knots}, return_type="dataframe")

# 最小二乘法求解系数
beta = np.linalg.lstsq(B, y_train, rcond=None)[0]
print(f"beta:{beta}")

# 预测新数据
# x_train_new = np.linspace(0, 10, 500)
B_train_new = dmatrix(formula, {"x": x_train, "knots": knots}, return_type="dataframe")
y_train_pred = np.dot(B_train_new, beta)
loss_train = 1/(2*len(y_train))*np.sum((y_train_pred-y_train)**2)

# 预测测试集新数据
# x_test_new = np.linspace(0, 10, 500)
B_test_new = dmatrix(formula, {"x": x_test, "knots": knots}, return_type="dataframe")
y_test_pred = np.dot(B_test_new, beta)
loss_test = 1/(2*len(y_test))*np.sum((y_test_pred-y_test)**2)
print(f"TrainingDataLoss:{loss_train}\nTestDataLoss:{loss_test}")

plt.figure(figsize=(12, 4))

# plot1训练数据和拟合直线
plt.subplot(1, 2, 1)
plt.scatter(x_train, y_train, color='blue', label='Training Data')
plt.plot(x_train, y_train_pred, color="red", label='Cubic Spline Fit')
plt.title('Cubic Spline Regression via OLS')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.annotate(f'Test loss: {loss_train:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, color='red', ha='right', va='top')

# plot2测试数据和拟合直线
plt.subplot(1, 2, 2)
plt.scatter(x_test, y_test, color='blue', label='Test Data')
plt.plot(x_test, y_test_pred, color='red', label='Predicted Line')
plt.title('Test Data and Predicted Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.annotate(f'Test loss: {loss_test:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, color='red', ha='right', va='top')

plt.savefig('D:\code\py\PRML\homework1\csr.png')
plt.show()