import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件中的数据
data = np.loadtxt('heat_data.csv', delimiter=',')

# 获取行列数
rows, cols = data.shape

# 设置网格大小
dh = 1 / rows
x = np.arange(0, cols, 1) * dh + 0.5*dh
y = np.arange(0, rows, 1) * dh + 0.5*dh

# 创建网格
X, Y = np.meshgrid(x, y)

# 绘制热力图
plt.pcolormesh(X, Y, data, clim=[0, 1])

# 添加颜色条
plt.colorbar()

# 设置坐标轴标签
plt.xlabel('x')
plt.ylabel('y')

# 显示图像
plt.show()
