
import numpy as np 
from typing import List, Dict
### 1 获取真实user-item数据，里面有些地方是空缺的。
### 2 带入到矩阵分解的函数里面。
### 3 梯度下降法 优化得到结果
# + (beta / 2) * (p.sum(axis=1)**2).reshape((-1, 1)) + (beta / 2) * q.sum(axis=0)**2
class Decomposition:
    """
    矩阵分解
    """
    def __init__(self):
        pass
        self.data_matrix = None 
    def load_data(self, path: str):
        """
        加载矩阵数据
        """
        data_list = []
        with open(path, "r") as f:
            for line in f.readlines():
                sub_list = []
                for num in line.split():
                    if num == "-":
                        sub_list.append(float(0))
                    else:
                        sub_list.append(float(num))
                data_list.append(sub_list)
        self.data_matrix = np.array(data_list)
    
    def gradient_descent(self, K):
        if self.data_matrix is None:
            print("data is None, please load data file ...")
            return 
        # print(self.data_matrix.sum())
        m, n = self.data_matrix.shape
        epoches = 1000
        lr = 0.0002
        beta = 0.02# 如果不用正则 则不需要这行代码
        p = np.random.random((m, K))
        q = np.random.random((K, n))
        none_zero_list = []
        zero_list = []
        for i in range(m):
                for j in range(n):
                    if self.data_matrix[i, j] != 0:
                        none_zero_list.append((i, j)) ##获取非零元素的下标
                    else :
                        zero_list.append((i, j))

        for epoch in range(epoches):
            temp_matrix = p.dot(q)
            loss_matrix = (self.data_matrix - temp_matrix)  ## 损失矩阵
            for (i, j) in zero_list:
                loss_matrix[i, j] = 0.0 ## 排除掉 '-' 缺失值对loss的影响
            ## 根据损失求p， q 的梯度
            for (i, j) in none_zero_list:
                pi = p[i, :]
                qj = q[:, j]
                dpi = (-2 * loss_matrix[i, j] * qj) ## 每次更新一行
                dqj = (-2 * loss_matrix[i, j] * pi)
                pi = pi - lr * dpi
                qj = qj - lr * dqj
                p[i, :] = pi
                q[:, j] = qj
                # p = p - lr * dp
                # q = q - lr * dq
            # dp = (-2 * loss_matrix.dot(q.T) + beta * p) / n
            # dq = (-2 * p.T.dot(loss_matrix) + beta * q) / m
            # p = p - lr * dp 
            # q = q - lr * dq 
            loss = loss_matrix.sum() / (m*n)## 求解损失值            
            if loss < 0.01:
                break 
            if epoch % 10 == 0:
                # print(loss_matrix)
                print(loss)
        return p.dot(q)


if __name__ == "__main__":
    pass
    decomposition = Decomposition()
    decomposition.load_data("./recommend-system/矩阵分解/data.txt")
    print(decomposition.gradient_descent(5))
    