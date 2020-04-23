## 生成一些模拟数据
import numpy as np 
import pandas as pd 
from typing import List, Dict, Tuple

def cons_data():
    """
    构建原始数据，手动加一些"-"符号，表示没有进行打分的题目 
    """
    a = np.ceil(np.random.rand(50) * 5)
    b = np.ceil(np.random.rand(50) * 5)
    c = np.ceil(np.random.rand(50) * 5)
    d = np.ceil(np.random.rand(50) * 5)
    e = np.ceil(np.random.rand(50) * 5)
    for i in range(20):
        ii = np.random.randint(0, 50)
        a[ii] = 0
    for i in range(20):
        ii = np.random.randint(0, 50)
        b[ii] = 0
    for i in range(20):
        ii = np.random.randint(0, 50)
        c[ii] = 0
    for i in range(20):
        ii = np.random.randint(0, 50)
        d[ii] = 0
    for i in range(20):
        ii = np.random.randint(0, 50)
        e[ii] = 0
    print(a)
    data = {"线性表中结点的前驱和后继": a, "树的带权路径长度": b, "循环队列中的元素个数": c,
            "单链表插入": d, "出栈序列": e
    }

    df = pd.DataFrame(data)
    df.to_csv("./data.csv", encoding="utf-8", index=False)

def get_matrix(df: pd.DataFrame):

    matrix = np.zeros((50, 5))
    for index, row in df.iterrows():
        for i in range(5):
            matrix[index, i] = row[i]
    return matrix

def gradient_descent(data_matrix, K):
    if data_matrix is None:
        print("data is None, please load data file ...")
        return 
    m, n = data_matrix.shape
    epoches = 10000
    lr = 0.002
    beta = 0.02# 如果不用正则 则不需要这行代码
    p = np.random.random((m, K))
    q = np.random.random((K, n))
    zero_list = [] ## 获取非零下标
    for i in range(m):
            for j in range(n):
                if data_matrix[i, j] == 0:
                    zero_list.append((i, j))
    for epoch in range(epoches):
        temp_matrix = p.dot(q)
        loss_matrix = (data_matrix - temp_matrix)  ## 损失矩阵
        for (i, j) in zero_list:
            loss_matrix[i, j] = 0.0 ## 排除掉 '-' 缺失值对loss的影响
        ## 根据损失求p， q 的梯度
        dp = -2 * loss_matrix.dot(q.T) / n
        dq = -2 * p.T.dot(loss_matrix) / m
        ## 更新参数
        p = p - lr * dp 
        q = q - lr * dq 
        loss = np.square(loss_matrix).sum() / (m*n)## 求解损失值            
        if loss < 0.001:
            print(loss)
            break 
        if epoch % 100 == 0:
            print(loss)

    return zero_list, p.dot(q)

def recommend(stu_id: int, matrix: "np.array", zero_list: List[Tuple[int]]):
    ## 输入补全的matrix矩阵，输出推荐结果
    items = list(matrix[stu_id])
    sort_result = []
    for index, item in enumerate(items):
        if (stu_id, index) in zero_list:
            ## 说明这个商品没有评分过,把商品的index和潜在评分加到list 要排序待会
            sort_result.append((index, item))
    
    return sorted(sort_result, key=lambda item: item[1], reverse=True)

def filter_recommend(matrix: "np.array", K):
    """
    K 是选取K个近邻
    使用协同过滤算法，计算相似度，计算潜在评分，进行推荐
    """
    item_num = matrix.shape[1] ## 得到一共几个商品
    user_num = matrix.shape[0] # 一共几个用户
    resemble_matrix = np.zeros((item_num, item_num))
    for i in range(item_num):
        for j in range(item_num):
            temp_i_matrix = np.copy(matrix[:, i])
            temp_j_matrix = np.copy(matrix[:, j])
            ## 单独拿出来两列，过滤有0的行
            for index, each in enumerate(temp_i_matrix):
                if each == 0 or temp_j_matrix[index] == 0 :
                    temp_i_matrix[index] = 0
                    temp_j_matrix[index] = 0
            
            # resemble_matrix[i, j] = np.sum((temp_i_matrix - temp_j_matrix)**2)
            resemble_matrix[i, j] = 1 / (1 + np.sqrt(np.sum((temp_i_matrix - temp_j_matrix)**2)))
    # print(resemble_matrix) ## 得到相似度矩阵了
    ## 下面进行推荐
    # student = matrix[stu_id]
    # recommend_list = []
    # for index, score in enumerate(student):
    #     if score == 0:
    #         # 则需要进行推荐
    #         result = np.sum(resemble_matrix[index] * student) ## 得到加权值
    #         recommend_list.append((index, result))
    # print(recommend_list)
    for stu_id in range(user_num):
        student = matrix[stu_id]
        row_res = np.zeros(item_num)
        for index, score in enumerate(student):
            if score == 0:
                # print(np.sort(resemble_matrix[index] * student))
                result = np.sum(np.sort(resemble_matrix[index] * student)[-K: ] )
                # print(np.sort(resemble_matrix[index] * student)[-K: ])
                row_res[index] = result
                # matrix[stu_id, index] = result
            else :
                row_res[index] = matrix[stu_id, index]
        matrix[stu_id] = row_res    
    # print(resemble_matrix)
    return matrix


def filter_recommend_user(matrix: "np.array", K):
    """
    基于user的协同过滤算法
    K 是选取K个近邻
    使用协同过滤算法，计算相似度，计算潜在评分，进行推荐
    """
    item_num = matrix.shape[1] ## 得到一共几个商品
    user_num = matrix.shape[0] # 一共几个用户
    resemble_matrix = np.zeros((user_num, user_num))
    for i in range(user_num):
        for j in range(user_num):
            # 拿出两行
            temp_i_matrix = np.copy(matrix[i, :])
            temp_j_matrix = np.copy(matrix[j, :])
            # ## 单独拿出来两行，过滤有0的列
            # for index, each in enumerate(temp_i_matrix):
            #     if each == 0 or temp_j_matrix[index] == 0 :
            #         temp_i_matrix[index] = 0
            #         temp_j_matrix[index] = 0
            
            resemble_matrix[i, j] = np.sum(temp_i_matrix * temp_j_matrix) / (np.sqrt(np.sum(temp_i_matrix**2)) * np.sqrt(np.sum(temp_j_matrix**2)))
            if np.isnan(resemble_matrix[i, j]):
                resemble_matrix[i, j] = 0
            # print(resemble_matrix[i, j])

    ## 筛选k近邻
    for index, row in enumerate(resemble_matrix):
        theta = np.sort(row)[-K]
        resemble_matrix[index][row < theta] = 0
    res_matrix = np.dot(resemble_matrix, matrix)
    print(resemble_matrix)
    with open("./resemble_matrix.txt", "w") as f :
        for line in resemble_matrix:
            f.write(str(line.tolist()))
            f.write("\n")
    return res_matrix
            

def main():
    
    # df = pd.read_csv("./data.csv")
    # matrix = get_matrix(df)
    # # print(matrix)
    # ## zero_list就是判断这个位置是否打过分了 如果打过分，肯定就不再推荐了
    # matrix = np.array([[1, 2, 3], [4, 0, 6], [3, 1, 0]])
    # zero_list, result_matrix = gradient_descent(matrix, 3)
    # # 给定一个学生的序号，直接循环所有商品
    # stu_id = 1
    # print(result_matrix)
    # recommend_list = recommend(stu_id, result_matrix, zero_list)

    # print(recommend_list)


    print("~~~~~ testing filter recommend ~~~~~~~~~")
    matrix = np.array([[1, 0, 0, 0, 0], [1, 0, 1, 0, 1], [1, 1, 0, 1, 0]])
    res = filter_recommend_user(matrix, K=3)
    print(res)

    
    ## 一些测试函数
    # n1 = np.array([1, 2, 3, 2, 0, 5])
    # print(np.sort(n1, 2))




if __name__ == "__main__":

    # cons_data()
    main()
    