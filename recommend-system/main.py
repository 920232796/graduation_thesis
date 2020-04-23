## 使用公共数据集，943用户 1682个电影评分数据，共10万条
import pandas as pd 
import numpy as np
from data import gradient_descent, filter_recommend_user
import matplotlib.pyplot as plt 
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

def compute_gini(matrix, max_item_id):
    """
    计算基尼系数, 传入user-item 矩阵
    """
    hot_value = np.sum(matrix, axis=0)
    hot_magnitude = hot_value / np.max(hot_value) # 归一化，得到流行程度
    # print(hot_magnitude)

    hot_list = []
    for index, item in enumerate(hot_magnitude):
        hot_list.append((index, item)) # 方便排序
    # 排序
    hot_list = sorted(hot_list, key=lambda x: x[1])
    print(hot_list)

    plot_array = np.zeros(max_item_id)
    for index, item in enumerate(hot_list):
        plot_array[index] = item[1]
    print("plot_array: " + str(plot_array))
    gini = 1 - (2 * np.sum(plot_array) + 1) / max_item_id

    ## index 是排好序的 商品，按照流行度从小到大排序， plot array的值是对应流行度
    return plot_array, gini

if __name__ == "__main__":
    
    # with open("./u.data", "r") as f:
    #     lines = f.readlines()
    
    # data = [line.split() for line in lines]
    # # 将数据根据user id 进行排个序
    # data = sorted(data, key=lambda x: int(x[0]))
    # # print(data[:10])

    # ## 取1000条进行测试
    # clear_data = data[:3000]
    # for index, each in enumerate(clear_data):
    #     clear_data[index] = [int(each[0]) - 1, each[1], each[2]]
    # print(clear_data[: 10])

    with open("self_data.txt") as f:
        lines = f.readlines()
        lines = [line.strip("\n").split(" ") for line in lines]
    matrix = np.array(lines, dtype=np.float)
    # print(matrix)
    # max_movie_id = max([int(d[1]) for d in clear_data]) # 求最大的电影id 构建矩阵
    # max_user_id = max([int(d[0]) for d in clear_data]) # 求最大用户id 构建矩阵
    max_movie_id = len(matrix[0])
    max_user_id = len(matrix)
    print(f"max_user_id is {max_user_id}")
    print(f"max_movie_id is {max_movie_id}")

    # 主要关注zero_list 就ok
    zero_list, _ = gradient_descent(matrix, K=8)
    # print(res)

    res2 = filter_recommend_user(matrix, K=4)
    # print(res2)
    with open("./res.txt", "w") as f :
        for line in res2:
            f.write(str(line.tolist()))
            f.write("\n")
    
    ## 求覆盖率
    ### 先计算推荐前的基尼系数
    plot_array, pre_gini = compute_gini(matrix, max_movie_id)
    plt.xlabel("题目")
    plt.ylabel("流行度")
    plt.plot(plot_array)
    plt.show()
    print(pre_gini)

    ### 计算推荐后的基尼系数
    hot_user_list = []
    for user in res2:
        tmp_list = []
        for index, score in enumerate(user):
            tmp_list.append((index, score)) ## index 表示商品的id
        hot_user_list.append(tmp_list)
    
    # 接下来需要针对每个用户，推荐前五个商品，作为这个用户的推荐列表
    for index, each_user_score in enumerate(hot_user_list):
        hot_user_list[index] = sorted(each_user_score, key=lambda x: x[1], reverse=True)
    
    # 求用户的推荐列表
    user_recommend_list = []
    for index, user_score_sorted in enumerate(hot_user_list):
        count = 0
        for score in user_score_sorted:
            if (index, score[0]) in zero_list:
                # score[0] 表示item id  如果没在zero list 表示这个是系统推荐的商品
                # 
                user_recommend_list.append((index, score[0]))
                count += 1
                if count == 3:
                    # 已经构造完成了 本用户的推荐列表
                    break
    # 根据构造的用户推荐列表，往原始的matrix里面补 1 表示这道题目是系统新推荐的～
    #
    # pre_matrix = []
    # for index, row in enumerate(matrix):
    #     for index_c, score in enumerate(row):
    #         if score != 0:
    #             pre_matrix.append((index, index_c))
    # print(pre_matrix[:30])


    # print(user_recommend_list)
    for each_recommend in user_recommend_list:
        matrix[each_recommend[0], each_recommend[1]] = 1
    
    # 计算推荐后的gini系数
    new_plot_array, new_gini = compute_gini(matrix, max_movie_id)
    plt.plot(new_plot_array)
    plt.xlabel("题目")
    plt.ylabel("流行度")
    plt.show()
    print(new_gini)

    


    



