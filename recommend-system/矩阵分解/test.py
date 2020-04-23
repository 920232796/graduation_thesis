import numpy as np 


if __name__ == "__main__":
    a1 = np.array([[1, 2, 3], [4, 5, 6]])
    print(a1)

    print(a1.sum(axis=1, keepdims=True).shape)

    a2 = np.array([[1, 2, 3], [4, 5, 6]])
    print(a2[:, 1])