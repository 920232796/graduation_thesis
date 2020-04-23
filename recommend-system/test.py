import numpy as np 


if __name__ == "__main__":
    a = np.arange(20).reshape(4, 5)
    il2 = np.tril_indices(4, -1, 5)
    print(a)
    print(a[il2])

    print(np.hypot(0, 3))