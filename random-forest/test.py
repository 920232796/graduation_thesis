
import torch 


if __name__ == "__main__":
    # t1 = torch.tensor([[1, 2, 3], [4, 5, 16], [7, 8, 9]])
    # labels = torch.tensor([[0, 1, 2]])
    # paths = labels
    # paths = torch.cat((paths[:, torch.tensor([2, 2, 2])], labels), dim=0)
    # print(paths)
    # print(t1.max(1))

    t2 = torch.tensor([[2, 3, 1], [0, 5, 2]])
    # ids = torch.tensor([2, 1, 2])
    # print(t2[:, ids])
    argmax_ids = torch.tensor([1])
    print(t2[:, argmax_ids])