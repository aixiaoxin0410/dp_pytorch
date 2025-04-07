import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

input = torch.tensor([1,2,3],dtype=torch.float32)
target = torch.tensor([1,2,5],dtype=torch.float32)

input = torch.reshape(input,(1,1,1,3))
target = torch.reshape(target,(1,1,1,3))

lose = L1Loss(reduction="sum")
result = lose(input,target)

lose_mse = MSELoss(reduction="sum")
result_mse = lose_mse(input,target)

# print(result)
# print(result_mse)

x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])

x = torch.reshape(x,(1,3))
lose_cross = CrossEntropyLoss()
result_cross = lose_cross(x,y)
print(result_cross)