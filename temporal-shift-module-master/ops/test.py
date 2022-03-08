from utils import accuracy_sens_prec, accuracy
import torch

output=torch.tensor([[0.2,0.8],[0.3,0.7],[0.8,0.2],[0,1],[0.9,0.1],[0.1,0.9]])
target=torch.tensor([1,1,1,0,0,0])

print(accuracy(output,target,topk=(1,1)))

sens, prec , cm=accuracy_sens_prec(output.data, target)

print(sens)
print( prec)
print(cm)

# nb_classes = 2

# confusion_matrix = torch.zeros(nb_classes, nb_classes)
# print("*******" * 30)
# with torch.no_grad():
#     _, preds = torch.max(output,1)
#     print(preds)
#     for t, p in zip(target.view(-1), preds.view(-1)):
#         print(t.long())
#         print(p.long())
#         print("*******" * 15)
#         confusion_matrix[t.long(), p.long()] += 1

# print(confusion_matrix)

