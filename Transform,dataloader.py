from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = FashionITIS("/content/fashion-mnist_train.csv",transform= transform)
test_dataset = FashionITIS("/content/fashion-mnist_test.csv",transform= transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle= True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle= True)
