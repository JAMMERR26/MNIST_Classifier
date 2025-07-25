import torch.optim as optim
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

epochs = 25
for epoch in range(epochs):
  running_loss= 0
  correct=0
  total=0
  for images, labels in train_loader:
     optimizer.zero_grad()
     outputs = model(images)
     loss = criterion(outputs,labels)

     loss.backward()
     optimizer.step()

     running_loss += loss.item()

     _,predicted = torch.max(outputs.data,1)
     correct += (predicted == labels).sum().item()
     total += labels.size(0)

  epoch_accuracy = 100 * correct/total

  print(f"Epoch{epoch+1}/{epochs},Loss: {running_loss/len(train_loader)}")
  print(f"Accuracy :{epoch_accuracy : 2f}%")
