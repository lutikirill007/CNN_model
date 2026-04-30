import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data 
from tqdm import tqdm
import os
import json
from PIL import Image
import torchvision.transforms.v2 as tfs 



class CNN_datase(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join('CNN_model',path, 'train' if train else 'test')
        self.transform = transform

        with open(os.path.join(self.path, 'format.json'), 'r') as fp:
            self.format = json.load(fp)

        self.lenght = len(self.format)
        self.files = tuple(self.format.keys())
        self.target = tuple(self.format.values())

    def __getitem__(self, index):
        path_file = os.path.join(self.path, self.files[index])
        img = Image.open(path_file).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(self.target[index], dtype=torch.float32)
    

    def __len__(self):
        return self.lenght
    
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 8, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(8, 4, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(4096, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
)

transform = tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)])

d_train = CNN_datase('dataset_reg', transform=transform)

train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)
loss_func = nn.MSELoss()

epochs = 5
model.train()


for _e in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        loss = loss_func(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1/lm_count * loss.item() + (1-1/lm_count)*loss_mean
        train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.3f}")


st = model.state_dict()
torch.save(st, 'CNN_model/model_sun.tar')


d_test = CNN_datase('dataset_reg',train=False, transform=transform)
test_data = data.DataLoader(d_test, batch_size=50, shuffle=False)

Q = 0
count = 0
model.eval()


test_tqdm = tqdm(test_data, leave=True)
for x_test, y_test in test_tqdm:
    with torch.no_grad():
        p = model(x_test)
        Q += loss_func(p, y_test)
        count += 1

Q = Q/count
print(Q)