import numpy as np
import os
import os.path as osp
import random
import shutil

from PIL import Image
from matplotlib import pyplot as plt
import torch
import torchvision

# torch.manual_seed(0)

# magic numbers
MEAN = [0.485, 0.456, 0.406]
STD_DEV = [0.229, 0.224, 0.225]
PIC_SIZE = (224, 224)
class_names = ['normal', 'viral', 'covid']


# don't think this is necessary, directory structure is already laid out.
def create_dirs():
    root_dir = 'COVID-19 Radiography Database'
    source_dirs = ['NORMAL', 'Viral Pneumonia', 'COVID-19']
    

    if osp.isdir(osp.join(root_dir, source_dirs[1])):
        os.mkdir(osp.join(root_dir, 'test'))

    for i, d in enumerate(source_dirs):
        os.rename(osp.join(root_dir, d), osp.join(root_dir, class_names[i]))

    for c in class_names:
        os.mkdir(osp.join(root_dir, 'test', c))

    for c in class_names:
        images = [x for x in os.listdir(osp.join(root_dir, c)) if x.lower().endswith('png')]
        selected_images = random.sample(images, 30)
        for image in selected_images:
            source_path = osp.join(root_dir, c, image)
            target_path = osp.join(root_dir, 'test', c, image)
            shutil.move(source_path, target_path)

class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x[-3:].lower().endswith('png')]
            print(f'Found {len(images)} {class_name} examples')
            return images
        
        self.images = {}
        self.class_names = ['normal', 'viral', 'covid']
        
        for class_name in self.class_names:
            self.images[class_name] = get_images(class_name)
            
        self.image_dirs = image_dirs
        self.transform = transform
        
    
    def __len__(self):
        return sum([len(self.images[class_name]) for class_name in self.class_names])
    
    
    def __getitem__(self, index):
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)


def show_images(images, labels, preds):
    plt.figure(figsize=(8, 4))
    for i, image in enumerate(images):
        plt.subplot(1, 6, i + 1, xticks=[], yticks=[])
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array(MEAN)
        std = np.array(STD_DEV)
        image = image * std + mean
        image = np.clip(image, 0., 1.)
        plt.imshow(image)
        col = 'green'
        if preds[i] != labels[i]:
            col = 'red'
            
        plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
        plt.ylabel(f'{class_names[int(preds[i].numpy())]}', color=col)
    plt.tight_layout()
    plt.show()

def show_preds(model, dl):
    model.eval()
    images, labels = next(iter(dl))
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    # show_images(images, labels, preds)

def train(epochs):
    # training data
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=PIC_SIZE),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=MEAN, std=STD_DEV)
    ])
    train_dirs = {
        'normal': 'COVID-19 Radiography Database/normal',
        'viral': 'COVID-19 Radiography Database/viral',
        'covid': 'COVID-19 Radiography Database/covid'
    }
    train_dataset = ChestXRayDataset(train_dirs, train_transform)
    class_names = train_dataset.class_names

    batch_size = 6
    dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # test data
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=PIC_SIZE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=MEAN, std=STD_DEV)
    ])

    test_dirs = {
        'normal': 'COVID-19 Radiography Database/test/normal',
        'viral': 'COVID-19 Radiography Database/test/viral',
        'covid': 'COVID-19 Radiography Database/test/covid'
    }
    test_dataset = ChestXRayDataset(test_dirs, test_transform)
    dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    resnet18 = torchvision.models.resnet18(pretrained=True)
    resnet18.fc = torch.nn.Linear(in_features=512, out_features=3)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet18.parameters(), lr=3e-5)

    print('Starting training..')
    for e in range(0, epochs):
        print('='*20)
        print(f'Starting epoch {e + 1}/{epochs}')
        print('='*20)

        train_loss = 0.
        val_loss = 0.

        resnet18.train() # set model to training phase

        for train_step, (images, labels) in enumerate(dl_train):
            optimizer.zero_grad()
            outputs = resnet18(images)
            loss = loss_fn(outputs, labels)
            # this step causes the "no cuda" message to be printed
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if train_step % 20 == 0:
                print('Evaluating at step', train_step)

                accuracy = 0

                resnet18.eval() # set model to eval phase

                for val_step, (images, labels) in enumerate(dl_test):
                    outputs = resnet18(images)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()

                    _, preds = torch.max(outputs, 1)
                    accuracy += sum((preds == labels).numpy())

                val_loss /= (val_step + 1)
                accuracy = accuracy/len(test_dataset)
                print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

                show_preds(resnet18, dl_test)

                resnet18.train()

                if accuracy >= 0.95:
                    print('Performance condition satisfied, stopping..')
                    return

        train_loss /= (train_step + 1)

        print(f'Training Loss: {train_loss:.4f}')


if __name__  == '__main__':
    print('Using PyTorch version', torch.__version__)
    train(epochs=1)
    print('Training complete.')
