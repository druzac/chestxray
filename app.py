from flask import Flask, render_template, request

import torchvision

from PIL import Image
import train
import torch

MEAN = [0.485, 0.456, 0.406]
STD_DEV = [0.229, 0.224, 0.225]
PIC_SIZE = (224, 224)

app = Flask(__name__)
class_names = ['normal', 'viral', 'covid']

# dumb stuff to load a single image.
# i had to do this because images are (3, 224, 224) tensors, but the model wants
# (x, 3, 224, 224) tensors. it'd probably be simpler to just do some tensor manipulation
# to add an extra dimension.
class SingletonTestDataSet(torch.utils.data.Dataset):
    def __init__(self, image, transform):
        self.transform = transform
        self.image = image

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        return self.transform(self.image)


def build_test_dl(image):
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=PIC_SIZE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=MEAN, std=STD_DEV)
    ])
    dataset = SingletonTestDataSet(image, test_transform)
    dl_test = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    return dl_test


def do_file_stuff():
    if 'image_file' not in request.files:
        print("argh, no file!")
        return "oops, we messed up"
    # flash('No file part')
    f = request.files['image_file']
    print("file object: ", f)
    print("type of file object: ", type(f))
    print("dir of file object: ", dir(f))
    # if user does not select file, browser also
    # submit an empty part without filename
    if f.filename == '':
        print("argh, no selected file!")
        return "oops, you messed up: no selected file"
    im = Image.open(f.stream).convert('RGB')
    model = train.initialize_model()
    train.load_model(model, "model_parameters/r18_lo3.pth")
    model.eval()
    dl = build_test_dl(im)
    first_image = next(iter(dl))
    print(f"type of first_image: {type(first_image)}, size: {first_image.size()}")
    output = model(first_image)

    print(f"output looks like: {output}, type: {type(output)}, first element looks like: {output[0]}")
    _, preds = torch.max(output[0], 0)
    print(f"preds looks like: {preds}, type: {type(preds)}")
    print(f"i think it is: {class_names[preds]}")
    return f"i think it is: {class_names[preds]}"

@app.route('/image', methods=['POST'])
def image():
    return do_file_stuff()

@app.route('/', methods=('GET', 'POST'))
def main():
    return render_template("index.html")
