import torch
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

img_size = (64, 64)

def load_data():
    df = pd.read_csv("archive/age_detection.csv")
    train = df[df["split"] == "train"].sample(frac=1)

    X = []
    X_file = train["file"]
    for img_path in X_file:
        with Image.open("archive/" + img_path) as img:
            img = ImageOps.exif_transpose(img)
            img.convert("RGB")
            img = img.resize(img_size)
            vec = np.asarray(img, dtype=np.uint8).flatten()
            X.append(vec)
            
    y = train["age"].to_numpy()

    return X, y

def display_picture(vec, label = ""):
    plt.imshow(vec.reshape(img_size[0], img_size[1], 3))
    plt.title(label)
    plt.show()

class NN_Regression(torch.nn.Module):

    def __init__(self):
        super(NN_Regression, self).__init__()
        self.l1 = torch.nn.Linear(64 ** 2, 200)
        self.l2 = torch.nn.Linear(200, 5)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x

if __name__ == "__main__":
    X_train, y_train = load_data()
    # for i in range(50, 53):
    #     display_picture(X_train[i], label=y_train[i])
    Model = NN_Regression()