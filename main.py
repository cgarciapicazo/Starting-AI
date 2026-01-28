import torch
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

img_size = (64, 64)

def load_data():
    df = pd.read_csv("archive/age_detection.csv")

    X = []
    X_file = df[df["split"] == "train"]["file"]
    for img_path in X_file:
        with Image.open("archive/" + img_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.resize(img_size)
            vec = np.asarray(img, dtype=np.uint8).flatten()
            X.append(vec)
            
    y = df[df["split"] == "train"]["age"]

    return X, y

def display_picture(vec):
    plt.imshow(vec.reshape(img_size[0], img_size[1], 3))
    plt.show()

if __name__ == "__main__":
    X_train, y_train = load_data()
    for f in X_train[:5]:
        display_picture(f)