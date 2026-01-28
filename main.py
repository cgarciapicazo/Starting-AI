import torch
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

IMG_SIZE = (64, 64)
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-3
EPOCH_CHECK_INTERVAL = 20
AGE_CLASSES = ["18-20", "21-30", "31-40", "41-50", "51-60"]
AGE_TO_INDEX = {label: i for i, label in enumerate(AGE_CLASSES)}

def load_data():
    df = pd.read_csv("archive/age_detection.csv")
    train = df[df["split"] == "train"].sample(frac=1)

    X = []
    X_file = train["file"]
    for img_path in X_file:
        with Image.open("archive/" + img_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            img = img.resize(IMG_SIZE)
            # Model-friendly format: float32 in [0, 1], flattened (H*W*C)
            vec = (np.asarray(img, dtype=np.float32) / 255.0).flatten()
            X.append(vec)
            
    y_str = train["age"].to_numpy()
    y = np.array([AGE_TO_INDEX[label] for label in y_str], dtype=np.int64)

    return X, y

def display_picture(vec, label = ""):
    plt.imshow(vec.reshape(IMG_SIZE[0], IMG_SIZE[1], 3))
    plt.title(label)
    plt.show()

class NN_Regression(torch.nn.Module):

    def __init__(self):
        super(NN_Regression, self).__init__()
        # Input is 64*64 RGB pixels flattened
        self.l1 = torch.nn.Linear(64 * 64 * 3, 200)
        self.l2 = torch.nn.Linear(200, 200)
        self.l3 = torch.nn.Linear(200, 5)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x

if __name__ == "__main__":
    X_train, y_train = load_data()
    # for i in range(50, 53):
    #     display_picture(X_train[i], label=y_train[i])
    X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    Model = NN_Regression()
    loss = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(Model.parameters(), lr=LEARNING_RATE)

    loss_graph = []
    for epoch in range(NUM_EPOCHS):
        optimiser.zero_grad()

        predictions = Model(X_train_tensor)
        crossent = loss(predictions, y_train_tensor)

        crossent.backward()
        optimiser.step()
        if epoch % EPOCH_CHECK_INTERVAL == 0:
            print(f"epoch: {epoch}, loss: {crossent}")
            loss_graph.append(float(crossent.values()))
    plt.plot(np.linspace(EPOCH_CHECK_INTERVAL, NUM_EPOCHS, EPOCH_CHECK_INTERVAL), loss_graph)
    plt.show()
