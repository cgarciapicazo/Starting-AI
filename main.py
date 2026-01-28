import torch
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

IMG_SIZE = (64, 64)
AGE_CLASSES = ["18-20", "21-30", "31-40", "41-50", "51-60"]
AGE_TO_INDEX = {label: i for i, label in enumerate(AGE_CLASSES)}

NUM_EPOCHS = 1000
LEARNING_RATE = 1e-3
BATCH_SIZE = 10
EPOCH_CHECK_INTERVAL = 50
SHOW_GRAPH = True


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
            vec = (np.asarray(img, dtype=np.float32) / 255.0).flatten()
            X.append(vec)
            
    y_str = train["age"].to_numpy()
    y = np.array([AGE_TO_INDEX[label] for label in y_str], dtype=np.int64)

    return X, y

def display_picture(vec, label = ""):
    plt.imshow(vec.reshape(IMG_SIZE[0], IMG_SIZE[1], 3))
    plt.title(label)
    plt.show()

def train_model():
    loss_graph = []
    for epoch in range(NUM_EPOCHS):
        optimiser.zero_grad()

        idx = torch.randint(0, X_train_tensor.size(0), (BATCH_SIZE,))
        xb = X_train_tensor[idx]
        yb = y_train_tensor[idx]

        predictions = Model(xb)
        crossent = loss(predictions, yb)

        crossent.backward()
        optimiser.step()

        if epoch % EPOCH_CHECK_INTERVAL == 0:
            print(f"epoch: {epoch}, loss: {crossent}")
            loss_graph.append(float(crossent.item()))
    if SHOW_GRAPH:
        plt.plot(np.linspace(0, NUM_EPOCHS, int(NUM_EPOCHS / EPOCH_CHECK_INTERVAL)), loss_graph)
        plt.show()
    
def evaluate_model(X, y):
    Model.eval()
    with torch.no_grad():
        predictions = Model(X)
        testloss = loss(predictions, y)
        return testloss

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
    train_model()

