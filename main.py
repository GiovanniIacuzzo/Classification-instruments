from utils.train import train_model
from utils.evaluate import evaluate_model

if __name__ == "__main__":
    train_model("dataset_split/train", "dataset_split/val", epochs=10, batch_size=16, lr=0.001)
    evaluate_model("dataset_split/test")
