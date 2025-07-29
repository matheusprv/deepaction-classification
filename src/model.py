using_colab = True
dataset_folder = "/content"

try:
    from google import colab
except:
    using_colab = False
    dataset_folder = "/media/work/matheusvieira/deepaction"


import torch
import timm
from torch import Tensor
from torch.nn import Linear, ReLU, Dropout, Sequential, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from PIL import Image

from skimage import util, transform, exposure

from datetime import datetime

import cv2
import numpy as np
import os
import pandas as pd
import random
import json
import gc
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

videos_df = pd.read_csv("videos.csv").values.tolist()
torch_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
with open("real_data_augmentation.json", "r") as f:
    real_videos_data_augmentation = json.load(f)

if using_colab:
    videos_df = videos_df[:10]

videos_dict = dict()

def read_fake_video(video_path):
    video_frames = []
    cap = cv2.VideoCapture(f"{dataset_folder}{video_path}")
    for i in range(16):
        ret, current_frame = cap.read()
        if not ret: break
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        video_frames.append(current_frame)
    cap.release()
    video_frames = [torch_transform(Image.fromarray(frame)) for frame in video_frames]
    stacked = torch.stack(video_frames)
    videos_dict[video_path] = [stacked]

def augment_video(video_frames, augment):
    rotation = augment["rotation"]
    brightness = augment["brightness"]
    noise = augment["noise"]
    flip_horizontal = augment["flip_horizontal"]
    flip_vertical = augment["flip_vertical"]

    augmented_frames = []

    for img in video_frames:
        img = transform.rotate(img, rotation)
        img = exposure.adjust_gamma(img, brightness)
        img = util.random_noise(img, mode="gaussian", var=noise)
        img = (img * 255).astype(np.uint8)
        if flip_horizontal:
            img = np.fliplr(img)
        if flip_vertical:
            img = np.flipud(img)
        augmented_frames.append(img)

    return augmented_frames

def read_real_video(video_path):
    video_frames = []
    cap = cv2.VideoCapture(f"{dataset_folder}{video_path}")
    for i in range(400):
        ret, current_frame = cap.read()
        if not ret: break
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        video_frames.append(current_frame)
    cap.release()

    subclips = []

    video = real_videos_data_augmentation[video_path]
    for clip in video.values():
        start = clip["start"]
        end = clip["end"]
        augment = clip["augment"]

        clip_frames = video_frames[start:end]
        if augment is not None:
            clip_frames = augment_video(clip_frames, augment)

        clip_frames = [torch_transform(Image.fromarray(frame)) for frame in clip_frames]    
        stacked = torch.stack(clip_frames)
        subclips.append(stacked)

    videos_dict[video_path] = subclips

    del clip_frames
    del stacked
    gc.collect()

for data in videos_df:
    data = data[0]
    if not "Pexels" in data:
        read_fake_video(data)
    else:
        read_real_video(data)

current_time = datetime.now()
print(current_time)


with open("real_data_augmentation.json", "r") as f:
    real_videos_data_augmentation = json.load(f)

REAL = [1.0, 0.0]
FAKE = [0.0, 1.0]

class CustomDataset(Dataset):
    num_frames = 16

    def __init__(self, videos_names, train=False):
        if train: self.dataset = self.make_dataset_train(videos_names)
        else:     self.dataset = self.make_dataset_test(videos_names)


    def make_dataset_train(self, videos_names):
        dataset = []
        for video in videos_names:
            if "Pexels" not in video:
                dataset.append((video, 0, Tensor(FAKE)))
            else:
                for i in range(25):
                    dataset.append((video, i, Tensor(REAL)))
        return dataset

    def make_dataset_test(self, videos_names):
        dataset = []
        for video in videos_names:
            label = Tensor(FAKE) if "Pexels" not in video else Tensor(REAL)
            dataset.append((video, 0, label))
        return dataset


    def __len__(self):
        return len(self.dataset)

    def dataset_info(self):
        real_videos = 0
        fake_videos = 0
        for i in self.dataset:
            if np.argmax(i[2]) == 0:
                real_videos += 1
            else:
                fake_videos += 1
        print(f"Real: {real_videos} - Fake: {fake_videos}")


    def __getitem__(self, index):
        video_key, frame, label = self.dataset[index]
        video = videos_dict[video_key][frame] 
        return video, label

class ViTVideoClassifier(torch.nn.Module):
    def __init__(self, backbone_name='vit_base_patch16_224', num_frames=16):
        super().__init__()
        self.frames = num_frames

        # 1) Pre-trained ViT – keep everything *except* the classification head
        vit = timm.create_model(backbone_name, pretrained=True)
        self.feature_dim = vit.head.in_features
        vit.reset_classifier(0)
        self.frame_encoder = vit

        # 2) Temporal avg-pool + custom head
        self.head = Sequential(
            Linear(self.feature_dim, 512),
            ReLU(inplace=True),
            Dropout(0.3),
            Linear(512, 128),
            ReLU(inplace=True),
            Dropout(0.3),
            Linear(128, 2)
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)                # treat every frame as an image
        feats = self.frame_encoder(x)           # [B*T, D]
        feats = feats.view(B, T, -1)            # [B, T, D]
        feats = feats.mean(dim=1)               # global *temporal* average pooling
        logits = self.head(feats)               # [B, 2]
        return logits


def get_KFold(i, K):
    dataset = list(videos_dict.keys())
    dataset_size = len(dataset)
    batch_size = dataset_size // K

    limits = np.arange(0, dataset_size+1, batch_size)

    start = limits[i]
    end = limits[i+1]

    train_df =  dataset[:start] + dataset[end:]
    test_df = dataset[start:end]

    return train_df, test_df

def run_epoch(model, dataloader, criterion=CrossEntropyLoss(), train=False, optimizer=None):
    """
    If `train=True`, back-propagate and update.
    Returns   loss, acc, auc, precision, recall
    """
    epoch_loss = 0
    all_labels, all_preds, all_probs = [], [], []

    model.train() if train else model.eval()

    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)

        if train:
            logits = model(xb)
            loss   = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(xb)
                loss   = criterion(logits, yb)

        epoch_loss += loss.item() * xb.size(0)

        probs  = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds  = logits.argmax(1).detach().cpu().numpy()
        labels = yb.argmax(1).detach().cpu().numpy()

        all_labels.extend(labels)
        all_preds.extend(preds)
        all_probs.extend(probs)

    avg_loss = epoch_loss / len(dataloader.dataset)
    acc      = accuracy_score(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')

    try:
        precision = precision_score(all_labels, all_preds)
    except ValueError:
        precision = float('nan')

    try:
        recall = recall_score(all_labels, all_preds)
    except ValueError:
        recall = float('nan')

    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, auc, precision, recall, cm


def train_loop(model, train_dl, epochs=15, save_path="best_model.pth", patience=3, min_delta=0.001):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    criterion = CrossEntropyLoss()

    best_loss = float('inf')
    curr_patience = patience

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, tr_auc, tr_prec, tr_rec, _ = run_epoch(
            model, train_dl, criterion,
            train=True, optimizer=optimizer
        )
        print(
            f"[{ep:02}/{epochs}] "
            f"Loss={tr_loss:.4f} "
            f"Acc={tr_acc:.3f} "
            f"AUC={tr_auc:.3f} "
            f"Prec={tr_prec:.3f} "
            f"Rec={tr_rec:.3f}"
        )

        if (best_loss - tr_loss) >= min_delta:
            best_loss = tr_loss
            torch.save(model.state_dict(), save_path)
            print(f"\tSaved new best model at epoch {ep} with loss {best_loss:.4f}")
            curr_patience = patience
        else:
            curr_patience -= 1
            print(f"\tNo significant improvement. Patience left: {curr_patience}")
            if curr_patience == 0:
                print(f"Model did not improve by at least {min_delta:.4f} for {patience} consecutive epochs. Finishing training.")
                break

    # Restoring best weights
    model.load_state_dict(torch.load(save_path))

def kfold_train(K=5, epochs=2):

    losses = np.empty(K)
    accuracies = np.empty(K)
    aucs = np.empty(K)
    precisions = np.empty(K)
    recalls = np.empty(K)


    for i in range(K):
        print(f"K: {i}")

        model = ViTVideoClassifier().to(device)
        train, test = get_KFold(i, 5)

        train = CustomDataset(train, True)
        train.dataset_info()
        test = CustomDataset(test)
        test.dataset_info()
        train_dataloader = DataLoader(train, batch_size=4, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test, batch_size=4, shuffle=True, drop_last=True)

        save_path = dataset_folder + f"/best_model_k{i+1}.pth"
        train_loop(model, train_dataloader, save_path=save_path, epochs=epochs)

        avg_loss, acc, auc, precision, recall, cm = run_epoch(model, test_dataloader)
        print(
            f"Teste "
            f"Loss={avg_loss:.4f} "
            f"Acc={acc:.4f} "
            f"AUC={auc:.4f} "
            f"Prec={precision:.4f} "
            f"Rec={recall:.4f}"
        )

        print("Confusion Matrix:")
        cm_df = pd.DataFrame(cm, index=["Actual REAL", "Actual FAKE"], columns=["Predicted REAL", "Predicted FAKE"])
        print(cm_df)

        losses[i] = avg_loss
        accuracies[i] = acc
        aucs[i] = auc
        precisions[i] = precision
        recalls[i] = recall

        del train
        del train_dataloader
        del test
        model.to("cpu")
        del model
        torch.cuda.empty_cache()

        print("="*20)

    print(f"Avg_loss: {np.mean(losses)}")
    print(f"Avg_acc: {np.mean(accuracies)}")
    print(f"Avg_auc: {np.mean(aucs)}")
    print(f"Avg_precisions: {np.mean(precisions)}")
    print(f"Avg_recalls: {np.mean(recalls)}")

kfold_train(K=5, epochs=10)
