import os
import cv2
import numpy as np
import pandas as pd


# RunwayML possui 10 vídeos para a classe 77, com prompt "A person holding a pet.". Mas cinco deles são da classe 27, "A person ironing clothes.", e já existem lá. Então serão deletados
miss_classified_videos = "/content/deepaction_v1/RunwayML/77"
files_to_delete = [os.path.join(miss_classified_videos, f"{chr(i)}.mp4") for i in range(ord('f'), ord('j')+1)]

for file in files_to_delete:
    if os.path.exists(file):
        os.remove(file)




dataset = "/content/deepaction_v1/"

# Selecionando somente os subdiretórios do "dataset"
videos_source = [
    (source_name, os.path.join(dataset, source_name))
    for source_name in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, source_name)) and '.git' not in source_name
]

videos_source = sorted(videos_source)

for v in videos_source:
    print(v)

print(len(videos_source))

dataset_info = {k:{} for k in range(104)}
list_of_videos = []

for source_name, source_path in videos_source:
    human_actions = [(human_action, os.path.join(source_path, human_action)) for human_action in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, human_action))]

    for human_action, human_action_path in human_actions:
        videos_paths = [os.path.join(human_action_path, x) for x in os.listdir(human_action_path)]
        dataset_info[int(human_action)][source_name] = videos_paths
        list_of_videos += videos_paths

for k in dataset_info:
    print(f"{k}: ", end="")
    datasources = dataset_info[k].keys()
    for source in datasources:
        print(f"{source}: {len(dataset_info[k][source])} - ", end ='')
    print("")





def check_number_of_frames(video_path):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return total_frames

def videos_statistcs(list_of_videos):
    videos_frames = [check_number_of_frames(x) for x in list_of_videos]
    videos_frames = sorted(videos_frames)
    print(videos_frames[:50])
    videos_frames = np.array(videos_frames)
    print(f"Min: {np.min(videos_frames)}", list_of_videos[np.argmin(videos_frames)])
    print(f"Primeiro quartil: {np.quantile(videos_frames, 0.25)}")
    print(f"Mediana: {np.median(videos_frames)}")
    print(f"Média: {np.mean(videos_frames)}")
    print(f"Max: {np.max(videos_frames)}", list_of_videos[np.argmax(videos_frames)])

    print(f"Menos de 16x25=400 frames: {(videos_frames[videos_frames < 400]).shape}")

videos_statistcs(list_of_videos)
print("="*50)
videos_statistcs(list(filter(lambda x: "Pexels" in x, list_of_videos)))





# Making a csv dataset
videos_list = []
for action in dataset_info.values():
    for source in action.values():
        videos_list.extend(source)

videos_path = [x.replace("/content", "") for x in videos_list]
train_df = pd.DataFrame(videos_path, columns=['video_path'])
train_df.to_csv("videos.csv",index=False)





# Real videos data augmentation
import random
import json
random.seed(42)

SUBCLIPS_SIZE = 16
real_videos = list(filter(lambda x: "Pexels" in x, list_of_videos))

videos_modifications = dict()

for real_video in real_videos:
    number_of_frames = check_number_of_frames(real_video)

    real_video = real_video.replace("/content", "")
    videos_modifications[real_video] = dict()

    upper_bound = min(401, number_of_frames+1)
    limits = np.arange(0, upper_bound, SUBCLIPS_SIZE)

    # Allow a circular list without including the last element
    len_limits = len(limits) - 1
    # Becomes True when the circular list start repeating
    repeating = False
    for i in range(25):
        start = i % len_limits
        end = start + 1
        if i >= number_of_frames // 16: repeating = True
        videos_modifications[real_video][i] = {
            "start": int(limits[start]),
            "end": int(limits[end]),
            "augment": None if not repeating else {
                "rotation": random.uniform(-10, 10),
                "brightness": random.uniform(0.8, 1.2),
                "noise": random.uniform(0.0, 0.05),
                "flip_horizontal": random.choice([True, False]),
                "flip_vertical": random.choice([True, False])
            }
        }

with open("real_data_augmentation.json", "w") as f:
    json.dump(videos_modifications, f, indent=4)