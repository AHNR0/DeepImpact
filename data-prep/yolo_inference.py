"""
This code is used to run inference on a dataset directory formed as TSM format, for each label foder. 
"""

import os
import subprocess
import random

videos_path = ""  # videos path of the dataset
yolo5_detectpy_path = ""  # path to the detect.py file of yolov5
yolo5_weights_path = ""  # path to where trained weights of the ball detection network are saved
name = 'ball_detection'
save_direction = ""  # where to save the ball detection labels

videos = os.listdir(videos_path)
##################################################################################
# random.shuffle(videos)
# del videos[3000:]
# print('***' *20)
# print(len(videos))
# print('***' *20)
##################################################################################

for video in videos:
    video_path = os.path.join(videos_path,video)
    cmd = "python {} --source \"{}\" --weights \"{}\" --img 640 --project \"{}\" --name \"{}\" --save-txt --device 1".format(
        yolo5_detectpy_path, video_path, yolo5_weights_path, save_direction, video
    )
    subprocess.call(cmd, shell=True)