# DeepImpact
This repo contains the codes used for preparation of the soccer header dataset as well as training the DeepImpact network for automatic detection of soccer headers. 

### Requirements
Python3.8 is required. In the main folder, run:

`python -m pip install -r requirement.txt`


## Dataset preparation steps
Use the codes in data-prep folder to prepapre the dataset for TSM

1. `prepare_soccerNet`: changes labels of soccerNet-v2 dataset to frame numbers.
2. `video2segFrames`: for each class, extracts and saves the specified number of frames for each event.
3. `yolo_inference`: for each event, applies the trained ball detector model on each frame to find ball position.
4. `kalman_4state`: applies kalman filtering on the detected locations of the ball to improve ball tracking.
5. `crop_dataset`: uses the tracked ball location to crop each frame around the location of the ball.
6. `HeaderDataset`: after cropped images are put in the proper format to be compatible with TSM, use this code to generate the train/val split of the data.
7. `create_testDataset_kalman`: use this code to generate the dataset of the unseen test videos.

## Train TSM model:
Once dataset is ready and put in the compatible format with TSM code, run the following code in temporal-shift-module folder to train the DeepImpact model. Note: follow the instructions of TSM code to add the pre-trained models.

```
python main_DeepImpact.py header RGB --arch resnet50 --num_segments 8 --gd 20 \
       --lr 0.001 --lr_steps 10 20 --epochs 30 --batch-size 32 -j 16 --dropout 0.6 \
       --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --shift_place=blockres \
       --tune_from=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
```

Once model is trained and saved, follow the instruction in the TSM repository to test the model on the unseen videos.

