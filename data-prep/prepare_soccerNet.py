import json, cv2, os, sys
import pandas as pd


class SoccerNet:
    def __init__(self, path):
        self.mainFolder = path
        self.HQvideoList = []
        self.LQvideoList = []
        self.get_video_list()
        
    def get_video_list(self):
        directory = iter(os.walk(self.mainFolder))
        for i in range(2000):
            try:
                direct = next(directory)
                if direct[1] == []:
                    self.LQvideoList.append(direct[0])
                    if os.path.isfile(os.path.join(direct[0], '1_HQ.mkv')) and os.path.isfile(os.path.join(direct[0], '2_HQ.mkv')) and os.path.isfile(os.path.join(direct[0], 'video.ini')):
                        self.HQvideoList.append(direct[0])
            except:
                break

    def extract_label_soccerNet(self, video_folder,dist_folder,mode='high'):
        """
        This function changes the labels of soccerNet to proper format to be used for generating non-header events:
        Generates and saves a csv file with the frame number of nonheader events.
        video folder: is the path to the folder containing videos of two halves (low or high quality),
        mode: to select between high or low quality videos
        """
        video_folder=os.path.join(video_folder,'')
        video_name=os.path.basename(os.path.dirname(video_folder)).replace(' ','')
        if mode == 'high':
            video_path = [os.path.join(video_folder, '1_HQ.mkv'), os.path.join(video_folder, '2_HQ.mkv')]
            video_fps = [self.get_fps(video_path[0]), self.get_fps(video_path[1])]
            HQvideo_detail = os.path.join(video_folder,'video.ini')
            try:
                with open(HQvideo_detail,'r') as f:
                    lines=f.readlines()
                    start_times=[int(float(lines[1].split()[2])*1000),int(float(lines[5].split()[2])*1000)]
#                     print(start_times)
            except:
                raise FileNotFoundError('video.ini file does not exist in video folder!')
        elif mode == 'low':
            video_path = [os.path.join(video_folder, '1.mkv'), os.path.join(video_folder, '2.mkv')]
            video_fps = [self.get_fps(video_path[0]), self.get_fps(video_path[1])]
        else:
            raise NameError('this mode is not defined!')
        label_path=os.path.join(video_folder, 'Labels-v2.json')
        with open(label_path,'rb') as f:
            labels=json.load(f)
        # print(labels['annotations'][0])
        label_df=pd.DataFrame(columns=['Event Number', 'Frame Number', 'Video Address'])

        for i, label in enumerate(labels['annotations']):
            if label['visibility'] == "visible":
                game_half=int(label['gameTime'][0])-1
                FPS= video_fps[game_half]
                if mode == 'high':
                    event_frame=self.ms_to_frameNumber(FPS, int(label['position'])+start_times[game_half])
        #         print(event_frame)
                else:
                    event_frame=self.ms_to_frameNumber(FPS, int(label['position']))
                
                label_df.loc[i] = [i+1, event_frame, video_path[game_half]]

        dis_folder = os.path.join(dist_folder, video_name)
        file_name = video_name+'.csv'
        if os.path.exists(dis_folder):
            label_df.to_csv(os.path.join(dis_folder, file_name), index=False)
        else:
            os.mkdir(dis_folder)
            label_df.to_csv(os.path.join(dis_folder, file_name), index=False)

    def get_fps(self, video_path):    
        vid = cv2.VideoCapture(video_path)
        fps = vid.get(cv2.CAP_PROP_FPS)
        return fps

    def ms_to_frameNumber(self, fps, time_stamp):
        return int(time_stamp/1000*fps)


if __name__ == "__main__":
    soccernet_folder = sys.argv[1]  # path to the main folder of soccerNet dataset
    dist_folder = sys.argv[2]   # path to where csv files of labels are saved
    
    soccerNet = SoccerNet(soccernet_folder)

    for i, path in enumerate(soccerNet.HQvideoList):
        soccerNet.extract_label_soccerNet(path, dist_folder, mode='high')
    
