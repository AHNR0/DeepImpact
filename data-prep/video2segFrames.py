# Based on the Code from "TSM: Temporal Shift Module for Efficient Video Understanding"
#

import os, subprocess
import numpy as np
import pandas as pd


n_thread = 100


def class_process(dir_path, dst_dir_path, class_name, length, mode):
    """
    Multiprocess all videos of a class of events(header, nonheader, test/header, test/nonheader) to extract frames of each event.

    dir_path: directory of the videos
    dst_dir_path: directory to save the data
    class_name: the folder name(event name) in dir_path
    length: number of frames to be extracted for each event
    mode: the extraction method, set it to None for faster process.

    """

    print('*' * 20, class_name, '*' * 20)
    class_path = os.path.join(dir_path, class_name)
    if not os.path.isdir(class_path):
        print('*** is not a dir {}'.format(class_path))
        return

    dst_class_path = os.path.join(dst_dir_path, class_name)
    if not os.path.exists(dst_class_path):
        os.mkdir(dst_class_path)

    vid_list = os.listdir(class_path)
    vid_list.sort()
    print(vid_list)
    print(class_path)
    print(dst_class_path)
    # for video in vid_list:
    #     vid2jpg(video,class_path,dst_class_path,length,mode)
    p = Pool(n_thread)
    from functools import partial
    worker = partial(vid2jpg, class_path=class_path, dst_class_path=dst_class_path, length=length, mode=mode)
    for _ in tqdm(p.imap_unordered(worker, vid_list), total=len(vid_list)):
        pass
    # p.map(worker, vid_list)
    p.close()
    p.join()

    print('\n')


def vid2jpg(file_name, class_path, dst_class_path, length, mode=None, PATH=''):
    '''
    length: the number of frames to be extracted before and after each labeled frame.

    mode: decides how frames are extracted from video
    None: ffmpeg directly extracts only the selected range of frames for each event
    Partial: ffmpeg firstly extracts frames from all
    '''

    # if '.mp4' not in file_name:
    #     return

    # name, ext = os.path.splitext(file_name)
    dst_directory_path = os.path.join(dst_class_path, file_name)

    label_file_path = os.path.join(class_path, file_name)
    label_file_path = os.path.join(label_file_path, file_name + '.csv')
    csv_df = pd.read_csv(label_file_path, delimiter=',')

    #PATH = "/media/simpl/123E772F3E770ACD/Users/Ahmad/Desktop/dataset/nonheader"
    # PATH= "/media/simpl/Data/videos_temp"

    if mode == 'partial':
        video_paths = csv_df['Video Address'].unique()
        video_names = []
        images_folders = []
        for i, video_path in enumerate(video_paths):
            video_names.append(os.path.splitext(os.path.basename(video_path))[0])

            # images_folders.append(os.path.join(dst_directory_path,'video_images_'+video_names[i]))
            images_folders.append(os.path.join(PATH, file_name + '_video_images_' + video_names[i]))

            if os.path.exists(images_folders[i]):
                if len(os.listdir(images_folders[i])) > 1000:  # assuming that video at least has 1000 frames.
                    print('frames of the video are extracted before!')
                else:
                    subprocess.call('rm -r \"{}\"'.format(images_folders[i]), shell=True)
                    os.mkdir(images_folders[i])
                    print('**' * 12)
                    print(video_path)
                    print('**' * 12)
                    print(images_folders[i])
                    video_extractFrames(video_path, images_folders[i])
            else:
                os.mkdir(images_folders[i])
                print('**' * 12)
                print(video_path)
                print('**' * 12)
                print(images_folders[i])
                video_extractFrames(video_path, images_folders[i])

    for i in range(len(csv_df)):
        subfolder_name = file_name + '_' + (4 - len(str(i))) * '0' + str(i)
        dst_directory_subfolder = os.path.join(dst_directory_path, subfolder_name)
        # print(dst_directory_subfolder)
        try:
            if os.path.exists(dst_directory_subfolder):
                if not os.path.exists(os.path.join(dst_directory_subfolder, 'img_00001.jpg')):
                    subprocess.call('rm -r \"{}\"'.format(dst_directory_subfolder), shell=True)
                    print('remove {}'.format(dst_directory_subfolder))
                    os.mkdir(dst_directory_subfolder)
                else:
                    print('*** convert has been done for incident number {} at : {}'.format(i, dst_directory_path))
                    continue
                    # return
            else:
                os.mkdir(dst_directory_subfolder)
        except:
            print(dst_directory_subfolder)
            return
        # print(csv_df.iloc[i,1])
        if mode == 'partial':
            if csv_df.iloc[i, 2] == video_paths[0]:
                copy_frames(int(csv_df.iloc[i, 1]), images_folders[0], dst_directory_subfolder, length)
            else:
                copy_frames(int(csv_df.iloc[i, 1]), images_folders[1], dst_directory_subfolder, length)
        else:
            try:
                if int(csv_df.iloc[i, 1]) > length:
                    start_frame = int(csv_df.iloc[i, 1]) - length
                else:
                    start_frame = 1
            except:
                print(label_file_path)
                # print(csv_df)
                print(i)
                print(csv_df.iloc[63, 1])
                print(csv_df.iloc[i, 1])
            # print(csv_df.iloc[i,1])
            end_frame = int(csv_df.iloc[i, 1]) + length
            # # cmd = 'ffmpeg -i \"{}\" -threads 1 -vf scale=-1:331 -q:v 0 \"{}/img_%05d.jpg\"'.format(video_file_path, dst_directory_path)
            cmd = 'ffmpeg -i \"{}\" -threads 1 -vf select=\'between(n\,{}\,{})\' -vsync 0 \"{}/img_%05d.jpg\"'.format(
                csv_df.iloc[i, 2], str(start_frame), str(end_frame), dst_directory_subfolder)
            # print(cmd)
            subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # if mode == 'partial':

    #     for image_folder in images_folders:
    #         cmd = 'rm -r \"{}\"'.format(image_folder)
    #         subprocess.call(cmd,shell=True)


def video_extractFrames(videoPath, Distination_path):
    cmd = 'ffmpeg -i \"{}\" \"{}/img_%06d.jpg\"'.format(videoPath, Distination_path)
    subprocess.call(cmd, shell=True)


def copy_frames(FrameNumber, ImagesPath, DistinationPath, length):
    start_frame = FrameNumber - length
    for i in range(length * 2 + 1):
        image_path = os.path.join(ImagesPath,
                                  'img_' + '0' * (6 - len(str(start_frame + i))) + str(start_frame + i) + '.jpg')
        target_image_path = os.path.join(DistinationPath, 'img_' + '0' * (5 - len(str(i + 1))) + str(i + 1) + '.jpg')
        shutil.copyfile(image_path, target_image_path)


if __name__ == "__main__":
    dir_path = sys.argv[1]
    dst_dir_path = sys.argv[2]
    length = int(sys.argv[3])
    try:
        Mode = sys.argv[4]
        PATH = sys.argv[5]
    except:
        Mode = None
        PATH = ''


    class_name = 'header'
    class_process(dir_path, dst_dir_path, class_name, length, mode=Mode, PATH='')
