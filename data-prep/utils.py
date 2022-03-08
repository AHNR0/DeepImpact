'''
This function checks lists all the txt files inside input_dir and overwrites them on the file with duplicate name inside input_dir/labels 
'''

import glob
import os
import subprocess

def rename_images(folder):
    '''
    folder: the folder in which there are some images and we want to rename these images to start from 1 and increase one by one.
    '''
    images=sorted(os.listdir(folder))
    print(images)
    for i, image in enumerate(images):
        new_img_name='img_'+(5-len(str(i+1)))*'0'+str(i+1)+'.jpg'
        print(new_img_name)
        os.rename(os.path.join(folder,image),os.path.join(folder,new_img_name))

def false_organizer(txt_file,save_file=False):
    """
    Organizes and saves the false positives and false negatives of the TSM result after testing.
    txt_file: TSM result
    returns:
        false positive and false negatives as separate lists
    """
    with open(txt_file,'r') as f:
        lines=f.readlines()
    false_pos=[]
    false_neg=[]

    for i, line in enumerate(lines):
        # print(line)
        if line[-18:] == ' header nonheader\n':
            false_pos.append(line)
        elif line[-18:] == ' nonheader header\n':
            false_neg.append(line)
    if save_file:
        with open(os.path.join(os.path.dirname(txt_file),'test_falPos.txt'),'w') as f:
            f.writelines(false_pos)
        with open(os.path.join(os.path.dirname(txt_file),'test_falNeg.txt'),'w') as f:
            f.writelines(false_neg)
    return false_pos, false_neg


def soccerdb_to_ballonly(soccerdb_path, ball_only_path):

    """
    This code changes the format of the labels in the original soccerdb dataset that has three classes to
    to a new format with only 1 class, which is ball. Both new label files and images are copied in ball_only_path.
    """
    # soccerdb_path=""
    # ball_only_path=""

    folders=['train','validation']

    for folder in folders:
        path_label=os.path.join(soccerdb_path,folder,'labels')
        path_images=os.path.join(soccerdb_path,folder,'images')

        sub_folders=os.listdir(path_label)
        for sub_folder in sub_folders:
            path=os.path.join(path_label,sub_folder)
            files=os.listdir(path)
            for file_name in files:
                print(file_name)
                file_path=os.path.join(path,file_name)
                with open(file_path,'r') as f:
                    copy_image=False
                    for line in f:
                        if line[0] == str(1):
                            copy_image=True
                            copy_path_folder=os.path.join(ball_only_path,folder,'labels',sub_folder)
                            # copy_path_file=os.path.join(ball_only_path,folder,'labels',sub_folder,file_name)
                            cmd = "mkdir -p \"{}\"".format(copy_path_folder)   #  && cp \"{}\" \"{}\"".format(copy_path_folder,file_path,copy_path_file)
                            subprocess.call(cmd,shell=True)
                            with open(os.path.join(ball_only_path,folder,'labels',sub_folder,file_name),'w') as f1:
                                f1.write('0'+line[1:])
                    if copy_image:
                        image_file_path=os.path.join(path_images,sub_folder,os.path.splitext(os.path.basename(file_name))[0])+'.jpg'
                        copy_path_folder=os.path.join(ball_only_path,folder,'images',sub_folder)
                        copy_path_file=os.path.join(ball_only_path,folder,'images',sub_folder,os.path.splitext(os.path.basename(file_name))[0])+'.jpg'
                        cmd = "mkdir -p \"{}\" && cp \"{}\" \"{}\"".format(copy_path_folder,image_file_path,copy_path_file)
                        subprocess.call(cmd,shell=True)


def shorten_dataset(dataset_path, dest_dataset_path, extract_length):
    """
    This function generates a shorter version of the given dataset by reducing the length of each event to the given length
    dataset_path: main directory of the main dataset that contains videos/ and labels/
    dest_dataset_path: path to save the shortened version of the dataset.
    extract_length: the number of frames to be included before and after the center frame. Should be less than the original dataset.
    NOTE: any cropping should later be applied to the shorter dataset.
    """
    
    ## for header events:
    video_folder = os.path.join(dataset_path, 'videos/header')
    label_folder = os.path.join(dataset_path, 'ball_labels/header')
    dest_video_folder = os.path.join(dest_dataset_path, 'videos/header')
    dest_label_folder = os.path.join(dest_dataset_path, 'ball_labels/header')

    events=os.listdir(video_folder)
    
    for i, event in enumerate(events):
        event_folder=os.path.join(video_folder,event)
        event_label_folder=os.path.join(label_folder,event)
        
        os.mkdir(os.path.join(dest_video_folder,event))
        
        for jj, img_num in enumerate(range(6-extract_length,6+extract_length+1)):
            img_name='img_'+'0'*(5-len(str(img_num)))+str(img_num)+'.jpg'
            label_name='img_'+'0'*(5-len(str(img_num)))+str(img_num)+'.txt'
            
            dir_img_name='img_'+'0'*(5-len(str(jj+1)))+str(jj+1)+'.jpg'
            cmd ='cp \"{}\" \"{}\"'.format(os.path.join(event_folder,img_name),os.path.join(dest_video_folder,event,dir_img_name))
    #         print(cmd)
            subprocess.call(cmd,shell=True)
            if os.path.exists(os.path.join(event_label_folder,label_name)):
                if not os.path.exists(os.path.join(dest_label_folder,event)):
                    os.mkdir(os.path.join(dest_label_folder,event))
                dir_label_name='img_'+'0'*(5-len(str(jj+1)))+str(jj+1)+'.txt'
                cmd ='cp \"{}\" \"{}\"'.format(os.path.join(event_label_folder,label_name),os.path.join(dest_label_folder,event,dir_label_name))
                subprocess.call(cmd,shell=True)
    
    ## for nonheader events:
    video_folder = os.path.join(dataset_path, 'videos/nonheader')
    label_folder = os.path.join(dataset_path, 'ball_labels/nonheader')
    dest_video_folder = os.path.join(dest_dataset_path, 'videos/nonheader')
    dest_label_folder = os.path.join(dest_dataset_path, 'ball_labels/nonheader')

    events=os.listdir(video_folder)
    
    for i, event in enumerate(events):
        event_folder=os.path.join(video_folder,event)
        event_label_folder=os.path.join(label_folder,event)
        
        os.mkdir(os.path.join(dest_video_folder,event))
        
        for jj, img_num in enumerate(range(6-extract_length,6+extract_length+1)):
            img_name='img_'+'0'*(5-len(str(img_num)))+str(img_num)+'.jpg'
            label_name='img_'+'0'*(5-len(str(img_num)))+str(img_num)+'.txt'
            
            dir_img_name='img_'+'0'*(5-len(str(jj+1)))+str(jj+1)+'.jpg'
            cmd ='cp \"{}\" \"{}\"'.format(os.path.join(event_folder,img_name),os.path.join(dest_video_folder,event,dir_img_name))
    #         print(cmd)
            subprocess.call(cmd,shell=True)
            if os.path.exists(os.path.join(event_label_folder,label_name)):
                if not os.path.exists(os.path.join(dest_label_folder,event)):
                    os.mkdir(os.path.join(dest_label_folder,event))
                dir_label_name='img_'+'0'*(5-len(str(jj+1)))+str(jj+1)+'.txt'
                cmd ='cp \"{}\" \"{}\"'.format(os.path.join(event_label_folder,label_name),os.path.join(dest_label_folder,event,dir_label_name))
                subprocess.call(cmd,shell=True)
    
def putTogether_4Yolo(videos_path):
    """
    Puts images of all events of a class in one folder for faster processing by Yolo for ball detection. After detection, labels are put back in the original format.
    """
    path = videos_path 
    target_path_images=os.path.join(videos_path, 'alltogether')
    
    videos=os.listdir(path)
    videos.remove('alltogether')
    # print(len(videos))
    count=0
    for video in videos:
    #     print(video)
        videoPath=os.path.join(path,video)
        
        images=os.listdir(videoPath)
        
        if 'labels' in images:
            images.remove('labels')
    #     print(images)
        for image in images:
            count += 1
            image_name=video+'_'+image
            home_path=os.path.join(videoPath,image)
            tar_path=os.path.join(target_path_images,image_name)
            subprocess.call('cp {} {}'.format(home_path,tar_path),shell=True)
            print('{}/320144'.format(count))

def reformat_label_Yolo(labels_path,dest_path):
    path = labels_path
    txt_files=glob.glob(os.path.join(path,'*.txt'))
    if os.path.join(path,'classes.txt') in txt_files:
        txt_files.remove(os.path.join(path,'classes.txt'))
    for i, txt_file in enumerate(txt_files):
        folder_name=os.path.basename(txt_file)[:-14]
        txt_file_name=os.path.basename(txt_file)[-13:]
        dir_txt_file=os.path.join(dest_path,folder_name,txt_file_name)
        print(folder_name)
        print(txt_file_name)
        
        cmd= 'cp {} {}'.format(txt_file,dir_txt_file)
        subprocess.call(cmd,shell=True)
    
            

