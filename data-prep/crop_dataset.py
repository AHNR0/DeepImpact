import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
from kalman_4state import *

def show_boxes(img,boundingBox,label):
    plt.imshow(img)
    print(img.shape)
    height,width,_= img.shape
    bounding_box=np.copy(boundingBox)
    bounding_box[:,[0,2]]=bounding_box[:,[0,2]]*width
    bounding_box[:,[1,3]]=bounding_box[:,[1,3]]*height
    ax=plt.gca()
    for i in range(bounding_box.shape[0]):
        center=[bounding_box[i,0]-bounding_box[i,2]/2,bounding_box[i,1]-bounding_box[i,3]/2]
        if label=='player':
            rectangle=patches.Rectangle(center,bounding_box[i,2],bounding_box[i,3],linewidth=1, edgecolor='r')
        if label=='ball':
            rectangle=patches.Rectangle(center,bounding_box[i,2],bounding_box[i,3],linewidth=1, edgecolor='b')
        ax.add_patch(rectangle)


## define the image cropper function

def image_crop(img_path,bounding_box,enlarge_factor=20,newSize=None,save_path=False):
    """
    img_path: path to the image
    boundingBox: path to the txt file containing the bounding box
    """
    #generate center coordinates and 
    img=Image.open(img_path)
    width, height = img.size
    


    #center coordinates:
    bounding_box[:,0]=bounding_box[:,0]*width
    bounding_box[:,1]=bounding_box[:,1]*height
    # enlarged box width and height
    bounding_box[:,2]=bounding_box[:,2]*width*enlarge_factor
    bounding_box[:,3]=bounding_box[:,3]*height*enlarge_factor
    
    #generate coordinates of ball bounding box
    left_corner=bounding_box[0,0]-bounding_box[0,2]/2
    right_corner=bounding_box[0,0]+bounding_box[0,2]/2
    top_corner=bounding_box[0,1]-bounding_box[0,3]/2
    bottom_corner=bounding_box[0,1]+bounding_box[0,3]/2
    

    if left_corner < 0:
        left_corner = 0
    if right_corner > width:
        right_corner = width
    if top_corner < 0:
        top_corner = 0
    if bottom_corner > height:
        bottom_corner = height

    #crop the image
    cropped_image=img.crop((left_corner,top_corner,right_corner,bottom_corner))
    
    if newSize:
        cropped_image=cropped_image.resize(newSize,resample=Image.BICUBIC)
    if save_path:
        img_name=os.path.splitext(os.path.basename(img_path))[0]+'_cropped.jpg'
        cropped_image.save(os.path.join(save_path,img_name))
    return cropped_image


'''
Processes images of a video and generates a dictionary of detected balls in the video
'''
import glob, os
import numpy as np
def video_BallData(imgs_folder,label_folder):
    '''
    inputs:
           imgs_folder: where all images of a video are saved.
           label_folder: where ball detection labels of all frames are saved.
    output:
          labels: a dictionary containing label information of all frames001
            for each key: 'img_name'={'ball_number':?,'bb_info':numpy array (ball_number,bb_info)}
            if ball_number=0, then returns full frame as bounding box.
    '''
    images=sorted(glob.glob(os.path.join(imgs_folder,'*.jpg')))
    video_data={}
    video_data['imgs_path']=imgs_folder
    labels={}
    for image in images:
        img_name=os.path.splitext(os.path.basename(image))[0]
        info={}
        if os.path.isfile(os.path.join(label_folder,img_name+'.txt')):
            with open(os.path.join(label_folder,img_name+'.txt'),'r') as f:
                lines=f.readlines()
            if lines == []:
                print('Label file empty. No ball detected for image: {}, full frame bb will be used'.format(img_name))
                info['ball_number']=0
                info['bb_infor']=np.array([0.5,0.5,1,1]).reshape(1,4)
                labels[img_name]=info
            else:
                # print(lines)
                lines=[lines[-1]]
                # print(lines)
                bb_info=np.zeros((len(lines),4))
                for i, line in enumerate(lines):
                    line=line.split()[1:]
                    for j, item in enumerate(line):
                        bb_info[i,j]=float(item)
                info['ball_number']=len(lines)
                info['bb_info']=bb_info
                labels[img_name]=info
                
        else:
            print('No ball detected for image: {}, full frame bb will be used'.format(img_name))
            info['ball_number']=0
            info['bb_info']=np.array([0.5,0.5,1,1]).reshape(1,4)
            labels[img_name]=info
    video_data['labels']=labels
    return video_data
    


def coordinate_to_map(coordinates,map_size):
    '''
    coordinates: a numpy array containing n points with coordinates of all points: shape (n,2)
    map: the size of coordinates 
    output:
         H: the count of each point in each cell of the grid.
    '''
    xedges=np.linspace(0,1,map_size[0]+1)
    yedges=np.linspace(0,1,map_size[1]+1)
    x=coordinates[:,0]
    y=coordinates[:,1]
    
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))

    return H

def find_number_address(number,map_size):
    xedges=np.linspace(0,1,map_size[0]+1)
    yedges=np.linspace(0,1,map_size[1]+1)
    
    x_ids=np.digitize(number[:,0],xedges)
    y_ids=np.digitize(number[:,1],yedges)
    
    return np.transpose(np.append([y_ids-1],[x_ids-1],axis=0))

def address_to_id(cell_location,map_size):
    idxs=np.arange(map_size[0]*map_size[1])
#     print(np.reshape(idxs,map_size))
    return np.reshape(idxs,map_size)[cell_location[:,0],cell_location[:,1]]

def id_to_string(idxs,map_size):
    zones=range(map_size[0]*map_size[1])
    strings=[]
    for zone in zones:
        string=''
        for i, ID in enumerate(idxs):
            if ID == zone:
                string += str(i)
        strings.append(string)
    return(strings)

def find_mutations(video_balldata,zones=(2,2),mode='all'):
    imgs_path=video_balldata['imgs_path']
    imgs=list(video_balldata['labels'].keys())
    
    bb_map_matrix=np.zeros((len(imgs),zones[0]*zones[1])).astype(str) ## (number of images, number of zones)
    
    for i in range(len(imgs)):
        detections_xy_address=find_number_address(video_balldata['labels'][imgs[i]]['bb_info'],zones)
        detections_ids=address_to_id(detections_xy_address,zones)
        zones_strings=id_to_string(detections_ids,zones)
        bb_map_matrix[i,:]=np.array(zones_strings)
#     print(bb_map_matrix)
    # Use vectorize function of numpy
    length_checker = np.vectorize(len)
    # Find the length of each element
    bb_map_matrix_len = np.vectorize(len)(bb_map_matrix)
#     print(bb_map_matrix_len)
    
    ### check if there is any ball detected:
    if np.count_nonzero(bb_map_matrix) == 0:
        print('There is no ball detected!')
    ### computing all possible sequences
    else:
        ### delete a sequence if there is no ball in it (all zeros):
        empty_seq=[]
        for i in range(bb_map_matrix.shape[1]):
            if np.array_equal(bb_map_matrix[:,i], np.array(['']*len(imgs))):
                empty_seq.append(i)
#         print(empty_seq)
        bb_map_matrix=np.delete(bb_map_matrix,empty_seq,1)
        bb_map_matrix_len=np.delete(bb_map_matrix_len,empty_seq,1)
#         print(bb_map_matrix)
        
        if mode== 'all':
            Mutations=np.tile(np.array(['']),(1,len(imgs)))
#             print(Mutations)
            for i in range(bb_map_matrix_len.shape[1]):
#                 print('*****' * 50)
                ## count number of all sequences:
                count=np.prod(bb_map_matrix_len[:,i][bb_map_matrix_len[:,i] != 0])
                
                ## generate the template of possible sequencecs
                mutations=np.tile(bb_map_matrix[:,i],(count,1))
                mutations_len=np.vectorize(len)(mutations)
#                 print(mutations)
                
                ## generate all possible mutations of orders between zones with more than 1 detections:
                multiple_detections=np.ones(tuple(mutations_len[0,mutations_len[0,:]>1]))
                
                ##### find the addresss of these mutations
                address=np.argwhere(multiple_detections>0)
                
                #### replace the template without possible mutations
                mutes=np.tile(np.array(['']),address.shape)
                more_than1_values=list(mutations[0,mutations_len[0,:]>1])
                
                for jjjj in range(mutes.shape[0]):
                    for kkkk in range(mutes.shape[1]):
                        mutes[jjjj,kkkk]=np.array(more_than1_values[kkkk][address[jjjj,kkkk]])
    
                mutations[:,mutations_len[0,:]>1]=mutes
                
                #### set the maximum allowable number of mutations
                if count > 5:  
                    mutations=mutations[:5,:]
                Mutations=np.append(Mutations,mutations,axis=0)
#                 print(mutations)
            Mutations=np.delete(Mutations,0,axis=0)
            
    return Mutations

def video_crop(mutations,video_ball_data,enlarge_factor=10,newSize=None,save_path=False):
    imgs_path=video_balldata['imgs_path']
    video_name=os.path.basename(imgs_path)
    print(video_name)
    imgs=list(video_balldata['labels'].keys())
    for i in range(mutations.shape[0]):
        video_name_mut=video_name+'_mu'+str(i)
        for j, img in enumerate(imgs):

            img_path=os.path.join(imgs_path,img+'.jpg')
            bb_id=mutations[i,j]
            if bb_id == '':
                bb = np.array([0.5,0.5,1,1]).reshape(1,4)
            else:
                bb=video_balldata['labels'][img]['bb_info'][int(bb_id),:].reshape(1,4).copy()
                
            cropped_image=image_crop(img_path,bb,enlarge_factor,newSize)
#             print(cropped_image)
#             cropped_image.show()
            if save_path:
                save_folder=os.path.join(save_path,video_name_mut)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                cropped_image.save(os.path.join(save_folder,img+'.jpg'))


if __name__ == "__main__":

    videos_path = ''  # path to the videos folder of the uncropped dataset
    labels_path = ''  # path to the ball labels folder for the dataset (from yolo)
    save_path = ''  # where to save the new dataset

    label_mode = 'from_detection'
    enlarge_factor = 10
    frame_division_size = (1,1)

    videos = os.listdir(videos_path)
    
    if 'alltogether' in videos:
        videos.remove('alltogether')
    # for i, video in enumerate(videos):
    #     print(video)
    #     print('{}/{}'.format(i,len(videos)))

    unsuccess_videos = []

    for i, video in enumerate(videos):
        imgs_path=os.path.join(videos_path, video)
        if label_mode == 'from_detection':
            labels_folder = os.path.join(labels_path, video, 'labels')
        if label_mode == 'from_label':
            labels_folder = os.path.join(labels_path, video)
        

        try:
            ### without Kalman filter:
            # video_balldata=video_BallData(imgs_path,labels_folder)
            ### add Kalman filter:
            video_balldata = kalman_filter(imgs_path,labels_folder)
            # print(video_balldata)
            mutations = find_mutations(video_balldata, frame_division_size)  ### (1,1) because we don't want to have any mutations. if desired to have put (2,2)
            video_crop(mutations, video_balldata, enlarge_factor, newSize=(640, 480), save_path=save_path)
        except:
            print('{}/{}'.format(i,len(videos)))
            unsuccess_videos.append(video)
            continue
        print('{}/{}'.format(i,len(videos)))

    print("The unsuccessful videos for cropping are: {}".format(unsuccess_videos))

    