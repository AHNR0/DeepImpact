import numpy as np
from scipy import linalg
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os, math

from crop_dataset import video_BallData


#### for printing the images:
def show_boxes(img,boundingBox,label):
    plt.imshow(img)
    # print(img.shape)
    height,width,_= img.shape
    bounding_box=np.copy(boundingBox)
    bounding_box[:,[0,2]]=bounding_box[:,[0,2]]*width
    bounding_box[:,[1,3]]=bounding_box[:,[1,3]]*height
    ax=plt.gca()
    for i in range(bounding_box.shape[0]):
        center=[bounding_box[i,0]-bounding_box[i,2]/2,bounding_box[i,1]-bounding_box[i,3]/2]
        if label=='player':
            rectangle=patches.Rectangle(center,bounding_box[i,2],bounding_box[i,3],linewidth=1,edgecolor='r')
        if label=='ball':
            rectangle=patches.Rectangle(center,bounding_box[i,2],bounding_box[i,3],linewidth=1,edgecolor='b')
        ax.add_patch(rectangle)

def show_video_images(video_data,row=3,show_bb=False):
    imgs_path=video_data['imgs_path']
    imgs_num = len(video_data['labels'].keys())
    column = math.ceil(imgs_num / row)
    print(imgs_path)
    print(imgs_num)
    print(column)
    fig=plt.figure()
    for item, key in enumerate(video_data['labels'].keys()):
        fig.add_subplot(row,column,item+1)
        img = plt.imread(os.path.join(imgs_path,key+'.jpg'))
        if show_boxes:
            show_boxes(img,video_data['labels'][key]['bb_info'],'ball')
            plt.axis('off')
        else:
            plt.imshow(img)
            plt.axis('off')
    fig.tight_layout()
    plt.show()

### for Kalman filtering:

def video_ball_number(video_data):
    ball_count=0
    first_ball_exists = False
    for item in video_data['labels'].keys():
        if video_data['labels'][item]['ball_number'] >0: 
            ball_count += 1 
    if video_data['labels']['img_00001']['ball_number'] > 0: first_ball_exists = True

    return ball_count, first_ball_exists

def cal_threshold(bb_size,threshold):
    return threshold * bb_size[0]

def kalman_filter(imgs_folder,label_folder):
    # read the ball information data:
    video_data = video_BallData(imgs_folder,label_folder)
    # print(video_data)
    ball_count, first_ball_exists = video_ball_number(video_data)
    # print(video_data)
    if ball_count >= 4 and first_ball_exists:

        
        ##initialization of the filter:
        # if first_ball_exists:
        bb_info = video_data['labels']['img_00001']['bb_info'][-1,:]
        AR = bb_info[2]/bb_info[3]

        x_k = np.array([[bb_info[0]],[bb_info[1]],[0.0],[0.0]])
        

        A = np.eye(4)
        A[0,2]=1
        A[1,3]=1

        H =np.array([[1.,0.,0.,0.],[0.,1.,0.,0.]])
        P = 0.1 * np.eye(4)
        R = 0.01 * np.eye(2)     
        Q =  0.000001 * np.eye(4)

        for step, item in enumerate(video_data['labels'].keys()):
            print(f'##################### step {step+1}')
            x_k = A @ x_k
            P = (A @ P @ np.transpose(A)) + Q
            temp= (H @ P @ np.transpose(H)) + R
            S = linalg.inv(temp)
            K = P @ np.transpose(H) @ S
            P = (np.ones((4,4))- K @ H) @ P
            # print(f'predicted x_k:\n {x_k}')            
            
            ## calculate the square root of the innovation covariance matrix:
            r = linalg.sqrtm(S)
            
            r = 5 * r.diagonal()
            print(r)
            is_outlier = False
            ## choose measurement:
            ## there is no ball detected, so we assume the prediction is the same as the measurement
            if video_data['labels'][item]['ball_number'] == 0:  
                print('No ball was detected in this frame.')
                print(f'predicted locations: {np.reshape(x_k[:4,:],(4,))}')
                print(f'square root of the covariance matrix: {r}')
                innovation = np.zeros((2,1))
            ## There is one or more ball detected: if one of them is close to the first detecten, use that, otherwise use the prediction as measurement
            else:
                ## check the distance between the ball and the prediction:
                distance=np.linalg.norm(video_data['labels'][item]['bb_info'][:,:2]-np.reshape(x_k[:2,:],(1,2)),axis=1)
                measurement = video_data['labels'][item]['bb_info'][np.argmin(distance),:]
                measurement = np.array([[measurement[0]],[measurement[1]]])
                # measurement = np.array([[measurement[0]],[measurement[1]],[measurement[2]*measurement[3]]])
               
                ### outlier rejection:
                
                ### threshold based on the square root of the innovation covariance matrix:
                # if innovation[0,0] > r[0] or innovation[1,0] > r[1]:
                #     innovation = np.zeros((4,1))
                # else:
                #     innovation = measurement - H @ x_k
                
                ### threshold based on the bb size:
                if distance.min() < 5*bb_info[2]:
                    innovation = measurement - H @ x_k
                else:
                    innovation = np.zeros((2,1))

            print(x_k)
            print(K)

            print(innovation)
            x_k = x_k + K @ innovation
            print(x_k)
            video_data['labels'][item]['ball_number'] = 1
            video_data['labels'][item]['bb_info']=np.array([[x_k[0,0],x_k[1,0],bb_info[2],bb_info[3]]])
                
                

        return video_data
    else:
        return video_data

if __name__ == "__main__":

    #### checking one single video:
    game_name = ""
    imgs_folder = ""
    label_folder = ""

    video_data=video_BallData(imgs_folder,label_folder)
    print(video_data)
    # show_video_images(video_data,show_bb=True)

    
    video_data = kalman_filter(imgs_folder,label_folder)
    show_video_images(video_data, show_bb=True)


    