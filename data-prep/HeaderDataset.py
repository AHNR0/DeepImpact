"""
This code process the videos folder in soccerheader dataset. videos folder contains: vidoes/header and videos/nonheader
This code generates a csv file that contains list of all events in both header and nonheader folders and saves this csv file in the labels folder of the dataset.

Also the split data function can split dataset to validation and train sections for cross-validation manner.
"""
import os, shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import subprocess

class HeaderDataset:
    def __init__(self, path, path_test=None):
        self.dataset_folder = path
        self.dataset_testfolder = path
        self.labels = ['header', 'nonheader']
        # self.videos_folder=os.path.join(self.dataset_folder,'short_videos')
        self.videos_folder = os.path.join(self.dataset_folder, 'videos')
        self.labels_folder = os.path.join(self.dataset_folder, 'labels')
        self.dataset_lists = []   # two list that each list includes events of one label
        # for label in self.labels: 
        #     if not os.path.isfile(os.path.join(self.labels_folder,label+'_EventList.csv')):
        #         self.dataset_lists=[]
        #         self.gen_list_events()
        #         break
        #     else:
        #         self.dataset_lists.append(pd.read_csv(os.path.join(self.labels_folder,label+'_EventList.csv'),delimiter=','))
        self.data_folds = []  ## this is for cross_validation
    
    def gen_list_videos(self):
        ### this functions generates a list of all videos in the dataset for header and nonheader events:
        
        self.videos_list=pd.DataFrame()

        for label in self.labels:
            events = sorted(os.listdir(os.path.join(self.videos_folder, label)))
            cleaned_events = []
            cleaned_events_df = pd.DataFrame()
            for event in events:
                event = event[:event.find('mu', -9)-6]
                # event=event[:-9]
                if event not in cleaned_events:
                    cleaned_events.append(event)
            cleaned_events_df[label] = cleaned_events
            self.videos_list = pd.concat([self.videos_list, cleaned_events_df], axis=1)
            print(self.videos_list)
        ##check for testfolder
        if os.path.exists(os.path.join(self.videos_folder, 'test')):
            for label in self.labels:
                if os.path.exists(os.path.join(self.videos_folder, 'test', label)):
                    events = sorted(os.listdir(os.path.join(self.videos_folder, 'test', label)))
                    cleaned_events = []
                    cleaned_events_df = pd.DataFrame()
                    for event in events:
                        event = event[:-9]
                        if event not in cleaned_events:
                            cleaned_events.append(event)
                    cleaned_events_df['test/'+label] = cleaned_events
                    self.videos_list = pd.concat([self.videos_list, cleaned_events_df], axis=1)
        
        self.videos_list.to_csv(os.path.join(self.labels_folder, 'videosList.csv'), index=False)

        return self.videos_list

    def gen_list_events(self, shuffle=True):
        print('hello')
        for label in self.labels:
            dataset_df = pd.DataFrame(columns=['label', 'name'])
            events = sorted(os.listdir(os.path.join(self.videos_folder, label)))
            indexes = np.arange(len(events))
            if shuffle == True:
                np.random.shuffle(indexes)
            for i, index in enumerate(indexes):
                dataset_df.loc[i]=[label,events[index]]
            print(dataset_df)
            dataset_df.to_csv(os.path.join(self.labels_folder,label+'_EventList.csv'),index=False)
            self.dataset_lists.append(dataset_df)
    
    def split_data(self,folds=5,nonheader_length=10000,CV_mode='test_equal'):
        if len(self.dataset_lists) != 2:
            self.dataset_lists=[]
            self.gen_list_events()
        # print('*****' * 10)
        # print(len(self.dataset_lists[0]))
        fold_length=len(self.dataset_lists[0])/folds
        if CV_mode == 'test_equal':
            nonheader_fixed_train_index=np.array(list(range(len(self.dataset_lists[0]),nonheader_length)))
            # print(nonheader_fix_index)
            kf = KFold(n_splits=folds)
            kf.get_n_splits()
            count=0
            for train_index, test_index in kf.split(self.dataset_lists[0]):
                # print("TRAIN:", train_index, "TEST:", test_index)
                nonheader_train_index=np.append(train_index,nonheader_fixed_train_index)
                header_train=self.dataset_lists[0].loc[train_index]
                nonheader_train=self.dataset_lists[1].loc[nonheader_train_index]
                train_data=header_train.append(nonheader_train,ignore_index=True)
                train_data=train_data.sample(frac=1).reset_index(drop=True)
                
                header_val=self.dataset_lists[0].loc[test_index]
                nonheader_val=self.dataset_lists[1].loc[test_index]
                val_data=header_val.append(nonheader_val,ignore_index=True)
                val_data=val_data.sample(frac=1).reset_index(drop=True)
                
                self.data_folds.append([val_data,train_data])
                train_data.to_csv(os.path.join(self.labels_folder,'train_'+str(count)+'.csv'),index=False)
                val_data.to_csv(os.path.join(self.labels_folder,'val_'+str(count)+'.csv'), index=False)
                count += 1

        elif CV_mode == 'nonheader_EqualDivision':
            kf = KFold(n_splits=folds)
            kf.get_n_splits()
            count=0
            header_indexes=[]
            nonheader_indexes=[]
            for train_index, test_index in kf.split(self.dataset_lists[0]):
                header_indexes.append([train_index,test_index])
            for train_index, test_index in kf.split(self.dataset_lists[1].loc[:nonheader_length]):
                nonheader_indexes.append([train_index,test_index])

            for i in range(folds):
                header_train=self.dataset_lists[0].loc[header_indexes[i][0]]
                header_val=self.dataset_lists[0].loc[header_indexes[i][1]]

                nonheader_train=self.dataset_lists[1].loc[nonheader_indexes[i][0]]
                nonheader_val=self.dataset_lists[1].loc[nonheader_indexes[i][1]]
                
                train_data=header_train.append(nonheader_train,ignore_index=True)
                train_data=train_data.sample(frac=1).reset_index(drop=True)
                val_data=header_val.append(nonheader_val,ignore_index=True)
                val_data=val_data.sample(frac=1).reset_index(drop=True)

                self.data_folds.append([val_data,train_data])
                train_data.to_csv(os.path.join(self.labels_folder,'train_'+str(i)+'.csv'),index=False)
                val_data.to_csv(os.path.join(self.labels_folder,'val_'+str(i)+'.csv'), index=False)
        else:    
            pass
        return self.data_folds


    def gen_label(self,fold_instance,folds=5,nonheader_length=2460,CV_mode='test_equal'):
        if self.data_folds == []:
            self.split_data(folds,nonheader_length,CV_mode)
        with open(os.path.join(self.labels_folder,'header_label_map.txt')) as f:
            categories = f.readlines()
            categories = [c.strip().replace(' ', '_').replace('"', '').replace('(', '').replace(')', '').replace("'", '') for c in categories]
        assert len(set(categories)) == 2
        dict_categories = {}
        for i, category in enumerate(categories):
            dict_categories[category] = i
        print(dict_categories)

        files_input = ['kinetics_val.csv', 'kinetics_train.csv']
        files_input = self.data_folds[fold_instance]
        files_output = ['val_videofolder', 'train_videofolder']
        for (file_input, filename_output) in zip(files_input, files_output):
            count_cat = {k: 0 for k in dict_categories.keys()}
            # with open(os.path.join(label_path, filename_input)) as f:
            #     lines = f.readlines()[1:]
            folders = []
            idx_categories = []
            categories_list = []
            for i in range(len(file_input)):
            # for line in lines:
                # print('**'*30)
                # print(line)
                # line = line.rstrip()
                # print(line)
                # items = line.split(',')
                # print(items)
                # print(items[1] + '_' + items[2])
                # folders.append(items[1] + '_' + items[2])
                folders.append(file_input.iloc[i,1])
                # this_catergory = items[0].replace(' ', '_').replace('"', '').replace('(', '').replace(')', '').replace("'", '')
                this_category= file_input.iloc[i,0]
                categories_list.append(this_category)
                idx_categories.append(dict_categories[this_category])
                count_cat[this_category] += 1
            print(count_cat.values())
            print(max(count_cat.values()))
            print(folders[123])
            print(idx_categories[123])
            # print(categories_list)

            assert len(idx_categories) == len(folders)
            missing_folders = []
            output = []
            for i in range(len(folders)):
            # for i in range(10):
                # print('**'*50)
                curFolder = folders[i]
                # print(i)
                # print(curFolder)
                # print(categories_list[i])
                curIDX = idx_categories[i]
            #     # counting the number of frames in each video folders
                img_dir = os.path.join(self.videos_folder, categories_list[i], curFolder)
                # print(img_dir)
                if not os.path.exists(img_dir):
                    missing_folders.append(img_dir)
                    print(missing_folders)
                else:
                    dir_files = os.listdir(img_dir)
                    output.append('%s %d %d'%(os.path.join(categories_list[i], curFolder), len(dir_files), curIDX))
                    # print(output)
                print('%d/%d, missing %d'%(i, len(folders), len(missing_folders)))
            with open(os.path.join(self.labels_folder, filename_output+'.txt'),'w') as f:
                f.write('\n'.join(output))
            with open(os.path.join(self.labels_folder,'missing_'+ filename_output+'.txt'),'w') as f:
                f.write('\n'.join(missing_folders))

    def testset_gen(self, different_path, shuffle=True):
        self.testData=[]
        for label in self.labels:
            dataset_df=pd.DataFrame(columns=['label','name'])
            
            if different_path != '':
                events = os.listdir(os.path.join(different_path,label))
            else:
                events=os.listdir(os.path.join(self.videos_folder,'test',label))
            
            
            indexes=np.arange(len(events))
            if shuffle == True:
                np.random.shuffle(indexes)
            for i, index in enumerate(indexes):
                dataset_df.loc[i]=[label,events[index]]
                print(i)
            print(dataset_df)
            
            self.testData.append(dataset_df)
        
        test_list=self.testData[0].append(self.testData[1],ignore_index=True)
        if shuffle == True:
            test_list=test_list.sample(frac=1).reset_index(drop=True)               ###################### shuffling the whole set
        output = []
        dict_categories = {'nonheader':0,'header':1}
        if different_path == '':
            for i in range(len(test_list)):
                # print(i)
                dir_files = os.listdir(os.path.join(self.videos_folder,'test', test_list.iloc[i,0],test_list.iloc[i,1]))
                # print(dir_files)
                output.append('%s %d %d'%(os.path.join('test',test_list.iloc[i,0], test_list.iloc[i,1]), len(dir_files), dict_categories[test_list.iloc[i,0]]))
        else:
            for i in range(len(test_list)):
                # print(i)
                dir_files = os.listdir(os.path.join(different_path, test_list.iloc[i,0],test_list.iloc[i,1]))
                # print(dir_files)
                output.append('%s %d %d'%(os.path.join(test_list.iloc[i,0], test_list.iloc[i,1]), len(dir_files), dict_categories[test_list.iloc[i,0]]))
        print(output[0])
        with open(os.path.join(self.labels_folder, 'test_videofolder.txt'),'w') as f:
            f.write('\n'.join(output))
        

    
    
    def reduce_length(self, path,dir_path,final_length):
        """
        Reduces the number of extracted frames from each video. Use this to set the number of frames that should be used for each event. 
        'path': path to a folder containing events of one label. event1/frames, event2/frames, ...
        'final_length': the desired number of frames for each to be transfrred to the destination directory.  
        """
        half_length=int(final_length/2)
        videos=os.listdir(path)
        for i, event in enumerate(videos):
            print('*****' * 20)
            print('{}/{} videos'.format(i,len(videos)))
            event_path=os.path.join(path,event)
            dir_event_path=os.path.join(dir_path,event)
            if not os.path.exists(dir_event_path):
                os.mkdir(dir_event_path)
            else:
                try:
                    shutil.rmtree(dir_event_path)
                    os.mkdir(dir_event_path)
                except OSError as e:
                    print ("Error: %s - %s." % (e.filename, e.strerror))
            
            images_count=len(os.listdir(event_path))
            images=sorted(os.listdir(event_path))
            for index, j in enumerate(range(int(images_count/2)+1-half_length,int(images_count/2)+2+half_length)):
                image_name='img_'+(5-len(str(j)))*'0'+str(j)+'.jpg'
                target_image_name='img_'+(5-len(str(index+1)))*'0'+str(index+1)+'.jpg'
                cmd = 'cp \"{}\" \"{}\"\n'.format(os.path.join(event_path,image_name),os.path.join(dir_event_path,target_image_name))
                subprocess.call(cmd,shell=True)




if __name__ == '__main__':


    ###### path to the main folder of the dataset
    dataset_folder = ""

    ###### settings of train/val definition
    shuffle=False
    CV_division_mode ='nonheader_EqualDivision'
    nonheader_number = 32663
    n_fold = 5
    fold_instance = 0
    test_path = ''
    
    ###### define the dataset
    dataset=HeaderDataset(dataset_folder)
    _ = dataset.gen_list_videos()
    
    ###### create the list of header and nonheader events:
    dataset.gen_list_events(shuffle=shuffle)
    
    ###### create the folds of cross-validation:
    # nonheader_length: the maximum number of nonheader events that we want to use in the dataset.
    dataset.split_data(folds=n_fold,nonheader_length=nonheader_number,CV_mode=CV_division_mode)

    ##### create train and test data from one fold of the folds created for cross-validation: 
    dataset.gen_label(fold_instance=fold_instance, nonheader_length=nonheader_number,CV_mode=CV_division_mode)
    
    ### create the test_videofolder.txt file:
    dataset.testset_gen(test_path, shuffle=False)


    

    



