import os
import pandas as pd

labels_folder = ''  # path to the labels folder of the dataset
videos_folder = ''  # path to the videos folder of the dataset
fold_number = '4'  # the fold number to be saved

train_csv = pd.read_csv(os.path.join(labels_folder, 'train_' + fold_number + '.csv'), delimiter=',', header=0)
val_csv = pd.read_csv(os.path.join(labels_folder, 'val_' + fold_number + '.csv'), delimiter=',', header=0)

with open(os.path.join(labels_folder, 'header_label_map.txt')) as f:
    categories = f.readlines()
    categories = [c.strip().replace(' ', '_').replace('"', '').replace('(', '').replace(')', '').replace("'", '') for c
                  in categories]
assert len(set(categories)) == 2
dict_categories = {}
for i, category in enumerate(categories):
    dict_categories[category] = i
print(dict_categories)

files_input = ['kinetics_val.csv', 'kinetics_train.csv']
files_input = [val_csv, train_csv]
files_output = ['val_videofolder', 'train_videofolder']
for (file_input, filename_output) in zip(files_input, files_output):
    count_cat = {k: 0 for k in dict_categories.keys()}
    # with open(os.path.join(label_path, filename_input)) as f:
    #     lines = f.readlines()[1:]
    folders = []
    idx_categories = []
    categories_list = []
    for i in range(len(file_input)):

        folders.append(file_input.iloc[i, 1])
        # this_catergory = items[0].replace(' ', '_').replace('"', '').replace('(', '').replace(')', '').replace("'", '')
        this_category = file_input.iloc[i, 0]
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
        img_dir = os.path.join(videos_folder, categories_list[i], curFolder)
        # print(img_dir)
        if not os.path.exists(img_dir):
            missing_folders.append(img_dir)
            print(missing_folders)
        else:
            dir_files = os.listdir(img_dir)
            output.append('%s %d %d' % (os.path.join(categories_list[i], curFolder), len(dir_files), curIDX))
            # print(output)
        print('%d/%d, missing %d' % (i, len(folders), len(missing_folders)))
    with open(os.path.join(labels_folder, filename_output + '.txt'), 'w') as f:
        f.write('\n'.join(output))
    with open(os.path.join(labels_folder, 'missing_' + filename_output + '.txt'), 'w') as f:
        f.write('\n'.join(missing_folders))
