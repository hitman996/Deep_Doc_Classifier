import os
from PIL import Image
import shutil, random
import numpy as np
import pandas as pd

def to3_channel(train_dir):
    for folders in os.listdir(train_dir):
        current_dir = os.path.join(train_dir,folders)
        if(os.path.isdir(current_dir)):
            for files in os.listdir(current_dir):
                if(files.endswith('.tif')):
                    img = Image.open(os.path.join(current_dir,files))
                    imarray = np.array(img)
                    stacked_img = np.stack((imarray,)*3, axis=-1)
                    three_channel_image = Image.fromarray(stacked_img,'RGB')
                    three_channel_image.save(os.path.join(current_dir,files), "JPEG")
                    
def create_train_test(train_dir,test_dir):
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    for folders in os.listdir(train_dir):
        current_dir = os.path.join(train_dir,folders)
        test_folders = os.path.join(test_dir,folders)
        
        if not os.path.exists(test_folders):
            os.mkdir(test_folders)
            
        if(os.path.isdir(current_dir)):
            total_files = 0
            for files in os.listdir(current_dir):
                if(files.endswith('.tif')):
                    total_files+=1
                    
            for i in range(0,int(.2*total_files)):
                test_image = random.choice(os.listdir(current_dir))
                
                src_path = os.path.join(current_dir, test_image)
                dst_path = os.path.join(test_folders, test_image)
                try:
                    shutil.move(src_path, dst_path)
                except IOError as e:
                    print('Unable to copy file {} to {}'.format(src_path, dst_path))
                except:
                    print('When try copy file {} to {}, unexpected error: {}'.format(src_path, dst_path, sys.exc_info()))

def label_creation(train_dir, test_dir):
    class_names = ['ADVE','Email','Form','Letter','Memo','News','Note','Report','Resume','Scientific']
    if os.path.exists(train_dir):
        classes = list()
        labels = list()
        names = list()
        for folders in os.listdir(train_dir):
            src_dir = os.path.join(train_dir,folders)
            for files in os.listdir(src_dir):
                if(files.endswith('.tif')):
                    if folders in class_names:
                        labels.append(class_names.index(folders))
                        names.append(files[:-4])
                        classes.append(folders)
        df_train = pd.DataFrame({'Name':names,'Class':classes, 'Labels':labels})
    if os.path.exists(test_dir):
        classes = list()
        labels = list()
        names = list()
        for folders in os.listdir(test_dir):
            src_dir = os.path.join(test_dir,folders)
            for files in os.listdir(src_dir):
                if(files.endswith('.tif')):
                    if folders in class_names:
                        labels.append(class_names.index(folders))
                        names.append(files[:-4])
                        classes.append(folders)
        df_test = pd.DataFrame({'Name':names,'Class':classes, 'Labels':labels})
        
    training_set = 'Data//train_set.csv'
    testing_set = 'Data//test_set.csv'
    
    df_train.to_csv(training_set, sep=',', encoding='utf-8',index=False)
    df_test.to_csv(testing_set, sep=',', encoding='utf-8',index = False)

def data_set_for_bayesian_nn(train_dir, test_dir):
    train_dir_new = os.path.join('Data','Bayesian_Data','Train')
    test_dir_new = os.path.join('Data','Bayesian_Data','Test')
    
    for folders in os.listdir(train_dir):
        src_dir = os.path.join(train_dir,folders)
        for files in os.listdir(src_dir):
            if(files.endswith('.tif')):
                src_file = os.path.join(src_dir, files)
                des_dir = train_dir_new
                try:
                    shutil.copy(src_file, des_dir)
                except IOError as e:
                    print('Unable to copy file {} to {}'.format(src_path, dst_path))
                except:
                    print('When try copy file {} to {}, unexpected error: {}'.format(src_path, dst_path, sys.exc_info()))
     
    for folders in os.listdir(test_dir):
        src_dir = os.path.join(test_dir,folders)
        for files in os.listdir(src_dir):
            if(files.endswith('.tif')):
                src_file = os.path.join(src_dir, files)
                des_dir = test_dir_new
                try:
                    shutil.copy(src_file, des_dir)
                except IOError as e:
                    print('Unable to copy file {} to {}'.format(src_path, dst_path))
                except:
                    print('When try copy file {} to {}, unexpected error: {}'.format(src_path, dst_path, sys.exc_info()))
