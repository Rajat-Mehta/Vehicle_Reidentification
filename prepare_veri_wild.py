import os
from shutil import copyfile
import patoolib
import pandas as pd 
import shutil

# You only need to change this line to your dataset download path
download_path = '../Datasets/VeRI-Wild'
images_path = '../Datasets/VeRI-Wild/images_new/images.part01'

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = os.path.join(download_path, 'pytorch')
if not os.path.isdir(save_path):
    os.mkdir(save_path)


vehicle_info_path = os.path.join(download_path, 'train_test_split', 'vehicle_info.txt')
data = pd.read_csv(vehicle_info_path, sep=";")
data['id'], data['image'] = data['id/image'].str.split('/', 1).str
data = data.drop(['id/image'], axis=1)
cols = data.columns.tolist()
cols.insert(0, cols.pop(cols.index('id')))
cols.insert(1, cols.pop(cols.index('image')))
vehicle_info = data.reindex(columns= cols)


#-----------------------------------------
#query
query_save_path = os.path.join(save_path, 'query')
query_image_list = os.path.join(download_path, 'train_test_split', 'test_10000_query.txt')
with open(query_image_list, "r") as q:
    lines = q.read().split('\n')

if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)

for img in lines:
    file_name = ''
    img_class = img.split('/')[0]
    img_name = img.split('/')[1]

    if not os.path.isdir(os.path.join(query_save_path, img_class)):
        os.mkdir(os.path.join(query_save_path, img_class))

    src_path = os.path.join(images_path, img_class, img_name + '.jpg')
    file_name = img_class + '_c' + str("{:03d}".format((vehicle_info['Camera ID'][vehicle_info['image'] == img_name]).iloc[0])) + '_' +img_name +'.jpg'
    dst_path = os.path.join(query_save_path, img_class, file_name)
    copyfile(src_path, dst_path)

#-----------------------------------------
#gallery
gallery_save_path = os.path.join(save_path, 'gallery')
gallery_image_list = os.path.join(download_path, 'train_test_split', 'test_10000.txt')
with open(gallery_image_list, "r") as q:
    lines = q.read().split('\n')

if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)
i=0
print("Gallery length", len(lines))
for img in lines:
    if i % 1000 ==0:
        print(i)
    i+=1
    file_name = ''
    img_class = img.split('/')[0]
    img_name = img.split('/')[1]

    if not os.path.isdir(os.path.join(gallery_save_path, img_class)):
        os.mkdir(os.path.join(gallery_save_path, img_class))

    src_path = os.path.join(images_path, img_class, img_name + '.jpg')
    file_name = img_class + '_c' + str("{:03d}".format((vehicle_info['Camera ID'][vehicle_info['image'] == img_name]).iloc[0])) + '_' + img_name +'.jpg'
    dst_path = os.path.join(gallery_save_path, img_class, file_name)
    copyfile(src_path, dst_path)

# splitting query and gallery set into 3 parts, which makes it easy to the model (because of memory issues)
data_parts = ['0-3000', '3001-6000', '6001-10000']
for part in data_parts:
    print("Splitting query part ", part)
    #-----------------------------------------
    #query_3000
    query_save_path = os.path.join(save_path, 'query_' + part)
    query_image_list = os.path.join(download_path, 'train_test_split', 'test_query_' + part + '.txt')
    with open(query_image_list, "r") as q:
        lines = q.read().split('\n')

    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)

    for img in lines:
        file_name = ''
        img_class = img.split('/')[0]
        img_name = img.split('/')[1]

        if not os.path.isdir(os.path.join(query_save_path, img_class)):
            os.mkdir(os.path.join(query_save_path, img_class))

        
        file_name = img_class + '_c' + str("{:03d}".format((vehicle_info['Camera ID'][vehicle_info['image'] == img_name]).iloc[0])) + '_' +img_name +'.jpg'
        src_path = os.path.join(save_path, 'query', img_class, file_name)
        dst_path = os.path.join(query_save_path, img_class, file_name)
        copyfile(src_path, dst_path)

    print("Splitting gallery part ", part)
    #-----------------------------------------
    #gallery_3000
    gallery_save_path = os.path.join(save_path, 'gallery_' + part)
    gallery_image_list = os.path.join(download_path, 'train_test_split', 'test_' + part + '.txt')
    with open(gallery_image_list, "r") as q:
        lines = q.read().split('\n')

    if not os.path.isdir(gallery_save_path):
        os.mkdir(gallery_save_path)
    i=0
    print("Gallery length", len(lines))
    for img in lines:
        if i % 1000 ==0:
            print(i)
        i+=1
        file_name = ''
        img_class = img.split('/')[0]
        img_name = img.split('/')[1]

        if not os.path.isdir(os.path.join(gallery_save_path, img_class)):
            os.mkdir(os.path.join(gallery_save_path, img_class))

        file_name = img_class + '_c' + str("{:03d}".format((vehicle_info['Camera ID'][vehicle_info['image'] == img_name]).iloc[0])) + '_' + img_name +'.jpg'
        src_path = os.path.join(save_path, 'gallery', img_class, file_name)
        dst_path = os.path.join(gallery_save_path, img_class, file_name)
        copyfile(src_path, dst_path)


#-----------------------------------------
#train
train_save_path = os.path.join(save_path, 'train')
train_image_list = os.path.join(download_path, 'train_test_split', 'train_list.txt')
with open(train_image_list, "r") as q:
    lines = q.read().split('\n')

if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
i=0
print("Train length", len(lines))
for img in lines:
    if i % 1000 ==0:
        print(i)
    i+=1
    file_name = ''
    img_class = img.split('/')[0]
    img_name = img.split('/')[1]

    if not os.path.isdir(os.path.join(train_save_path, img_class)):
        os.mkdir(train_save_path + '/' + img_class)

    src_path = os.path.join(images_path, img_class, img_name + '.jpg')
    file_name = img_class + '_c' + str("{:03d}".format((vehicle_info['Camera ID'][vehicle_info['image'] == img_name]).iloc[0])) + '_' + img_name +'.jpg'
    dst_path = os.path.join(train_save_path, img_class, file_name)
    copyfile(src_path, dst_path)


#-----------------------------------------
#train_all
train_all_path = os.path.join(save_path, 'train_all')
shutil.copytree(train_save_path, train_all_path)

#-----------------------------------------
#val
val_save_path = os.path.join(save_path, 'val')
train_save_path = os.path.join(save_path, 'train')

if not os.path.isdir(val_save_path):
    os.mkdir(val_save_path)

for subdir, dirs, files in os.walk(train_save_path):
    for item in dirs:
        for subdir, dirs, files in os.walk(os.path.join(train_save_path, item)):
            if not os.path.isdir(os.path.join(val_save_path, item)):
                os.mkdir(os.path.join(val_save_path, item))
            shutil.move(os.path.join(os.path.join(train_save_path, item), files[0]), os.path.join(os.path.join(val_save_path, item),files[0])) 
            break
