import numpy as np
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import math
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from tqdm import tqdm
def make_dataset(image_list, label_list, au_relation=None):
   
    len_ = len(image_list)
    choice=len(label_list)
    if choice==0:
        images = [(image_list[i].strip()) for i in range(len_)]
        return images
    if au_relation is not None:
        images = [(image_list[i].strip(),  label_list[i, :],au_relation[i,:]) for i in range(len_)]
    else:
        images = [(image_list[i].strip(),  label_list[i, :]) for i in range(len_)]
    return images


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)
def erase(image):
    pos_x=np.random.randint(0,224-80)
    pos_y=np.random.randint(0,224-80)
    mask_size=80
    image[pos_x:pos_x + mask_size, pos_y:pos_y + mask_size] = 0
    image=Image.fromarray(image.astype('uint8'))
    return image
def sunny(image):
    radius=80
    pos_x=np.random.randint(0,224-radius)
    pos_y=np.random.randint(0,224-radius)
    strength = 100
    for j in range(pos_y - radius, pos_y + radius):
        for i in range(pos_x-radius, pos_x+radius):
            distance = math.pow((pos_x - i), 2) + math.pow((pos_y - j), 2)
            distance = np.sqrt(distance)
            if distance < radius:
                result = 1 - distance / radius
                result = result*strength
                image[i, j, 0] = min((image[i, j, 0] + result),255)
                image[i, j, 1] = min((image[i, j, 1] + result),255)
                image[i, j, 2] = min((image[i, j, 2] + result),255)
    image=Image.fromarray(image.astype('uint8'))
    return image

def shallow(image):
    radius=80
    pos_x=np.random.randint(0,224-radius)
    pos_y=np.random.randint(0,224-radius)
    strength = 100
    for j in range(pos_y - radius, pos_y + radius):
        for i in range(pos_x-radius, pos_x+radius):

            distance = math.pow((pos_x - i), 2) + math.pow((pos_y - j), 2)
            distance = np.sqrt(distance)
            if distance < radius:
                result = 1 - distance / radius
                result = result*strength
                # print(result)
                image[i, j, 0] = max((image[i, j, 0] - result),0)
                image[i, j, 1] = max((image[i, j, 1] - result),0)
                image[i, j, 2] = max((image[i, j, 2] - result),0)
    image=Image.fromarray(image.astype('uint8'))
    return image
def AddNoise(image):
    image=np.array(image)
    prob=np.random.randint(0,100)
    if prob>50:#挖空
        
        image=erase(image)
    else:#光照或者阴影
        prob1=np.random.randint(0,100)
        if prob1>50:
            
            image=sunny(image)
        else:
            
            image=shallow(image)
    return image
class BP4D(Dataset):
    def __init__(self, root_path, train, fold = 1, transform=None, crop_size = 224, stage=1, loader=default_loader):

        assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        self._root_path = root_path
        self._train = train
        self._stage = stage
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        #self.img_folder_path = os.path.join(root_path,'img')
        self.img_folder_path=''
        list='list'
        if self._train==1:
            # img
            print("datasetTrain1")
            train_image_list_path = os.path.join(root_path, list, 'BP4D_train_img_path_fold' + str(fold) +'.txt')
            train_image_list = open(train_image_list_path).readlines()
            train_image_list=[x.replace('_AU','') for x in train_image_list]
            print("images_num:")
            print(len(train_image_list))
            # img labels
            train_label_list_path = os.path.join(root_path, list, 'BP4D_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)

            # AU relation
            if self._stage == 2:
                au_relation_list_path = os.path.join(root_path, list, 'BP4D_train_AU_relation_fold' + str(fold) + '.txt')
                au_relation_list = np.loadtxt(au_relation_list_path)
                self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list)
            else:
                self.data_list = make_dataset(train_image_list, train_label_list)
        if self._train==2:
            print("datasetTrain2")
            # img
            train_image_list_path = os.path.join(root_path, list, 'BP4D_train_img_path_fold' + str(fold) +'.txt')
            train_image_list = open(train_image_list_path).readlines()
            train_image_list=[x.replace('_AU','') for x in train_image_list]
            print("images_num:")
            print(len(train_image_list))
            # img labels
            train_label_list_path = os.path.join(root_path, list, 'BP4D_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)

            # AU relation
            if self._stage == 2:
                au_relation_list_path = os.path.join(root_path, list, 'BP4D_train_AU_relation_fold' + str(fold) + '.txt')
                au_relation_list = np.loadtxt(au_relation_list_path)
                self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list)
            else:
                self.data_list = make_dataset(train_image_list, train_label_list)
        if self._train==0:
            print("datasetTest1")
            # img
            test_image_list_path = os.path.join(root_path, list, 'BP4D_test_img_path_fold' + str(fold) + '.txt')
            test_image_list = open(test_image_list_path).readlines()
            print("images_num:")
            print(len(test_image_list))
            test_image_list=[x.replace('_AU','') for x in test_image_list]
            # img labels
            test_label_list_path = os.path.join(root_path, list, 'BP4D_test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            self.data_list = make_dataset(test_image_list, test_label_list)

        
    def __getitem__(self, index):
        resizex=256
        resizey=256
        if self._stage == 2 and self._train:
            img, label, au_relation = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))
            img=img.resize((resizex,resizey))
            w, h = img.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)
            flip = random.randint(0, 1)
            if self._transform is not None:
                img = self._transform(img, flip, offset_x, offset_y)
            return img, label, au_relation
        else:
            img, label = self.data_list[index]
            path=img
            
            img = self.loader(os.path.join(self.img_folder_path, img))
            img=img.resize((resizex,resizey))#224  224  3
            if self._train:
                w, h = img.size
                offset_y = random.randint(0, h - self.crop_size)
                offset_x = random.randint(0, w - self.crop_size)
                flip = random.randint(0, 1)
                if self._transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
            else:
                if self._transform is not None:
                    img = self._transform(img)
            ll=[
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,0,0,0],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]
            ]
            p_for_6=torch.from_numpy(np.array(ll))
            return img, label,path,p_for_6

    def __len__(self):
        return len(self.data_list)
class NoiseBP4D(Dataset):
    def __init__(self, root_path, train, fold = 1, transform=None, crop_size = 224, stage=1, loader=default_loader):

        assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        self._root_path = root_path
        self._train = train
        self._stage = stage
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        #self.img_folder_path = os.path.join(root_path,'img')
        self.img_folder_path=''
        if self._train==1:
            # img
            print("datasetTrain1")
            train_image_list_path = os.path.join(root_path, 'list', 'BP4D_train_img_path_fold' + str(fold) +'.txt')
            train_image_list = open(train_image_list_path).readlines()
            train_image_list=[x.replace('_AU','') for x in train_image_list]
            # img labels
            train_label_list_path = os.path.join(root_path, 'list', 'BP4D_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)

            # AU relation
            if self._stage == 2:
                au_relation_list_path = os.path.join(root_path, 'list', 'BP4D_train_AU_relation_fold' + str(fold) + '.txt')
                au_relation_list = np.loadtxt(au_relation_list_path)
                self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list)
            else:
                self.data_list = make_dataset(train_image_list, train_label_list)
        if self._train==2:
            print("datasetTrain2")
            # img
            train_image_list_path = os.path.join(root_path, 'list', 'BP4D_train_img_path_fold' + str(fold) +'.txt')
            train_image_list = open(train_image_list_path).readlines()
            train_image_list=[x.replace('_AU','') for x in train_image_list]
            # img labels
            train_label_list_path = os.path.join(root_path, 'list', 'BP4D_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)

            # AU relation
            if self._stage == 2:
                au_relation_list_path = os.path.join(root_path, 'list', 'BP4D_train_AU_relation_fold' + str(fold) + '.txt')
                au_relation_list = np.loadtxt(au_relation_list_path)
                self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list)
            else:
                self.data_list = make_dataset(train_image_list, train_label_list)
        if self._train==0:
            print("datasetTest1")
            # img
            test_image_list_path = os.path.join(root_path, 'list', 'BP4D_test_img_path_fold' + str(fold) + '.txt')
            test_image_list = open(test_image_list_path).readlines()
            test_image_list=[x.replace('_AU','') for x in test_image_list]
            # img labels
            test_label_list_path = os.path.join(root_path, 'list', 'BP4D_test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            self.data_list = make_dataset(test_image_list, test_label_list)

    def __getitem__(self, index):
        if self._stage == 2 and self._train:
            img, label, au_relation = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))
            img=img.resize((224,224))
            ########add  noise
            img=AddNoise(img)
            ##################
            w, h = img.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)
            flip = random.randint(0, 1)
            if self._transform is not None:
                img = self._transform(img, flip, offset_x, offset_y)
            return img, label, au_relation
        else:
            img, label = self.data_list[index]
            path=img
            
            img = self.loader(os.path.join(self.img_folder_path, img))
            img=img.resize((224,224))#224  224  3
            ########add  noise
            img=AddNoise(img)
            ##################
            if self._train:
                w, h = img.size
                offset_y = random.randint(0, h - self.crop_size)
                offset_x = random.randint(0, w - self.crop_size)
                flip = random.randint(0, 1)
                if self._transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
            else:
                if self._transform is not None:
                    img = self._transform(img)
            ll=[
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,0,0,0],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]
            ]
            p_for_6=torch.from_numpy(np.array(ll))
            return img, label,path,p_for_6

    def __len__(self):
        return len(self.data_list)
class vox(Dataset):
    def __init__(self, root_path, transform=None, crop_size = 224, loader=default_loader):
        self._root_path = root_path
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        #self.img_folder_path = os.path.join(root_path,'img')
        self.img_folder_path=''
        print("datasetTrain1")
        train_image_list_path ='./BYOL/vox_train_list_remove_id.txt'
        #/data2/vox_process/EmoVoxCeleb/vox_align/processed_align/Armand_Assante/AEtSynxqitc
        #读文件夹名称
        train_image_list = open(train_image_list_path).readlines()
        images=[]
        self.images_point=[]
        #随机选图片
        for dir_path in train_image_list:
            dir_path='/data2/vox_process/EmoVoxCeleb/vox_align/processed_align/'+dir_path.replace('\n','')
            file_list = os.listdir(dir_path)
            file_name = random.choice(file_list)
            file_path = os.path.join(dir_path,file_name)

            img_list = os.listdir(file_path)
            img_name = random.choice(img_list)
            images.append(file_path+'/'+img_name)
            ######point
            point_path=file_path+'/'+img_name
           # print(point_path)
            point_path=point_path.replace('.jpg','.txt').replace('/data2/vox_process/EmoVoxCeleb/vox_align/processed_align/','./BYOL/landmarks/')
           # print(point_path)
            point=np.loadtxt(point_path)
            self.images_point.append(point)
            ####
            img_list.remove(img_name)
            img_name2 = random.choice(img_list)
            images.append(file_path+'/'+img_name2)
            ######point
            point_path2=file_path+'/'+img_name2
            point_path2=point_path2.replace('.jpg','.txt').replace('/data2/vox_process/EmoVoxCeleb/vox_align/processed_align/','./BYOL/landmarks/')
            point2=np.loadtxt(point_path2)
            self.images_point.append(point2)
            ####
            # while 1:
            #     i=1
        self.data_list = make_dataset(images,[])
        print("dataset_num:")
        print(len(self.data_list))
        #train_image_list=[x.replace('_AU','') for x in train_image_list]

            

    def __getitem__(self, index):
        img = self.data_list[index]
        point=self.images_point[index]
        img = self.loader(os.path.join(self.img_folder_path, img))
        resizex=256
        resizey=256
        img=img.resize((resizex,resizey))
        w, h = img.size
        offset_y = random.randint(0, h - self.crop_size)
        offset_x = random.randint(0, w - self.crop_size)
        flip = random.randint(0, 1)
        if self._transform is not None:
            #img = self._transform(img, flip, offset_x, offset_y)
            img = self._transform(img)
        # print(img.size())
        # to_pil = ToPILImage()
        # processed_pil_img = to_pil(img)
        # processed_pil_img.save('./test.png','PNG')
        # while 1:
        #     continue
        return img,point

    def __len__(self):
        return len(self.data_list)
class DISFA(Dataset):
    def __init__(self, root_path, train, fold = 1, transform=None, crop_size = 224, stage=1, loader=default_loader):

        assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        self._root_path = root_path
        self._train = train
        self._stage = stage
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        #self.img_folder_path = os.path.join(root_path,'img')
        self.img_folder_path=''
        list='list'
        if self._train==1:
            # img
            print("datasetTrain1")
            train_image_list_path = os.path.join(root_path, list, 'DISFA_train_img_path_fold' + str(fold) +'.txt')
            train_image_list = open(train_image_list_path).readlines()
            train_image_list=[x.replace('_AU','') for x in train_image_list]
            # img labels
            train_label_list_path = os.path.join(root_path, list, 'DISFA_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)

            # AU relation
            if self._stage == 2:
                au_relation_list_path = os.path.join(root_path, list, 'DISFA_train_AU_relation_fold' + str(fold) + '.txt')
                au_relation_list = np.loadtxt(au_relation_list_path)
                self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list)
            else:
                
                self.data_list = make_dataset(train_image_list, train_label_list)
        if self._train==2:
            print("datasetTrain2")
            # img
            train_image_list_path = os.path.join(root_path, list, 'DISFA_train_img_path_fold' + str(fold) +'.txt')
            train_image_list = open(train_image_list_path).readlines()
            train_image_list=[x.replace('_AU','') for x in train_image_list]
            # img labels
            train_label_list_path = os.path.join(root_path, list, 'DISFA_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)

            # AU relation
            if self._stage == 2:
                au_relation_list_path = os.path.join(root_path, list, 'DISFA_train_AU_relation_fold' + str(fold) + '.txt')
                au_relation_list = np.loadtxt(au_relation_list_path)
                self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list)
            else:
                self.data_list = make_dataset(train_image_list, train_label_list)
        if self._train==0:
            print("datasetTest1")
            # img
            test_image_list_path = os.path.join(root_path, list, 'DISFA_test_img_path_fold' + str(fold) + '.txt')
            test_image_list = open(test_image_list_path).readlines()
            test_image_list=[x.replace('_AU','') for x in test_image_list]
            # img labels
            test_label_list_path = os.path.join(root_path, list, 'DISFA_test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            self.data_list = make_dataset(test_image_list, test_label_list)

    def __getitem__(self, index):
        if self._stage == 2 and self._train:
            img, label, au_relation = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))
            img=img.resize((224,224))
            ########add  noise
            #img=AddNoise(img)
            ##################
            w, h = img.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)
            flip = random.randint(0, 1)
            if self._transform is not None:
                img = self._transform(img, flip, offset_x, offset_y)
            return img, label, au_relation
        else:
            img, label = self.data_list[index]
            path=img
            
            img = self.loader(os.path.join(self.img_folder_path, img))
            img=img.resize((224,224))
            ########add  noise
           # img=AddNoise(img)
            ##################
            if self._train:
                w, h = img.size
                offset_y = random.randint(0, h - self.crop_size)
                offset_x = random.randint(0, w - self.crop_size)
                flip = random.randint(0, 1)
                if self._transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
            else:
                if self._transform is not None:
                    img = self._transform(img)
            ll=[
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,0,0,0],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]
            ]
            p_for_6=torch.from_numpy(np.array(ll))
            return img, label,path,p_for_6

    def __len__(self):
        return len(self.data_list)
class initDISFA(Dataset):
    def __init__(self, root_path, train=True, fold = 1, transform=None, crop_size = 224, stage=1, loader=default_loader):

        assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        self._root_path = root_path
        self._train = train
        self._stage = stage
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.img_folder_path = os.path.join(root_path,'img')
        if self._train:
            # img
            train_image_list_path = os.path.join(root_path, 'list', 'DISFA_train_img_path_fold' + str(fold) + '.txt')
            train_image_list = open(train_image_list_path).readlines()
            # img labels
            train_label_list_path = os.path.join(root_path, 'list', 'DISFA_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)

            # AU relation
            if self._stage == 2:
                au_relation_list_path = os.path.join(root_path, 'list', 'DISFA_train_AU_relation_fold' + str(fold) + '.txt')
                au_relation_list = np.loadtxt(au_relation_list_path)
                self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list)
            else:
                self.data_list = make_dataset(train_image_list, train_label_list)

        else:
            # img
            test_image_list_path = os.path.join(root_path, 'list', 'DISFA_test_img_path_fold' + str(fold) + '.txt')
            test_image_list = open(test_image_list_path).readlines()

            # img labels
            test_label_list_path = os.path.join(root_path, 'list', 'DISFA_test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            self.data_list = make_dataset(test_image_list, test_label_list)

    def __getitem__(self, index):
        if self._stage == 2 and self._train:
            img, label, au_relation = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))

            w, h = img.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)
            flip = random.randint(0, 1)
            if self._transform is not None:
                img = self._transform(img, flip, offset_x, offset_y)
            return img, label, au_relation
        else:
            img, label = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path,img))

            if self._train:
                w, h = img.size
                offset_y = random.randint(0, h - self.crop_size)
                offset_x = random.randint(0, w - self.crop_size)
                flip = random.randint(0, 1)
                if self._transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
            else:
                if self._transform is not None:
                    img = self._transform(img)
            return img, label

    def __len__(self):
        return len(self.data_list)




# path='./data/imgs/BP4D/2F02_03/00000.png'
# img=pil_loader(path)
# img=AddNoise(img)
# # img=shallow(np.array(img))
# img.save('./tool/1.png')