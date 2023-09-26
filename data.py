import numpy as np 
import matplotlib.pyplot as plt 
 
#Load the 5 files    
a0 = np.load('/home/pankhil/Desktop/dtu_mlops/data/corruptmnist/train_0.npz')
a1 = np.load('/home/pankhil/Desktop/dtu_mlops/data/corruptmnist/train_1.npz')
a2 = np.load('/home/pankhil/Desktop/dtu_mlops/data/corruptmnist/train_2.npz')
a3 = np.load('/home/pankhil/Desktop/dtu_mlops/data/corruptmnist/train_3.npz')
a4 =np.load('/home/pankhil/Desktop/dtu_mlops/data/corruptmnist/train_4.npz')

#check how many arrays these files have 
a = [a0,a1,a2,a3,a4]
#print(a0.files)
#print(a1.files)
#print(a2.files)
#print(a3.files)
#print(a4.files)
#merge each of arrays of these files 
arr_0 = np.array([a0['images'],a1['images'],a2['images'],a3['images'],a4['images']])
arr_1 = np.array([a0['labels'],a1['labels'],a2['labels'],a3['labels'],a4['labels']])
arr_2 = np.array([a0['allow_pickle'],a1['allow_pickle'],a2['allow_pickle'],a3['allow_pickle'],a4['allow_pickle']])


np.savez('data_new.npz', arr_0,arr_1,arr_2)
data  = np.load('/home/pankhil/Desktop/dtu_mlops/data/corruptmnist/data_new.npz')
print(data.files)
print(a0.files)
img = a0['images']
print(img.shape)
img1 = torch.from_numpy(img)