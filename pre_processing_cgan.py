import os
import shutil
import numpy as np

abs_dirname="/disk/scratch/datasets/50_400Hz_pure/0"
end_folder="/disk/scratch/datasets/fourier_signals"
files = [os.path.join(abs_dirname, f) for f in os.listdir(abs_dirname)]
tile=np.tile([0,1,2,3], 25000)
class_0=np.argwhere(tile==0).flatten()
class_1=np.argwhere(tile==1).flatten()
class_2=np.argwhere(tile==2).flatten()
class_3=np.argwhere(tile==3).flatten()
first_class=[f for f in files if int(f.split("/")[-1][:-4]) in class_0]
second_class=[f for f in files if int(f.split("/")[-1][:-4]) in class_1]
third_class=[f for f in files if int(f.split("/")[-1][:-4]) in class_2]
fourth_class=[f for f in files if int(f.split("/")[-1][:-4]) in class_3]

for i in first_class:
    # print(os.path.join(end_folder, str(0)+ "/" + i.split('/')[-1]))
    shutil.move(i, os.path.join(end_folder, str(0)+ "/" + i.split('/')[-1]))
for i in second_class:
    shutil.move(i, os.path.join(end_folder, str(1)+ "/" + i.split('/')[-1]))

for i in third_class:
    shutil.move(i, os.path.join(end_folder, str(2)+ "/" + i.split('/')[-1]))

for i in fourth_class:
    shutil.move(i, os.path.join(end_folder,  str(3)+ "/" + i.split('/')[-1]))





    # i = 0
    # curr_subdir = None
    # files.sort()
    #
    # for f in files:
    #     # create new subdir if necessary
    #     if i % N == 0:
    #         subdir_name = os.path.join(abs_dirname, '{0:03d}'.format(i // N + 1))
    #         os.mkdir(subdir_name)
    #         curr_subdir = subdir_name
    #
    #     # move file to current dir
    #     f_base = os.path.basename(f)
    #     shutil.move(f, os.path.join(subdir_name, f_base))
    #     i +=