import rampy as rp
from functools import reduce
from scipy import sparse
from tkinter import filedialog
from tkinter import Tk
from scipy.signal import medfilt
import numpy as np
import os

from tqdm import tqdm


root = Tk()


from scipy.signal import medfilt

def cosrad_eli(spec: np.ndarray):
    # 去除宇宙射线的函数
    lam = 1000
    N = len(spec)
    D = sparse.csc_matrix(np.diff(np.eye(N), 2))
    w = np.ones(N)  # 行向量
    while True:
        W = sparse.spdiags(w, 0, N, N)
        Z = W + lam * D.dot(D.transpose())
        z = sparse.linalg.spsolve(Z, w * spec)

        d = spec - z
        c = W @ d
        s = np.std(c)

        if np.any(c > 5 * s):
            w[c >= 5 * s] = 0
        else:
            break

    spec[w == 0] = z[w == 0] + np.random.randn() * s
    return spec

def preprocess_one_spectrum(originSpec):
    #
    spec = originSpec.copy().transpose()
    spec_orignal = spec.copy()
    # 去除宇宙射线
    spec_orignal[1, :] = cosrad_eli(spec_orignal[1, :])
    return spec_orignal.T




# data_path_original =  filedialog.askdirectory(initialdir = "/",title = "选择原始文件夹",mustexist = True)
data_path_original = 'E:\Pythonproject\拉曼光谱分类\dataset\肠杆菌\init\Proteus\PMI01'
data_path_norm =  data_path_original+'_cos'

has_subfolders = False
has_sub_subfolders = False

for dirpath, dirnames, filenames in os.walk(data_path_original):
    if len(dirnames) > 0:
        has_subfolders = True
        # Check if the first subfolder has sub-subfolders
        subfolder_path = os.path.join(dirpath, dirnames[0])
        for _, sub_subdirectories, _ in os.walk(subfolder_path):
            if len(sub_subdirectories) > 0:
                has_sub_subfolders = True
                break
        break

if has_subfolders and has_sub_subfolders:
    print("\n该文件夹内有子文件夹且有子子文件夹")

    datas = os.listdir(data_path_original)
    for sub_folder in tqdm(datas):
        if sub_folder == '.DS_Store':
            continue

        sub_folder_path = os.path.join(data_path_original, sub_folder)
        sub_sub_folders = [f for f in os.listdir(sub_folder_path) if os.path.isdir(os.path.join(sub_folder_path, f))]

        # Process files in sub-sub-folders
        for sub_sub_folder in sub_sub_folders:
            sub_sub_folder_path = os.path.join(sub_folder_path, sub_sub_folder)
            files = os.listdir(sub_sub_folder_path)

            newFolder = os.path.join(data_path_norm, sub_folder, sub_sub_folder)
            if not os.path.exists(newFolder):
                os.makedirs(newFolder)

            for file in files:
                file_path = os.path.join(sub_sub_folder_path, file)

                try:
                    originSpec = np.loadtxt(file_path, skiprows=0)
                except:
                    originSpec = np.loadtxt(file_path, skiprows=0, encoding='utf')
                spectra_norm = preprocess_one_spectrum(originSpec)  # 将单个光谱预处理得到归一化的单个光谱,再转置为(862,2)
                new_file_path = os.path.join(newFolder, file[:-9] + '_cos' + file[-9:])
                np.savetxt(new_file_path, spectra_norm, delimiter='\t', fmt=['%.3f', '%f'], encoding='utf', comments='')

elif has_subfolders and not has_sub_subfolders:
    print("\n该文件夹内有子文件夹且没有子子文件夹")

    datas = os.listdir(data_path_original)  # 用于返回指定的文件夹包含的文件或文件夹的名字的列表。datas存放几个子文件夹的名字
    for sub_folder in tqdm(datas):          #sub_folder代表每个子文件夹的名字（1s,5s,10s,15s）
        if sub_folder == '.DS_Store':
            continue
        files_path = os.path.join(data_path_original, sub_folder)   #代表每个子文件夹的完整路径
        files = os.listdir(files_path)                              #files代表每个子文件夹下，所有单个文件的名字集合

        newFolder = os.path.join(data_path_norm, sub_folder)        #在另一个文件夹内的同名子文件夹的完整路径
        if not os.path.exists(newFolder):
            os.makedirs(newFolder)                                  #在另一个文件夹内，新建同名子文件夹

        for file in files:                                          #file代表单个光谱文件的名字
            file_path = os.path.join(files_path, file)              #file_path代表单个光谱文件的完整路径
            # if sub_folder=='肺克_15s' or '肺克_10s':
            #     lamada = 1e8
            # elif sub_folder==:

            try:
                originSpec = np.loadtxt(file_path, skiprows=0)      # 将单个光谱文件读入到np.array
            except:
                originSpec = np.loadtxt(file_path, skiprows=0,encoding='utf')  # 将单个光谱文件读入到np.array

            spectra_norm = preprocess_one_spectrum(originSpec)    #将单个光谱预处理得到归一化的单个光谱,再转置为(862,2)
            new_file_path = os.path.join(newFolder, file[:-9]+'_cos'+file[-9:])
            np.savetxt(new_file_path, spectra_norm,  delimiter='\t', fmt=['%.3f', '%f'],encoding='utf', comments='')
            # new_file = np.loadtxt(new_file_path, skiprows=0)        #确认一下是否可以读入，确认过没问题



elif not has_subfolders and not has_sub_subfolders:
#如果打开的是单个文件夹，无子文件夹
    print("\n目前只有一个文件夹")
    files = os.listdir(data_path_original)  # 用于返回指定的文件夹包含的文件或文件夹的名字的列表。datas存放几个子文件夹的名字
    for file in files:  # file代表单个光谱文件的名字
        file_path = os.path.join(data_path_original, file)  # file_path代表单个光谱文件的完整路径
        # if sub_folder=='肺克_15s' or '肺克_10s':
        #     lamada = 1e8
        # elif sub_folder==:

        try:
            originSpec = np.loadtxt(file_path, skiprows=0)  # 将单个光谱文件读入到np.array
        except:
            originSpec = np.loadtxt(file_path, skiprows=0, encoding='utf')  # 将单个光谱文件读入到np.array

        spectra_norm = preprocess_one_spectrum(originSpec)
        new_file_path = os.path.join(data_path_norm, file[:-9] + '_cos' + file[-9:])

        # 检查目录是否存在，不存在则创建
        if not os.path.exists(data_path_norm):
            os.makedirs(data_path_norm)
        np.savetxt(new_file_path, spectra_norm, delimiter='\t', fmt=['%.3f', '%f'], encoding='utf', comments='')
        # new_file = np.loadtxt(new_file_path, skiprows=0)        #确认一下是否可以读入，确认过没问题
