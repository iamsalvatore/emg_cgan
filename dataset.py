class CustomDataset(Dataset):
    def __init__(self):
        # self.imgs_path = "/Users/salvatoreesposito/Documents/copy_dummy/"
        self.imgs_path = "/disk/scratch/datasets/fourier_class/"
        file_list = glob.glob(self.imgs_path + "*")
        # print(file_list)
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for npy_path in glob.glob(class_path + "/*.npy"):
                self.data.append([npy_path, class_name])
        # print(self.data)
        self.class_map = {"0" : 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npy_path, class_name = self.data[idx]
        class_id = self.class_map[class_name]
        # img_tensor = torch.from_numpy(np.load(npy_path))
        img_tensor = torch.unsqueeze(torch.from_numpy(np.load(npy_path)), dim=0)
        class_id = torch.tensor([class_id])

        return img_tensor, class_id
