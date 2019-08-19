class DriveData(Dataset):
    def __init__(self, datasetf, transform=None):
        self.__xs = []
        self.__ys = []
        self.transform = transform
        # Open and load text file including the whole training data
        with open(datasetf) as f:
            #print(datasetf)
            i = 0
            for line in f:
                # the following i>0 is for skipping the first line which is the column names 
                if i > 0:
                    # checked the resizing to 50*300 and it's correct
                    #self.__xs.append(torch.from_numpy(np.asarray([float(x) for x in line.split(',')[0:15000]])).view(1,50,300))
                    self.__xs.append(torch.from_numpy(np.asarray([float(x) for x in line.split(',')[0:15000]])))
                    #if i == 5:
                        #print(i, self.__xs)
                    self.__ys.append(line.split(',')[-1])
                i += 1
        f.close()

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        message = self.__xs[index]
        if self.transform is not None:
            message = self.transform(message)

        # Convert image and label to torch tensors
        #message = torch.from_numpy(np.asarray(message))
        # the subtraction of 1 is to make the target values range from 0 to 25 instead of 1 to 26
        label = torch.from_numpy(np.asarray(int(self.__ys[index])-1).reshape(1))

#        if int(self.__ys[index]) in big_levels:
#            label = torch.from_numpy(np.asarray(big_level_conv_dic[int(self.__ys[index])]-1).reshape(1))
#        elif int(self.__ys[index]) in mid_levels:
#            label = torch.from_numpy(np.asarray(mid_level_conv_dic[int(self.__ys[index])]-1).reshape(1))
#        else:
#            label = torch.from_numpy(np.asarray(3-1).reshape(1))

        label = label.long()
        return message, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)
