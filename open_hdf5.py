import h5py
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # Open an HDF5 file
    with h5py.File('power-rabi.hdf5', 'r') as file:


        print(file.keys())
        # Access the 'Data' group and then the 'Data' dataset
        data_group = file['Data']
        data_dataset = data_group['Data']

        # Print dataset information
        print("Dataset 'Data' info:")
        print("Shape:", data_dataset.shape)
        print("Datatype:", data_dataset.dtype)

        data = data_dataset[()]
        x = np.array(data).T[0][0]
        y = np.array(data).T[0][1]

        plt.plot(x,y)
        plt.show()

        commends = file.attrs['comment']
        import ast

        dict_obj = ast.literal_eval(commends)

        # Print the resulting dictionary
        print(dict_obj.keys())

        args = dict_obj['args']
        n_avg = dict_obj['n_avg']

        pprint(args)
