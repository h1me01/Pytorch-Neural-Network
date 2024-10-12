import ctypes
import numpy as np

lib = ctypes.CDLL('lib/Astra-Dataloader.dll')

class DenseResult(ctypes.Structure):
    _fields_ = [
        ("input1", ctypes.c_float * 768),
        ("input2", ctypes.c_float * 768),
        ("target", ctypes.c_float)
    ]

class DataLoader:
    def __init__(self, path):
        lib.create_data_loader.restype = ctypes.POINTER(ctypes.c_void_p)
        lib.delete_data_loader.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.get_size.restype = ctypes.c_size_t
        lib.get_data.restype = DenseResult 

        self.loader = lib.create_data_loader(path.encode('utf-8'))

    def __del__(self):
        if hasattr(self, 'loader') and self.loader:
            lib.delete_data_loader(self.loader)

    def get_size(self):
        return lib.get_size(self.loader)

    def get_data(self, idx):
        result = lib.get_data(self.loader, ctypes.c_size_t(idx))
        
        input1 = np.array(result.input1)
        input2 = np.array(result.input2)
        target = result.target
        return input1, input2, target

# test
if __name__ == "__main__":
    data_path = r'C:\Users\semio\Downloads\chess_data.bin'  
    loader = DataLoader(data_path)

    size = loader.get_size()
    print(f"Dataset size: {size}")

    input1, input2, score = loader.get_data(0)

    print("Input1 Array:")
    print(input1) 
    print("\nInput2 Array:")
    print(input2)  
    print("Scores:", score)
