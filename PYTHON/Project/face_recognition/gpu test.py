import dlib
print("GPU enabled?", dlib.DLIB_USE_CUDA)
print("Number of CUDA devices:", dlib.cuda.get_num_devices())
