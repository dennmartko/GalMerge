import math
import numpy as np

from numba import cuda, jit, prange, vectorize, guvectorize
from sys import getsizeof
from multiprocessing import cpu_count, Pool

@cuda.jit(device=True)
def haversine_cuda(s_lat,s_lng,e_lat,e_lng):
    '''
    This is now a non-vectorized version of the haversine distance function. 
    All inputs are expected to be scalars.
    '''
    # approximate radius of earth in km
    R = 6373.0

    s_lat = s_lat * math.pi / 180                     
    s_lng = s_lng * math.pi / 180 
    e_lat = e_lat * math.pi / 180                    
    e_lng = e_lng * math.pi / 180 

    d = math.sin((e_lat - s_lat)/2)**2 + math.cos(s_lat)*math.cos(e_lat) * math.sin((e_lng - s_lng)/2)**2

    return 2 * R * math.asin(math.sqrt(d))

@cuda.jit
def get_nearby_kernel(coord1, coord2, max_dist, out):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    lat_filter = max_dist / 100
    
    for i in range(start, coord1.shape[0], stride):
        ct = 0
        _lat1 = coord1[i,0]
        _lng1 = coord1[i,1]
        
        for j in range(coord2.shape[0]):
            _lat2 = coord2[j,0]
            _lng2 = coord2[j,1]
            # escape condition if latitudes are too far apart
            if math.fabs(_lat1 - _lat2) <= lat_filter:
                dist = haversine_cuda(_lat1, _lng1, _lat2, _lng2)
                if dist < max_dist:ct += 1
                
        out[i] = ct
        
threads_per_block = 512
blocks_per_grid = 36


# Initial Data
n = 10_000_00
k = 1_000_000

coord1 = np.zeros((n, 2), dtype=np.float32)
coord2 = np.zeros((k, 2), dtype=np.float32)

coord1[:,0] = np.random.uniform(-90, 90, n).astype(np.float32)
coord1[:,1] = np.random.uniform(-180, 180, n).astype(np.float32)
coord2[:,0] = np.random.uniform(-90, 90, k).astype(np.float32)
coord2[:,1] = np.random.uniform(-180, 180, k).astype(np.float32)

coord1 = np.sort(coord1,axis=0)
coord2 = np.sort(coord2,axis=0)

# CUDA Arrays
coord1_gpu = cuda.to_device(coord1)
coord2_gpu = cuda.to_device(coord2)
out_gpu = cuda.device_array(shape=(n,), dtype=np.int32)
get_nearby_kernel[blocks_per_grid, threads_per_block](coord1_gpu, coord2_gpu, 1.0, out_gpu)
gpu_solution = out_gpu.copy_to_host()
print(gpu_solution)