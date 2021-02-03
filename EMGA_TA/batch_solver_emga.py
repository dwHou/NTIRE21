import sys
import subprocess
import os
from multiprocessing import Pool
import numpy as np
import time

def run_cmd(cmd):
    retcode = subprocess.call(cmd, shell=True)
    return retcode

def mycallback(arg):
    print("async process test", arg)


def test(gpu_idx, dir_idx):
    print('gpu_idx:', gpu_idx, 'dir_idx', dir_idx)
    sleep(6)
    print('Child Process id : %s, Parent Process id : %s' % (os.getpid(), os.getppid()))

def inference(gpu_idx, dir_idx):
    # print('gpu_idx:', gpu_idx, 'dir_idx', dir_idx)
    # print('Child Process id : %s, Parent Process id : %s' % (os.getpid(), os.getppid()))
    root =  "/cfs_data/devonnhou/NTIRE2021/Dataset/FixedQP/valuation_fixed-QP/"
    input_video = os.path.join(root, 'fixedqp_png', dir_idx)
    input_info = os.path.join(root, 'fixedqp_info', dir_idx)
    output_video = os.path.join(root, 'EMGA_out', dir_idx)
    print(input_video)

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_idx} python3 src_online/EMGA_solver.py --cuda --input_video {input_video} --output_video {output_video} --input_info {input_info}"
    print(cmd)
    run_cmd(cmd)
    # os.system(cmd)
    print("======>done handle", dir_indx)


allvideos = [1000 + i for i in range(1, 21)]
threads = 4
pool = Pool(threads)

for idx in allvideos:
    # print(str(idx)[1:])
    while(True):
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
        m_idx = np.argmax(memory_gpu)
        mem = memory_gpu[m_idx]
        if mem != 15079:
            time.sleep(2 + int(idx - 1000) * 3)
            continue
        else:
            break 
        
    
    idx = str(idx)[1:]
    # gpu_idx = int(idx) % threads 
    gpu_idx = m_idx
    print(f'=================> video {idx} using gpu idx {gpu_idx}')
    pool.apply_async(inference, args=(gpu_idx, idx), callback=mycallback)
    # pool.apply_async(test, args=(gpu_idx, idx), callback=mycallback)
    time.sleep(5)    

pool.close()
pool.join()

