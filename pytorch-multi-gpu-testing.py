# --------------------------------------------------------
# Pytorch-Multi-GPU-Testing
# Written by Jingyun Liang
# --------------------------------------------------------

import os
import random
import time
import warnings
import torch
import multiprocessing

# global variables
total_gpu_num = 8
max_process_per_gpu = 1  # todo: support max_process_per_gpu > 1
used_gpu_list = multiprocessing.Manager().list([0] * total_gpu_num)
lock = multiprocessing.Lock()


class CNN(torch.nn.Module):
    '''
    A toy CNN.
    '''
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


def multi_gpu_testing_wrapper(model, input, index, gpu_id=None, available_gpu_num=1):
    '''
    Multi-GPU testing wrapper.
    :param input: Model input, e.g. an image.
    :param gpu_id: Given gpu_id. Only used in debugging.
    :param available_gpu_num: Available GPU number.
    :return: Model output, used GPU id and process id.
    '''
    # GPU assignment
    lock.acquire()
    if gpu_id is None:
        for i in range(available_gpu_num):
            if used_gpu_list[i] < max_process_per_gpu:
                gpu_id = i
                break

    used_gpu_list[gpu_id] += 1
    lock.release()
    torch.cuda.set_device(gpu_id)
    device = torch.device('cuda')
    print(f'testing   input {index} on GPU {gpu_id}. Overall GPU usages: ', list(used_gpu_list))

    # model testing
    input = input.to(device)
    model = model.to(device)
    time.sleep(random.randrange(0, 10)) # used in this toy example to avoid deadlock
    output = model(input).detach().cpu()

    # release GPU memory manually (multiprocessing.Pool may not release GPU memory of a process in time)
    del input, model
    torch.cuda.empty_cache()

    # release GPU
    lock.acquire()
    used_gpu_list[gpu_id] -= 1
    lock.release()
    print(f'releasing input {index} on GPU {gpu_id}. Overall GPU usages: ', list(used_gpu_list))

    # return output
    return output, gpu_id, os.getpid()


def main():

    # setup GPU
    available_gpu_num = 4
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    assert available_gpu_num <= total_gpu_num
    if max_process_per_gpu > 1:
        warnings.warn("The program may get stuck when max_process_per_gpu > 1.")

    # initialize input and model
    total_input_num = 10
    input = [torch.ones(1, 1, 2, 2)] * total_input_num
    model = CNN()
    output = []

    def mycallback(arg):
        output.append(arg[0])

    # test the model on multiple GPUs distributedly
    pool = multiprocessing.Pool(available_gpu_num * max_process_per_gpu)
    for i in range(total_input_num):
        # hint: pool.apply_async cannot output informative debugging logs. Use pool.apply() for debugging.
        pool.apply_async(multi_gpu_testing_wrapper, args=(model, input[i], i,  None, available_gpu_num), callback=mycallback)
        # pool.apply(multi_gpu_testing_wrapper, args=(model, input[i], i,  0, available_gpu_num))

    pool.close()
    pool.join()
    print('All subprocesses done.')

    # check output quality (sometimes a process may fail due to out-of-momory error, but there is no error!)
    print(f'\n{len(output)}/{len(input)} processes succeeded.')
    assert len(input) == len(output)


if __name__ == '__main__':
    main()
