# A Script for PyTorch Multi-GPU Testing

In this repository, We provide a multi-GPU multi-process testing script that enables distributed testing in PyTorch (should also work for TensorFlow).

## Problem
PyTorch distributed training is easy to use. However, we have to test the model sample by sample on a single GPU, since different testing samples often have different sizes. When we have multiple GPUs as same as training, this is a waste of time.

## Solution
We use the `multiprocessing` package for distributed testing on multiple GPUs. It supports multiple process on multiple GPUs and each GPU can run multiple processes if you have large enough GPU memory. Note that each process is an independent execution of the testing function.

## Important To Know
1, As each process is independent in terms of PyTorch, we need to load the model into GPU memory repeatedly before testing every sample, so our distirubted testing script may **NOT** save your time. It is suitable for cases where testing a single sample needs long runtime. For example, zero-shoting learning tasks and video recognition tasks.

2, When we only has one testing process per GPU (i.e., `max_process_per_gpu==1`), it always works fine. But when we try to start multiple processes per GPU (i.e., `max_process_per_gpu>=2`), it may **get stuck** on some computers or clusters.

3, When there is a runtime error (e.g., out-of-memory error) in one testing process, it will **NOT** impact other processes. Be sure to check that the output number and input number are equal.
