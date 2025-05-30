
## bank conflicts

Bank conflicts occur when multiple threads in a warp try to access different addresses within the same memory bank of shared memory simultaneously, causing serialized access instead of parallel access.

How Shared Memory Banks Work
CUDA shared memory is divided into 32 banks (on most modern GPUs):

Each bank is 4 bytes wide (32-bit)
Threads in a warp can access different banks simultaneously
Conflict: Multiple threads access the same bank → serialized access
No conflict: Each thread accesses a different bank → parallel access