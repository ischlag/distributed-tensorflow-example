# Distributed Tensorflow 1.2 Example (DEPRECATED)

This example uses python>=3.6 features.

Using data parallelism with shared model parameters while updating parameters
asynchronous.  See comment for some changes to make the parameter updates
synchronous (not sure if the synchronous part is implemented correctly though).

Trains a simple sigmoid Neural Network on MNIST for 20 epochs on `n` machines
using one parameter server.

Run it like this: 

First, change the hardcoded host names with your own and run the following commands on the respective machines.

On the machine running the parameter server with dns `server.example.edu`, run
```
server.example.edu$ python example.py \
--job-name ps --task-index 0 \
--ps-host server.example.edu \
--worker-host worker_0.example.edu worker_1.example.edu worker_2.example.edu
```
On the machine(s) running the workers with dns `worker_<j>.example.edu`, run 
```
worker_0.example.edu$ python example.py \
--job-name worker --task-index 0 \
--ps-host server.example.edu \
--worker-host worker_0.example.edu worker_1.example.edu worker_2.example.edu
worker_1.example.edu$ python example.py \
--job-name worker --task-index 1 \
--ps-host server.example.edu \
--worker-host worker_0.example.edu worker_1.example.edu worker_2.example.edu
...
```

Thanks to snowsquizy for updating the script to TensorFlow 1.2.
