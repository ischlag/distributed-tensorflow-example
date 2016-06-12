# Distributed Tensorflow Example 

Using data parallelism with shared model parameters while updating parameters asynchronous. See comment for some changes to make the parameter updates synchronous. Not sure if this implemented correctly though.

Trains a simple sigmoid neural network on mnist for 20 epochs on three machines using one parameter server. The goal was not to achieve high accurcy but to get to know tensorflow.

Run like this: 

First, change the hardcoded host urls below with your own hosts and run the following commands on the respective machines.

'''
pc-01$ python example.py --job-name="ps" --task_index=0 
pc-02$ python example.py --job-name="worker" --task_index=0 
pc-03$ python example.py --job-name="worker" --task_index=1 
pc-04$ python example.py --job-name="worker" --task_index=2 
'''

More details here: [ischlag.github.io](http://ischlag.github.io/)