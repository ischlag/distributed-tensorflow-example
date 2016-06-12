# Distributed Tensorflow example of using data parallelism and share model parameters.
Trains a simple sigmoid neural network on mnist for 20 epochs on three machines using one parameter server. 

Change the hardcoded host urls below with your own hosts. 
Run like this: 

pc-01$ python example.py --job-name="ps" --task_index=0 
pc-02$ python example.py --job-name="worker" --task_index=0 
pc-03$ python example.py --job-name="worker" --task_index=1 
pc-04$ python example.py --job-name="worker" --task_index=2 

More details here: [ischlag.github.io](http://ischlag.github.io/)