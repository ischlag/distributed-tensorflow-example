'''
Distributed Tensorflow 1.2.0 example of using data parallelism and share model parameters.
Trains a simple sigmoid neural network on mnist for 20 epochs on three machines using one parameter server. 

Change the hardcoded host urls below with your own hosts. 
Run like this: 

pc-01$ python example.py --job_name="ps" --task_index=0 
pc-02$ python example.py --job_name="worker" --task_index=0 
pc-03$ python example.py --job_name="worker" --task_index=1 
pc-04$ python example.py --job_name="worker" --task_index=2 

More details here: ischlag.github.io
'''

import os
import sys
import time
import argparse
import tensorflow as tf

default_data_path = os.path.join(os.environ['DATA'], 'mnist')

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=2222)
parser.add_argument('--logdir', type=str, default='/tmp/mnist')
parser.add_argument('--ps-host', type=str, default='localhost')
parser.add_argument('--proceed', action='store_true')
parser.add_argument('--job-name', choices=['ps', 'worker'], default='ps')
parser.add_argument('--data-path', type=str, default=default_data_path)
parser.add_argument('--task-index', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--worker-host', type=str, nargs='*', default=['localhost'])
parser.add_argument('--learning-rate', type=float, default=0.0005)
parser.add_argument('--training-epochs', type=int, default=20)
args = parser.parse_args()

assert len(args.worker_host) > 0, f"Specify one or more worker hosts. {args.worker_host}"

parameter_server = [f"{args.ps_host}:{args.port}"]
workers = [f"{host}:{args.port+1+h}" for h, host in enumerate(args.worker_host)]

cluster_spec = tf.train.ClusterSpec({
    "ps": parameter_server,
    "worker": workers
})

if 'ps' == args.job_name:
    print("ps", parameter_server)
    print("workers", workers)

# start a server for a specific task
server = tf.train.Server(
    cluster_spec,
    job_name=args.job_name,
    task_index=args.task_index)

if 'ps' == args.job_name:
    print("Starting server..  This process will run forever.")
    server.join()
elif 'worker' == args.job_name:
    # load mnist data set
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(args.data_path, one_hot=True)

    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
            worker_device = "/job:worker/task:%d" % args.task_index,
            cluster = cluster_spec
        )):

        # input images
        with tf.name_scope('input'):
            # None -> batch size can be any size, 784 -> flattened mnist image
            input_features = tf.placeholder(tf.float32, shape=(None, 784), name="features")
            # target 10 output classes
            input_target = tf.placeholder(tf.float32, shape=(None, 10), name="target")

        def dense(input_tensor, output_dim, activation=None, name="dense"):
            """Dense neural-network layer with output-dim dimensions."""
            input_dim = int(input_tensor.get_shape()[1])
            with tf.variable_scope(name):
                W = tf.Variable(tf.random_normal((input_dim, output_dim)), dtype=tf.float32, name='W')
                b = tf.Variable(tf.random_normal((output_dim,)), dtype=tf.float32, name='b')
                output = tf.add(tf.matmul(input_tensor, W), b)
                if activation is not None:
                    output = activation(output)
            return output

        tf.set_random_seed(1)
        x = dense(input_features, output_dim=100, activation=tf.nn.sigmoid, name="Dense_1")
        logits = dense(x, output_dim=10, name="Dense_2")
        y = tf.nn.softmax(logits)

        # specify cost function
        with tf.name_scope('cross_entropy'):
            xent = tf.nn.softmax_cross_entropy_with_logits
            cross_entropy = tf.reduce_mean(xent(labels=input_target, logits=logits))
            # cross_entropy = tf.reduce_mean(
            #     -tf.reduce_sum(input_target * tf.log(y), reduction_indices=[1]))

        # specify optimizer
        with tf.name_scope('train'):
            # count the number of updates
            global_step = tf.get_variable(
                'global_step',
                [],
                initializer = tf.constant_initializer(0),
                trainable = False
            )

            # optimizer is an "operation" which we can execute in a session
            grad_op = tf.train.GradientDescentOptimizer(args.learning_rate)
            '''
            rep_op = tf.train.SyncReplicasOptimizer(
                grad_op,
                replicas_to_aggregate = len(workers),
                replica_id = args.task_index, 
                total_num_replicas = len(workers),
                use_locking = True
            )
            train_op = rep_op.minimize(cross_entropy, global_step=global_step)
            '''
            train_op = grad_op.minimize(cross_entropy, global_step=global_step)
            
        '''
        init_token_op = rep_op.get_init_tokens_op()
        chief_queue_runner = rep_op.get_chief_queue_runner()
        '''

        with tf.name_scope('Accuracy'):
            # accuracy
            correct_prediction = tf.equal(tf.argmax(y, 1),
                    tf.argmax(input_target, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # create a summary for our cost and accuracy
        tf.summary.scalar("cost", cross_entropy)
        tf.summary.scalar("accuracy", accuracy)

        # merge all summaries into a single "operation" which we can execute in a session 
        summary_op = tf.summary.merge_all()
        # .. proceed with training
        init_op = None if args.proceed else tf.global_variables_initializer()
        print("Variables initialized ...")

    sv = tf.train.Supervisor(
        is_chief = (args.task_index == 0),
        global_step = global_step,
        init_op = init_op
    )

    begin_time = time.time()
    frequency = 100
    with sv.prepare_or_wait_for_session(server.target) as S:
        '''
        # is chief
        if args.task_index == 0:
            sv.start_queue_runners(S, [chief_queue_runner])
            S.run(init_token_op)
        '''
        # create log writer object (this will log on every machine)
        writer = tf.summary.FileWriter(args.logdir, graph=tf.get_default_graph())
                
        # perform training cycles
        start_time = time.time()
        for epoch in range(args.training_epochs):

            # number of batches in one epoch
            batch_count = int(mnist.train.num_examples/args.batch_size)

            count = 0
            for i in range(batch_count):
                features, target = mnist.train.next_batch(args.batch_size)
                
                # perform the operations we defined earlier on batch
                _, cost, summary, step = S.run(
                    [train_op, cross_entropy, summary_op, global_step],
                    feed_dict = {
                        input_features: features,
                        input_target: target
                    }
                )
                writer.add_summary(summary, step)

                count += 1
                if count % frequency == 0 or i+1 == batch_count:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print("Step: %d," % (step+1), 
                          "  Epoch: %2d," % (epoch+1), 
                          "  Batch: %3d of %3d," % (i+1, batch_count), 
                          "  Cost: %.4f," % cost, 
                          "  AvgTime: %3.2fms" % float(elapsed_time*1000/frequency))
                    count = 0

        acc = S.run(accuracy, feed_dict={input_features:
            mnist.test.images, input_target: mnist.test.labels})
        print(f"Test-Accuracy: {acc}")
        print(f"Total Time: {time.time()-begin_time} sec.")
        print(f"Final Cost: {cost}")

    sv.stop()
    print("done")
