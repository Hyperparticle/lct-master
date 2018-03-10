#!/usr/bin/env python3
import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.summary # Needed to allow importing summary operations

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args

# Hyperparameter optimization
# https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/19_Hyper-Parameters.ipynb
class Network:
    OBSERVATIONS = 4
    ACTIONS = 2

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, logdir):
        with self.session.graph.as_default():
            self.observations = tf.placeholder(tf.float32, [None, self.OBSERVATIONS], name="observations")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")

            # Define the model, with the output layers for actions in `output_layer`
            hidden_layer = self.observations
            for _ in range(args.num_dense_layers):
                hidden_layer = tf.layers.dense(hidden_layer, args.num_dense_nodes, activation=tf.nn.relu)
                hidden_layer = tf.nn.dropout(hidden_layer, keep_prob=args.dropout)

            output_layer = tf.layers.dense(hidden_layer, self.ACTIONS, activation=None, name="output_layer")
            self.predictions = tf.argmax(output_layer, axis=1)

            self.actions = tf.argmax(output_layer, axis=1, name="actions")

            # Global step
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss, global_step=global_step, name="training")

            # Summaries
            accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.actions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(logdir, flush_millis=10 * 1000)
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries = [tf.contrib.summary.scalar("train/loss", loss),
                                  tf.contrib.summary.scalar("train/accuracy", accuracy)]

            # Construct the saver
            tf.add_to_collection("end_points/observations", self.observations)
            tf.add_to_collection("end_points/actions", self.actions)
            self.saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, observations, labels):
        self.session.run([self.training, self.summaries], {self.observations: observations, self.labels: labels})

    def save(self, path):
        self.saver.save(self.session, path)

    def load(self, path):
        # Load the metagraph
        with self.session.graph.as_default():
            self.saver = tf.train.import_meta_graph(path + ".meta")

            # Attach the end points
            self.observations = tf.get_collection("end_points/observations")[0]
            self.actions = tf.get_collection("end_points/actions")[0]

        # Load the graph weights
        self.saver.restore(self.session, path)

    def predict(self, observations):
        return self.session.run(self.actions, {self.observations: [observations]})[0]

def fitness(x):
    args.learning_rate, args.num_dense_layers, args.num_dense_nodes, args.epochs, args.dropout = x

    global call_num
    call_num += 1

    # Print the hyper-parameters.
    print('Iteration', call_num)

    # Create logdir name
    logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    observations, labels = [], []
    with open("gym_cartpole-data.txt", "r") as data:
        for line in data:
            columns = line.rstrip("\n").split()
            observations.append([float(column) for column in columns[0:4]])
            labels.append(int(columns[4]))
    observations, labels = np.array(observations), np.array(labels)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, logdir)

    # Train
    for i in range(args.epochs):
        # Train for an epoch
        network.train(observations, labels)

    # Create the environment
    env = gym.make('CartPole-v1')

    # Evaluate the episodes
    total_score = 0
    for episode in range(args.episodes):
        observation = env.reset()
        score = 0
        for i in range(env.spec.timestep_limit):
            if args.render:
                env.render()
            observation, reward, done, info = env.step(network.predict(observation))
            score += reward
            if done:
                break

        total_score += score
        # print("The episode {} finished with score {}.".format(episode + 1, score))

    reward = total_score / args.episodes
    # print("The average reward per episode was {:.2f}.".format(reward))

    global best_reward
    if reward > best_reward:
        print()
        print('New best')
        print("Reward: {:.2f}".format(reward))
        print('learning rate: {0:.1e}'.format(args.learning_rate))
        print('num_dense_layers:', args.num_dense_layers)
        print('num_dense_nodes:', args.num_dense_nodes)
        print('num_epochs:', args.epochs)
        print('dropout: {0:.1e}'.format(args.dropout))
        print()

        network.save("gym_cartpole/model")
        best_reward = reward

    network.session.close()
    del network
    
    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.
    return -reward / 500.0

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", default=100, type=int, help="Number of iterations.")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--episodes", default=100, type=int, help="Number of episodes.")
    parser.add_argument("--render", default=False, action="store_true", help="Render the environment.")
    args = parser.parse_args()

    dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
    dim_num_dense_layers = Integer(low=1, high=3, name='num_dense_layers')
    dim_num_dense_nodes = Integer(low=5, high=512, name='num_dense_nodes')
    dim_num_epochs = Integer(low=20, high=500, name='num_epochs')
    dim_dropout = Real(low=0.9, high=1.0, name='dropout')
    dimensions = [dim_learning_rate,
                  dim_num_dense_layers,
                  dim_num_epochs,
                  dim_num_epochs,
                  dim_dropout]
    default_parameters = [1e-5, 1, 16, 50, 1.0]

    best_reward = 0.0
    call_num = 0

    res_gp = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=args.iter)
    
    print("Best score=%.4f" % res_gp.fun)
    print("""Best parameters:
            - learning_rate=%d
            - num_dense_layers=%.6f
            - num_dense_nodes=%d
            - num_epochs=%d
            - dropout=%d""" % (res_gp.x[0], res_gp.x[1], 
                                        res_gp.x[2], res_gp.x[3], 
                                        res_gp.x[4]))