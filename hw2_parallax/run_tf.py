'''
Setting of Environment
parameter server : localhost:12345
worker 1 server : localhost:12346
worker 2 server : localhost:12347

Example command for running the script:
parmeter : python run_tf.py --ps_hosts localhost:12345 --worker_hosts localhost:12346 localhost:12347 --job_name ps --task_index 0
worker 1 : python run_tf.py --ps_hosts localhost:12345 --worker_hosts localhost:12346 localhost:12347 --job_name worker --task_index 0
worker 2 : python run_tf.py --ps_hosts localhost:12345 --worker_hosts localhost:12346 localhost:12347 --job_name worker --task_index 1

Example command for examining the checkpoint file:
python <PARALLAX_HOME>/tensorflow/tensorflow/python/tools/inspect_checkpoint.py --file_name=vanilla_distributed/model.ckpt-0.data-00000-of-00001 --tensor_name=conv1/kernel
'''
import os
import sys
import time
import errno

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from src.dataset import DataSet
from src.model_tf import network, get_placeholders, inference
from src.utils import argparser, logging

if __name__ == '__main__':
  FLAGS = argparser()
  FLAGS.is_training = False
  logfile = "./logs/vanilla-tf/log.txt"
  try :
    os.makedirs(os.path.dirname(logfile))
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(os.path.dirname(logfile)):
      pass
    else :
      raise

  cluster = tf.train.ClusterSpec({ "ps":FLAGS.ps_hosts,
                                   "worker":FLAGS.worker_hosts })
  server = tf.train.Server( cluster, 
                            job_name=FLAGS.job_name,
                            task_index=FLAGS.task_index )
  
  if FLAGS.job_name == "ps":
    server.join()
  
  assert FLAGS.job_name == "worker"

  train_dataset = DataSet( fpath=FLAGS.train_file,
                           seqlen=FLAGS.seq_len,
                           n_classes=FLAGS.num_classes,
                           need_shuffle=False )
  test_dataset = DataSet( fpath=FLAGS.test_file,
                          seqlen=FLAGS.seq_len,
                          n_classes=FLAGS.num_classes,
                          need_shuffle=False )
  FLAGS.charset_size = train_dataset.charset_size
    
  # Assign operator to each of the devices in the worker set
  with tf.device(tf.train.replica_device_setter(
         ps_device="/job:ps/cpu:0",
         worker_device="/job:worker/task:{}".format(FLAGS.task_index),
         cluster=cluster)):
    ops, global_step = get_placeholders( FLAGS )
    
    seq = ops['data']
    label = ops['labels']
    logits, _ = inference(seq, FLAGS)

    tf.losses.softmax_cross_entropy(label, logits)
    loss = tf.losses.get_total_loss()

    _acc_op = tf.equal( tf.argmax(logits, 1), tf.argmax(label, 1))
    acc_op = tf.reduce_mean( tf.cast( _acc_op, tf.float32 ) )
    _hit_op = tf.equal( tf.argmax(logits, 1), tf.argmax(label, 1))
    hit_op = tf.reduce_sum( tf.cast( _hit_op, tf.float32 ) )

    optimizer = tf.train.AdamOptimizer( FLAGS.learning_rate )
    optimizer = tf.train.SyncReplicasOptimizer(
        optimizer,
        replicas_to_aggregate=len(FLAGS.worker_hosts),
        total_num_replicas=len(FLAGS.worker_hosts)
    )
    train_op = optimizer.minimize(loss, global_step=global_step)

    is_chief = (FLAGS.task_index == 0)
    sync_replicas_hook = optimizer.make_session_run_hook(is_chief)

    saver = tf.train.Saver( tf.global_variables(), save_relative_paths=False,
                            allow_empty=True, max_to_keep=10 )
    tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
    scaffold = tf.train.Scaffold(saver=saver)
    ckpt_hook = tf.train.CheckpointSaverHook( 'vanilla_distributed',
                                              save_steps=100,
                                              scaffold=scaffold )
  fw = open(logfile, 'w')
  with tf.train.MonitoredTrainingSession( master=server.target,
                                          is_chief=is_chief,
                                          hooks=[sync_replicas_hook],
                                          chief_only_hooks=[ckpt_hook] ) as sess:
    start = time.time()
    train_batches = train_dataset.iter_batch(FLAGS.batch_size, FLAGS.max_step)
    
    for i, train_data in enumerate( train_batches ):
      if i >= FLAGS.max_step :
        break
      
      fw.write("{} steps started : {}\n".format(i, time.time()))
      # Code for running train code
      FLAGS.is_training = True
      _, loss_val, acc_val = sess.run([train_op, loss, acc_op], feed_dict={
          seq : train_data[0],
          label : train_data[1]
      })
      fw.write("{}th step done : {}\n".format(i, time.time()))
      print("{} batch done ".format(i))

      # Code for running test code
      FLAGS.is_training = False
      hit_count = 0.0
      total_count = 0

      if (i+1) % 10 == 0:
        for test_data, test_labels in test_dataset.iter_once( FLAGS.batch_size ):
          hits = sess.run( hit_op, feed_dict={ seq:test_data,
                                               label:test_labels })
          hit_count += np.sum( hits )
          total_count += len( test_data )
        test_dataset = DataSet( fpath=FLAGS.test_file,
                                seqlen=FLAGS.seq_len,
                                n_classes=FLAGS.num_classes,
                                need_shuffle=False )
        fw.write("{} : micro-precision = {}\n".format(i, hit_count/total_count))
    fw.write("\n\nTotal Time Consumed : {}\n".format(time.time() - start))
  fw.close()
