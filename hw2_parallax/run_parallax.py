'''
Example command for running this script
python run_parallax.py 

Example command for examing the checkpoint file:
python <PARALLAX_HOME>/tensorflow/tensorflow/python/tools/inspect_checkpoint.py --file_name=parallax_distributed/model.ckpt-0.data-00000-of-00001 --tensor_name=conv1/kernel
'''
import os 
import sys
import time
import errno

import numpy as np
import tensorflow as tf
import parallax

from src.dataset import DataSet
from src.model_tf import network, get_placeholders, inference
from src.utils import argparser, logging

logfile = './logs/parallax-tf/logs.txt'
try : 
  os.makedirs(os.path.dirname(logfile))
except OSError as exc:
  if exc.errno == errno.EEXIST and os.path.isdir(os.path.dirname(logfile)):
    pass
  else :
    raise

FLAGS = argparser()
FLAGS.is_training = False

train_dataset = DataSet( fpath=FLAGS.train_file,
                         seqlen=FLAGS.seq_len,
                         n_classes=FLAGS.num_classes,
                         need_shuffle=False )
test_dataset = DataSet( fpath=FLAGS.test_file,
                        seqlen=FLAGS.seq_len,
                        n_classes=FLAGS.num_classes,
                        need_shuffle=False )
FLAGS.charset_size = train_dataset.charset_size
FLAGS.sync = True

resource_info = os.path.abspath(os.path.join( os.path.dirname(__file__),
                                              '.',
                                              FLAGS.resource_info_file ))

single_graph = tf.Graph()

with single_graph.as_default():
  ops, global_step = get_placeholders( FLAGS )

  seq = ops['data']
  label = ops['labels']
  logits, _ = inference(seq, FLAGS)

  tf.losses.softmax_cross_entropy(label, logits)
  loss = tf.losses.get_total_loss()
    
  optimizer = tf.train.AdamOptimizer( FLAGS.learning_rate )
  train_op = optimizer.minimize(loss, global_step=global_step)

  _acc_op = tf.equal( tf.argmax(logits, 1), tf.argmax(label,  1))
  acc_op = tf.reduce_mean( tf.cast(_acc_op, tf.float32 ))
  _hit_op = tf.equal( tf.argmax(logits, 1), tf.argmax(label, 1))
  hit_op = tf.reduce_sum( tf.cast(_hit_op, tf.float32 ))

parallax_config = parallax.Config()
ckpt_config = parallax.CheckPointConfig(ckpt_dir='parallax_ckpt',
                                        save_ckpt_steps=100)
parallax_config.ckpt_config = ckpt_config

sess, num_workers, worker_id, num_replicas_per_worker = parallax.parallel_run(
    single_graph,
    resource_info,
    sync=FLAGS.sync,
    parallax_config=parallax_config
)
  
fw = open(logfile, 'w')
start = time.time()
  
train_batches = train_dataset.iter_batch( FLAGS.batch_size * num_replicas_per_worker, 
                                          FLAGS.max_step)

for i, train_data in enumerate(train_batches):
  if i >= FLAGS.max_step:
    break
  fw.write("{} steps started : {}\n".format(i, time.time()))
    
  # Train
  FLAGS.is_training = True
  _, loss_val, acc_val = sess.run([train_op, loss, acc_op], feed_dict={
      seq : np.split(train_data[0], num_replicas_per_worker),
      label : np.split(train_data[1], num_replicas_per_worker)
  })
  fw.write("{} step done : {}\n".format(i, time.time()))
  print("{} batch done".format(i))
    
  # Test
  FLAGS.is_training = False
  hit_count = 0.0
  total_count = 0
      
  if (i+1) % 10 == 0 :
    for test_data, test_labels in test_dataset.iter_once( FLAGS.batch_size ):
      hits = sess.run([hit_op], feed_dict={ seq: [test_data],
                                            label : [test_labels] })
      hit_count += np.sum(hits)
      total_count += len( test_data )
    test_dataset = DataSet( fpath=FLAGS.test_file,
                            seqlen=FLAGS.seq_len,
                            n_classes=FLAGS.num_classes,
                            need_shuffle=False )
    fw.write("{} : micro-precision = {}\n".format(i, hit_count/total_count))
    parallax.log.info("{} : micro-precision = {}\n".format(i, hit_count/total_count))
fw.write("\n\nTotal Time Consumed : {}\n".format(time.time() - start))
fw.close()
      

