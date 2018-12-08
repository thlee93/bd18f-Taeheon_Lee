import re
import time
from datetime import datetime
import os
import errno

import numpy as np
import tensorflow as tf

from src.utils import argparser, logging
from src.model_tf import network, get_placeholders, inference
from src.dataset import DataSet

if __name__ == '__main__':
  FLAGS = argparser()
  FLAGS.is_training = False
  logfile = "./logs/tensorflow/log.txt"
  try :
    os.makedirs(os.path.dirname(logfile))
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(os.path.dirname(logfile)):
      pass
    else : 
      raise

  train_dataset = DataSet( fpath=FLAGS.train_file,
                           seqlen=FLAGS.seq_len,
                           n_classes=FLAGS.num_classes,
                           need_shuffle=True )
  test_dataset = DataSet( fpath=FLAGS.test_file,
                          seqlen=FLAGS.seq_len,
                          n_classes=FLAGS.num_classes,
                          need_shuffle=True )
  FLAGS.charset_size = train_dataset.charset_size

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
  train_op = optimizer.minimize(loss, global_step=global_step)

  saver = tf.train.Saver(max_to_keep=None)
  fw = open(logfile, 'w')

  ### iterate over max_epoch
  with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )
    
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

      if i % 10 == 0:
        for test_data, test_labels in test_dataset.iter_once( FLAGS.batch_size ):
          hits = sess.run( hit_op, feed_dict={ seq:test_data,
                                               label:test_labels })
          hit_count += np.sum( hits )
          total_count += len( test_data )
        test_dataset = DataSet( fpath=FLAGS.test_file,
                                seqlen=FLAGS.seq_len,
                                n_classes=FLAGS.num_classes,
                                need_shuffle=True )
        fw.write("{} : micro-precision = {}\n".format(i, hit_count/total_count))
    fw.write("\n\nTotal Time Consumed : {}\n".format(time.time() - start))
  fw.close()