import argparse
import os

def argparser():
  parser = argparse.ArgumentParser()
  # for model
  parser.add_argument(
      '--window_lengths',
      type=int,
      nargs='+',
      default=[256, 256, 256, 256, 256, 256, 256, 256],
      help='Space seperated list of motif filter lengths. (ex, --window_lengths 4 8 12)'
  )
  parser.add_argument(
      '--num_windows',
      type=int,
      nargs='+',
      default=[8, 12, 16, 20, 24, 28, 32, 36],
      help='Space seperated list of the number of motif filters corresponding to length list. (ex, --num_windows 100 200 100)'
  )
  parser.add_argument(
      '--num_hidden',
      type=int,
      default=2000,
      help='Number of neurons in hidden layer.'
  )
  parser.add_argument(
      '--regularizer',
      type=float,
      default=0.001,
      help='(Lambda value / 2) of L2 regularizer on weights connected to last layer (0 to exclude).'
  )
  parser.add_argument(
      '--keep_prob',
      type=float,
      default=0.7,
      help='Rate to be kept for dropout.'
  )
  parser.add_argument(
      '--num_classes',
      type=int,
      default=1000,
      help='Number of classes (families).'
  )
  parser.add_argument(
      '--seq_len',
      type=int,
      default=1000,
      help='Length of input sequences.'
  )
  # for learning
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.001,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_epoch',
      type=int,
      default=25,
      help='Number of epochs to train.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=64,
      help='Batch size. Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--train_file',
      type=str,
      default='./data/train.txt',
      help='Directory for input data.'
  )
  parser.add_argument(
      '--test_file',
      type=str,
      default='./data/test.txt',
      help='Directory for input data.'
  )
  parser.add_argument(
      '--prev_checkpoint_path',
      type=str,
      default='',
      help='Restore from pre-trained model if specified.'
  )
  parser.add_argument(
      '--checkpoint_path',
      type=str,
      default='',
      help='Path to write checkpoint file.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/tmp/logs',
      help='Directory for log data.'
  )
  parser.add_argument(
      '--log_interval',
      type=int,
      default=100,
      help='Interval of steps for logging.'
  )
  parser.add_argument(
      '--save_interval',
      type=int,
      default=100,
      help='Interval of steps for save model.'
  )
  # test
  parser.add_argument(
      '--fine_tuning',
      type=bool,
      default=False,
      help='If true, weight on last layer will not be restored.'
  )
  parser.add_argument(
      '--fine_tuning_layers',
      type=str,
      nargs='+',
      default=["fc2"],
      help='Which layers should be restored. Default is ["fc2"].'
  )

  # Arguments for cluster spec setting
  parser.add_argument(
      '--worker_hosts',
      type=str,
      nargs='+',
      default=[],
      help='list of worker nodes for current settings'
  )
  parser.add_argument(
      '--ps_hosts',
      type=str,
      nargs='+',
      default=[],
      help='list of parameter servers for current settings'
  )
  parser.add_argument(
      '--job_name',
      type=str,
      default=None,
      help='type of job for this device'
  )
  parser.add_argument(
      '--task_index',
      type=int,
      default=None,
      help='index of task within the job'
  )
  parser.add_argument(
      '--max_step',
      type=int,
      default=200
  )
  parser.add_argument(
      '--resource_info_file',
      type=str,
      default='./resource_info'
  )
  FLAGS, unparsed = parser.parse_known_args()

  # check validity
  assert( len(FLAGS.window_lengths) == len(FLAGS.num_windows) )

  return FLAGS




def logging(msg, FLAGS):
  fpath = os.path.join( FLAGS.log_dir, "log.txt" )
  with open( fpath, "a" ) as fw:
    fw.write("%s\n" % msg)
  print(msg)
