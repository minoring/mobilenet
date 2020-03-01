# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Build and train mobilenet_v1 with options for quantization"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import slim as contrib_slim

from datasets import dataset_factory
from nets import mobilenet_v1
from preprocessing import preprocessing_factory

slim = contrib_slim

flags = tf.compat.v1.app.flags

flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('task', 0, 'Task')
flags.DEFINE_integer('ps_tasks', 0, 'Number of ps')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('num_classes', 1001, 'Number of classes to classify')
flags.DEFINE_integer('number_of_steps', None,
                     'Number of training steps')
flags.DEFINE_integer('image_size', 224, 'Input image resolution')
flags.DEFINE_float('depth_multiplier', 1.0, 'Depth multiplier for mobilenet')
flags.DEFINE_bool('quantize', False, 'Quantize training')
flags.DEFINE_string('fine_tune_checkpoint', '',
                    'Checkpoint from which to start finetuning')
flags.DEFINE_string('checkpoint_dir', '',
                    'Directory for writing training checkpoints and logs')
flags.DEFINE_string('dataset_dir', '', 'Location of dataset')
flags.DEFINE_integer('log_every_n_steps', 100, 'Number of steps per log')
flags.DEFINE_integer('save_summaries_secs', 100
                     'How often to save summaries, secs')
flags.DEFINE_integer('save_interval_secs', 100,
                     'How often to save checkpoints, secs')

FLAGS = flags.FLAGS

_LEARNING_RATE_DECAY_FACTOR = 0.94


def get_learning_rate():
  if FLAGS.fine_tune_checkpoint:
    # If we are fine tuning a checkpoint we need to start at lower learning
    # rate since we are farther along on training.
    return 1e-4
  else:
    return 0.045


def get_quant_delay():
  if FLAGS.fine_tune_checkpoint:
    # We can start quantizing immediately if we are finetuning.
    return 0
  else:
    # We need to wait for the model to train a bit before we quantize if we are
    # training from scratch.
    return 250000 # TODO(minho): What is this number???


def imagenet_inputs(is_training):
  """Data reder for imagenet.

  Reads in imagenet data and performs pre-processing on the images.

  Args:
    is_training: bool specifying if train or validation dataset is needed.

  Returns:
    A batch of images and labels.
  """
  if is_training:
    dataset = dataset_factory.get_dataset('imagenet', 'train', 
                                          FLAGS.dataset_dir)
  else:
    dataset = dataset_factory.get_dataset('imagenet', 'validation',
                                          FLAGS.dataset_dir)
  
  provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      shuffle=is_training,
      common_queue_capacity=2*FLAGS.batch_size,
      common_queue_min=FLAGS.batch_size)
  [image, label] = provider.get(['image', 'label'])

  image_preprocessing_fn = preprocessing_factory.get_preprocessing(
      'mobilenet_v1', is_training=is_training)
