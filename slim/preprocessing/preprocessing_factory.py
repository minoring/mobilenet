# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a factory of preprocessing for various models."""

from tensorflow.contrib import slim as contrib_slim

from preprocessing import inception_preprocessing


slim = contrib_slim


def get_preprocessing(name, is_training=False, use_grayscale=False):
  """Returns preprocessing_fn(image, height, width, **kwargs).

  Args:
    name: The name of the preprocessing function.
    is_training: `True` if the model is being used for training and `False`
      otherwise.
    use_grayscale: Whether to convert the image from RGB to grayscale.

  Returns:
    preprocessing_fn: A function that preprocess a single image (per-batch).
      If has the following signature:
        image = preprocessing_fn(image, output_height, output_width, ...).

  Raises:
    ValueError: If preprocessing `name` is not recognized.
  """
  preprocessing_fn_map = {
    'mobilenet_v1': inception_preprocessing
  }

  if name not in preprocessing_fn_map:
    raise ValueError('Preprocessing name [%s] was not recognized' % name)

  def preprocessing_fn(image, output_height, output_width, **kwargs):
    return preprocessing_fn_map[name].preprocess_image(
        image,
        output_height,
        output_width,
        is_training=is_training,
        use_grayscale=use_grayscale,
        **kwargs)

  return preprocessing_fn
