from __future__ import division

import tensorflow as tf

from per_example_gradients import PerExampleGradients



def clipped_gradients(loss, params, clipbound = None, epsilon = None, delta = None) :
  xs = [tf.convert_to_tensor(x) for x in var_list]
  px_grads = per_example_gradients.PerExampleGradients(loss, xs)
  sanitized_grads = []
  for px_grad, v in zip(px_grads, var_list):
    sanitized_grad = self._sanitizer.sanitize(px_grad, clipbound = clipboud, 
      epsilon = epsilon, delta =  delta)
    sanitized_grads.append(sanitized_grad)

  return sanitized_grads





def sanitize(x, clipbound = None, epsilon=None, delta= None):
    """Sanitize the given tensor.
    This sanitizes a given tensor by first applying l2 norm clipping and then
    adding noise. 
    Args:
      x: the tensor to sanitize.
      clipbound: the bound on l2 norm beyond which we clip example-wise
        gradients
      epsilon: eps for (eps,delta)-DP. Use it to compute sigma.
      delta: delta for (eps,delta)-DP. Use it to compute sigma.
    Returns:
      a sanitized tensor
    """

    eps, delta = (epsilon, delta)
    with tf.control_dependencies(
        [tf.Assert(tf.greater(eps, 0),
                   ["eps needs to be greater than 0"]),
         tf.Assert(tf.greater(delta, 0),
                   ["delta needs to be greater than 0"])]):
      # The following formula is taken from
      #   Dwork and Roth, The Algorithmic Foundations of Differential
      #   Privacy, Appendix A.
      #   http://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
      sigma = tf.sqrt(2.0 * tf.log(1.25 / delta)) / eps

    l2norm_bound = clipbound
    if l2norm_bound is not None:
      x = BatchClipByL2norm(x, l2norm_bound)

    num_examples = tf.slice(tf.shape(x), [0], [1])
    saned_x = AddGaussianNoise(tf.reduce_sum(x, 0),
                                         sigma * l2norm_bound)
    #else:
    #  saned_x = tf.reduce_sum(x, 0)
    return saned_x





#TAKEN FROM UTILS OF OP


def AddGaussianNoise(t, sigma):
  """Add i.i.d. Gaussian noise (0, sigma^2) to every entry of t.
  Args:
    t: the input tensor.
    sigma: the stddev of the Gaussian noise.
    name: optional name.
  Returns:
    the noisy tensor.
  """

  with tf.name_scope(values=[t, sigma],
                     default_name="add_gaussian_noise") as name:
    noisy_t = t + tf.random_normal(tf.shape(t), stddev=sigma)
  return noisy_t





#TAKEN FROM UTILS OF OP


def BatchClipByL2norm(t, upper_bound):
  """Clip an array of tensors by L2 norm.
  Shrink each dimension-0 slice of tensor (for matrix it is each row) such
  that the l2 norm is at most upper_bound. Here we clip each row as it
  corresponds to each example in the batch.
  Args:
    t: the input tensor.
    upper_bound: the upperbound of the L2 norm.
  Returns:
    the clipped tensor.
  """

  assert upper_bound > 0
  with tf.name_scope(values=[t, upper_bound], name=name,
                     default_name="batch_clip_by_l2norm") as name:
    saved_shape = tf.shape(t)
    batch_size = tf.slice(saved_shape, [0], [1])
    t2 = tf.reshape(t, tf.concat(axis=0, values=[batch_size, [-1]]))
    upper_bound_inv = tf.fill(tf.slice(saved_shape, [0], [1]),
                              tf.constant(1.0/upper_bound))
    # Add a small number to avoid divide by 0
    l2norm_inv = tf.rsqrt(tf.reduce_sum(t2 * t2, [1]) + 0.000001)
    scale = tf.minimum(l2norm_inv, upper_bound_inv) * upper_bound
    clipped_t = tf.matmul(tf.diag(scale), t2)
    clipped_t = tf.reshape(clipped_t, saved_shape, name=name)
  return clipped_t
