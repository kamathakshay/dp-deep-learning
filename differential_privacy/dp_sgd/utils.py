import tensorflow as tf



# parameters for the training
tf.flags.DEFINE_integer("batch_size", 100,
                        "The training batch size.")
tf.flags.DEFINE_integer("batches_per_lot", 1,
                        "Number of batches per lot.")
# Together, batch_size and batches_per_lot determine lot_size.
tf.flags.DEFINE_integer("num_training_steps", 100,
                        "The number of training steps."
                        "This counts number of lots.")

tf.flags.DEFINE_bool("randomize", True,
                     "If true, randomize the input data; otherwise use a fixed "
                     "seed and non-randomized input.")
tf.flags.DEFINE_bool("freeze_bottom_layers", False,
                     "If true, only train on the logit layer.")
tf.flags.DEFINE_bool("save_mistakes", False,
                     "If true, save the mistakes made during testing.")
tf.flags.DEFINE_float("lr", 0.05, "start learning rate")
tf.flags.DEFINE_float("end_lr", 0.05, "end learning rate")
tf.flags.DEFINE_float("lr_saturate_epochs", 0,
                      "learning rate saturate epochs; set to 0 for a constant "
                      "learning rate of --lr.")

# For searching parameters
tf.flags.DEFINE_integer("projection_dimensions", 60,
                        "PCA projection dimensions, or 0 for no projection.")
tf.flags.DEFINE_integer("num_hidden_layers", 1,
                        "Number of hidden layers in the network")
tf.flags.DEFINE_integer("hidden_layer_num_units", 1000,
                        "Number of units per hidden layer")
tf.flags.DEFINE_float("default_gradient_l2norm_bound", 4.0, "norm clipping")
tf.flags.DEFINE_integer("num_conv_layers", 0,
                        "Number of convolutional layers to use. (0, 1, 2)")

tf.flags.DEFINE_string("training_data_path",
                       "../../data/mnist/mnist_train.tfrecord",
                       "Location of the training data.")
tf.flags.DEFINE_string("eval_data_path",
                       "../../data/mnist/mnist_train.tfrecord",
                       "Location of the eval data.")
tf.flags.DEFINE_integer("eval_steps", 10,
                        "Evaluate the model every eval_steps")

# Parameters for privacy spending. We allow linearly varying eps during
# training.
tf.flags.DEFINE_string("accountant_type", "Amortized", "Moments, Amortized.")

# Flags that control privacy spending during training.
tf.flags.DEFINE_float("eps", 0.0,
                      "Start privacy spending for one epoch of training, "
                      "used if accountant_type is Amortized.")
tf.flags.DEFINE_float("end_eps", 0.0,
                      "End privacy spending for one epoch of training, "
                      "used if accountant_type is Amortized.")
tf.flags.DEFINE_float("eps_saturate_epochs", 0,
                      "Stop varying epsilon after eps_saturate_epochs. Set to "
                      "0 for constant eps of --eps. "
                      "Used if accountant_type is Amortized.")
tf.flags.DEFINE_float("delta", 1e-5,
                      "Privacy spending for training. Constant through "
                      "training, used if accountant_type is Amortized.")
tf.flags.DEFINE_float("sigma", 0.0,
                      "Noise sigma, used only if accountant_type is Moments")

# Flags that control privacy spending for the pca projection
# (only used if --projection_dimensions > 0).
tf.flags.DEFINE_float("pca_eps", 0.5,
                      "Privacy spending for PCA, used if accountant_type is "
                      "Amortized.")
tf.flags.DEFINE_float("pca_delta", 0.005,
                      "Privacy spending for PCA, used if accountant_type is "
                      "Amortized.")

tf.flags.DEFINE_float("pca_sigma", 7.0,
                      "Noise sigma for PCA, used if accountant_type is Moments")

tf.flags.DEFINE_string("target_eps", "0.125,0.25,0.5,1,2,4,8",
                       "Log the privacy loss for the target epsilon's. Only "
                       "used when accountant_type is Moments.")
tf.flags.DEFINE_float("target_delta", 1e-5,
                      "Maximum delta for --terminate_based_on_privacy.")
tf.flags.DEFINE_bool("terminate_based_on_privacy", False,
                     "Stop training if privacy spent exceeds "
                     "(max(--target_eps), --target_delta), even "
                     "if --num_training_steps have not yet been completed.")

tf.flags.DEFINE_string("save_path", "results/",
                       "Directory for saving model outputs.")


#Optimizers
tf.flags.DEFINE_string("optimizer", "adam",
                       "Optimizer for training: momentum/adam/SGD default")

tf.flags.DEFINE_float("momentum", 0.99,
                       "momentum parameter")

tf.flags.DEFINE_float("beta1", 0.9,
                       "beta1 parameter for Adam")

tf.flags.DEFINE_float("beta2", 0.999,
                       "beta2 parameter for Adam")

tf.flags.DEFINE_float("adam_epsilon", 0.00000001,
                       "epsilon parameter in Adam")




#Transfer Learning
tf.flags.DEFINE_bool("transfer_learn", False,
                     "True if you want to transfer learn from CIFAR100"
                     "if --num_training_steps have not yet been completed.")

tf.flags.DEFINE_string("transfer_checkpoint",
                       "../results_cifar100/",
                       "Location of the checkpoint data.")



FLAGS = tf.flags.FLAGS