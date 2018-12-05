# dp-deep-learning
Class Project for CS395T (Deep Learning Seminar)

The relevant files for the differentially private optimizers are in /differential_privacy/dp_sgd/

## Train
First train a network with 2 convolutional layers and 1 hidden layer using Cifar100 dataset from scrach:

`cd ./differential_privacy/dp_sgd/dp_cifar100`  
`python dp_cifar100.py --save_path ../results_cifar100 \
                     --num_training_steps 150000  \
                     --projection_dimensions 0  \
                     --num_conv_layers 2  \
                     --accountant_type Moments \
                     --sigma 0 \
                     --eval_steps 5000`  
                     
Then use the pretrained weights to initialize and then train the same network on Cifar10 dataset.

`cd ../dp_cifar10`  
`MODEL_NAME=m_18-11-29-15-44_aAmortized_b100_lr0.05_eps0.0_delta1e-05`  
`python dp_cifar10.py --save_path ../results_cifar/SGD/ \
                     --num_training_steps 70000  \
                     --projection_dimensions 0  \
                     --num_conv_layers 2  \
                     --accountant_type Amortized \
                     --eps 32.0 \
                     --delta 1e-6 \
                     --eval_steps 500 \
                     --transfer_learn True \
                     --transfer_checkpoint ../results_cifar100/$MODEL_NAME/\
                     --optimizer SGD \`

To train on TACC, load the following modules and follow the instructions above:  

`export PYTHONPATH="."`  
`module load gcc/4.9.3  cuda/8.0  cudnn/5.1  python/2.7.12`  
`module load tensorflow-gpu/1.0.0 `  
       
Or use the shell scripts `run` under ./dp_cifar100 and ./dp_cifar10 to submit job.

To train the model using MNIST dataset, follow the instructions in dp_sgd/README.md.

## Visualization
To produce the plots in our report, run:

`cd ./differential_privacy/dp_sgd/`  
`python visualize.py`

This will plot "Accuracy vs Epochs" for the newest model under `./differential_privacy/dp_sgd/results_cifar`.
Run `bash visual_all` to plot for every models under the same directory.
Or set `--model_name` to plot for a specific model, set `--save_dir` to change the directory where models are saved.