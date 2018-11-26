#For running on TACC:

cd /work/05945/kamath/dp-project2
export PYTHONPATH="."
module load gcc/4.9.3  cuda/8.0  cudnn/5.1  python/2.7.12
module load tensorflow-gpu/1.0.0


# For downloading data 
cd slim/
DATA_DIR="../differential_privacy/data"
python download_and_convert_data.py --dataset_name=mnist --dataset_dir="${DATA_DIR}"
mkdir /tmp/mnist_dir

TO RUN:
cd ../
python differential_privacy/dp_sgd/dp_mnist/dp_mnist.py