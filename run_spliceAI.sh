echo 'Standard Training.'
module load eth_proxy python_gpu/3.7.1 cudnn/8.0.5 cuda/11.0.3
python -u train.py
python -u predict.py
echo 'Script completed!'
