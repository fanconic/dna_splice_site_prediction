echo 'Standard Training.'
module load eth_proxy python_gpu/3.7.1 cudnn/8.0.5 cuda/11.0.3
python -u train_spliceAI.py
python -u predict_spliceAI.py
echo 'Script completed!'
