
ENV_NAME=.molmo-train
conda create -n $ENV_NAME python=3.11 -y
conda activate $ENV_NAME
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory
pip3 install -e ".[torch,deepspeed,vllm,bitsandbytes,metrics,liger-kernel]"
pip3 install ujson decord tensorflow tf-keras wandb natsort
pip3 install flash-attn --no-build-isolation