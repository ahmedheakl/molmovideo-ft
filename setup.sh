
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory
pip3 install -e ".[torch,deepspeed,vllm,bitsandbytes,metrics,liger-kernel]"
pip3 install ujson decord tensorflow tf-keras wandb natsort
pip3 install flash-attn --no-build-isolation