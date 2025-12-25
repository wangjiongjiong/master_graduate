项目部署步骤
1.conda env create -f environment.yaml 这块要注意有两个github安装的，网络不行就手动安装
2.conda activate aerogen
3.下载ckpt https://huggingface.co/Sonetto702/AeroGen baseline的ckpt
4.注意这里clip也需要下载一个
5.下载数据集
6.训练 
conda activate aerogen
python src/train/prepare_weight_r.py
bash configs/stable-diffusion/dual/train_r.sh
7.推理python src/inference/inference.py
demo/txt里面存放文本数据