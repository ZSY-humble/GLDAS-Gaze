git clone https://github.com/cvlab-stonybrook/HAT.git

conda create --name gldas --file requirements.txt 
 pip install -r a.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 

 git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2


下载这个项目的代码GitHub - fundamentalvision/Deformable-DETR: Deformable DETR: Deformable Transformers for End-to-End Object Detection.
cd ./models/ops
sh ./make.sh

git clone https://github.com/cocodataset/panopticapi.git
python setup.py build_ext --inplace  
python setup.py build_ext install


Install MSDeformableAttn，下面目录存在于HAT文件中
cd ./hat/pixel_decoder/ops
sh make.sh


apt-get update
apt-get install libgl1-mesa-glx
 pip install -r a.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 