环境参考requirements.txt
如果还是无法安装请
```python
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install scipy
pip install simpleitk
pip install scikit-learn
```

主要的代码就是demo.py

1. 将测试数据放入demo_dataset中，或者自己设定的路径。测试数据可以是一个文件夹，文件夹中都是.dcm文件，或者是一个nii.gz的图像。测试数据： https://www.dropbox.com/scl/fo/cvipp09e6u12bynzpnidc/AF7z5H_gfeOjfixkuGcI_nI?rlkey=n7393o4mqh8phd8magu82i6g1&st=ggs71bge&dl=0
   
2. 训练的weight放在model_weight中，或者自己设定的路径，weight:https://www.dropbox.com/scl/fo/nm0udlfnrvg40f8olkm3j/AG6O_QId7_KQoWPXqj_pAi8?rlkey=33g5o5i84rskneordpts3q6e1&st=5ayak7o8&dl=0
   

命令为

```python
python demo.py --resume_path model_weight/save_96.pth --image_path demo_dataset/image_dicom_0001_01
```
--resume_path就是需要load的weight, image_path就是测试的图像的路径