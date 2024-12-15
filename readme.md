1. 首先安装miniconda，去官网下载：[https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)

2. 确认miniconda已经安装完成，并配置环境变量，在编辑环境变量的用户变量的Path中添加你安装的miniconda的路径：

   ```bash
   D:\ProgramData\miniconda3
   D:\ProgramData\miniconda3\Library\bin
   D:\ProgramData\miniconda3\Scripts
   ```

   你需要找到你自己的miniconda3的位置，将前面的路径做替换。配置好环境变量后打开cmd，输入

   ```bash
   conda --version
   ```

   应该显示你安装的conda版本，如：

   ```bash
   conda 24.9.2
   ```

   这样就表示安装成功

3. 打开`CTS_classify`根目录，在这个路径下打开命令行，使用conda创建一个名字叫`cts`的python3.9环境，它询问需要下载的东西都输入y表示接受。

   ```bash
   conda create -n cts python==3.9
   ```

   然后激活这个环境（如果激活不成功可以重新打开输入下面的命令）

   ```bash
   conda activate cts
   ```

4. 激活环境后开始下载所依赖的包，下面都是在cmd里面输入的命令：

   ```bash
   conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
   pip install -r .\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

   然后将模型文件`save_96.pth`放到目录`static\resources\model_weight`下面。

5. 下载后启动main函数

   ```
   start python main.py
   ```

   然后使用**火狐浏览器(其它浏览器很可能失败)**打开[http://127.0.0.1:5000/login](http://127.0.0.1:5000/login)输入admin和password作为账号密码，登录后，输入你所在的病人的文件夹的绝对路径，并将多个你想要上传的文件夹拖拽到指定为止，就可以等待输出结果，输出的结果会在右侧以表格的形式显示，并提供下载功能，下载的excel可以上传到后续的云端以供病人查询。