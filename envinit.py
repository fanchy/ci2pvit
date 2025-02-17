import os
#os.system('pip install wget')
#import wget
#wget.download(url, filename)
#import ssl
# 取消ssl全局验证
#ssl._create_default_https_context = ssl._create_unverified_context
import os


def cmd_batch(cmdlist):
    for cmd in cmdlist:
        print(cmd)
        os.system(cmd)

def envinit():
    cmd_init_cmd = [
        'apt-get update',
        'apt-get install -y subversion',
        'mkdir -p dataset',
        'git config --global credential.helper store',
        #'git clone https://gitee.com/evanown/paper.git'
        'cd dataset && wget -c https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        'tar -zxvf flower_photos.tgz',
        'mkdir -p ~/.pip',
        'touch ~/.pip/pip.conf',
    ]
    cmd_batch(cmd_init_cmd)
    f = open('/root/.pip/pip.conf', 'w')
    strdata = '''
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple/
[install]
trusted-host=pypi.tuna.tsinghua.edu.cn
    '''
    f.write(strdata)
    f.close()
    return
def mminit():
    cmd_init_cmd = [
        #'conda init bash',
        #'conda create -n open-mmlab python=3.8 -y',
        #'conda activate open-mmlab',
        #'conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia',
        'pip install -U openmim',
        'mim install "mmpretrain>=1.0.0rc8"',
        'pip install compressai',
        'sudo apt-get update',
        'sudo apt-get install libgl1',
    ]
    cmd_batch(cmd_init_cmd)
"""
export CUDA_HOME=/usr/share/user/cuda116
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
"""
mminit()