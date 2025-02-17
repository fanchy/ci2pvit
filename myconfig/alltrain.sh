#
CUDA_VISIBLE_DEVICES=2 mim train mmpretrain config_ci2pvitplus.py
CUDA_VISIBLE_DEVICES=2 mim train mmpretrain config_ci2pvit.py 
CUDA_VISIBLE_DEVICES=2 mim train mmpretrain config_posembed_ci2pvit.py 
CUDA_VISIBLE_DEVICES=2 mim train mmpretrain config_vitp16.py

# mim analysis_tools/get_flops  config_ci2pvit.py --shape 256
# pip install compressai
# 画精度图
# python C:\myai\mmpretrain\tools/analysis_tools/analyze_logs.py plot_curve scalars.json scalars2.json  --keys accuracy/top1 --legend exp1 exp2 --out plot.pdf
# 生成测试数据
# python C:\myai\mmpretrain\tools/test.py config_train.py ./work_dirs/config_train/epoch_300.pth
# 混淆矩阵
# python C:\myai\mmpretrain\tools/analysis_tools/confusion_matrix.py config_train.py ./work_dirs/config_train/epoch_300.pth  --out output --show --show-path output 
