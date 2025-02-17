CUDA_VISIBLE_DEVICES=4 mim train mmpretrain config_ci2pvitplus_efficientnet_imagenet.py

#--work-dir work_imagenetplus
# python C:\myai\mmpretrain\tools/analysis_tools/get_flops.py  config_ci2pvit.py --shape 256
# pip install compressai
# 画精度图
# python C:\myai\mmpretrain\tools/analysis_tools/analyze_logs.py plot_curve scalars.json scalars2.json  --keys accuracy/top1 --legend exp1 exp2 --out plot.pdf
# 生成测试数据
# python C:\myai\mmpretrain\tools/test.py config_train.py ./work_dirs/config_train/epoch_300.pth
# 混淆矩阵
# python C:\myai\mmpretrain\tools/analysis_tools/confusion_matrix.py config_train.py ./work_dirs/config_train/epoch_300.pth  --out output --show --show-path output 
