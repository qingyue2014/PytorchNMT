# PytorchNMT
1. 模型参数设置：epoch number: 10 hidden_size=256 embedding_size=256
2. 训练：将要训练的文本保存在data/train_data.txt，运行MyMain.py文件, mode设为train，训练模型保存在model文件夹中，训练日志保存在all.log。
3. 测试：将要测试的文本保存为test_data.txt，运行MyMain.py文件时，mode设为test.测试结果保存在result.txt.
4. 结果：由于数据量有限，因此本次数据测试结果的BLEU值只有0.46
