
Ark
项目介绍：
Ark  高效的自动化机器学习工具。原型来自 4paradigm/AutoX
ref :https://github.com/4paradigm/AutoX

它的特点包括:
效果出色: Ark在多个kaggle数据集上，效果显著优于其他解决方案(见效果对比)。
简单易用: Ark的接口和sklearn类似，方便上手使用。
通用: 适用于分类和回归问题。
自动化: 无需人工干预，全自动的数据清洗、特征工程、模型调参等步骤。
灵活性: 各组件解耦合，能单独使用，对于自动机器学习效果不满意的地方，可以结合专家知识，Ark提供灵活的接口。

主要内容包括
autox_competition: 主要针对于表格类型的数据挖掘
autox_server: 用于上线部署的automl服务
autox_interpreter: 机器学习可解释功能
autox_nlp: 对文本列进行处理的自动化工具
autox_recommend: 推荐系统的自动机器学习
快速上手
    python3 run.py input_path out_path
    python3 naive_decison_tree.py

