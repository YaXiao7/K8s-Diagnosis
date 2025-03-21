# K8s-Diagnosis
## 各文件/文件夹功能描述：
### assets文件夹存储Kubernetes YAML文件
### data文件夹存储相关对数据集处理后的JSON数据
### LLM4ConfigDetect文件夹存储相关的检测数据集构建、微调数据集构建、微调训练模型相关代码
### LLM4ConfigDetect文件夹下的weights文件夹存储微调训练模型后权重的相关代码
### model文件夹存放LLM模型
## 
## 目录结构说明：
```
.
├── assets
│   └── XXX.yaml
├── data
│   ├── instruction_templates_update.json  （存放的是微调数据集的模板文件）
│   ├── cleaned_files.csv （assets文件夹中，有的YAML文件没有被检测出问题，因此进行一个清理）
│   └── split_files.json （assets文件夹被清理后，但是每个YAML文件中以"---"来划分不同资源的定义。并不是每个YAML的资源定义都会出现问题，分割出每个YAML中有问题的具体资源定义，并记录下索引，这个索引就是指以"---"划分后的顺序）
│   └──top_1000_split_files.json （分割后的文件，选取前1000个YAML文件，选取1000个YAML文件用于构造数据集）
├── LLM4ConfigDetect
│   ├── Clean_yaml.py （assets文件夹中，有的YAML文件没有被检测出问题，因此进行一个清理，主要是判断YAML中是否会被检测出问题，来判断）
│   ├── Download_model.py （下载LLM模型）
│   └── Instruction_dataset_construction_update.py （微调数据集构建）
│   └── Splite_clean_yaml.py （将清理后的YAML文件进行分割，以便构造微调数据集）
│   └── Train.py （微调训练模型）
```