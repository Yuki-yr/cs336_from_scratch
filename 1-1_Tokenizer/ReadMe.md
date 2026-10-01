# README

## 一、代码命名说明

文件名以“pure”结尾的代码文件里面是简洁注释的代码（可以更清晰地观察代码结构与逻辑）

同名没有“pure”结尾的代码文件里面是详细注释的代码（可以快速了解代码细节和部分陌生的语法知识）

 

## 二、代码功能说明

train_BPE.py：通过语料训练得到词表和merge文件。

tokenizer.py: 实现了Tokenzier类, 可以实现原始字符串与ID列表的相互转换。

preprocess.py: 通过调用tokenizer.py实现的Tokenzier类，并进行一些其他处理，将原始语料txt文件转换为bin文件。



## 三、训练数据获取方式

Download the TinyStories data and a subsample of OpenWebText

```
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```



## 四、notebook文件夹

里面是本节学习笔记，包含学习资源。

