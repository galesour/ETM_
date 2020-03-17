<!--
 * @Author: 一蓑烟雨任平生
 * @Date: 2020-03-13 11:09:39
 * @LastEditTime: 2020-03-15 13:50:39
 * @LastEditors: Please set LastEditors
 * @Description: ETM论文读书笔记
 * @FilePath: /ETM_note.md
 -->
# ETM读书笔记
论文还剩有一点没看懂，但是论文中对应的代码都看了一遍了。机器学习的课程看到15了。做了笔记。
***
## LDA
## Word Embedding
word2vec包括两种，一种是skip-gram, 一种是CBOW。在论文中，我们是使用的CBOW来做的向量化。这里简单介绍一下
## ETM
ETM = LDA + Word Embedding, ETM相当于在传统LDA的基础上，增加了Word_Embedding，在论文中是使用的CBOW来对单词向量化的。