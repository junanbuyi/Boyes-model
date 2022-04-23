import pandas as pd
import numpy as np
train = pd.read_csv("C:/Users/13279/Desktop/train.csv")
test = pd.read_csv("C:/Users/13279/Desktop/test.csv")
print("shape:",train.shape,"shape:",test.shape)
#合并数据集，对数据进行清洗
full = train.append(test,ignore_index = True)
print(full)
#查看数据
print(full.head())
#填充缺失数据
full['Age'] = full['Age'].fillna(full['Age'].mean())
full['Fare'] = full['Fare'].fillna(full['Fare'].mean())
full['Cabin'] = full['Cabin'].fillna("N")#N代表不知道
#统计各个港口出现的次数
from collections import Counter
print(Counter(full['Embarked']))
full['Embarked'] = full['Embarked'].fillna("s")#s港口上的人数最多，所以将缺失值填充为s
#将数据清洗之后，再次查看确认是否清洗完毕
print(full.info())

#将性别映射成数值
sex_map = {"male":1,"female":0}
full['Sex'] = full['Sex'].map(sex_map)
print(full['Sex'])

# 存放提取后的特征
pclassDf = pd.DataFrame()
# 使用get_dummies进行one-hot编码，列名前缀为Pclass
pclassDf = pd.get_dummies(full['Pclass'], prefix = 'Pclass')
full = pd.concat([full,pclassDf],axis = 1)
full.drop('Pclass',axis = 1,inplace = True)
print(full)

# 存放提取后的特征
embarkedDF = pd.DataFrame()
# 使用get_dummies进行one-hot编码，列名前缀为Embarked
embarkedDF = pd.get_dummies(full['Embarked'],prefix = 'Embarked')
print(embarkedDF)
# 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full, embarkedDF], axis = 1)
# 因已对登船港口（Embarked）进行了one-hot编码产生虚拟变量，故删除 Embarked
full.drop('Embarked', axis = 1, inplace = True)
print(full)



# 从姓名中获取头衔
# split（）通过制定分隔符对字符串进行切片
def getTitle(name):
    str1 = name.split(',')[1]
    str2 = str1.split('.')[0]
    # strip()移除字符串头尾制定的字符（默认为空格）
    str3 = str2.strip()
    return str3

titleDf = pd.DataFrame()
titleDf['Title'] = full['Name'].map(getTitle)
titleDf.head()

# 从姓名中头衔字符串与自定义头衔类别的映射
title_mapDict = {
    'Capt':'Officer',
    'Col':'Officer',
    'Major':'Officer',
    'Jonkheer':'Royalty',
    'Don':'Royalty',
    'Sir':'Royalty',
    'Dr':'Officer',
    'Rev':'Officer',
    'the Countess':'Royalty',
    'Dona':'Royalty',
    'Mme':'Mrs',
    'Mlle':'Miss',
    'Mr':'Mr',
    'Mrs':'Mrs',
    'Miss':'Miss',
    'Master':'Master',
    'Lady':'Royalty'
}
# print(title_mapDict)
titleDf['Title'] = titleDf['Title'].map(title_mapDict)
# 使用get——dummies进行one-hot编码
titleDf = pd.get_dummies(titleDf['Title'])
titleDf.head()

# 添加one-hot编码产生的虚拟变量到泰坦尼克号数据集full
full = pd.concat([full, titleDf], axis = 1)
# 删除姓名（Name）这一列
full.drop('Name', axis = 1, inplace = True)
full.head()

# 存放客舱号信息
cabinDf = pd.DataFrame()
# 客舱号的类别值是首字母， eg：C85
#定义匿名函数 lambda，用于查找首字母
full['Cabin'] = full['Cabin'].map(lambda c:c[0])
# 使用get_dummies 进行one-hot 编码， 列名前缀为Cabin
cabinDf = pd.get_dummies(full['Cabin'], prefix = 'Cabin')
cabinDf.head()

# 添加one-hot编码产生的虚拟变量到泰坦尼克号数据集full
full = pd.concat([full, cabinDf], axis = 1)
# 删除客舱号等级（Pclass）这一列
full.drop('Cabin', axis = 1, inplace = True)
full.head()

# 存放家庭信息
familyDf = pd.DataFrame()
# 家庭人数 = 同代直系亲属数（SibSp）+ 不同代直系亲属数（Parch）+ 乘客自己
familyDf['Familysize'] = full['SibSp'] + full['Parch'] + 1
"""
家庭类别：
小家庭Family_Single：家庭人数 = 1
中等家庭Family_Small：2<家庭人数<=3
大家庭Family_Large:家庭人数>3
"""
# if条件为真是返回if前面内容， 否则返回0
familyDf['Family_Single'] = familyDf['Familysize'].map(lambda s : 1 if s == 1 else 0)
familyDf['Family_Small'] = familyDf['Familysize'].map(lambda s : 1 if 2 < s <=3 else 0)
familyDf['Family_Large'] = familyDf['Familysize'].map(lambda s : 1 if s > 3 else 0)
familyDf.head()

# 存放年龄信息
ageDf = pd.DataFrame()
"""
年龄类别：
儿童Child：0<=年龄<8
青少年Teenager：8<=年龄<16
青年Youth：16<=年龄<=35
中年Middle_age：35<年龄<55
老年Older:55<=年龄
"""
ageDf['Child'] = full['Age'].map(lambda a : 1 if 0 < a < 4 else 0)
ageDf['Teenager'] = full['Age'].map(lambda a : 1 if 4 <= a < 20 else 0)
ageDf['Youth'] = full['Age'].map(lambda a : 1 if 20 <= a < 35 else 0)
ageDf['Middle_age'] = full['Age'].map(lambda a : 1 if 35 <= a < 55 else 0)
ageDf['Older'] = full['Age'].map(lambda a : 1 if a >= 55 else 0)
ageDf.head()

# 添加one-hot编码产生的虚拟变量到泰坦尼克号数据集full
full = pd.concat([full, ageDf], axis = 1)
# 删除Age这一列
full.drop('Age', axis = 1, inplace = True)
full.head()

# 相关矩阵
corrDf = full.corr()
# 查看各个特征与生成情况（Survived）的相关系数，ascending = False表示按降序排列
corrDf['Survived'].sort_values(ascending = False)


# 特征选择
full_X = pd.concat([
    titleDf,  # 头衔
    pclassDf,
    full['Fare'],
    full['Sex'],
    cabinDf,
    embarkedDF
], axis = 1)
full_X.head()

# 原始数据共有891行
sourceRow = 891
"""
原始数据集sourceRow是从Kaggle下载的训练集，可知共有891条数据从特征集
full_X中提取原始数据前891行数据时需减去1，因为行号是从0开始
"""
# 原始数据集：特征
source_X = full_X.loc[0:sourceRow-1,:]
# 原始数据集：标签
source_y = full.loc[0:sourceRow-1,'Survived']
# 预测数据集：特征
pred_X = full_X.loc[sourceRow:,:]
# 查看原始数据集有多少行
print('原始数据集：', source_X.shape[0])
# 查看预测数据集有多少行
print('预测数据集：',pred_X.shape[0])

# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

# 建立模型所需的训练数据集合测试集
train_X,test_X,train_y,test_y = train_test_split(source_X,source_y,train_size=0.85)
# 输出数据集大小
print('原始数据集特征：',source_X.shape,
     '训练数据集特征：',train_X.shape,
     '测试数据集特征：',test_X.shape,)
print('原始数据集标签：',source_y.shape,
     '训练数据集标签：',train_y.shape,
     '测试数据集标签：',test_y.shape,)

# 第一步：导入算法
from sklearn.linear_model import LogisticRegression
# 第二步：创建模型：逻辑回归
model = LogisticRegression()

# 第三步：训练模型
model.fit(train_X, train_y)


# 第四步评估模型
# 分类问题 score 得到的是模型正确率
print(model.score(test_X, test_y))

# 使用机器学习模型，对预测数据集中的生存情况进行预测
pred_y = model.predict(pred_X)

# 生成的预测值是浮点数，但是Kaggle要求提交的结果是整数型
# 使用astype对数据类型进行转换
pred_y = pred_y.astype(int)
# 乘客id
passenger_id = full.loc[sourceRow:,'PassengerId']
# 数据框：乘客id， 预测生存情况
predDf = pd.DataFrame({'PassengerId':passenger_id, 'Survived':pred_y})
predDf.shape
print(predDf)
# 保存结果
predDf.to_csv('C:\\Users\\13279\\Desktop\\1.csv', index=False)
