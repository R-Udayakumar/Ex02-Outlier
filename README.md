# Ex02-Outlier
You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

(i) Using IQR detect weight outliers and print them

(ii) Using IQR, detect height outliers and print them
# Explanation :
An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.

# ALGORITHM
# STEP 1
Read the given Data.

# STEP 2
Get the information about the data.

# STEP 3
Detect the Outliers using IQR method and Z score.

# STEP 4
Remove the outliers.

# STEP 5
Plot the datas using Box Plot.

# PROGRAM

 // Developed by : UDAYAKUMAR R (22008609)
```python
# [1]
import pandas as ps
import numpy as np
import seaborn as sns
df=ps.read_csv("bhp.csv")
df
df.head()
df.describe()
df.info()
df.isnull().sum()
df.shape
sns.boxplot(x="price_per_sqft",data=df)

# [2]
q1=df['price_per_sqft'].quantile(0.35)
q3=df['price_per_sqft'].quantile(0.65)
print("First Quantile =",q1,"Second quantile =",q3)
IQR=q3-q1 #INTERQUARTILE RANGE
ul =q3+0.5*IQR
ll =q1-1.5*IQR
df1=df[((df['price_per_sqft']<=l1)&(df['price_per_sqft']>u1))]
df1
df1.shape
sns.boxplot(x='price_per_sqft',data=df1)

# [3]
from scipy import stats
z=np.abs(stats.zscore(df['price_per_sqft']))
df2=df[(z<3)]
df2
print(df2.shape)
sns.boxplot(x='price_per_sqft',data=df2)

# [4] (i) 
df3=ps.read_csv('height_weight.csv')
df3
df3.head()
df3.info()
df3.describe()
df3.isnull().sum()
df3.shape
sns.boxplot(x='weight',data=df3)

# [4] (ii)
q1=df3['weight'].quantile(0.25)
q3=df3['weight'].quantile(0.75)
print('First Quantile =',q1,'Second Quantile =',q3)
IQR=q3-q1
u1=q3+1.5*IQR
l1=q1-1.5*IQR
df4 =df3[((df3['height']>=l1)&(df3['height']<=u1))]
df4.shape
sns.boxplot(x='height',data=df4)
```
# OUTPUT
## DATA FOR BHP.CSV :
![image](https://user-images.githubusercontent.com/118708024/227533147-acf5fea3-bf75-4bdc-a782-17d742b11774.png)

## DATASET HEAD :
![image](https://user-images.githubusercontent.com/118708024/227533012-606b5b33-f1ed-47f5-bd4f-289e335a4e98.png)

## DATASET DESCRIBE :
![image](https://user-images.githubusercontent.com/118708024/227532895-4ca5c4fb-e1d7-40ab-981e-439eb368305a.png)

## DATASET INFO :
![image](https://user-images.githubusercontent.com/118708024/227532764-d2aa2731-ad53-418f-96bb-358a35cade39.png)

## NULL VALUES :
![image](https://user-images.githubusercontent.com/118708024/227532639-816ac244-1e18-4928-9aa8-25bd942303dc.png)

## DATASET SHAPE WITH OUTLIERS :
![image](https://user-images.githubusercontent.com/118708024/227532533-aff2fb63-f00a-4ffc-9019-bd09a6ec5e64.png)

## DATASET BOXPLOT WITH OUTLIERS :
![image](https://user-images.githubusercontent.com/118708024/227532442-d24287a6-63d4-4241-b1df-11784af83df5.png)

## DATASET WITHOUT OUTLIERS :
![image](https://user-images.githubusercontent.com/118708024/227532311-f11dc97e-d1c5-4ba8-b3de-366fa7ac0a99.png)
![image](https://user-images.githubusercontent.com/118708024/227532201-859df000-e7e7-4ea3-94fb-0cd7a5524216.png)

## DATASET SHAPE WITHOUT OUTLIERS :
![image](https://user-images.githubusercontent.com/118708024/227532112-c15fd237-bf49-4de9-9bc8-afec505419ba.png)

## DATASET BOXPLOT WITHOUT OUTLIERS :
![image](https://user-images.githubusercontent.com/118708024/227532000-239e6666-e4a0-40ed-acb9-e6301329d375.png)

## DATASET AFTER REMOVAL OF OUTLIERS USING Z-SCORE :
![image](https://user-images.githubusercontent.com/118708024/227531879-da03eafb-e3f9-498c-8946-7452193fb886.png)

## DATA SHAPE AFTER REMOVAL OF OUTLIERS :
![image](https://user-images.githubusercontent.com/118708024/227531686-0c589529-6d15-4136-8e5c-af8edcc36332.png)

## DATA AFTER REMOVAL OF OUTLIERS USING Z-SCORE :
![image](https://user-images.githubusercontent.com/118708024/227531586-37c03caf-3016-42ac-bcdf-e69cde0d7929.png)

## DATASET FOR HEIGHT_WEIGHT.CSV :
![image](https://user-images.githubusercontent.com/118708024/227531474-46dce348-5ca6-4950-8a8e-f7469c58bdb1.png)

## DATASET HEAD :
![image](https://user-images.githubusercontent.com/118708024/227531356-27e94aba-635d-421d-ba01-cab84c5649c5.png)

## DATASET INFO :
![image](https://user-images.githubusercontent.com/118708024/227529874-a4f6ec22-f780-4c06-8aa1-ffda1c6a3de4.png)

## DATASET DESCRIBE :
![image](https://user-images.githubusercontent.com/118708024/227529470-47eba2d7-fd5b-4b7c-91d6-2e9dbeabb71d.png)

## NULL VALUES :
![image](https://user-images.githubusercontent.com/118708024/227529315-ee0a34ec-017b-412a-acbf-bc11fbd521ed.png)

## DATA BOXPLOT WITH OUTLIERS :
![image](https://user-images.githubusercontent.com/118708024/227529164-6144b014-e2d1-4daa-b8d9-8db634fc9e47.png)

## DATA AFTER REMOVING OUTLIERS USING IQR METHOD :
![image](https://user-images.githubusercontent.com/118708024/227528401-6a1be11e-b0c7-42ec-9741-8729a5266929.png)
![image](https://user-images.githubusercontent.com/118708024/227528605-4485f6d0-b412-4cbd-9d6d-03881076562f.png)

## DATA SHAPE :
![image](https://user-images.githubusercontent.com/118708024/227528810-9049e9eb-0f7b-46b4-be2d-3bfa4a2cd456.png)

## DATA BOXPLOT AFTER REMOVING OUTLIERS USING IQR METHOD :
![image](https://user-images.githubusercontent.com/118708024/227527687-970e8b96-cd66-458d-849c-d5a59085de7a.png)

# RESULT
Thus the given datasets are readed and outliers are detected and removed using IQR and z-score method.

