# Understanding Customers: Setup, Wrangling, & EDA

*Company Background*
* An electronics retailer
* In business for 40 years
* 30 physical stores in Southeast
* Started selling online 1+ year ago

***Task: mine data for insights that will help Blackwell grow their online sales.***

Danielle's Questions:
1. Do customers in different regions spend more per transaction?
2. Which regions spend the most/least?
3. Is there a relationship between number of items purchased and amount spent?

Additional Questions:
* Do customers spend more in online or in-store transactions?
* How much do customers spend per item in online vs. in-store transactions?
* What is the proportion of online vs. in-store transactions in each region?

## Python Project Setup

At the beginning of every new project you need to do a few things before creating your project jupyter notebook:
1. create a folder for your project in a directory that anaconda will be able to see. ben recommended creating all project folders in the root c:\  dir instead of the default c:\users\kpiat
    * i got a permission denied error when i tried to create my first jup nb from c:\   to fix the problem i have to run the anaconda prompt as an administrator (this gets arount the windows UAC security features, allowing you to make changes to the files in your computer's OS —i think)
2. virtual environment— you may also want to create a virtual environment for your project. by default, when you open the conda prompt, you are in the (base) environment, which has access to all the pkgs, IDEs, etc. that come with your anaconda installation. 
    * but it's recommended that you don't use the (base) env to do your work for a few reasons.
        * you may need to install packages that didn't come pre-installed with your anaconda installation. and installing those packages could alter the pre-installed packages, which could cause other projects created in and using the base env to break (**verify**)
        * you want to alter/customize some of the pre-installed pkgs for a particular project but you don't want alter every subsequent installation of those pkgs. (**verify** it occurs to me that future installations of altered pkgs would only be altered if pkgs installed into ve are installed from the base environment, as opposed to the conda repo, eg. and i'm not sure that's the case)
        * you want to have dedicated environments for different purposes (e.g. machine learning, data analysis)
        * the pkgs in base may be updated with conda, and those updates may break stuff in your projects. by creating a ve, you keep the stable, unchanging version of pkgs that you installed in the ve (**verify**)
    * to create and activate a ve:
        1. open anaconda prompt
        2. change directories to the dir where you have the project folder
        3. enter ```conda create --name nameofenv```
        4. activate the ve ```activate nameofenv```
        5. install pkgs in ve ```conda install pandas, numpy, jupyter notebook, ipykernel```
        6. (optional) create jup nb kernel for ve ```ipython kernel install --user --name = nameofenv```
        7. deactivate the ve ```conda deactivate```
        8. re-activate the ve ```activate nameofenv```
    * with your ve activated, launch jupyter nb ```jupyter notebook``` 

### Jupyter Setup

*create a juypter notebook for your project*  
once jupyter homepage launches in your browser, click New, and select the kernel you want to use for the nb:
* if using the base env, select 'Python 3'
* if using a ve and created a jup nb kernel (see above),select nameofenv from the dropdown menu
* if using a virtual env and launched jup nb from activated ve, select Python 3 from dropdown. (if you launch jup nb from a ve, jup nb you create will automatically be in (using?) that ve)

## Import Packages & Data


```python
#import data science pkgs I need
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
#import data file
df = pd.read_csv('Demographic_Data.csv')
```

*Notes:*  
  
 ☼ error fix — first time i ran the code i got a file not found error because the data file was not in the same directory/folder as this jup nb. when i moved the data file into the same dir and re-ran, it worked.


```python
#view first 5 lines of dataframe to confirm it looks as expected
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>in-store</th>
      <th>age</th>
      <th>items</th>
      <th>amount</th>
      <th>region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>37</td>
      <td>4</td>
      <td>281.03</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>35</td>
      <td>2</td>
      <td>219.51</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>45</td>
      <td>3</td>
      <td>1525.70</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>46</td>
      <td>3</td>
      <td>715.25</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>33</td>
      <td>4</td>
      <td>1937.50</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#view characteristics of each variable in the dataframe
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 80000 entries, 0 to 79999
    Data columns (total 5 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   in-store  80000 non-null  int64  
     1   age       80000 non-null  int64  
     2   items     80000 non-null  int64  
     3   amount    80000 non-null  float64
     4   region    80000 non-null  int64  
    dtypes: float64(1), int64(4)
    memory usage: 3.1 MB
    

*Notes*  
* 80,000 transactions
* 5 variables:
    * in-store — whether each transaction occurred online or in-store. 1=in-store, 0=online
    * age — customer's age.
    * items — number of items purchased in each transaction.
    * amount — dollar amount spent for each transaction.
    * region — region each transaction occured. 1=North, 2=South, 3=East, 4=West

*can you easily tell which features/variables are numeric and which are nominal?*  
* i can tell all the features have numeric values b/c their dtypes are either int64 (integers) or float64 (real numbers)
* ***nominal*** — categorical values that don't contain any order (eg. 0=online, 1=instore)
* ***ordinal*** —categorical values that do have an order, but the intervals between values are not uniform (e.g. ecomomic standing = low, medium, high)
* ***numerical*** — like ordinal values, but the intervals between values are uniform (e.g. 
    * 2 types *interval*(has no fixed zero point) and *ratio*(has a fixed zero point)
* what type of data you're analyzing is important because it partially determines what kind of statistical tests you use (parametric vs. non-parametic tests)
* i think *in-store* and *region* are nominal variables 
* i think age, items, and amount are numerical (ratio) variables 
* [this](https://pbpython.com/pandas_dtypes.html) website explains pandas data types and why they matter
               

# Data Wrangling 

Commonly includes:
* data-cleaning
* data transformation
* data reduction
* data discretization
* text cleaning  

i asked about a data prep checklist? ben directed me to ["Easy Guide to Data Preprocessing in Python"](https://www.kdnuggets.com/2020/07/easy-guide-data-preprocessing-python.html)
which walks through:
 * splitting the data into training and validation sets (presumably this only applies in situations where the analysis involves some kind of machine learning) using scikit-learn```train_test_split```
 * taking care of missing values
     * see the missing values ```data.isna().sum()```
     * shows how to drop missing values or fill with mean, median, mode for relevant variable(s)
 * taking care of categorical features
     * turning variables with string values into categorical or ordinal variables by adding number labels to strings
     * also shows how to create a new categorical variable (e.g. continent) from other categorical variables in the dataset. 
     * discusses using OneHotEncoder. But i don't get why.
 * normalization of data set
     * the goal of normalization is get variables to 
  

## Data Cleaning


```python
#get rid of duplicate transactions
df = df.drop_duplicates()
```

Notes: this command seems to be getting rid of duplicate rows. it seems to me that i should look at the duplicates before i delete them in case they might tell me something interesting and/or concerning about the quality of the data.
* what is the command if i want to know, eg, how many and/or where the duplicates are before i delete them?


```python
#check for missing values
print(df.isnull().sum())
```

    in-store    0
    age         0
    items       0
    amount      0
    region      0
    dtype: int64
    

Notes: the output tells us there is no missing data in this dataset
* the course links to a page about 6 ways to impute missing data, but it's not clear to me under what circumstances imputation of missing values should be used. i know in academia, there are guidelines about when to use imputation, vs. list-deletion vs. something else. but maybe the standards/rules are different for data analysis in business.


# Data Exploration (EDA)

## *How is the data distributed?*


```python
#view descriptive statistics of dataset 
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>in-store</th>
      <th>age</th>
      <th>items</th>
      <th>amount</th>
      <th>region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>79979.000000</td>
      <td>79979.000000</td>
      <td>79979.000000</td>
      <td>79979.000000</td>
      <td>79979.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.500006</td>
      <td>45.758512</td>
      <td>4.505133</td>
      <td>835.825727</td>
      <td>2.674915</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.500003</td>
      <td>15.715158</td>
      <td>2.061250</td>
      <td>721.263650</td>
      <td>1.126642</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>18.000000</td>
      <td>1.000000</td>
      <td>5.004700</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>33.000000</td>
      <td>3.000000</td>
      <td>285.120000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>45.000000</td>
      <td>4.000000</td>
      <td>582.140000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>56.000000</td>
      <td>6.000000</td>
      <td>1233.400000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>85.000000</td>
      <td>8.000000</td>
      <td>3000.000000</td>
      <td>4.000000</td>
    </tr>
  </tbody>
</table>
</div>



*Notes:*  
  
  ```data.describe()``` generates basic descriptive stats on the dataframe,including central tendency, dispersion, and shape of distribution  
* in the ( ) you can specify ```(percentiles =, include =, exclude =, datetime_is_numeric =)```  

*Inferences/Observations*
 
- *in-store* 
    * i know from the documentation that 1= in-store purchase, 0 = online purchase
    * it's a nominal variable, all values should be 0 or 1.
    * mean, std are not meaningful with nominal variables
    * nomimal variables can be used in cross-tabs and chi square tests
- *age*
    * mean of 45.8 means the average consumer in the data set is around 45
    * std tells us about how spread out the observations are. std of 15.7 means that 68% of customers in the dataset are between 29.9 and 61.5, and 95% are between 14.2 and 77.2
    * the minimum age is 18 (uh-oh there's a problem) and the max is 86, 25% are 33 or below, 50% are 45 or below (why is this # diff than the mean?)
- *items*
    * most customers purchased between 2.5 and 6.5 items, and most spent between 114.6 and 1,557
- *region*
    * the regions are 1=north, 2=south, 3=east, 4=west.this is a nominal variable, so mean etc aren't meaningful


```python
#view histograms of each var in the dataframe
plt.subplots(figsize = (15,15))

#create in-store plot
plt.subplot(3,3,1)
plt.hist(df['in-store'])
plt.title('In-Store')
plt.ylabel('Transaction Count')

#create plot of age
plt.subplot(3,3,2)
plt.hist(df['age'])
plt.title('Age')


#create plot of items
plt.subplot(3,3,3)
plt.hist(df['items'], bins=8)
plt.title('Items')


#create plot of amount
plt.subplot(3,3,4)
plt.hist(df['amount'])
plt.title('Amount')
plt.ylabel('Transaction Count')

#create plot of region
plt.subplot(3,3,5)
plt.hist(df['region'])
plt.title('Region')


plt.show

# export subplots as .jpeg
plt.savefig('blackwell_histogram_subplots.jpeg')
```


    
![png](output_23_0.png)
    


*Coding Notes*  

☼ fix: first time ran, the charts were too crowded, so i searched how to fix and found an [answer on stackoverflow](https://stackoverflow.com/questions/45014099/making-plot-less-crowded-in-matplotlib) about how to specify the size of your subplots using ```plt.subplots(figsize=(18,10))```

I used [these](https://nickmccullum.com/python-visualization/subplots/) instructions to figure out the code for the above subplots. I think that website may have more helpful info and tutorials.  

<blockquote><strong><em>Observations</strong></em>

* 50% of transactions are online and 50% are in-store.
* Customers' ages skew slightly younger, with the bulk between roughly 25-60.
* In most transactions, customers buy 2-7 items.
* The amount customers spend per transaction is significantly skewed to the low end, the most common amount per transaction is less than $250.00.
* The North has the fewest transactions and the West has the most transactions. The South and East fall in between and have roughly the same number of transactions. </blockquote>

---

## *How are the variables related?*


```python
data_sample=df.sample(1000)
data_sample.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>in-store</th>
      <th>age</th>
      <th>items</th>
      <th>amount</th>
      <th>region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>31982</th>
      <td>1</td>
      <td>38</td>
      <td>5</td>
      <td>868.79</td>
      <td>4</td>
    </tr>
    <tr>
      <th>18860</th>
      <td>1</td>
      <td>50</td>
      <td>7</td>
      <td>861.83</td>
      <td>3</td>
    </tr>
    <tr>
      <th>79605</th>
      <td>1</td>
      <td>33</td>
      <td>2</td>
      <td>522.93</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26902</th>
      <td>0</td>
      <td>56</td>
      <td>1</td>
      <td>2163.00</td>
      <td>4</td>
    </tr>
    <tr>
      <th>52313</th>
      <td>0</td>
      <td>50</td>
      <td>7</td>
      <td>314.13</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
#create cross plot of all variables in df
sns.pairplot(data_sample, hue='in-store', kind='scatter')

# save pairplot as jpeg
plt.savefig('blackwell_pairplot.jpeg')
```


    
![png](output_28_0.png)
    



```python
#distribution plot of amount variable in each region for in-store & online
sns.set_theme(style="darkgrid")
sns.displot(
    df, x="amount", col="region", row="in-store",
    binwidth=3, height=3, facet_kws=dict(margin_titles=True),
)
```




    <seaborn.axisgrid.FacetGrid at 0x15c29577cd0>




    
![png](output_29_1.png)
    


### Correlation
2 Types
   * Positive Correlation—when A increases, so does B. (0.5-0.7 is slight to high positive correlation, 0.9 - 1 is perfectly positively correlated.  
   * Negative Correlation—when A increases, B decreases.
   
2 Tests:
   * Pearson Correlation Coefficient—use when testing continuous variables that have a linear relationship.
   * Spearman Correlation Coefficient—use when testing variables with a non-linear relationship



```python
#generate correlation matrix
corr_mat = df.corr()
print(corr_mat)
```

              in-store       age     items    amount    region
    in-store  1.000000 -0.178180 -0.003897 -0.085573 -0.133171
    age      -0.178180  1.000000  0.000657 -0.282033 -0.235370
    items    -0.003897  0.000657  1.000000  0.000384 -0.001904
    amount   -0.085573 -0.282033  0.000384  1.000000  0.403486
    region   -0.133171 -0.235370 -0.001904  0.403486  1.000000
    

*Notes*  

* I don't think the correlation co-efficients for in-store and region are meaningful.
* There is a slight negative correlation between amount spent per transaction and age, suggesting that as age increases, the amount spent per transaction decreases. But I know from the scatter plot I generated earlier that the negative correlation between age and amount spent doesn't kick in until around age 62-65, when you see a sharp dropoff. Prior to that there doesn't appear to be any correlation between the two variables. 

### Covariance


```python
#generate covariance matrix
cov_mat = df.cov()
print(cov_mat)
```

               in-store          age     items         amount      region
    in-store   0.250003    -1.400071 -0.004017     -30.860425   -0.075019
    age       -1.400071   246.966189  0.021270   -3196.782841   -4.167305
    items     -0.004017     0.021270  4.248751       0.570791   -0.004421
    amount   -30.860425 -3196.782841  0.570791  520221.252295  327.874873
    region    -0.075019    -4.167305 -0.004421     327.874873    1.269321
    

## Visualization Options

This is primarily a learning exercise in how to create core types of data vizualizations in Python.

### *Line Plots*


```python
fig, ax = plt.subplots()
ax.plot(df['age'], df['amount'])

#add a title
plt.title("Amount Spent by Age")

#add labels to axes
plt.xlabel('Age')
plt.ylabel('Amount')

#this command not nec. in jupyter nb, but does supress lines of code appearing in output 
plt.show()
```


    
![png](output_37_0.png)
    


*Coding Notes*  

☼ *fix*: title and axes labels appeared on a empty figure below the one with the one above because the ```plt.show()``` was below the ```ax.plot()``` command. Fixed by moving ```plt.show()``` to the end.

i can also gerate the above graph using the following code:  
```x = data['age']
y = data['amount']
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()```  

⌂ Learning: You don't need to use ```plt.show()``` in Jupyter notebooks to generate matplotlib visualisation. But including it suppresses several lines of code that would otherwise appear above your graph/plot. 

### *Histograms*

### *Scatter Plots*


```python
#create a var that is a random sample of 100 observations from the dataset
data_sample100 = df.sample(100)

#set x-axis to 'age' and y-axis to 'amount'
x = data_sample['age']
y = data_sample['amount']

#create a scatter plot
plt.scatter(x,y, marker='o')

#add a title
plt.title('Amount Spent Per Purchase by Age, Sample of 100')
#add labels to x and y axes
plt.xlabel('Age')
plt.ylabel('Spending/Purchase')

plt.show()

```


    
![png](output_41_0.png)
    


*Coding Notes*  

☼ SyntaxError fix: after i put in code to add title and labels, got a syntax error. message said problem was ```plt.xlabel('Age')``` was the problem. but i successfully used that code to add label to above graph. After a few minutes I saw that a close paren was missing from the add a title line.  

⌂ info: the ```data.sample(n)``` takes a *random* sample of n observations from the dataset when creating the scatter plot. since the sample is different every time you run the code, the resulting plot with be correspondingly different. You can apparently use ```random_state``` to get reproducibility. See the [Pandas documentation for(dataframe.sample)](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html)  

§Insights: can't tell much from this scatter plot alone, except there is less variability in spending for customers 70+. *Could that be an artifact of the sampling from the larger dataset?* I'm going to look at a larger sample.


```python
#create a var that is a random sample of 100 observations from the dataset
data_sample1000 = df.sample(1000)

#set x-axis to 'age' and y-axis to 'amount'
x = data_sample['age']
y = data_sample['amount']

#create a scatter plot
plt.scatter(x,y, marker='o')

#add a title
plt.title('Amount Spent Per Purchase by Age, Sample of 1000')
#add labels to x and y axes
plt.xlabel('Age')
plt.ylabel('Spending/Purchase')

plt.show()

```


    
![png](output_43_0.png)
    


*Notes*  

§Insights: you can see more clearly in this graph how percipitously the variation in spending drops off after age 63 or so.  

The [matplotlib](https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.scatter.html)
documentation for scatter plots shows example code for creating plots with differed colored and sized dots, a legend, and a number of othe features. 

### *Boxplots*

Box plots are especially good for:  
* finding outliers
* getting rough idea of how your data is distributed
* how tightly your data is grouped


```python
#create a variable to represent the amount feature
A = df['amount']

#generate boxplot of amount feature
plt.boxplot(A,0,'gD')

#add plot title and label y-axis
plt.title('Amount Spent Per Purchase')
plt.ylabel ('$ Spent')

plt.show()
```


    
![png](output_47_0.png)
    


*Notes*  

☼NameError fix: got NameError first time because I used a capital O instead of 0.  

```plt.boxplot(A, 0, 'gD')``` 
* The 0 parameter is determining whether the plot has notches. 0=No, 1=Yes.
* The 'gD' parameter is determinint the color and shape of the smybol representing outliers.
    * To supress symbols altogether use ```plt.boxplot(A, 0, ' ')```
* To generate a basic box plot just use ```plt.boxplot(A)```
* To generate a horizontal box plot use ```plt.boxplot(A, 0, 'rs', 0)```
* To generate boxplot *without* definining a new object ```plt.boxplot(data['amount'])```
    

¿*What is the green pill shape at the top representing?*
>it's a representation of the outliers in the dataset. the reason it's an extended shape, instead of individual points, is the fact that there are a group of outliers bunched up at the top. if there was only a single outlier in the dataset, instead of the pill shape there would be a point. 

*Resources*  

* [This](https://matplotlib.org/examples/pylab_examples/boxplot_demo.html) matplotlib documentation shows how to change various elements of plot's appearance.
* [This](https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51) post on understanding boxplots on towardsdatascience is a clear intro to generating and interepreting box plots.

# Regional Analysis

In this section I'll use the information about the data I've gathered so far and do some additional analysis to answer Blackwell's questions about regional differences in customer spending. 

## *Do customers in different regions spend more per transaction?*


```python
 df.groupby('region').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>in-store</th>
      <th>age</th>
      <th>items</th>
      <th>amount</th>
    </tr>
    <tr>
      <th>region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.000000</td>
      <td>43.704132</td>
      <td>4.510283</td>
      <td>744.990939</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>56.609083</td>
      <td>4.512804</td>
      <td>252.097741</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.611056</td>
      <td>45.646944</td>
      <td>4.494000</td>
      <td>917.969637</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.500000</td>
      <td>38.752424</td>
      <td>4.503771</td>
      <td>1283.938357</td>
    </tr>
  </tbody>
</table>
</div>



*Notes*
* It appears by looking at the avg. amount spent per transaction, that the South spends the least per transaction (about 252) and the East spends the most (about 1284). However, we need to look at the mean and mode as well to check whether those numbers are being skewed by outliers. 
* All transactions in the North occurred in-store. *This doesn't make sense since Blackwell's stores are in the Southeast. 
* All transactions in the South region occur online. *This also doesn't make a lot of sense. 
* The average customer in the East  is 15 years younger than the average customer in the South.
* The avg. number of items purchased in each transaction is 4.5 in all regions.


```python
df.groupby('region').median()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>in-store</th>
      <th>age</th>
      <th>items</th>
      <th>amount</th>
    </tr>
    <tr>
      <th>region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>43.0</td>
      <td>4.0</td>
      <td>662.860</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>57.0</td>
      <td>5.0</td>
      <td>250.205</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>45.0</td>
      <td>4.0</td>
      <td>697.825</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.5</td>
      <td>39.0</td>
      <td>5.0</td>
      <td>1230.250</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby('region').std()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>in-store</th>
      <th>age</th>
      <th>items</th>
      <th>amount</th>
    </tr>
    <tr>
      <th>region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>14.085525</td>
      <td>2.050402</td>
      <td>492.989025</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>16.537368</td>
      <td>2.065467</td>
      <td>142.676946</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.487524</td>
      <td>14.417935</td>
      <td>2.058095</td>
      <td>751.409992</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.500010</td>
      <td>11.886239</td>
      <td>2.066920</td>
      <td>756.302513</td>
    </tr>
  </tbody>
</table>
</div>



*Notes*
* The median amount spent per transaction is significantly lower than the mean amount spent per transaction in the North and West regions, suggesting there are some expensive outlier transactions in those regions. 
* The relatively small std in amount spent per transaction in the East region suggests most purchases are in a small range around the mean of 250.


```python
#add a new variable to df to map region numbers to their names 
#and temporarily set all = to North
df['region_name']='North'
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>in-store</th>
      <th>age</th>
      <th>items</th>
      <th>amount</th>
      <th>region</th>
      <th>region_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>37</td>
      <td>4</td>
      <td>281.03</td>
      <td>2</td>
      <td>North</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>35</td>
      <td>2</td>
      <td>219.51</td>
      <td>2</td>
      <td>North</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>45</td>
      <td>3</td>
      <td>1525.70</td>
      <td>4</td>
      <td>North</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>46</td>
      <td>3</td>
      <td>715.25</td>
      <td>3</td>
      <td>North</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>33</td>
      <td>4</td>
      <td>1937.50</td>
      <td>1</td>
      <td>North</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[df['region']==2,'region_name']='South'
```


```python
df.loc[df['region']==3,'region_name']='East'
```


```python
df.loc[df['region']==4,'region_name']='West'

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>in-store</th>
      <th>age</th>
      <th>items</th>
      <th>amount</th>
      <th>region</th>
      <th>region_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>37</td>
      <td>4</td>
      <td>281.03</td>
      <td>2</td>
      <td>South</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>35</td>
      <td>2</td>
      <td>219.51</td>
      <td>2</td>
      <td>South</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>45</td>
      <td>3</td>
      <td>1525.70</td>
      <td>4</td>
      <td>West</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>46</td>
      <td>3</td>
      <td>715.25</td>
      <td>3</td>
      <td>East</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>33</td>
      <td>4</td>
      <td>1937.50</td>
      <td>1</td>
      <td>North</td>
    </tr>
  </tbody>
</table>
</div>




```python
#create violin plot of amount variable in each region
abr=sns.catplot(x = 'region_name', y = 'amount', data = df, kind = 'violin',
            aspect = 1.5)
abr.set_xticklabels( rotation = 45) #argument({'tic 1', 'tic 2',...}, rotation=45)
abr.fig.suptitle('Regional Spending')
abr.fig.subplots_adjust(top=0.9)
abr.set_axis_labels('Region', 'Spending per Transaction')

# save plot as jpeg--save command must be in same cell as plot code
plt.savefig('blackwell_regional_spending_violin.jpeg')
```


    
![png](output_60_0.png)
    



```python
#create box plot of amount variable in each region
abr_box=sns.catplot(x = 'region_name', y = 'amount', data = df, 
                    kind = 'box', aspect = 1)
abr_box.set_xticklabels(rotation = 45)
abr_box.fig.suptitle('Regional Spending')
abr_box.fig.subplots_adjust(top=0.9)
abr_box.set_axis_labels('', 'Spending per Transaction')

# save jpeg of boxplot
plt.savefig('blackwell_regional_spending_boxplot.jpeg')
```


    
![png](output_61_0.png)
    


***Summary of Findings***  

*Are there regional differences in spending per transaction?*  

Short answer:YES. A quick look at the average amount spent in each region reveals that the Southern region spends the least (about $252.00$). Although it looks like the Eastern region spends the most per transaction (about $1284), further investigation reveals that number is heavily skewed by several outlier transactions. When the outliers are left aside, we can see that typical spendintg in the Eastern and Northern regions are very similar. We can also see that customers in the Western region typically spend the most. One final thing to note is that the amount spent per transaction in the Southern region spans a noteably smaller range than the other three areas, and is concentrated at the low end. 


---
---

## *Which regions spend the most/least?*

This question could be understood to in several different ways. Here I will take it to mean: *which regions spend the most/least as a % of total amount spent in all regions?*



```python
#create bar chart of y=amount, x=region <--this is a stupid bar chart
sns.set_theme(style="darkgrid")
bad=sns.catplot(x = 'region_name', y = 'amount', data = df, kind = 'bar',
            aspect = 1.5)
#add space between top of graph and title
bad.fig.subplots_adjust(top=0.9)
#add and format x-axis tick labels
bad.set_xticklabels( rotation=45)
#set x and y axis labels
bad.set_axis_labels('Region','Amount', fontsize=12)
#add plot title
bad.fig.suptitle('Revenue by Region', fontsize=16)

#save hist as jpeg
plt.savefig('blackwell_revenue_by_region_hist.jpeg')

"""
Notes

i keep making bar charts when they don't make any sense. 
in the above chart i wanted to see how much spending was happening in each region. but the above bar chart does not show that. what is the y-axis representing?
what i want is a pie chart showing the percent of total spending that occurs in each region.
steps:
first i need to get the sum of the amount variable for each region.
then i need to figure out what percent of the total amount each regional sum is.
then create a pie chart

"""
```




    "\nNotes\n\ni keep making bar charts when they don't make any sense. \nin the above chart i wanted to see how much spending was happening in each region. but the above bar chart does not show that. what is the y-axis representing?\nwhat i want is a pie chart showing the percent of total spending that occurs in each region.\nsteps:\nfirst i need to get the sum of the amount variable for each region.\nthen i need to figure out what percent of the total amount each regional sum is.\nthen create a pie chart\n\n"




    
![png](output_64_1.png)
    



```python
robsum=df.groupby(['region']).sum()
print(robsum)
```

            in-store      age   items        amount
    region                                         
    1          15997   699135   72151  1.191762e+07
    2              0  1131842   90229  5.040442e+06
    3          10999   821645   80892  1.652345e+07
    4          12994  1007098  117044  3.336699e+07
    


```python
amountotal = df['amount'].sum()
print(amountotal)
```

    66848505.79050001
    


```python
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'North', 'South', 'East', 'West'
sizes = [17.8, 7.5, 24.7, 49.9]
explode = (0, 0.1, 0, 0.1)  # "explode" 2nd and 3rd slices

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#add plot title
fig1.suptitle('Revenue by Region', fontsize=16)

#save pie chart as jpeg
plt.savefig('blackwell_revenue_byregion_pie.jpeg')
```


    
![png](output_67_0.png)
    



```python
df.groupby('region').median()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>in-store</th>
      <th>age</th>
      <th>items</th>
      <th>amount</th>
    </tr>
    <tr>
      <th>region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>43.0</td>
      <td>4.0</td>
      <td>662.860</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>57.0</td>
      <td>5.0</td>
      <td>250.205</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>45.0</td>
      <td>4.0</td>
      <td>697.825</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.5</td>
      <td>39.0</td>
      <td>5.0</td>
      <td>1230.250</td>
    </tr>
  </tbody>
</table>
</div>



***Summary of Findings***  

*Which regions spend the most/least?*

* As we saw in the previous question, customers in the **Southern region** typically spend to **least** per transaction (about 250.00), and customers in the **Western region** typically spend the most (about 1230.00).
* But we can also look at this from the perspective of which regions have the most/least % of the total amount. Perhaps unsurprisingly, we see that the Western region is responsible for almost 50% of all Blackwell sales.

---
---

## *Are there regional differences in online vs. in-store spending?*


```python
df.groupby('in-store').describe()['amount']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>in-store</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39989.0</td>
      <td>897.546576</td>
      <td>880.146928</td>
      <td>5.0047</td>
      <td>228.1500</td>
      <td>441.89</td>
      <td>1542.7</td>
      <td>3000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>39990.0</td>
      <td>774.106421</td>
      <td>508.118520</td>
      <td>50.0500</td>
      <td>368.6225</td>
      <td>688.18</td>
      <td>1024.4</td>
      <td>1999.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
#revenue online vs. in-store by region 
oir = sns.catplot(data=df, kind='box', 
                  x='region_name', y='amount', hue='in-store', height=6)
oir.despine(left=True)
#oir.title("Online vs. In-Store Spending by Region")<--doesn't work
oir.set_xticklabels( rotation=45)
oir.set_axis_labels('Region', 'Spending')
oir.legend.set_title('')
#NA--oir.subplots_adjust(top=0.9)
oir.fig.suptitle("Online vs. In-Store Spending",
                  fontsize=16, fontdict={"weight": "bold"}, y = 1.05)
oir.savefig('oir_plot.png')
```


    
![png](output_72_0.png)
    


***Summary of findings***

* As you can see in the above plot, the Southern region has **only online** sales, and the Northern region has **no online** sales. This makes me wonder whether the data has been corrupted. But assuming the data is sound, the complete lack of online sales in the South presents a big opportunity to establish and grow Blackwell's online sales in the region. 
* Something else we can see in the above graph is that online sales account for all the high dollar transactions. Almost all the sales above $2,100.00$ are online transactions. This suggests two possible stategies (or a combination of the two) to improve Blackwell's online sales: deploy a strategy aimed at growing the high-dollar purchases that currently only take place online. (2) compare the low dollar purchases in-store with low dollar purchases online to see if there are any significant differences to exploit.

# Age Analysis

## *Create Categorical Age Variables*

* *Discretization* is the process of taking a continuous variable, like age or weight, that could (theoretically) take on any value and putting it into discreet bins (e.g. below 100 lbs, between 140-160, over 200 lbs.)
     * *when you discretize data, don't you need to make sure the bins you create are the same size (i.e.cover the same interval).so the resulting variable will be ordinal instead of nominal, and thus it is appropriate to use various types of statistical tests?* 
     * [this](https://s3.amazonaws.com/gbstool/courses/1094/docs/An%20Introduction%20to%20Discretization%20Techniques%20for%20Data%20Scientists.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20201023T205053Z&X-Amz-SignedHeaders=host&X-Amz-Expires=36900&X-Amz-Credential=AKIAJBIZLMJQ2O6DKIAA%2F20201023%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=4e9e02e0403840c946cd9a5a9fe5270c8146878715dcdb9853fa908ee0821602) is the explainer article linked in the course, it contains the pkgs and commands needed to do each kind of discretization.
* ***why discretization is important***
    * often results are *more meaningful and easier to understand* when the data has been binned into a smaller number of categories that have been thoughtfully constructed to fit the question we are trying to answer. (e.g. you might care about generational differences in attitudes, so you could bin observations by age into groups corresponding to baby boomers, millenials, gen z, etc.)
    * certain methods/models may not be used with continous data (e.g. alternative decision-tree models like random forest models, feature engineering methods)
    * fitting continuous data into a small # of bins reduces the impact of small variations in the data (i.e. noise) which could reduce the strength of the signal we get from the data.
* ***approaches to discretization***
    * *equal width* — separate all values into 'N' number of bins, each having the same width (i.e. span of values)
    * *equal frequency* separate the values in 'N' number of bins, each containing the same number of observations.
    * *K-means* — apply k-means clustering to the continuous variable. i don't know what this means
    * *decision-trees* — 


```python
#define new discritized age variable, two bins: under 65 and 65+
young = pd.cut(df.age, bins=[18,63,84], labels=['Under 65', '65+'])

#add new young variable to df
df['young']=young

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>in-store</th>
      <th>age</th>
      <th>items</th>
      <th>amount</th>
      <th>region</th>
      <th>region_name</th>
      <th>young</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>37</td>
      <td>4</td>
      <td>281.03</td>
      <td>2</td>
      <td>South</td>
      <td>Under 65</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>35</td>
      <td>2</td>
      <td>219.51</td>
      <td>2</td>
      <td>South</td>
      <td>Under 65</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>45</td>
      <td>3</td>
      <td>1525.70</td>
      <td>4</td>
      <td>West</td>
      <td>Under 65</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>46</td>
      <td>3</td>
      <td>715.25</td>
      <td>3</td>
      <td>East</td>
      <td>Under 65</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>33</td>
      <td>4</td>
      <td>1937.50</td>
      <td>1</td>
      <td>North</td>
      <td>Under 65</td>
    </tr>
  </tbody>
</table>
</div>




```python
#define new generation variable
gen = pd.cut(df.age, bins=[18,23,39,55,74,85], labels =['Gen Z (8-23)',
                                                    'Millenial (24-39)',
                                                     'Gen X (40-55)',
                                                     'Boomer (56-74)',
                                                     'Silent Gen (75-95)'])

#add new gen variable to df
df['gen']=gen
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>in-store</th>
      <th>age</th>
      <th>items</th>
      <th>amount</th>
      <th>region</th>
      <th>region_name</th>
      <th>young</th>
      <th>gen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>37</td>
      <td>4</td>
      <td>281.03</td>
      <td>2</td>
      <td>South</td>
      <td>Under 65</td>
      <td>Millenial (24-39)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>35</td>
      <td>2</td>
      <td>219.51</td>
      <td>2</td>
      <td>South</td>
      <td>Under 65</td>
      <td>Millenial (24-39)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>45</td>
      <td>3</td>
      <td>1525.70</td>
      <td>4</td>
      <td>West</td>
      <td>Under 65</td>
      <td>Gen X (40-55)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>46</td>
      <td>3</td>
      <td>715.25</td>
      <td>3</td>
      <td>East</td>
      <td>Under 65</td>
      <td>Gen X (40-55)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>33</td>
      <td>4</td>
      <td>1937.50</td>
      <td>1</td>
      <td>North</td>
      <td>Under 65</td>
      <td>Millenial (24-39)</td>
    </tr>
  </tbody>
</table>
</div>




```python
ca = sns.countplot(data=df, 
                  x='region_name', y=None, hue='young')
ca.set(xlabel='Region', ylabel='Count')
ca.legend().set_title('')

#ca.savefig("customerbyregionE.png", format='png',dpi=150)
fig=ca.get_figure()
fig.savefig('pic.png')
```


    
![png](output_78_0.png)
    


*Notes*
* This bar chart tells us that (1) all customers in the Western region are under 65, and the Southern region has the most number of customers over 65. This suggests marketing target at age groups may be more successful in those two regions.
* That being said, the relatively small size of Blackwell's customer base that is 65+ makes it an area for big potential gains.


```python
gen_plot = sns.catplot(data=df, 
                  x='region_name', y=None, hue='gen', kind='count', 
                      height=4, aspect=1)
gen_plot.set(xlabel='Region', ylabel='Count',
            title='Regional Transactions by Age Group')
gen_plot.set_xticklabels(rotation=45)

#NA- gen_plot_leg=gen_plot.legend()
#NA- gen_plot.fig.legend(title='', labels=[])
gen_plot.legend.set_title('')
#NA- legend = gen_plot.legend()
#NA- legend.texts[0].set_text('')

#NA- fig=gen_plot.get_figure()
gen_plot.savefig('pic_gen_plot.png')

#save plot as jpeg
plt.savefig('blackwell_generational_spending_byregion.jpeg')
```


    
![png](output_80_0.png)
    


***Summary of findings***

*Are there regional differences in age amoung our customers?* Yes

* All Blackwell customers of the Silent Generation are in the South.
* Generation Z (18-24 in our sample) is the smallest or next to smallest cohort of customers in every region. In the South, there are no Gen Z customers. When combined with the spending patterns of each cohort this presents a significant opportunity.

---

## *Do customers differ in age by reigion?*


```python
sns.set_theme(style="darkgrid")
sns.displot(
    df, x="age", col="region",
    binwidth=3, height=3, facet_kws=dict(margin_titles=True),
)
```




    <seaborn.axisgrid.FacetGrid at 0x15c4dde1d90>




    
![png](output_83_1.png)
    


## *Do customers of different ages differ in their spending?*


```python
#violin plot of spending by customers under 65 and 65+ in each region
car = sns.catplot(data=df, kind='violin', 
                  x='region_name', y='amount', hue='young',
                  palette='rocket', legend_out=True, ci = None)
car.despine(left=True)
car.set_xticklabels( rotation=45)
car.set_axis_labels('', 'Spending per Transaction')
car.fig.suptitle("Customer Spending by Age: Under 65 vs 65+",
                  fontsize=16, fontdict={"weight": "bold"})
#car.fig.legend(title="", labels=['Online', 'In-store'])
car.legend.set_title('')
car.savefig("customerbyregion.png",
                  format='png',dpi=150)
```


    
![png](output_85_0.png)
    



```python
#calculate mean of age, items, & amount for each gen within each region
gen_mean=df.groupby(['region_name', 'gen'])['age', 'items', 'amount'].mean()
print(gen_mean)

```

                                          age     items       amount
    region_name gen                                                 
    East        Gen Z (8-23)        20.954082  4.503827  1584.896097
                Millenial (24-39)   31.656688  4.467841   958.103526
                Gen X (40-55)       47.476020  4.508320   927.122623
                Boomer (56-74)      63.478210  4.496804   751.545588
                Silent Gen (75-95)        NaN       NaN          NaN
    North       Gen Z (8-23)        21.157093  4.478312  1018.099817
                Millenial (24-39)   31.661903  4.515131   804.479170
                Gen X (40-55)       47.337918  4.499066   771.135680
                Boomer (56-74)      64.708195  4.529785   520.992073
                Silent Gen (75-95)        NaN       NaN          NaN
    South       Gen Z (8-23)              NaN       NaN          NaN
                Millenial (24-39)   33.685424  4.500248   250.351972
                Gen X (40-55)       47.550467  4.505747   252.129675
                Boomer (56-74)      64.981328  4.522211   252.277828
                Silent Gen (75-95)  79.886164  4.520128   253.608984
    West        Gen Z (8-23)        21.098558  4.564197  1266.141474
                Millenial (24-39)   31.466787  4.485318  1264.815006
                Gen X (40-55)       47.407644  4.504028  1251.748461
                Boomer (56-74)      59.264597  4.491659  1545.107915
                Silent Gen (75-95)        NaN       NaN          NaN
    

    <ipython-input-40-2efe97d6df6e>:2: FutureWarning: Indexing with multiple keys (implicitly converted to a tuple of keys) will be deprecated, use a list instead.
      gen_mean=df.groupby(['region_name', 'gen'])['age', 'items', 'amount'].mean()
    


```python
gen_spending_plot = sns.catplot(data=df,
                  x='region_name', y='amount', hue='gen', kind='box', 
                      height=4, aspect=1)
gen_spending_plot.set_xticklabels(rotation=45)
gen_spending_plot.legend.set_title('')
gen_spending_plot.set(xlabel='Region', ylabel='Amount per Transaction',
            title='Spending by Generation')

gen_spending_plot.savefig('gen_spending_plot.png')
```


    
![png](output_87_0.png)
    



```python
#calculate sum of amount & item for each gen within each region
gen_sum=df.groupby(['region_name', 'gen'])['items', 'amount'].sum()
print(gen_sum)

```

                                    items        amount
    region_name gen                                    
    East        Gen Z (8-23)         3531  1.242559e+06
                Millenial (24-39)   26119  5.601073e+06
                Gen X (40-55)       27636  5.683262e+06
                Boomer (56-74)      23217  3.880230e+06
                Silent Gen (75-95)      0  0.000000e+00
    North       Gen Z (8-23)         3820  8.684391e+05
                Millenial (24-39)   26856  4.785042e+06
                Gen X (40-55)       26495  4.541218e+06
                Boomer (56-74)      14980  1.722921e+06
                Silent Gen (75-95)      0  0.000000e+00
    South       Gen Z (8-23)            0  0.000000e+00
                Millenial (24-39)   18154  1.009920e+06
                Gen X (40-55)       25088  1.403858e+06
                Boomer (56-74)      30032  1.675377e+06
                Silent Gen (75-95)  16955  9.512873e+05
    West        Gen Z (8-23)        13615  3.776900e+06
                Millenial (24-39)   47199  1.330965e+07
                Gen X (40-55)       45842  1.274030e+07
                Boomer (56-74)       9693  3.334343e+06
                Silent Gen (75-95)      0  0.000000e+00
    

    <ipython-input-42-e02cd5cdffab>:2: FutureWarning: Indexing with multiple keys (implicitly converted to a tuple of keys) will be deprecated, use a list instead.
      gen_sum=df.groupby(['region_name', 'gen'])['items', 'amount'].sum()
    


```python
gen_spending_violin = sns.catplot(data=df,
                  x='region_name', y='amount', hue='gen', kind='violin', 
                      height=8, aspect=2)
gen_spending_violin.set(xlabel='Region', ylabel='Amount Spent')
gen_spending_violin.fig.suptitle('Spending by Generation', fontsize = 20)

# save plot as jpeg
plt.savefig("blackwell_spending_by_generation_violin.jpeg")
```


    
![png](output_89_0.png)
    



```python
ageamount =sns.scatterplot(data=df, x='amount',y='age')
```


    
![png](output_90_0.png)
    



```python
gen_spending_boxen = sns.catplot(data=df,
                  x='region_name', y='amount', hue='gen', kind='boxen', 
                      height=6, aspect=2)
gen_spending_boxen.set(xlabel='Region', ylabel='Amount Spent',
            title='Spending by Age Group')
```




    <seaborn.axisgrid.FacetGrid at 0x15c2f6e3640>




    
![png](output_91_1.png)
    



```python
#group data by region
df_grouped_rg=df.groupby(['region'])

df_grouped_rg['age'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>15997.0</td>
      <td>43.704132</td>
      <td>14.085525</td>
      <td>19.0</td>
      <td>32.0</td>
      <td>43.0</td>
      <td>53.0</td>
      <td>74.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19994.0</td>
      <td>56.609083</td>
      <td>16.537368</td>
      <td>28.0</td>
      <td>42.0</td>
      <td>57.0</td>
      <td>71.0</td>
      <td>85.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18000.0</td>
      <td>45.646944</td>
      <td>14.417935</td>
      <td>18.0</td>
      <td>34.0</td>
      <td>45.0</td>
      <td>57.0</td>
      <td>74.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25988.0</td>
      <td>38.752424</td>
      <td>11.886239</td>
      <td>18.0</td>
      <td>29.0</td>
      <td>39.0</td>
      <td>49.0</td>
      <td>63.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#sort by age the df grouped by region
df_grouped_rg.sort_values(by ='age', ascending=False)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-54-d51d35db1c75> in <module>
          1 #sort by age the df grouped by region
    ----> 2 df_grouped_rg.sort_values(by ='age', ascending=False)
    

    ~\anaconda3\lib\site-packages\pandas\core\groupby\groupby.py in __getattr__(self, attr)
        750             return self[attr]
        751 
    --> 752         raise AttributeError(
        753             f"'{type(self).__name__}' object has no attribute '{attr}'"
        754         )
    

    AttributeError: 'DataFrameGroupBy' object has no attribute 'sort_values'



```python
#sort df by region then by age 
df.sort_values(by=['region', 'age'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>in-store</th>
      <th>age</th>
      <th>items</th>
      <th>amount</th>
      <th>region</th>
      <th>region_name</th>
      <th>young</th>
      <th>gen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>96</th>
      <td>1</td>
      <td>19</td>
      <td>7</td>
      <td>1995.000</td>
      <td>1</td>
      <td>North</td>
      <td>Under 65</td>
      <td>Gen Z (8-23)</td>
    </tr>
    <tr>
      <th>1125</th>
      <td>1</td>
      <td>19</td>
      <td>8</td>
      <td>991.720</td>
      <td>1</td>
      <td>North</td>
      <td>Under 65</td>
      <td>Gen Z (8-23)</td>
    </tr>
    <tr>
      <th>1435</th>
      <td>1</td>
      <td>19</td>
      <td>2</td>
      <td>991.600</td>
      <td>1</td>
      <td>North</td>
      <td>Under 65</td>
      <td>Gen Z (8-23)</td>
    </tr>
    <tr>
      <th>1545</th>
      <td>1</td>
      <td>19</td>
      <td>2</td>
      <td>83.041</td>
      <td>1</td>
      <td>North</td>
      <td>Under 65</td>
      <td>Gen Z (8-23)</td>
    </tr>
    <tr>
      <th>1643</th>
      <td>1</td>
      <td>19</td>
      <td>3</td>
      <td>151.390</td>
      <td>1</td>
      <td>North</td>
      <td>Under 65</td>
      <td>Gen Z (8-23)</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78603</th>
      <td>0</td>
      <td>63</td>
      <td>7</td>
      <td>2891.200</td>
      <td>4</td>
      <td>West</td>
      <td>Under 65</td>
      <td>Boomer (56-74)</td>
    </tr>
    <tr>
      <th>78620</th>
      <td>0</td>
      <td>63</td>
      <td>7</td>
      <td>843.310</td>
      <td>4</td>
      <td>West</td>
      <td>Under 65</td>
      <td>Boomer (56-74)</td>
    </tr>
    <tr>
      <th>78684</th>
      <td>0</td>
      <td>63</td>
      <td>5</td>
      <td>1878.000</td>
      <td>4</td>
      <td>West</td>
      <td>Under 65</td>
      <td>Boomer (56-74)</td>
    </tr>
    <tr>
      <th>78801</th>
      <td>0</td>
      <td>63</td>
      <td>1</td>
      <td>772.170</td>
      <td>4</td>
      <td>West</td>
      <td>Under 65</td>
      <td>Boomer (56-74)</td>
    </tr>
    <tr>
      <th>79147</th>
      <td>0</td>
      <td>63</td>
      <td>3</td>
      <td>744.400</td>
      <td>4</td>
      <td>West</td>
      <td>Under 65</td>
      <td>Boomer (56-74)</td>
    </tr>
  </tbody>
</table>
<p>79979 rows × 8 columns</p>
</div>



**Observations**

## *What, if any, relationship is there between number of items purchased and amount spent?*


```python
corr_mat = df.corr()
print(corr_mat)
```

              in-store       age     items    amount    region
    in-store  1.000000 -0.178180 -0.003897 -0.085573 -0.133171
    age      -0.178180  1.000000  0.000657 -0.282033 -0.235370
    items    -0.003897  0.000657  1.000000  0.000384 -0.001904
    amount   -0.085573 -0.282033  0.000384  1.000000  0.403486
    region   -0.133171 -0.235370 -0.001904  0.403486  1.000000
    


```python
#amount by item 
item_amount = sns.swarmplot(data=df, 
                  x='items', y='amount', hue='in-store', palette='dark')
item_amount.despine(left=True)
item_amount.set_xticklabels( rotation=45)
item_amount.set_axis_labels('Items Purchased', 'Revenu')
item_amount.legend.set_title('')
```

    C:\Users\kpiat\anaconda3\lib\site-packages\seaborn\categorical.py:1296: UserWarning: 95.2% of the points cannot be placed; you may want to decrease the size of the markers or use stripplot.
      warnings.warn(msg, UserWarning)
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-56-8c67b4d566fc> in <module>
          1 #amount by item
    ----> 2 item_amount = sns.swarmplot(data=df, 
          3                   x='items', y='amount', hue='in-store', palette='dark')
          4 item_amount.despine(left=True)
          5 item_amount.set_xticklabels( rotation=45)
    

    ~\anaconda3\lib\site-packages\seaborn\_decorators.py in inner_f(*args, **kwargs)
         44             )
         45         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 46         return f(**kwargs)
         47     return inner_f
         48 
    

    ~\anaconda3\lib\site-packages\seaborn\categorical.py in swarmplot(x, y, hue, data, order, hue_order, dodge, orient, color, palette, size, edgecolor, linewidth, ax, **kwargs)
       3014                        linewidth=linewidth))
       3015 
    -> 3016     plotter.plot(ax, kwargs)
       3017     return ax
       3018 
    

    ~\anaconda3\lib\site-packages\seaborn\categorical.py in plot(self, ax, kws)
       1418     def plot(self, ax, kws):
       1419         """Make the full plot."""
    -> 1420         self.draw_swarmplot(ax, kws)
       1421         self.add_legend_data(ax)
       1422         self.annotate_axes(ax)
    

    ~\anaconda3\lib\site-packages\seaborn\categorical.py in draw_swarmplot(self, ax, kws)
       1414         for center, swarm in zip(centers, swarms):
       1415             if swarm.get_offsets().size:
    -> 1416                 self.swarm_points(ax, swarm, center, width, s, **kws)
       1417 
       1418     def plot(self, ax, kws):
    

    ~\anaconda3\lib\site-packages\seaborn\categorical.py in swarm_points(self, ax, points, center, width, s, **kws)
       1316 
       1317         # Do the beeswarm in point coordinates
    -> 1318         new_xy = self.beeswarm(orig_xy, d)
       1319 
       1320         # Transform the point coordinates back to data coordinates
    

    ~\anaconda3\lib\site-packages\seaborn\categorical.py in beeswarm(self, orig_xy, d)
       1268 
       1269             # Find the first candidate that does not overlap any neighbours
    -> 1270             new_xy_i = self.first_non_overlapping_candidate(candidates,
       1271                                                             neighbors, d)
       1272 
    

    ~\anaconda3\lib\site-packages\seaborn\categorical.py in first_non_overlapping_candidate(self, candidates, neighbors, d)
       1227             dy = neighbors_y - y_i
       1228 
    -> 1229             sq_distances = np.power(dx, 2.0) + np.power(dy, 2.0)
       1230 
       1231             # good candidate does not overlap any of neighbors
    

    KeyboardInterrupt: 



    
![png](output_98_2.png)
    



```python
f, ax = plt.subplots(figsize=(6.5,6.5))
sns.despine(f, left=True, bottom=True)
sns.scatter
```
