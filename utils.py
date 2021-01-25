# Python program to convert the currency 
# of one country to that of another country 

# Import the modules needed 
import requests
import numpy as np 
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import operator 
import re

def currency_converter(currencies_id,countries,original_prices,currency_to_convert='USD'):
    """
    Function to convert prices of a particular country currency to usd.
    Input: 
    - countries: array with the name of the country for each value 
    - original_prices: array with values(price) for each case

    Output:
    
    """ 
    
    # API to get currency exchange
    url = 'http://data.fixer.io/api/latest?access_key=' + 'cd313e520bb02e4de54931127bd763a5'
    r = requests.get(url).json()
    rates = r['rates']
    # taking values from array
    country=countries.values
    values = original_prices.values
    
    # initialize an array to full with values in usd
    values_usd=np.zeros(len(values))
    for i in range(len(country)):
        # Take the currency id for the country
        currency_id = currencies_id[country[i]]
        if currency_to_convert=='USD' and currency_id!='VES':
            try:
                #Convert to usd
                amount = values[i]
                amount = amount/rates[currency_id]
                # limiting the precision to 2 decimal places 
                amount = round(amount * rates['USD'], 9)
                # add the value in usd
                values_usd[i] = amount
            except:
                continue
        elif currency_id =='VES':
            # rate from USD to VES at 20/01/2021
            # 1 usd = 1'523.537,44 VES  taken from:  https://es.exchange-rates.org/Rate/USD/VES
            rate_usd_ves = float(1523537.44)
            try:
                amount=values[i]
                amount = np.round(amount/rate_usd_ves,decimals=9)
                values_usd[i] = amount
            except:
                continue

    #Check that both arrays have same length
    assert(len(values)==len(values_usd))
    return values_usd 




## Function to extract a brief version of the produt name
def product_name_extractor(products):
    """
    Input: array with the name for each product
    Output: array with a short name for each product
    """
    product_names = products.str.split(' ').values
    #
    short_product_names = []
    for i in range(len(product_names)):
        try:
            if len(str(product_names[i][0]))<=4 and len(product_names[i])>1:
                short_product_names.append(str(product_names[i][0]) + ' ' + str(product_names[i][1]) )
            else:
                short_product_names.append(str(product_names[i][0]))
        except Exception as e:
            print(f'problem in case {i}')
            continue
            
    return short_product_names




# function to create a tags freq dictionary
def tags_extractor(tags):
    """
    Input: array with the tags for each post
    Output: freq dictionary with key,value -> tag,freq
    """
    # tags for all post
    tags_list = tags.str.split(',').values
    # stacked tags
    stacked_tags=[]
    # tags frequency dictionary
    tags_freq_dict = {}
    # sorted dict
    sorted_dict={}
    for i in range(len(tags_list)):
        try:
            # tags for a particular post
            post_tag_list = tags_list[i]
            for j in range(len(post_tag_list)):
                #create a pattern to extract alhpa numeric characters
                pattern = re.compile('\w+')
                tag = pattern.findall(post_tag_list[j])[0]
                if tag in tags_freq_dict.keys():
                    tags_freq_dict[tag]+=1
                else:
                    tags_freq_dict[tag]=1
                
                # Finally add the tag to list stacked_tags
                stacked_tags.append((i,tag))
                    
        except Exception as e:
            #exception is relative to index out of bound, cuz some post doesn't have tags. 
            continue
    # Sorting the freq dictionary with key,value --> tag,frequency
    sorted_keys=sorted(tags_freq_dict,key=tags_freq_dict.get,reverse=True)
    for tag in sorted_keys:
        sorted_dict[tag] = tags_freq_dict[tag]
            
    return stacked_tags,sorted_dict
    

# function to add tags columns and fill those inside a dataframe
def add_tags_fields(df,stacked_tags,tags_freq_dict):
    """
    add and fill a set of  tags columns to a datafarme based on its original column tags
    getting an expanded version of this original tags column.
    
    Input: 
    - df: dataframe with the column tags where each row has all tags of a post collapsed
    - stacked_tags: list with elements of the form (row_in_df,tag)
    - tags_freq_dict: dictionary with tags frequency in df
    Output:
    - df_expanded: a dataframe with the set of "unique" tags found in all the df as columns 
    and filled with 0 or 1 depending if the the post has or not the tag
    """
    # make a copy of df
    df_expanded = df.copy(deep=True)
    # get the unique tags
    unique_tags = list(tags_freq_dict.keys())
    # add and fill the unique tags columns in df_expanded
    for tag in unique_tags:
        # add and initialize the tag field created
        df_expanded[tag] = np.zeros((df_expanded.shape[0]))
    
    
    # filling the fields added
    for element in stacked_tags:
        try:
            # get the index of the row to fill
            indx = element[0]
            # get the tag column name to fill
            tag_name = element[1]
            # fill the column=tag_name with 1
            try:
                df_expanded.loc[indx,tag_name]=1
            except:
                continue
            
        except Exception as e:
            print(f'Problems filling row {element[0]}')
            print('Exception',e)
            continue

    return df_expanded 


# function to change the dtype of a list of columns selected

def convert_column_dtype(df,columns,convert_to):
    """
    convert the dtype of the a set of columns
    Input: 
    - df: dataframe with the columns to change the dtype
    - columns: list with the names of columns to turn the dtype
    - convert_to: list with the dtype desire to the columns selected
    Output: dataframe with new_dtypes for the selected columns
    """
    if len(columns)==len(convert_to):
        for col,new_type in zip(columns,convert_to):
            df[[col]] = df[[col]].astype(new_type)
    else:
        new_type = convert_to[0]
        for col in columns:
            df[[col]] = df[[col]].astype(new_type)
    return df


# function to add the duplicated tag in some rows
def add_duplicated_field(df,indx_duplicated):
    """
    function to add the duplicated field that tag the rows with duplicated cases
    Input: 
    - df: dataframe to add the duplicated colum
    - indx_duplicated: indices where a duplicated was found.
    Output:
    - df_tatted: datframe with the binary column duplicated
    """
    # create a copy of df
    df_tagged = df.copy(deep=True)
    # initialize the column
    df_tagged['duplicated(Y/N)(1/0)'] = np.zeros(df_tagged.shape[0])
    for i in range(len(df_tagged['duplicated(Y/N)(1/0)'])):
        if i in indx_duplicated:
            df_tagged['duplicated(Y/N)(1/0)'].values[i]=1
    # change the dtype of the colum to categorical.
    df_tagged[['duplicated(Y/N)(1/0)']] = df_tagged[['duplicated(Y/N)(1/0)']].astype('category')  
    return df_tagged




def frequency_summary_table(df,columns):
    """
    Function to create a summarization for a list of columns,
    showing the values distribution for each of those columns
    
    Input: 
    - df: dataframe to extract summarization
    - columns: list with columns to extract values distribution
    """
    freq_dict ={}
    for col in columns:
        try:
            values_distribution = df[col].value_counts()
            # select values with a distribution 
            values = values_distribution.values
            mean = np.mean(values)
            std=np.std(values)
            # create a threshold to 
            thres= mean + 1.5*std
            if len(values[values>thres])>0 or len(values[values<thres])>0:
                freq_dict[col]=values_distribution
        except Exception as e:
            print(f'Column {col} not found!')
            continue
    return pd.DataFrame(freq_dict)



# function to run the elbow method used in kmeans to detect number of cluster to use
def kmeans_elbow(df,n_cluster_test_range):
    # number of cluster to test
    Nc = range(1,n_cluster_test_range)
    kmeanlist = [KMeans(n_clusters=i) for i in Nc]
    varianza = [kmeanlist[i].fit(df).inertia_ for i in range(len(kmeanlist)) ]
    plt.plot(Nc, varianza, 'o-')
    plt.xlabel('Numero de clusters')
    plt.ylabel('Varianza explicada Intraclases')
    plt.title('Elbow method')
    return varianza



def add_cutting_func(df, bins):
    df_cut = df.copy(deep=True)
    df['interval'] = pd.cut(df['sold_quantity'],bins=bins)
    
    # create a coding for the interval variable:
    le_interval = LabelEncoder()
    # list with intervals
    intervals=[]
    for i in range(len(bins)):
        try:
            interval = pd.Interval(left=bins[i],right =bins[i+1])
            intervals.append(interval)
        except:
            continue
    # Number of intervals
    print('Number of classes(intervals): ', len(intervals))
    # fit the encoder
    le_interval.fit(intervals)
    #create the new variable
    df['sold_quantity_range'] = le_interval.transform(df['interval'])
    return df




def stratified_sample_df(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_

def create_undersampled_df(df):
    classes = df['sold_quantity_range'].value_counts().index.to_list()
    n_samples = int(df['interval'].value_counts().mean())
    min_samples =df['interval'].value_counts().min()
    
    dfs = []
    for i in classes:
        try:
            df_class= stratified_sample_df(df[df['sold_quantity_range']==int(i)],'sold_quantity_range',n_samples)
            dfs.append(df_class)
        except:
            df_class = stratified_sample_df(df[df['sold_quantity_range']==int(i)],'sold_quantity_range',min_samples)
            df_class.append(df_class)
        continue
    
    df_sampled = pd.concat(dfs)
    return df_sampled



def best_model(df,test_prop=0.2):
    """
    function to compare classification models and select the best based on the accuracy score
    
    Input: 
    -df: dataframe with features and target variable.
    - test_prop: rate of test size to use in the train test split.
    
    Output:
    - model with the high accuracy
    """
    
    # Split df in X,y
    X=df.drop(columns=['sold_quantity_range','interval'])
    y = df[['sold_quantity_range']]
    
    
    #split the data in train,test
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=test_prop,random_state=42)
    
    print('x_train shape: ',x_train.shape)
    print('x_test: ',x_test.shape)
    
    


    # define models to compare:
    models = {'DecisionTree':DecisionTreeClassifier(),
             'RandomForest':RandomForestClassifier()}
    
    # evaluate models and choose the model with min error over test.
    accuracy_scores = {}
    pred_dict ={}
    
    for name,model in models.items():
        try:
            model.fit(x_train,y_train['sold_quantity_range'])
            prediction = model.predict(x_test)
            accuracy = accuracy_score(y_test,prediction)
            print(accuracy)
            print('#'*50)
            print('accuracy for '+ name + ':',accuracy)
            accuracy_scores[name]=accuracy
            pred_dict[name]=prediction
        except Exception as e:
            print('Warning!:')
            print(f'problem with model{name}\n')
            continue
            
    # # get the name of classifier with the max accuracy
    max_accuracy_classifier = max(accuracy_scores.items(),key=operator.itemgetter(1))[0]
    # select and train the estimator with minimun mse
    best_regressor = models[max_accuracy_classifier]
    print('\n')
    print('*'*36)
    print('best model selected:', max_accuracy_classifier)
    print('*'*36)
    best_regressor.fit(X,y)
    return best_regressor



## Scraper to get the data from categories
def get_data_by_category(categories_dict):
    """
    Function to collect the data from the MELI API by category.
    Input: 
    - categories_dict: dictionary with the categories available per country.
    Output: 

    """
    countries = categories_dict['Country']
    categories = categories_dict['Category_id']
    categories_name = categories_dict['Category_name']
    sites = categories_dict['country_id']
    data =[]
    for i in range(len(categories)):
        try:
            # first querie
            url=f'https://api.mercadolibre.com/sites/{sites[i]}/search?category={categories[i]}' 
            # make requets
            r = requests.get(url)
            if r.status_code==200:
                raw = r.json()
                total= int(raw['paging']['total'])
                   # Amount of records to get from the available, depending of total value.
            if total<1000:
                amount_records=int(0.5*total)
            elif total>1000 and total<10000:
                amount_records = int(0.1*total)
            elif total>10000 and total<100000:
                amount_records=int(0.01*total)
            elif total>100000:
                amount_records = int(0.001*total)
            print(f'Amount of extracted records for category {categories[i]}: {amount_records}')
        except Exception as e:
            print(e)
            continue

        for j in range(0,amount_records,50):
            try:
                url=f'https://api.mercadolibre.com/sites/{sites[i]}/search?category={categories[i]}&offset={int(j)}'    
                # make requets
                time.sleep(2.5)
                r = requests.get(url)
                #verify status code
                if r.status_code==200:
                    raw = r.json()
                    # get the results from request
                    res = raw['results']
                    for l in range(50):
                        # info extracted for each result
                        post_id = res[l]['id']
                        seller_id=res[l]['seller']['id']
                        title = res[l]['title']
                        price = res[l]['price']
                        original_price = res[l]['original_price']
                        aval_quantity=res[l]['available_quantity']
                        sold = res[l]['sold_quantity']
                        condition = res[l]['condition']
                        accept_mercadopago = res[l]['accepts_mercadopago']
                        shipping_state = res[l]['shipping']['free_shipping']
                        order_backed = res[l]['order_backend']
                        tags=res[l]['tags']
                        try:
                            state=res[l]['seller_address']['city']['name']
                        except:
                            state='No_available'
                        # adding all the info as a field
                        data.append((post_id,seller_id,countries[i],state, categories_name[i],sites[i],
                        title,categories_name[i],price,original_price,aval_quantity,sold,condition,
                        accept_mercadopago,shipping_state,order_backed,tags))

                        # create a dataframe with the data collected
                        df_meli = pd.DataFrame(data,columns=['post_id','user_id','country','city','category_name',
                        'site_id','category','title','price','original_price','available_quantity','sold_quantity',
                        'condition','accepts_mercadopago','shipping_state','order_backend','tags'])

            except Exception as e:
                print(f'!Problem in case: {j}')
                print('Exception: ',e)
                continue  
    return df_meli