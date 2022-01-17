import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# predict the Canadian GDP based on time using linear regression 
def predict_GDP_onTime(data_path, industry, train_covid):
    
    # convert the data into a DataFrame object
    data = pd.read_csv(data_path)
    
    # preprocessing the data
    data.dropna()
    
    # show the correlation between the dataset
    print(data.corr())
    
    # define features and targets
    if(train_covid==1): # training set contain covid period
        time_x = data['num'].values.reshape(-1,1)
        gdp_change_y = data[industry]
    else: # training set not contain covid period
        time_x = data.loc[:277, ['num']]
        gdp_change_y = data.loc[:277, [industry]]
        
    # store the covid-19 period data    
    covid_time_x = data.loc[278:289, ['num']]
    covid_gdp_y = data.loc[278:289, [industry]]
    
     # split dataset into the training and test data
    x_train, x_test, y_train, y_test = train_test_split(time_x, gdp_change_y, test_size=0.2, random_state = 1)
    
     # create a linear regression object
    lin_reg = linear_model.LinearRegression()
    
    # training the model
    lin_reg.fit(x_train, y_train)

    # even we are not training covid-19 period, we need to store these data for testing
    if(train_covid==0):
        x_test = np.row_stack((x_test, covid_time_x))
        y_test = np.row_stack((y_test, covid_gdp_y))
    
    # make our prediction
    gdp_value_predict = lin_reg.predict(x_test) # 1997/1--2020/12 
    gdp_value_predict_COVID = lin_reg.predict(covid_time_x) # covid-19 period   
    
    # renew value to get all the points
    time_x = data['num'].values.reshape(-1,1)
    gdp_change_y = data[industry]
    
    # build a string for graph
    info = ''
    if((train_covid==1)and(industry=='Canada')):
        info = ' with training covid period'
    elif(train_covid==0):
        info = ' without training covid period'
        
    #draw the scatter plot and prediction line
    plt.xlabel('Time/month')
    plt.ylabel('GDP value/million')
    plt.title('Trends in GDP in '+ industry + info)
    
    if(industry=='Canada'):
        plt.scatter(time_x, gdp_change_y, color = 'red', s=5)
    
    if(industry=='Canada'):
        plt.plot(x_test, gdp_value_predict, color = 'blue', linewidth = 1.5)
    else:
        plt.plot(time_x[:277], gdp_change_y[:277], color = 'blue', linewidth = 1.5, label='non-COVID-19 period')
        plt.plot(time_x[276:], gdp_change_y[276:], color = 'red', linewidth = 1.5, label='COVID-19 period')
        plt.legend(loc = 'upper center')
    
    plt.xticks([0, 50, 100, 150, 200, 250, 300], ['Jan/1997','Feb/2001','Apr/2005','June/2009','Aug/2013','Oct/2017','Dec/2021'])
    plt.show()
    
    # print the r2 value for checking the accuracy of our prediction
    print('R Square value: %.2f' % r2_score(y_test, gdp_value_predict)) # 1997/1--2020/12
    
    #draw the scatter plot and prediction line during covid-19 period
    plt.xlabel('Time/month')
    plt.ylabel('GDP value/million')
    plt.title('Trends in GDP in '+ industry + ' during covid-19 period' + info)
    plt.scatter(covid_time_x, covid_gdp_y, color = 'red', s=5)
    plt.plot(covid_time_x, gdp_value_predict_COVID, color = 'blue', linewidth = 1.5)
    plt.xticks([280, 282, 284, 286, 288], ['Apr/2020','June/2020','Aug/2020','Oct/2020','Dec/2020'])
    plt.show()
    
    # print the r2 value for checking the accuracy of our prediction
    print('R Square value: %.2f' % r2_score(covid_gdp_y, gdp_value_predict_COVID)) # covid-19 period
    
    
    
# predict the GDP value based on covid-19 new cases rate using linear regression
def predict_GDP_onCovid(data_path, industry):
    
    # build a new file path
    newfile = 'w' + data_path
    
    # sort the raw data in ascending order then write them in a new data file
    with open(data_path) as sample, open(newfile, "w") as out:
        csv1=csv.reader(sample)
        header = next(csv1, None)
        csv_writer = csv.writer(out)
        if header:
            csv_writer.writerow(header)
        csv_writer.writerows(sorted(csv1, key=lambda x:float(x[0])))
        
    # convert the data into a DataFrame object
    data = pd.read_csv(newfile)
    
    # preprocessing the data
    data.dropna()
    
    # show the correlation between the dataset
    print(data.corr())

    # define daily new cases rate as the feature
    new_casesRate_x = data['newCases'].values.reshape(-1,1)
 
    # define GDP value change as target
    GDP_change_y = data[industry]

    # split dataset into the training and test data
    x_train, x_test, y_train, y_test = train_test_split(new_casesRate_x, GDP_change_y, test_size=0.5, random_state = 1)

    # create a linear regression object
    lin_reg = linear_model.LinearRegression()
    
    # training the model
    lin_reg.fit(new_casesRate_x, GDP_change_y)
    
    # test the model
    gdp_value_predict = lin_reg.predict(new_casesRate_x)
    
    #draw the scatter plot and prediction line during covid-19 period
    plt.xlabel('New Cases rate')
    plt.ylabel(' GDP value changes/million')
    plt.title('Relationship Between New COVID-19 Cases rate and GDP value changes\n in Canadian '+ industry + ' industry')
    plt.scatter(new_casesRate_x, GDP_change_y, color = 'red')
    plt.plot(new_casesRate_x, gdp_value_predict, color = 'blue', linewidth = 1.5)
    plt.show()
    
    # print the r2 value for checking the accuracy of our prediction
    print('R Square value: %.2f' % r2_score(GDP_change_y.values, gdp_value_predict))
    
    #draw the scatter plot and prediction line during covid-19 period only show the the prediction of test set
    lin_reg.fit(x_train, y_train)
    gdp_value_predict2 = lin_reg.predict(x_test) 
    plt.xlabel('New Cases rate')
    plt.ylabel(' GDP value changes/million')
    plt.title('Relationship Between New COVID-19 Cases rate and GDP value changes in Canadian '+ industry + ' industry (test set)' )
    #plt.scatter(x_test, y_test, color = 'red')
    plt.scatter(new_casesRate_x, GDP_change_y, color = 'red')
    plt.plot(x_test, gdp_value_predict2, color = 'blue', linewidth = 1.5)
    plt.show()
    
    # print the r2 value for checking the accuracy of our prediction
    print('R Square value: %.2f' % r2_score(y_test.values, gdp_value_predict2))
    
    
    
# predict the GDP value based on covid-19 new cases rate using polynomial regression    
def predict_GDP_onCovid2(data_path, industry, d):
    
    # build a new file path
    newfile = 'w' + data_path
    
    # sort the raw data in ascending order then write them in a new data file
    with open(data_path) as sample, open(newfile, "w") as out:
        csv1=csv.reader(sample)
        header = next(csv1, None)
        csv_writer = csv.writer(out)
        if header:
            csv_writer.writerow(header)
        csv_writer.writerows(sorted(csv1, key=lambda x:float(x[0])))    
    
    # convert the data into a DataFrame object
    data = pd.read_csv(newfile)
    
    # preprocessing the data
    data.dropna()
    
    # show the correlation between the dataset
    print(data.corr())

    # define daily new cases rate as the feature
    new_casesRate_x = data['newCases'].values.reshape(-1,1)
 
    # define GDP value change as target
    GDP_change_y = data[industry]

    # split dataset into the training and test data
    x_train, x_test, y_train, y_test = train_test_split(new_casesRate_x, GDP_change_y, test_size=0.5, random_state = 1)
    
    # preprocess the data sets, sort them in ascending order
    Z = zip(x_train,y_train)
    Z = sorted(Z,reverse=False)
    x_train,y_train = zip(*Z)
    
    Z = zip(x_test,y_test)
    Z = sorted(Z,reverse=False)
    x_test,y_test = zip(*Z)

    # choose the degree of the polynomial features
    poly_reg = PolynomialFeatures(degree=d)
    X_poly = poly_reg.fit_transform(new_casesRate_x)

    # create a linear regression object
    lin_reg = linear_model.LinearRegression()
    
    # training the model
    lin_reg.fit(X_poly, GDP_change_y)
    
    # test the model
    gdp_value_predict = lin_reg.predict(poly_reg.fit_transform(new_casesRate_x))
    
    #draw the scatter plot and prediction line during covid-19 period
    plt.xlabel('New Cases rate')
    plt.ylabel(' GDP value changes/million')
    plt.title('Relationship Between New COVID-19 Cases rate and GDP value changes\n in Canadian '+ industry + ' industry')
    plt.scatter(new_casesRate_x, GDP_change_y, color = 'red')
    plt.plot(new_casesRate_x, gdp_value_predict, color = 'blue', linewidth = 1.5)
    plt.show()
    
    # print the r2 value for checking the accuracy of our prediction
    print('R Square: %.2f' % r2_score(GDP_change_y.values, gdp_value_predict))
    
    #draw the scatter plot and prediction line during covid-19 period only show the the prediction of test set
    X_poly2 = poly_reg.fit_transform(x_train)
    lin_reg.fit(X_poly2, y_train)
    gdp_value_predict2 = lin_reg.predict(poly_reg.fit_transform(x_test)) 
    plt.xlabel('New Cases rate')
    plt.ylabel(' GDP value changes/million')
    plt.title('Relationship Between New COVID-19 Cases rate and GDP value changes in Canadian '+ industry + ' industry (test set)')
    #plt.scatter(x_test, y_test, color = 'red')
    plt.scatter(new_casesRate_x, GDP_change_y, color = 'red')
    plt.plot(x_test, gdp_value_predict2, color = 'blue', linewidth = 1.5)
    plt.show()
    
    # print the r2 value for checking the accuracy of our prediction
    print('R Square: %.2f' % r2_score(y_test, gdp_value_predict2))

    

# predict the Canadian GDP based on time using polynomial regression 
def predict_GDP_onTime2(data_path, industry, train_covid, d):
    
    # convert the data into a DataFrame object
    data = pd.read_csv(data_path)
    
    # preprocessing the data
    data.dropna()
    
    # show the correlation between the dataset
    print(data.corr())
    
    # define features and targets
    if(train_covid==1): # training set contain covid period
        time_x = data['num'].values.reshape(-1,1)
        gdp_change_y = data[industry]
    else: # training set not contain covid period
        time_x = data.loc[:277, ['num']].values.reshape(-1,1)
        gdp_change_y = data.loc[:277, [industry]].values.reshape(-1,1)
        
    # store the covid-19 period data     
    covid_time_x = data.loc[278:289, ['num']]
    covid_gdp_y = data.loc[278:289, [industry]]
    
     # split dataset into the training and test data
    x_train, x_test, y_train, y_test = train_test_split(time_x, gdp_change_y, test_size=0.2, random_state = 1)
    
    # preprocess the data sets, sort them in ascending order
    Z = zip(x_train,y_train)
    Z = sorted(Z,reverse=False)
    x_train,y_train = zip(*Z)
    
    Z = zip(x_test,y_test)
    Z = sorted(Z,reverse=False)
    x_test,y_test = zip(*Z)
    
    # choose the degree of the polynomial features
    poly_reg = PolynomialFeatures(degree=d)
    X_poly = poly_reg.fit_transform(x_train)
    
     # create a linear regression object
    lin_reg = linear_model.LinearRegression()
    
    # training the model
    lin_reg.fit(X_poly, y_train)

    # even we are not training covid-19 period, we need to store these data for testing
    if(train_covid==0):
        x_test = np.row_stack((x_test, covid_time_x))
        y_test = np.row_stack((y_test, covid_gdp_y))
    
    # make our prediction
    gdp_value_predict = lin_reg.predict(poly_reg.fit_transform(x_test)) # 1997/1--2020/12 
    gdp_value_predict_COVID = lin_reg.predict(poly_reg.fit_transform(covid_time_x)) # covid-19 period   
    
    # renew value to get all the points
    time_x = data['num'].values.reshape(-1,1)
    gdp_change_y = data[industry]
    
    # build a string for graph
    info = ''
    if(train_covid==1):
        info = ' with training covid period'
    else:
        info = ' without training covid period'
        
    #draw the scatter plot and prediction line
    plt.xlabel('Time/month')
    plt.ylabel('GDP value/million')
    plt.title('Trends in GDP in '+ industry + info)
    plt.scatter(time_x, gdp_change_y, color = 'red', s=5)
    plt.plot(x_test, gdp_value_predict, color = 'blue', linewidth = 1.5)
    plt.xticks([0, 50, 100, 150, 200, 250, 300], ['Jan/1997','Feb/2001','Apr/2005','June/2009','Aug/2013','Oct/2017','Dec/2021'])
    plt.show()
    
    # print the r2 value for checking the accuracy of our prediction
    print('R Square value: %.2f' % r2_score(y_test, gdp_value_predict)) # 1997/1--2020/12
    
    #draw the scatter plot and prediction line during covid-19 period
    plt.xlabel('Time/month')
    plt.ylabel('GDP value/million')
    plt.title('Trends in GDP in '+ industry + ' during covid-19 period')
    plt.scatter(covid_time_x, covid_gdp_y, color = 'red', s=5)
    plt.plot(covid_time_x, gdp_value_predict_COVID, color = 'blue', linewidth = 1.5)
    plt.xticks([280, 282, 284, 286, 288], ['Apr/2020','June/2020','Aug/2020','Oct/2020','Dec/2020'])
    plt.show()
    
    # print the r2 value for checking the accuracy of our prediction
    print('R Square value: %.2f' % r2_score(covid_gdp_y, gdp_value_predict_COVID)) # covid-19 period   
    
  
    
# method callings
predict_GDP_onTime('all_on_time.csv','Canada',0)    
predict_GDP_onTime('all_on_time.csv','Canada',1)
predict_GDP_onTime2('all_on_time.csv','Canada',0, 9) # best degree = 9
predict_GDP_onTime2('all_on_time.csv','Canada',1, 9) # for compare with the last call    
predict_GDP_onTime('accommodation_alltime.csv','accommodation',1)
predict_GDP_onTime('entertainment_alltime.csv','entertainment',1)
predict_GDP_onTime('manufacturing_alltime.csv','manufacturing',1)
predict_GDP_onTime('agriculture_alltime.csv','agriculture',1)
predict_GDP_onTime('utilities_alltime.csv','utilities',1)
predict_GDP_onTime('cannabis_alltime.csv','cannabis',1)    
predict_GDP_onCovid('food_accommodation.csv','accommodation') 
predict_GDP_onCovid2('food_accommodation.csv','accommodation', 2) 
predict_GDP_onCovid('entertainment.csv','entertainment') 
predict_GDP_onCovid2('entertainment.csv','entertainment', 2) 
predict_GDP_onCovid('manufacturing.csv','manufacturing') 
predict_GDP_onCovid2('manufacturing.csv','manufacturing', 2) 
predict_GDP_onCovid('agriculture.csv','agriculture') 
predict_GDP_onCovid2('agriculture.csv','agriculture', 2) 
predict_GDP_onCovid('utilities.csv','utilities') 
predict_GDP_onCovid2('utilities.csv','utilities', 2) 
predict_GDP_onCovid('cannabis.csv','cannabis') 
predict_GDP_onCovid2('cannabis.csv','cannabis', 2)   
