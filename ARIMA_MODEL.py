from pandas import read_csv

from datetime import datetime

from matplotlib import pyplot

from pandas.plotting import autocorrelation_plot

from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error

from math import sqrt

from statsmodels.tsa.stattools import adfuller

from tkinter import messagebox

import csv

# load dataset

def parser(x):

#  return datetime.strptime(x,'%-m/%-d/%Y')  # shampoo.csv

 return datetime.strptime(x,'%Y-%m-%d')  # dataset.csv

 #return datetime.strptime('190'+x, '%Y-%m')  shampoo-sales

def adfuller_test(sales):

    result=adfuller(sales)

    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']

    for value,label in zip(result,labels):

        print(label+' : '+str(value) )

    if result[1] <= 0.05:

        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")

    else:

        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

series = read_csv('dataset.csv', header=0, index_col=0, parse_dates=True, date_parser=parser)

test_result=adfuller_test(series['Sales'])

# series.plot()

# pyplot.show()

#fig,subPLotAxis=pyplot.subplots(3)

#subPLotAxis[0].plot(series['Sales'])

#subPLotAxis[1].plot(series['Sales'].diff())

#subPLotAxis[2].plot(series['Sales'].diff().diff())




series.index = series.index.to_period('M')

# autocorrelation_plot(series)

# split into train and test sets

# fig, (ax1, ax2, ax3) = pyplot.subplots(3)

# plot_acf(series['Sales'], ax=ax1)

# plot_acf(series['Sales'].diff().dropna(), ax=ax2)

# plot_acf(series['Sales'].diff().diff().dropna(), ax=ax3)

# pyplot.show()

X = series.values

print(X)


size = int(len(X) * 0.66)


train, test = X[0:size], X[size:len(X)]

history = [x for x in train]

predictions = list()

thresholds = list()

inventory_lefts= list()

new_inventory=list()

old_new_inventorys=list()




# walk-forward validation

for t in range(len(test)):

    model = ARIMA(history, order=(8,1,0))

    model_fit = model.fit()

    output = model_fit.forecast()

    yhat = output[0]

    predictions.append(yhat)

    obs = test[t]

    history.append(obs)

    threshold = yhat/3

    inventory_left=250 - obs

    thresholds.append(threshold)

    inventory_lefts.append(inventory_left)
    
    if(inventory_left>80):

        old_new_inventorys.append(inventory_left)
    else:

        old_new_inventorys.append(inventory_left+100)

        print("restocked happened")

    if(inventory_left<=threshold):

        new_inventory.append(inventory_left+100)

    else:

        new_inventory.append(inventory_left)

    print('predicted=%f, expected=%f , threshold=%f , inventory_left=%f' % (yhat, obs,threshold,inventory_left))


# evaluate forecasts

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)



# plot forecasts against actual outcomes

pyplot.plot(test)

pyplot.legend(['Real Time Sales'])

pyplot.plot(predictions, color='red')

pyplot.legend(['Predicted Sales'])

pyplot.axhline(y=250,color='aqua')

pyplot.axhline(y=80,color='purple')

pyplot.axhline(y=0,color='black')

pyplot.plot(old_new_inventorys,color='brown')

pyplot.legend(['Inventory'])

pyplot.plot(thresholds, color='green')

pyplot.legend(['Inventory Threshold @ one third'])

# pyplot.plot(inventory_lefts, color='orange')

# pyplot.legend(['Remaining Inventory by RealTime Sales'])

# pyplot.plot(new_inventory, color='pink')

# pyplot.legend(['Remaining Inventory by RealTime Sales using ARIMA'])



fig1 = pyplot.figure()  
ax1 = fig1.add_subplot(111) 
ax1.plot(test, label='Without Threshold')
ax1.plot(inventory_lefts, color='orange')
ax1.axhline(y=250,color='aqua')
ax1.axhline(y=0,color='black')
ax1.set_title('Plot 1 : Without Threshold')
ax1.set_xlabel('Time')
ax1.set_ylabel('Sales')
ax1.legend(["Real Time Sales","Inventory Left","Inventory","Empty Inventory"])

fig2=pyplot.figure()
ax2=fig2.add_subplot(111)  
ax2.plot(test, label='With static threshold')
ax2.axhline(y=250,color='aqua')
ax2.axhline(y=0,color='black')
ax2.axhline(y=80,color='purple')
ax2.plot(old_new_inventorys,color='brown')
ax2.set_title('Plot 2 : With Static Threshold')
ax2.set_xlabel('Time')
ax2.set_ylabel('Sales')
ax2.legend(["Real Time Sales","Inventory","Empty Inventory","Static Threshold","Restocked Inventory"])

fig3 = pyplot.figure()  
ax3 = fig3.add_subplot(111)  
ax3.plot(test, label='Arima Threshold')
ax3.axhline(y=250,color='aqua')
ax3.axhline(y=0,color='black')
ax3.plot(predictions, color='red')
ax3.plot(new_inventory, color='pink')
ax3.plot(thresholds, color='green')
ax3.set_title('Plot 3: ARIMA Threshold')
ax3.set_xlabel('Time')
ax3.set_ylabel('Sales')
ax3.legend(["Real Time Sales","Inventory","Empty Inventory","Predicted Sales","Arima Inventory","Arima Threshold"])


pyplot.show()

with open('month1.csv', encoding="utf8") as f:
    csv_reader = csv.DictReader(f)
    # skip the header
    next(csv_reader)
    # show the data
    x=predictions[1]
    for line in csv_reader:
        x=x-float(line["1"])
        if(x>80):
            old_new_inventorys.append(inventory_left)
        elif(x<=10):
                messagebox.showinfo(title="SORRYY!!!", message="OUT OF STOCK!!!")
                break
        else:
            #x=x+100
            messagebox.showinfo(title="Hurry UP!!", message="Come On ! It's Selling Filling fast!!!")
            #print("restocked happened")

