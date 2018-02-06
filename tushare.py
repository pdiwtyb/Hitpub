 data = ts.get_hist_data('300386',start='2017-01-05',end='2018-02-05')
>>> data1 = ts.get_hist_data('300386',start='2017-01-04',end='2018-02-04')
>>> print(data.shape)
(268, 14)
>>> print(data1.shape)
(268, 14)
>>>
>>> x = data1[['open','high','close','low','volume','price_change','turnover']]
>>> y = data[['p_change']]
>>> fig, ax = plt.subplots()
>>>
>>>  predicted = cross_val_predict(linreg, x, y, cv=10)
  File "<stdin>", line 1
    predicted = cross_val_predict(linreg, x, y, cv=10)
    ^
IndentationError: unexpected indent
>>> predicted = cross_val_predict(linreg, x, y, cv=10)
>>> ax.scatter(y, predicted)
<matplotlib.collections.PathCollection object at 0x09FE6550>
>>> ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
[<matplotlib.lines.Line2D object at 0x09FE6D70>]
>>> plt.show()