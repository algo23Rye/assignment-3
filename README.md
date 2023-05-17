# 研报复现作业三

作业三主要参考论文de Jesus G S, Hoppe A F, Sartori A, et al, *An Approach to Forecasting Commodity Futures Prices Using
Machine Learning*，但我主要参考的是论文中因子的构建和处理以及模型搭建部分，数据使用和一些细节上的改动如下：

1. 该论文研究对象是巴西和国际商品期货市场，本研究针对的是中国期货市场。
2. 该论文使用Keras库搭建LSTM模型，本研究使用pytorch搭建。
3. 该论文模型中的目标函数是商品期货的价格，但个人觉得价格不是平稳的时间序列，所以在实际测试时候改成了预测收益率来作为目标函数(在prediction.py中LSTM_Prediction类中也写了把价格作为目标函数的方法get_price_as_objective)
4. 实际我用来预测的LSTM模型在该论文构建模型的基础上做了简化，减少了隐层数量，因为输入的特征也不多，个人觉得没必要用那么多隐层，训练速度慢而且dropout层太多结果不稳定（在prediction.py中Mylstm1是根据论文搭建的模型，Mylstm2是我做了简化后的模型)
5. 算力原因这里只进行每次向前一步滚动未来一天收益率的预测，该论文进行了未来1天和3天期货合约价格的预测。
6. 该论文中没有明确说预测时候是直接划分训练和测试集，还是窗口滚动，窗口拓展方法来进行预测的，我只测试了滚动窗口方法的预测，在prediction.py中LSTM_Prediction类中我也写了直接划分样本和窗口拓展方法来预测的方法。

## 预测结果

这里只进行了简单的测试，输入的特征参考论文中的做法只使用Talib库来构造技术指标，并且由于算力原因，这里未对lstm模型针对不同的期货品种进行精细地调参。参考论文，这里对玉米，强麦，白糖，强麦和豆一主力连续合约进行了测试，发现预测结果并不太好，在不同的期货品种中差异比较大。只有am和cm两个品种预测效果比较好，CFM和WHM的预测在2020年出现了模型预测失效，所以导致后面的净值曲线和真实曲线差异较大（虽然走势接近），但对于SRM来说，2021年之后模型几乎是失效了。个人认为可能有如下原因并且以后可以继续完善：一是目前只使用了Talib库中的技术指标，后续可以加入更复杂的技术指标以及基本面指标来更好地去刻画期货市场的收益来源，二是每个期货品种有自己的特点，未来应该根据每个期货品种来更有针对性的调参来提升模型效果。

1. 均方误差


|     | mean square error |
| --- | ----------------- |
| SRM | 0.0107            |
| CFM | 0.0167            |
| cm  | 0.0097            |
| am  | 0.0142            |
| WHM | 0.0143            |

2. 由所预测的收益率转化成的净值曲线

![ampredictionnav.png](Image/am_prediction_nav.png?t=1684325099721)

![cmpredictionnav.png](Image/cm_prediction_nav.png?t=1684325136684)

![CFMpredictionnav.png](Image/CFM_prediction_nav.png?t=1684325114114)

![WHMpredictionnav.png](Image/WHM_prediction_nav.png?t=1684325163353)

![SRMpredictionnav.png](Image/SRM_prediction_nav.png?t=1684325152332)
