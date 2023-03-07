import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 读取数据
city = "Lianyungang"

if city == "dongying":
    gdp = np.array([1376, 1329, 1288, 1167, 1189, 1350, 3887, 4327, 2958, 4469, 3572, 4097, 4175, 4658, 5222, 7422, 11090, 12998, 14333, 16227, 16672, 17040, 23267,
     24514, 25686, 31914, 39463, 48607, 53834, 59194, 71410, 71724, 80829, 90569, 100194, 107862, 113326, 113871, 114510, 122949, 128727, 134022, 133859, 156852])
elif city == "Weifang":
    gdp = np.array([335,367,394,443,538,594,698,821,989,1165,1550,1712,1917,2202,2604,3281,4132,5066,5961,6653,7170,7553,8277,9223,10239,11501,13448,16009,18100,
    21259,25183,27364,31295,35312,40007,44533,48220,50708,54859,56855,58780,60651,62731,74606])
elif city == "Yantai" :
    gdp = np.array([458, 488, 535, 542, 585, 693, 795, 962, 1100, 1394, 1867, 2010, 2362, 2720, 3624, 5043, 6730, 8451, 9589, 10466, 11439, 12345, 13537, 15060, 
    17131, 20221, 25061, 30923, 34623, 41271, 48656, 52311, 58367, 66311, 71865, 76575, 81815, 87243, 92180, 96099, 101820, 108199, 108843, 122818])
elif city == "Weihai" :
    gdp = np.array([440, 453, 551, 607, 604,803,962,1117,1211,1480,2168,2436,3041,3679,5332,7311,10889,13044,15170,16861,18090,18925,20751,22249,24472,27906,
    31660,35520,40826,46599,51208,55948,62012,67564,74572,80464,88264,94517,102099,109270,113189,102818,102897,118925])
elif city == "Qingdao" :
    gdp = np.array([663,742,819,822,857,1002,1150,1311,1483,1746,2199,2426,2714,3053,3856,5455,7436,9089,10130,11235,12443,13884,15900,17882,20236,23340,27577,
    31725,36743,42509,49183,53481,62148,70945,76636,82463,87677,92157,97430,104905,112129,118978,123828,138849])
elif city == "Rizhao" :
    gdp = np.array([256, 289, 332, 365, 418, 461, 508, 583, 677, 825, 1089, 1137, 1255, 1460, 1683, 2186, 3306, 4051, 4869, 5499, 6133, 6586, 7152, 7866, 8743, 
    10332, 12754, 14957, 17444, 21524, 25892, 28561, 32703, 37843, 41341, 45267, 48410, 48456, 52117, 57394, 61031, 66193, 67376, 74434])
elif city == "Lianyungang" :
    gdp = np.array([321, 351, 381, 424, 527, 585, 673, 833, 981, 1075, 1231, 1280, 1391, 1456, 1660, 2058, 2620, 3240, 3939, 4501, 4916, 5209, 5512, 5884, 
    6427, 7141, 8579, 10910, 13204, 15706, 18618, 21310, 27179, 32312, 36625, 41131, 44419, 51219, 56041, 61173, 63926, 68151, 71303, 81015])




# gdp = np.array([1376, 1329, 1288, 1167, 1189, 1350, 3887, 4327, 2958, 4469, 3572, 4097, 4175, 4658, 5222, 7422, 11090, 12998, 14333, 16227, 16672, 17040, 23267, 24514, 25686, 31914, 39463, 48607, 53834, 59194, 71410, 71724, 80829, 90569, 100194, 107862, 113326, 113871, 114510, 122949, 128727, 134022, 133859, 156852])
print(gdp.shape)
time = np.arange(1978, 2023)
print(time.shape)
# 数据归一化
# gdp_max = np.max(gdp)
# gdp_min = np.min(gdp)
# gdp = (gdp - gdp_min) / (gdp_max - gdp_min)

# # 构造训练数据
x_train = []
y_train = []
for i in range(10, len(gdp)):
    x_train.append(gdp[i-10:i])
    y_train.append(gdp[i])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape(34,10,1)
print(x_train.shape,y_train.shape)

# 构造模型
# model = keras.Sequential([
#     layers.Dense(64, activation='relu', input_shape=(10,)),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(1)
# ])

# model = keras.Sequential([
#     layers.LSTM(64, activation='relu', input_shape=(10,1 )),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(1)
# ])

model = keras.Sequential()
model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(10,1)))
model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mae')
model.summary()

# 训练模型
model.fit(x_train, y_train, epochs=5, batch_size=16)

# 预测未来十年的GDP
forecast = []
x_test = gdp[-10:]
year = 30
for i in range(year):
    y_pred = model.predict(x_test.reshape(1,10, -1))[0][0]
    forecast.append(y_pred)
    x_test = np.append(x_test[1:], y_pred)
    
# 反归一化
# forecast = forecast * (gdp_max - gdp_min) + gdp_min

# 输出结果
print(city + "未来{}年的GDP预测值为：".format(year), forecast)
 
# city = list(city)

# out_data = pd.DataFrame(data=forecast)
# out_data.to_csv("result.csv",header = None, index = None)

plt.plot(range(1978, 2022+year),np.array(gdp.tolist() + forecast))
plt.savefig("test.jpg")
