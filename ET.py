# interpreter: python3.9
# Critical package:scikit-learn 1.0.2
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
import time
import joblib

# display Chinese characters normally when drawing a picture
plt.rcParams['font.sans-serif'] = ['SimHei']
#display negative sign normally when drawing a picture
plt.rcParams['axes.unicode_minus'] = False

# Read the train set in Excel file format
trainset_dir = r'data\trainSet.xlsx'
df_training = pd.read_excel(trainset_dir).values
target_train = df_training[:, 5].tolist()
target_train = np.log10(target_train)
observation_train = df_training[:, :5]

regs = ExtraTreesRegressor(n_jobs=16, random_state=0)

# training
regs.fit(observation_train, target_train)

# predict
start_time1 = time.perf_counter()
re1 = regs.predict(observation_train)
end_time1 = time.perf_counter()
CPU_time1 = end_time1 - start_time1

# plot
plt.figure("Fitting")
plt.plot(target_train, label="targets")
plt.plot(re1, label="predictions")
plt.xlabel("Sample number")
plt.ylabel("Buildup factor")
plt.legend()
plt.show()

# Output the results in Excel file format
for i in range(len(re1)):
    re1[i] = 10 ** re1[i]
df_re = pd.read_excel(trainset_dir)
df_re['ET'] = re1
filename = os.path.basename(trainset_dir)
filename = filename.split('.')[0]
filename = filename + '-results.xlsx'
path = os.path.dirname(trainset_dir) + '\\' + filename
df_re.to_excel(path, encoding='utf-8', index=False)

# save model
joblib.dump(regs, "ET.joblib")

