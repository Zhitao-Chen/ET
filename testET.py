# interpreter: python3.9
# Critical package:scikit-learn 1.0.2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import joblib

# display Chinese characters normally when drawing a picture
plt.rcParams['font.sans-serif'] = ['SimHei']
#display negative sign normally when drawing a picture
plt.rcParams['axes.unicode_minus'] = False

# Read the test set in Excel file format
testset_dir = r'data\TAW-D.xlsx'
df_test = pd.read_excel(testset_dir).values
target_test = df_test[:, 5].tolist()
target_test = np.log10(target_test)
observation_test = df_test[:, :5]

# load model
regs = joblib.load("ET.joblib")

# predict
start_time1 = time.perf_counter()
re1 = regs.predict(observation_test)
end_time1 = time.perf_counter()
CPU_time1 = end_time1 - start_time1

# plot
plt.figure("Fitting")
plt.plot(target_test, label="targets")
plt.plot(re1, label="predictions")
plt.xlabel("Sample number")
plt.ylabel("Buildup factor")
plt.legend()
plt.show()

# Output the results in Excel file format
for i in range(len(re1)):
    re1[i] = 10 ** re1[i]
df_re = pd.read_excel(testset_dir)
df_re['ET'] = re1
filename = os.path.basename(testset_dir)
filename = filename.split('.')[0]
filename = filename + '-results.xlsx'
path = os.path.dirname(testset_dir) + '\\' + filename
df_re.to_excel(path, encoding='utf-8', index=False)