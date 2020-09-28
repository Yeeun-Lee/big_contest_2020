import pandas as pd
import numpy as np

from models.lgbm_reg import train_lgbm, pred, save_file


###### with final_performance #####

model = train_lgbm()
model.save_model("lgbm_reg.txt")

train = pd.read_csv("prep/data/final_performance_1.csv")

# train data
###### Load Trained Model ######
# model = lgb.Booster(model_file='lgbm_reg.txt')
prediction = pred(train, model)

save_file(prediction)


