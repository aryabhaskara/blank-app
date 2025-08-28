import xgboost as xgb

def predict(data):
    #model1 = load_model('model/ANN_Torque_FINAL.h5')
    model1 = xgb.XGBRegressor()
    model1.load_model('model/XGB_TQ.model')    
    model2 = xgb.XGBRegressor()
    model2.load_model('model/XGB_SFC_MAE_final.model')
    model3 = xgb.XGBRegressor()
    model3.load_model('model/XGB_TE.model')
    return model1.predict(data), model2.predict(data), model3.predict(data)
