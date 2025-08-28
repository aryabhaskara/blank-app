import xgboost as xgb

def predict(data):
    model = xgb.XGBClassifier()
    model.load_model('model/best_model_xgb.model')    
    return model.predict(data)
