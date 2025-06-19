from time import time
import sklearn
import pandas as pd

def get_classifier_metrics(y, preds, sub_name=""):
    metrics = {}
    metrics["accuracy"+sub_name]  = sklearn.metrics.accuracy_score(y, preds)
    metrics["precision"+sub_name] = sklearn.metrics.precision_score(y, preds)
    metrics["recall"+sub_name]    = sklearn.metrics.recall_score(y, preds)
    metrics["f1"+sub_name]        = sklearn.metrics.f1_score(y, preds)

    return metrics

def get_sklearn_params(instance, params=False):
    # instance can be a Model, Scaler, Vectorizer... and any other instance having the method "get_params()"
    #
    # params = False -> Nothing
    # params = True  -> Include all model parameters (will add NaN in the result if comparing models with different paramenters)
    # params = ['random_state'] -> Include model params in the list (will add NaN if model has not the param)
    result = {}
    if params == True:
        result |= model.get_params()
    elif isinstance(params, list) and params:
        result |= {param:value for param, value in instance.get_params().items() if param in params}

    return result

def evaluate_model(model, X_train, y_train, X_test, y_test, scaler=False, vectorizer=False, inc_params=False):
    vdata = sdata = {}
    if vectorizer:
        X_train = vectorizer.fit_transform(X_train)
        X_test  = vectorizer.transform(X_test)

        # Add model name and fit time
        vdata |= {"vectorizer": type(vectorizer).__name__}
        vdata |= get_sklearn_params(vectorizer, params=inc_params)
    
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Add model name and fit time
        sdata |= {"scaler": type(scaler).__name__}
        sdata |= get_sklearn_params(scaler, params=inc_params)

    # Fit model
    start = time()
    model.fit(X_train, y_train)

    mdata = {"model": type(model).__name__, "fit_time": time()-start} 
    mdata |= get_sklearn_params(model, params=inc_params) | vdata | sdata

    # Evaluate the model
    mdata |= get_classifier_metrics(y_train, model.predict(X_train), sub_name="_train")
    mdata |= get_classifier_metrics(y_test,  model.predict(X_test), sub_name="_test")

    return mdata

def compare_models(models, X_train, y_train, X_test, y_test, inc_params=False):
    results = []
    for model in models:
        results.append(evaluate_model(model, X_train, y_train, X_test, y_test, inc_params=inc_params))
    return pd.DataFrame(results)

def compare_scalers(scalers, models, X_train, y_train, X_test, y_test, inc_params=False):
    results = []
    for model in models:
        for scaler in scalers:
            results.append(evaluate_model(model, X_train, y_train, X_test, y_test, scaler=scaler, inc_params=inc_params))
    return pd.DataFrame(results)

def compare_vectorizers(vectorizers, models, X_train, y_train, X_test, y_test, inc_params=False):
    results = []
    for model in models:
        for vectrizer in vectorizers:
            results.append(evaluate_model(model, X_train, y_train, X_test, y_test, vectorizer=vectrizer, inc_params=inc_params))
    return pd.DataFrame(results)
