import joblib


def predict(data):
    clf = joblib.load('C:\\Users\\Oreof\\PycharmProjects\\Playzone\\Iris-App\\rf_model.sav')
    return clf.predict(data)
