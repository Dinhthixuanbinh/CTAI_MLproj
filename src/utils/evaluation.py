from sklearn.metrics import classification_report

def get_classification_report(golds, preds):
    return classification_report(golds, preds)