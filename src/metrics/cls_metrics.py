from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def acc_f1(y_true, y_pred, average="macro"):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average=average))
    }

def conf_mat(y_true, y_pred, labels=None):
    return confusion_matrix(y_true, y_pred, labels=labels)
