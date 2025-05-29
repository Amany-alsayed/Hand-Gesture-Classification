
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path
import logging

def evaluation(y_test,y_pred,output_dir):
    logging.info("Calculating evaluation metrics...")
    classes=['call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one',
       'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted',
       'three', 'three2', 'two_up', 'two_up_inverted']
    accuracy=accuracy_score(y_test,y_pred)
    pres=precision_score(y_test,y_pred,average='weighted')
    recall=recall_score(y_test,y_pred,average='weighted')
    f1_sc=f1_score(y_test,y_pred,average='weighted')
    conf=confusion_matrix(y_test,y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf,display_labels=classes)
    disp.plot(cmap="Blues")
    plt.xticks(rotation=90, ha="right")
    plt.title("Confusion Matrix")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "conf_matrix.png")
    plt.close()
    return accuracy,pres,recall,f1_sc
