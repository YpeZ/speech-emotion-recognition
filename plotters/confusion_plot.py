import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tikzplotlib
from sklearn.metrics import confusion_matrix


def confusion_plot(actual, pred, name):
    labels =['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

    cm = confusion_matrix(y_true=actual, y_pred=pred, labels=labels)

    plt.figure(figsize=(12, 10))
    cm = pd.DataFrame(cm, index=[i for i in labels], columns=[i for i in labels])
    ax = sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(name + ' Confusion Matrix', size=20)
    plt.xlabel('Predicted Labels', size=14)
    plt.ylabel('Actual Labels', size=14)
    plt.savefig(name + '_Confusion_Matrix.png')

    # tikzplotlib.save(name.replace(' ', '_') + '_confusion_matrix.tex')
    plt.clf()


if __name__ == '__main__':
    pass
