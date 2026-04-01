import sklearn.metrics as skm
import matplotlib.pyplot as plt
import numpy as np


def accuracy(y_true, y_pred):
    acc = skm.accuracy_score(y_true, y_pred)
    return acc

def recall(y_true, y_pred):
    rec = skm.recall_score(y_true, y_pred, average='macro', zero_division=0)
    return rec

def precision(y_true, y_pred):
    rec = skm.precision_score(y_true, y_pred, average='macro', zero_division=0)
    return rec

def f1_score(y_true, y_pred):
    f1 = skm.f1_score(y_true, y_pred, average='macro', zero_division=0)
    return f1

def get_iou(y_true, y_pred):
    if len(y_true.shape) > 1:
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
    C = skm.confusion_matrix(y_true, y_pred)
    iou = C[0, 0] / (C[0, 0] + C [0, 1] + C[1, 0])
    return iou

def draw_roc_curve(y_true, y_score, fig_size=None, save_path=None, label=None):
    fig_size_ = fig_size if fig_size is not None else (3,3)
    plt.figure(figsize=fig_size_)
    plt.rcParams.update({
        'font.family':'Arial',
        'figure.dpi':300,
        'savefig.dpi':300,
        'font.size':8,
        'legend.fontsize':'small'
        })
    # plt.margins(0,0)
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    # color_list = ['#66CCCC','#FF99CC','#CCFF66','#FF9999',
    #                 '#99CC66','#FF6666','#FF9900',
    #                 '#666699','#CC3399','#66CCFF','#6699CC',
    #                 '#009933','#FFCC33','#0066CC','#99CCCC','#666666',
    #                 '#FF0033','#993399']
    # linestyles = ['-','-','-','-','-','-','-','-','-','-']

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title(title)

    FPR, TPR, thresholds = skm.roc_curve(y_true, y_score, pos_label=1)
    AUC_ROC = skm.auc(FPR, TPR)
    plt.plot(FPR, TPR, lw=1.0, color='#FA7F6F', linestyle='-', label=f'{label} ({AUC_ROC:.3f})')
        
    plt.plot([0, 1.1], [0, 1.1], color='#82B0D2', lw=1.0, linestyle='--')
    plt.legend(loc="lower right", frameon=True, fontsize=6)
    
    if save_path is not None:
        # plt.savefig('./result_figure/diagnosis_roc.eps', format='eps', bbox_inches='tight', pad_inches=0)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return AUC_ROC

def confusion_matrix(y_true, y_pred, labels, fig_size=None, save_path=None):
    C = skm.confusion_matrix(y_true, y_pred, normalize='true')
    if save_path is not None:
        plt.rcParams.update({
            'font.family':'Times New Roman',
            'figure.dpi':300,
            'savefig.dpi':300,
            'font.size':6,
            'legend.fontsize':'small'
            })
        # plt.figure(figsize=(1, 1), dpi=300)
        fig_size_ = fig_size if fig_size is not None else (8,6)
        plt.figure(figsize=fig_size_)
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.matshow(C, fignum=1, cmap=plt.cm.Blues)
        plt.colorbar()
        for i in range(len(C)):
            for j in range(len(C)):
                plt.annotate(f'{C[j, i]:.3f}', xy=(i, j), horizontalalignment='center', verticalalignment='center', fontsize=6)
        plt.tick_params(labelsize=6)
        plt.ylabel('True label', loc='center')
        plt.xlabel('Predicted label', loc='center')
        plt.xticks(range(len(labels)), labels=labels, rotation=45)
        # plt.gca().xaxis.tick_bottom()
        plt.yticks(range(len(labels)), labels=labels, rotation=45) 
        plt.savefig(save_path)
        plt.close()
    return C

