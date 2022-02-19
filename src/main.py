import argparse
import yaml
import torch
import numpy as np
import far_ho as far
from collections import defaultdict, OrderedDict

from core.model_handler import ModelHandler

################################################################################
# Main #
################################################################################


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(config):
    print_config(config)
    set_random_seed(config['random_seed'])
    model = ModelHandler(config)
    model.train()
    test_metrics, output, gold = model.test()
    id_test = model.test_loader['idx_test']
    return test_metrics, output, gold, id_test

################################################################################
# ArgParse and Helper Functions #
################################################################################
def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    parser.add_argument('--multi_run', action='store_true', help='flag: multi run')
    args = vars(parser.parse_args())
    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


if __name__ == '__main__':
    
    cfg = {'config': 'config/HardinLoopPlatform/hilp.yml', 'multi_run':False}
    config = get_config(cfg['config'])

    test_metrics, output, gold, id_test = main(config)
    multi_label = output.max(1)[1].type_as(gold)    # from one-hot to label

    output = output[id_test].cpu().numpy()
    gold = gold[id_test].cpu().numpy()
    multi_label = multi_label[id_test].cpu().numpy()
    num_class = max(gold) + 1
    one_hot_gold = np.eye(num_class)[gold]

    from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report
    
    cfm = confusion_matrix(gold, multi_label)
    row_sums = np.sum(cfm, axis=1)
    err_matrix = (cfm.T / row_sums).T
    
    false_alarm_ratio = 1 - cfm[0,0] / sum(cfm[0,:])
    detect_ratio = 1 - sum(cfm[1:,0]) / sum(sum(cfm[1:,:]))
    print('False alarm ratio is: {}'.format(false_alarm_ratio))
    print('Detect ratio is: {}'.format(detect_ratio))
    
    
    # pr value
    precision0, recall0, thresholds0 = precision_recall_curve(one_hot_gold[:,0], output[:,0])
    precision1, recall1, thresholds1 = precision_recall_curve(one_hot_gold[:,1], output[:,1])
    precision2, recall2, thresholds2 = precision_recall_curve(one_hot_gold[:,2], output[:,2])
    print(precision0.shape)

    aupr_normal = auc(recall0, precision0)
    aupr_1 = auc(recall1, precision1)
    aupr_2 = auc(recall2, precision2)

    print('Normal aupr: {}'.format(aupr_normal))
    print('Abnormal 1 aupr: {}'.format(aupr_1))
    print('Abnormal 2 aupr: {}'.format(aupr_2))
    # roc value
    fpr0, tpr0, threshold0 = roc_curve(one_hot_gold[:,0], output[:,0])
    fpr1, tpr1, threshold1 = roc_curve(one_hot_gold[:,1], output[:,1])
    fpr2, tpr2, threshold2 = roc_curve(one_hot_gold[:,2], output[:,2])
    

    auroc_normal = auc(fpr0, tpr0)
    auroc_1 = auc(fpr1, tpr1)
    auroc_2 = auc(fpr2, tpr2)

    print('Normal auroc: {}'.format(auroc_normal))
    print('Abnormal 1 auroc: {}'.format(auroc_1))
    print('Abnormal 2 auroc: {}'.format(auroc_2))

    import matplotlib.pyplot as plt
    from decimal import *
    xlocations = np.array(range(8))
    labels_zh_cmn = ['0', '1', '2', '3', '4', '5', '6', '7']
    plt.xticks(xlocations, labels_zh_cmn, fontsize=10)
    plt.yticks(xlocations, labels_zh_cmn, fontsize=10)
    plt.ylabel('True Label', fontsize=10)
    plt.xlabel('Pred Label', fontsize=10)
    plt.imshow(err_matrix, interpolation='nearest', cmap="YlGnBu")
    plt.colorbar()

    # data visualization
    for first_index in range(len(err_matrix)):    # row
        for second_index in range(len(err_matrix[first_index])):    # column
            if second_index == first_index:
                fontColor = 'white'
            else:
                fontColor = 'black'
            plt.text(second_index, first_index, "%0.2f" %(err_matrix[first_index][second_index],), va='center', ha='center', color = fontColor)
    print('cfm matrix is: ')
    print(cfm)
    print('err_matrix is :')
    print(err_matrix)
    print(classification_report(gold, multi_label))
    plt.show()



