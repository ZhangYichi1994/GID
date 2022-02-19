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


def multi_run_main(config):
    print_config(config)
    set_random_seed(config['random_seed'])
    hyperparams = []
    for k, v in config.items():
        if isinstance(v, list):
            hyperparams.append(k)

    scores = []
    configs = grid(config)
    for cnf in configs:
        print('\n')
        for k in hyperparams:
            cnf['out_dir'] += '_{}_{}'.format(k, cnf[k])
        print(cnf['out_dir'])
        model = ModelHandler(cnf)
        dev_metrics = model.train()
        test_metrics = model.test()
        scores.append(test_metrics[model.model.metric_name])

    print('Average score: {}'.format(np.mean(scores)))
    print('Std score: {}'.format(np.std(scores)))



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


def grid(kwargs):
    """Builds a mesh grid with given keyword arguments for this Config class.
    If the value is not a list, then it is considered fixed"""

    class MncDc:
        """This is because np.meshgrid does not always work properly..."""

        def __init__(self, a):
            self.a = a  # tuple!

        def __call__(self):
            return self.a

    sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
    for k, v in sin.items():
        copy_v = []
        for e in v:
            copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
        sin[k] = copy_v

    grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
    return [far.utils.merge_dicts(
        {k: v for k, v in kwargs.items() if not isinstance(v, list)},
        {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
    ) for vv in grd]


if __name__ == '__main__':
    
    cfg = {'config': 'config/HardinLoopPlatform/hdlp.yml', 'multi_run':False}
    config = get_config(cfg['config'])
    if cfg['multi_run']:
        multi_run_main(config)
    else:
        test_metrics, output, gold, id_test = main(config)
        if not config['one_class']:
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
            plt.figure(1)
            plt.plot(recall0, precision0, 'b--')
            plt.plot(recall1, precision1, 'r--')
            plt.plot(recall2, precision2, 'g--')

            plt.figure(2)
            plt.plot(fpr0, tpr0, 'b--')
            plt.plot(fpr1, tpr1, 'r--')
            plt.plot(fpr2, tpr2, 'g--')

            plt.figure(3)
            
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
            plt.savefig("..//out//confusionMatrix.svg", format = 'svg')
            plt.show()



        if config['one_class']:
            output = output.cpu().numpy()
            gold = gold.cpu().numpy()
            from sklearn.metrics import precision_recall_curve, roc_curve, auc
            precision3, recall3, thresholds3 = precision_recall_curve(gold, output[:,0])
            precision4, recall4, thresholds4 = precision_recall_curve(gold, output[:,1])

            fpr3, tpr3, threshold3 = roc_curve(gold, output[:,0]) 
            fpr4, tpr4, threshold4 = roc_curve(gold, output[:,1]) 

            roc_auc3 = auc(fpr3, tpr3)
            roc_auc4 = auc(fpr4, tpr4)

            print('roc auc 3 is {}.'.format(roc_auc3))
            print('roc auc 4 is {}.'.format(roc_auc4))

            import matplotlib.pyplot as plt
            plt.figure(1)
            plt.plot(recall3, precision3, 'g*')
            plt.plot(recall4, precision4, 'y--')
            
            plt.figure(2)
            plt.plot(fpr3, tpr3, color='green', label='ROC curve (area = %0.2f)' % roc_auc3)
            plt.plot(fpr4, tpr4, color='yellow', label='ROC curve (area = %0.2f)' % roc_auc4)

            plt.show()


