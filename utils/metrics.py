import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score,label_ranking_average_precision_score


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}



def MRR(test_df, topk=20, user_col_name='item_id', label_col_name='labels', pred_col_name='scores'):
    # 按照item_id 分组
    user_groups = test_df.groupby(user_col_name)
    y_trues, y_preds = [], []
    # 遍历每个组
    for item_id in test_df[user_col_name].unique():
        cur_user = user_groups.get_group(item_id)

        if cur_user.shape[0] >= topk:
            y_trues.append(cur_user[label_col_name].values)
            y_preds.append(cur_user[pred_col_name].values)

    return label_ranking_average_precision_score(np.array(y_trues), np.array(y_preds))

def MAP(test_df, topk=20, user_col_name='item_id', label_col_name='labels', pred_col_name='scores'):
    # 按照item_id 分组
    user_groups = test_df.groupby(user_col_name)

    pred_map = 0
    user_cou = 0
    # 遍历每个组
    for user_id in test_df[user_col_name].unique():

        cur_user = user_groups.get_group(user_id)

        if cur_user.shape[0] >= topk:
            cur_ap = average_precision_score(cur_user[label_col_name].values.ravel(), cur_user[pred_col_name].values.ravel())
            if not np.isnan(cur_ap):
                pred_map += cur_ap
                user_cou += 1
    return pred_map / user_cou

def HitRate(test_df, topk=20, user_col_name='item_id', label_col_name='labels', pred_col_name='scores'):
    # 按照item_id 分组
    item_groups = test_df.groupby(user_col_name)

    hit = 0
    user_item_cou = 0
    # 遍历每个组
    for item_id in test_df[user_col_name].unique():

        group_df = item_groups.get_group(item_id)

        if group_df.shape[0] >= topk:
            group = group_df.sort_values(pred_col_name, ascending=False)[:topk]
            hit += sum(group[label_col_name])          # 前tokp里面击中的个数
            user_item_cou += group_df.shape[0]     # 每个用户测试集里面的被推荐item个数

    return hit / user_item_cou