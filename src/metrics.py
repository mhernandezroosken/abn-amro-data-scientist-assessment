import math

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix, \
    roc_auc_score


def mean_squared_error(y_target, y_pred):
    return np.mean((y_target - y_pred) ** 2)


def top_k_acc_1darray(y_target, y_pred, k):
    if max(y_target) == 0.0:
        return None
    top_k_vals = np.sort(y_pred)[-k:]
    top_k_bool = np.isin(y_pred, top_k_vals)
    correct = np.any(y_target[top_k_bool] >= 1.)
    return correct


def top_k_acc_month(y_target, y_pred, k):
    if y_pred.shape[0] != y_target.shape[0]:
        raise ValueError(f"Error: mismatching array shapes: {y_pred.shape} and {y_target.shape}.")
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape((1, -1))
    if len(y_target.shape) == 1:
        y_target = y_target.reshape((1, -1))

    results = [
        top_k_acc_1darray(y_target[i], y_pred[i], k) for i in range(y_pred.shape[0])
    ]
    results = [r for r in results if r is not None]

    if len(results) == 0:
        result = 1.0
    else:
        result = np.mean(results)

    return result


def get_metrics(y_val_total, y_pred_total, cutoff_prob=0.5):
    # total nr of (predicted) mortgage applications in the validation period
    # nr_applications_pred = np.sum(y_pred_total, axis=1)
    # nr_applications_val = np.sum(y_val_total, axis=1)

    # has_application_pred = nr_applications_pred >= 0.50
    # has_application_val = nr_applications_val >= 0.50

    # probabilities of a mortgage application occurring per month
    has_application_monthly_val = np.clip(
        y_val_total,
        a_min=0,
        a_max=1
    )
    has_application_monthly_pred = np.clip(
        y_pred_total,
        a_min=0,
        a_max=1
    )

    # probabilities of a mortgage application occurring in the period
    p_application_val = (1 - np.prod(1 - has_application_monthly_val, axis=-1))
    has_application_val = p_application_val >= cutoff_prob
    has_application_pred = np.max(has_application_monthly_pred, axis=-1) >= cutoff_prob

    # # find the optimal cutoff for classification
    # cutoffs = np.sort(p_application_val)
    #
    # fpr =

    # calculate and return metrics
    # rmse_month = math.sqrt(mean_squared_error(y_val_total, y_pred_total))
    # rmse_total = math.sqrt(mean_squared_error(nr_applications_val, nr_applications_pred))
    precision_has_appl, recall_has_appl, f1_has_appl, _ = precision_recall_fscore_support(
        has_application_val,
        has_application_pred,
        beta=1.0,
        average='binary'
    )
    bal_acc_has_appl = balanced_accuracy_score(has_application_val, has_application_pred)
    conf_mat = confusion_matrix(has_application_val, has_application_pred)
    tpr_has_appl = conf_mat[1, 1] / np.sum(conf_mat[1])
    tnr_has_appl = conf_mat[0, 0] / np.sum(conf_mat[0])
    roc_auc_has_appl = roc_auc_score(has_application_val, p_application_val)

    top1_acc_month = top_k_acc_month(y_val_total, y_pred_total, k=1)
    top2_acc_month = top_k_acc_month(y_val_total, y_pred_total, k=2)

    metrics = {
        # 'rmse_month': rmse_month,
        # 'rmse_total': rmse_total,
        'f1_has_appl': f1_has_appl,
        'precision_has_appl': precision_has_appl,
        'recall_has_appl': recall_has_appl,
        'balanced_acc_has_appl': bal_acc_has_appl,
        'tpr_has_appl': tpr_has_appl,
        'tnr_has_appl': tnr_has_appl,
        'conf_mat_has_appl': conf_mat,
        'roc_auc_has_appl': roc_auc_has_appl,
        'top1_acc_month': top1_acc_month,
        'top2_acc_month': top2_acc_month
    }
    return metrics


def print_metrics(metrics):
    # print('Applications per month: RMSE={:.3f}'.format(
    #     metrics['rmse_month']
    # ))
    # print('Applications period: RMSE={:.3f}'.format(
    #     metrics['rmse_total']
    # ))
    print(('Period had application: \n'
           '    ROC AUC={:.3f} \n'
           '    f1={:.3f} \n'
           '    precision={:.3f} \n'
           '    recall={:.3f}').format(
        metrics['roc_auc_has_appl'],
        metrics['f1_has_appl'],
        metrics['precision_has_appl'],
        metrics['recall_has_appl']
    ))
    print(('    Bal.Acc.={:.3f} \n'
           '    TPR={:.3f} \n'
           '    TNR={:.3f} \n'
           '    Confusion Matrix={}').format(
        metrics['balanced_acc_has_appl'],
        metrics['tpr_has_appl'],
        metrics['tnr_has_appl'],
        metrics['conf_mat_has_appl']
    ))
    print(('Which month had application:\n'
           '    Top-1 Acc.={:.3f}\n'
           '    Top-2 Acc.={:.3f}').format(
        metrics['top1_acc_month'],
        metrics['top2_acc_month'],
    ))


def create_prediction_csv(y_val_total, y_pred_total, client_nrs, month_nr_to_yearmonth,
                          nr_train, nr_val, nr_test, method_name):
    # predictions for the periods total
    y_pred_period = 1 - np.prod(1 - y_pred_total, axis=-1)
    y_val_period = np.max(y_val_total, axis=-1)

    # predict which month has a mortgage (if any)
    y_pred_month = np.argmax(y_pred_total, axis=-1)

    # print(y_val_period, y_pred_period, y_pred_month)

    period_start_month = month_nr_to_yearmonth[nr_train + nr_val]
    period_end_month = month_nr_to_yearmonth[nr_train + nr_val + nr_test - 1]

    df_predictions = pd.DataFrame({
        'client_nr': client_nrs,
        'most_likely_application_month': [
            month_nr_to_yearmonth[nr_train + nr_val + pr] for pr in y_pred_month
        ],
        'actual_application_months': [
            ", ".join([
                month_nr_to_yearmonth[nr_train + nr_val + j]
                for j in range(y_val_total.shape[1])
                if y_val_total[i, j] == 1.0
            ])
            for i in range(y_val_total.shape[0])
        ],
        'has_application': y_val_period,
        'has_application_prediction': y_pred_period
    })

    df_predictions.to_csv(f'{method_name}-mortgage-predictions.csv')
