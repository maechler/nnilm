import numpy as np


# pred[:,0] = start, pred[:,1] = end, pred[:,2] = power
def rectangular_metrics(pred, truth, activation_threshold=0.1, print_results=False):
    results = {}

    # clip values to be between 0 and 1
    np.putmask(pred[:, 0], pred[:, 0] <= 0, 0)
    np.putmask(pred[:, 1], pred[:, 1] >= 1, 1)

    # remove wrong solutions where start is before end
    np.putmask(pred[:, 0], pred[:, 1] < pred[:, 0], 0)
    np.putmask(pred[:, 1], pred[:, 1] < pred[:, 0], 0)

    # set everything below threshold to 0
    np.putmask(pred[:, 1], pred[:, 2] < activation_threshold, 0)
    np.putmask(pred[:, 0], pred[:, 2] < activation_threshold, 0)
    np.putmask(pred[:, 2], pred[:, 2] < activation_threshold, 0)

    # determine predicted and true negatives and positives
    predicted_negative = (pred[:, 2] == 0)
    truth_negative = (truth[:, 2] == 0)
    predicted_positive = ~predicted_negative
    truth_positive = ~truth_negative

    # determine true positives, false positives, true negatives and false negatives
    results['tp'] = truth_positive[predicted_positive].sum()
    results['fp'] = truth_negative[predicted_positive].sum()
    results['tn'] = truth_negative[predicted_negative].sum()
    results['fn'] = truth_positive[predicted_negative].sum()

    # Calculate recall, precision, f1 and accuracy
    results['recall'] = results['tp'] / float(results['tp'] + results['fn'])
    results['precision'] = results['tp'] / float(results['tp'] + results['fp'])
    results['f1'] = 2 * results['precision'] * results['recall'] / (results['precision'] + results['recall'])
    results['accuracy'] = (results['tp'] + results['tn']) / float(len(pred))

    truth_power_sum = np.sum(truth[:, 2])
    predicted_power_sum = np.sum(pred[:, 2])

    results['relative_error_in_total_energy'] = (predicted_power_sum - truth_power_sum) / float(max(truth_power_sum, predicted_power_sum))
    results['sum_abs_diff'] = np.sum(np.fabs(pred[:, 2] - truth[:, 2]))

    if print_results:
        print('TP: ' + str(results['tp']))
        print('FP: ' + str(results['fp']))
        print('TN: ' + str(results['tn']))
        print('FN: ' + str(results['fn']))

        print('F1: ' + str(results['f1']))
        print('Precision: ' + str(results['precision']))
        print('Recall: ' + str(results['recall']))
        print('Accuracy: ' + str(results['accuracy']))

        print('Relative error in total energy: ' + str(results['relative_error_in_total_energy']))
        print('Summed absolute difference: ' + str(results['sum_abs_diff']))

    return results
