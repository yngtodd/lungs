def print_progress(epoch, num_epochs, timemeter, lossmeter, aucmeter=None):
    """
    Prints current values and averages for AverageMeters

    Parameters:
    ----------
    epoch : int
        Current epoch.

    num_epochs : int
        Total number of epochs.

    timemeter : lungs.metrics.AverageMeter
        AverageMeter object recording runtime.

    lossmeter : lungs.metrics.AverageMeter
        AverageMeter object recording loss.

    aucmeter : lungs.metrics.AUCMeter
        Optional: AUCMeter object recording AUC, TPR, and FPR.
    """
    message = f"Epoch: [{epoch}/{num_epochs}] "\
              f"Time: {timemeter.val} [{timemeter.avg}] "\
              f"Loss: {lossmeter.val} [{lossmeter.avg}] "

    if aucmeter:
        auc = f"AUC: {aucmeter.area} TPR: {aucmeter.tpr} FPR: {aucmeter.fpr} "
        message += auc

    print(message)
