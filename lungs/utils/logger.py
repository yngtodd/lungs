def print_progress(context, epoch, num_epochs, batch, num_samples, timemeter, lossmeter, apmeter, aucmeter=None):
    """
    Prints current values and averages for AverageMeters

    Parameters:
    ----------
    context : str
        What progress we are tracking.
        Options - {'Train', 'Validation', 'Test'}

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
    message = f"{context} Epoch: [{epoch}/{num_epochs}] "\
              f"Batch: [{batch}/{num_samples}] "\
              f"Time: {timemeter.val:.2f} [{timemeter.avg:.2f}] "\
              f"Loss: {lossmeter.val:.4f} [{lossmeter.avg:.4f}] "\
              f"Avg Prec: {apmeter.val:.4f} "

    if aucmeter:
        auc = f"AUC: {aucmeter.area:.2f} TPR: {aucmeter.tpr:.2f} FPR: {aucmeter.fpr:.2f} "
        message += auc

    print(message)
