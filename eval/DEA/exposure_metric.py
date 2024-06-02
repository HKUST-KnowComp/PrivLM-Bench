import numpy as np


def calculate_single_exposure(canary_loss, reference_losses):
    n = reference_losses.shape[0]
    combination = np.append(reference_losses, canary_loss)
    exposure = np.log2(n) - np.log2(np.argsort(combination)[-1] + 1)
    return exposure


def calculate_exposures(canary_losses, reference_losses):
    n = reference_losses.shape[0]
    exposures = []
    for canary_loss in canary_losses:
        combination = np.append(reference_losses, canary_loss)
        exposure = np.log2(n) - np.log2(np.argsort(combination)[-1] + 1)
        exposures.append(exposure)
    return exposures


def expected_exposure():
    return 1 / np.log(2)


def get_lower_bound_epsilon_for_DP(canary_losses, reference_losses):
    mid = canary_losses.shape[0] // 2
    c_mid = np.sort(canary_losses)[mid]

    return np.log(2) * (calculate_single_exposure(c_mid) - 1)



