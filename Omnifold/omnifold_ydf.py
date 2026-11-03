# Yggdrasil Decision Forest (YDF) implementation of Omnifold.

import numpy as np
import ydf

def reweight_ydf(events_dict, model):
    probs = model.predict(events_dict)["probabilities"][:, 1]  # P(label=1)
    weights = probs / (1. - probs)
    return np.nan_to_num(weights)

def to_ydf_ds(X, feature_names, y, w):
    ds = {name: X[:, i] for i, name in enumerate(feature_names)}
    ds["label"] = y.astype(str)   # must be string for CATEGORICAL
    ds["weight"] = w
    return ds


def omnifold_ydf(theta0,
             theta_unknown_S, 
             iterations,  
             feature_names = None,
             verbose=0):

    weights = np.empty(shape=(iterations, 2, len(theta0)))
    # shape = (iteration, step, event)

    theta0_G = theta0[:,0]
    theta0_S = theta0[:,1]

    labels0 = np.zeros(len(theta0))
    labels_unknown = np.ones(len(theta_unknown_S))
    labels_unknown_step2 = np.ones(len(theta0_G))

    xvals_1 = np.concatenate((theta0_S, theta_unknown_S))
    yvals_1 = np.concatenate((labels0, labels_unknown))

    xvals_2 = np.concatenate((theta0_G, theta0_G))
    yvals_2 = np.concatenate((labels0, labels_unknown_step2))

    if (feature_names is None):
        feature_names = ["f{}".format(i) for i in range(theta0.shape[2])]

    # initial iterative weights are ones
    weights_pull = np.ones(len(theta0_S))
    weights_push = np.ones(len(theta0_S))

    for i in range(iterations):

        if (verbose>0):
            print("\nITERATION: {}\n".format(i + 1))
            pass

        # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
        # weights reweighted Sim. --> Data

        if (verbose>0):
            print("STEP 1\n")
            pass

        weights_1 = np.concatenate((weights_push, np.ones(len(theta_unknown_S))))

        ds_1 = to_ydf_ds(X=xvals_1, feature_names=feature_names, y=yvals_1, w=weights_1)
        learner = ydf.GradientBoostedTreesLearner(label="label", weights="weight")
        model = learner.train(ds_1)

        weights_pull = weights_push * reweight_ydf(events_dict={name: theta0_S[:, i] for i, name in enumerate(feature_names)}, 
                                                   model=model)
        weights[i, :1, :] = weights_pull

        # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.

        if (verbose>0):
            print("\nSTEP 2\n")
            pass

        weights_2 = np.concatenate((np.ones(len(theta0_G)), weights_pull))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.


        ds_2 = to_ydf_ds(X=xvals_2, feature_names=feature_names, y=yvals_2, w=weights_2)
        learner = ydf.GradientBoostedTreesLearner(label="label", weights="weight")
        model = learner.train(ds_2)

        weights_push = reweight_ydf(events_dict={name: theta0_G[:, i] for i, name in enumerate(feature_names)}, 
                                    model=model)
        weights[i, 1:2, :] = weights_push
        pass

    return weights
