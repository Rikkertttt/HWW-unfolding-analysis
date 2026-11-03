# Yggdrasil Decision Forest (YDF) implementation of Omnifold.

import numpy as np
import ydf

def reweight_ydf(events,model,batch_size=10000):
    f = model.predict(events, batch_size=batch_size)
    weights = f / (1. - f)
    return np.squeeze(np.nan_to_num(weights))


def omnifold_ydf(theta0,
             theta_unknown_S, 
             iterations,  
             verbose=0, 
             callbacks_fn=None):

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

        # zip ("hide") the weights with the labels
        Y_train_1 = np.stack((Y_train_1, w_train_1), axis=1)
        Y_test_1 = np.stack((Y_test_1, w_test_1), axis=1)   

        model.compile(loss=weighted_binary_crossentropy,
                      optimizer='Adam',
                      metrics=[weighted_accuracy],
                      jit_compile=False)

        callbacks = callbacks_fn(i) if callbacks_fn is not None else [[], []]

        model.fit(X_train_1, Y_train_1,
                  epochs=20,
                  batch_size=10000,
                  validation_data=(X_test_1, Y_test_1),
                  callbacks=callbacks[0],
                  verbose=verbose)

        weights_pull = weights_push * reweight(theta0_S,model)
        weights[i, :1, :] = weights_pull

        # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.

        if (verbose>0):
            print("\nSTEP 2\n")
            pass

        weights_2 = np.concatenate((np.ones(len(theta0_G)), weights_pull))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.

        X_train_2, X_test_2, Y_train_2, Y_test_2, w_train_2, w_test_2 = train_test_split(xvals_2, yvals_2, weights_2)

        # zip ("hide") the weights with the labels
        Y_train_2 = np.stack((Y_train_2, w_train_2), axis=1)
        Y_test_2 = np.stack((Y_test_2, w_test_2), axis=1)   

        model.compile(loss=weighted_binary_crossentropy,
                      optimizer='Adam',
                      metrics=[weighted_accuracy],
                      jit_compile=False)
        
        model.fit(X_train_2, Y_train_2,
                  epochs=20,
                  batch_size=2000,
                  validation_data=(X_test_2, Y_test_2),
                  callbacks=callbacks[1],
                  verbose=verbose)

        weights_push = reweight(theta0_G,model)
        weights[i, 1:2, :] = weights_push
        pass

    return weights
