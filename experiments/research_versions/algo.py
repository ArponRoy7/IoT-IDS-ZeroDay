Input:
    Dataset D
    Zero-day class Z
    AE epochs E
    Threshold percentile P
    RF parameters θ

Output:
    Final predictions Y_pred

Step 1: LOAO Split
    Remove all samples of class Z from training set
    Train_full = D \ Z
    Test_zero = samples of class Z

Step 2: Split Train_full
    Benign → Benign_train, Benign_val
    Known attacks → Attack_train, Attack_test

Step 3: Fit Scaler
    Fit StandardScaler using Benign_train

Step 4: Train Autoencoder
    Train AE on Benign_train for E epochs
    Validate using Benign_val
    Compute reconstruction error on validation set
    Set threshold T = percentile_P(validation_errors)

Step 5: Train Random Forest
    Compute residual feature for Attack_train
    Append residual to feature vector
    Train RF using parameters θ

Step 6: Hybrid Inference (Test Phase)
    For each sample x in test set:

        Compute reconstruction error e

        If e ≤ T:
            Predict BENIGN

        Else:
            Get RF probabilities
            If RF confidence < threshold:
                Predict ZERO_DAY
            Else:
                Predict predicted attack class

Return predictions


