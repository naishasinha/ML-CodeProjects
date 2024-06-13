if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    df_train, df_test = load_dataset("crime")

    print(df_train.head())

    y_train = df_train.iloc[:, 0].values
    X_train = df_train.iloc[:, 1:].values
    y_test = df_test.iloc[:, 0].values
    X_test = df_test.iloc[:, 1:].values

    n, d = X_train.shape
    initial_weight = np.zeros(d)
    
    avg_y = np.mean(y_train)
    _lambda = 2 * np.abs(np.matmul(X_train.T, (y_train - avg_y))).max()
    eta = 2e-5
    delta = 1e-4

    nonzeroes = np.array([])
    lambdas = np.array([])
    weight_record = np.array([])
    train_mse = np.array([])
    test_mse = np.array([])
    count = 0

    while _lambda >= 0.01:
        [initial_weight, start_bias] = train(X_train, y_train, _lambda, eta=eta, convergence_delta=delta, start_weight=initial_weight, start_bias=0)
        nonzeroes = np.append(nonzeroes, np.count_nonzero(initial_weight))
        lambdas = np.append(lambdas, _lambda)
        weight_record = np.append(weight_record, initial_weight)

        train_predictions = np.dot(X_train, initial_weight) + start_bias
        test_predictions = np.dot(X_test, initial_weight) + start_bias
        train_mse = np.append(train_mse, np.mean((train_predictions - y_train)**2))
        test_mse = np.append(test_mse, np.mean((test_predictions - y_test)**2))
        
        print('>>', count, ' lambda ', _lambda, ' nonzeroes ', nonzeroes)

        initial_weight, start_bias = train(X_train, y_train, _lambda, eta=eta, convergence_delta=delta, start_weight=initial_weight, start_bias=start_bias)
        _lambda *= 0.5
        count = nonzeroes[-1]
    
    labels = np.array(['agePct12t29','pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize'])
    loc = [df_train.columns.get_loc(label) for label in labels]

    weight_record = np.reshape(weight_record, (len(nonzeroes), len(initial_weight)))
    extracted_weights = weight_record[:, loc]

    plt.figure(1)
    plt.plot(lambdas, nonzeroes)
    plt.title(f'Plot 1 lambda v. non-zero weights - eta={eta}, delta={delta}')
    plt.xlabel('Lambda')
    plt.ylabel('Number of NonZeroes Features')
    plt.xscale('log')
    plt.grid(True)

    plt.figure(2)
    for i in range(extracted_weights.shape[1]):
        plt.plot(lambdas, extracted_weights[:, i], label=f'{labels[i]}')

    plt.title('Line Plots - Extracted Columns')
    plt.xlabel('Lambda')
    plt.xscale('log')
    plt.ylabel('Coefficient')
    plt.grid(True)
    plt.legend()

    plt.figure(3)
    plt.title('Training error v. Test error')
    plt.plot(lambdas, train_mse, label = 'Training MSE')
    plt.plot(lambdas, test_mse, label = 'Test MSE')
    plt.xscale('log')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

