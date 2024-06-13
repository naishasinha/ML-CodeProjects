if __name__ == "__main__":
    from k_means import lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    Run Lloyd's Algorithm for k=10, and report 10 centers returned.

    NOTE: This code might take a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. Make sure to change it back before submission!
    """
    (x_train, _), _ = load_dataset("mnist")

    x_train = x_train[:10000]

    num_centers = 10
    centers, errors = lloyd_algorithm(x_train, num_centers)

    print("Centers returned by Lloyd's Algorithm:")
    print(centers)

    plt.plot(errors)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Convergence of Lloyd\'s Algorithm')
    plt.show()


if __name__ == "__main__":
    main()
