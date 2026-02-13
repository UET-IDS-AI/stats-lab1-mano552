import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Question 1 – Generate & Plot Histograms
# -----------------------------------

def normal_histogram(n):
    """
    Generate n samples from Normal(0,1),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.normal(0, 1, n)

    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Normal Distribution (0,1)")
    plt.show()

    return data


def uniform_histogram(n):
    """
    Generate n samples from Uniform(0,10),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.uniform(0, 10, n)

    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Uniform Distribution (0,10)")
    plt.show()

    return data


def bernoulli_histogram(n):
    """
    Generate n samples from Bernoulli(0.5),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.binomial(1, 0.5, n)

    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Bernoulli Distribution (p = 0.5)")
    plt.show()

    return data


# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    """
    Compute sample mean.
    """
    data = np.array(data)
    return np.sum(data) / len(data)


def sample_variance(data):
    """
    Compute sample variance using n-1 denominator.
    """
    data = np.array(data)
    n = len(data)

    mean = sample_mean(data)

    variance = np.sum((data - mean) ** 2) / (n - 1)

    return variance


# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):
    """
    Return:
    - min
    - max
    - median
    - 25th percentile (Q1)
    - 75th percentile (Q3)
    """
    data = np.sort(np.array(data))
    n = len(data)

    minimum = data[0]
    maximum = data[-1]

    # Median
    if n % 2 == 0:
        median = (data[n//2 - 1] + data[n//2]) / 2
    else:
        median = data[n//2]

    # Quartiles (test-compatible method)
    q1 = data[n // 4]
    q3 = data[(3 * n) // 4]

    return minimum, maximum, median, q1, q3

    


# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):
    """
    Compute sample covariance using n-1 denominator.
    """
    x = np.array(x)
    y = np.array(y)

    n = len(x)

    mean_x = sample_mean(x)
    mean_y = sample_mean(y)

    cov = np.sum((x - mean_x) * (y - mean_y)) / (n - 1)

    return cov


# -----------------------------------
# Question 5 – Covariance Matrixdw
# -----------------------------------

def covariance_matrix(x, y):
    """
    Return 2x2 covariance matrix:
        [[var(x), cov(x,y)],
         [cov(x,y), var(y)]]
    """
    var_x = sample_variance(x)
    var_y = sample_variance(y)
    cov_xy = sample_covariance(x, y)

    matrix = np.array([
        [var_x, cov_xy],
        [cov_xy, var_y]
    ])

    return matrix
