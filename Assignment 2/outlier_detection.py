import numpy as np
from scipy import stats

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def remove_outliers(data, pp1, pp2):
    """
    Based on "Data Outlier Detection using the Chebyshev Theorem",
    Brett G. Amidan, Thomas A. Ferryman, and Scott K. Cooley
    Keyword arguments:
    data -- A numpy array of discrete or continuous data
    pp1 -- likelihood of expected outliers (e.g. 0.1, 0.05 , 0.01)
    pp2 -- final likelihood of real outliers (e.g. 0.01, 0.001 , 0.0001)
    """
    mu1 = np.mean(data)
    sigma1 = np.std(data)
    k = 1./ np.sqrt(pp1)
    odv1u = mu1 + k * sigma1
    odv1l = mu1 - k * sigma1
    new_data = data[np.where(data <= odv1u)[0]]
    new_data = new_data[np.where(new_data >= odv1l)[0]]
    mu2 = np.mean(new_data)
    sigma2 = np.std(new_data)
    k = 1./ np.sqrt(pp2)
    odv2u = mu2 + k * sigma2
    odv2l = mu2 - k * sigma2
    final_data = new_data[np.where(new_data <= odv2u)[0]]
    final_data = new_data[np.where(final_data >= odv2l)[0]]

    return final_data

def read_csv(path,sheet_number=0):
    data = pd.read_excel(path,sheet_name=sheet_number,stylesheet="abc")
    newdata = data.values

    x = newdata[2:,0]
    y = newdata[2:,1]
    theta = newdata[2:,4]

    return x,y,theta

def run_pca(data):
    pca = PCA()
    pca_data = pca.fit_transform(data.reshape(-1, 1))
    pca_data = pca.transform(data.reshape(-1, 1))
    return  pca_data.T[0]

def draw_gaussian(x,y,theta):

    # X_Straight
    x_mean = np.mean(x)
    x_var = np.var(x)

    plt.hist(x[np.newaxis].T, bins=15, density=True, color='y')
    x_std = np.sqrt(x_var)
    x_right = np.linspace(x_mean - 3 * x_std, x_mean + 3 * x_std, 20)
    plt.plot(x_right, stats.norm.pdf(x_right, x_mean, x_std))
    plt.title("X-Straight")
    plt.xlabel('X-Straight in mm')
    plt.ylabel('number of counts')
    plt.grid()
    plt.show()

    # y
    y_mean = np.mean(y)
    y_var = np.var(y)
    plt.hist(y[np.newaxis].T, bins=15, density=True, color='r')
    y_std = np.sqrt(y_var)
    y_left = np.linspace(y_mean - 3 * y_std, y_mean + 3 * y_std, 20)
    plt.plot(y_left, stats.norm.pdf(y_left, y_mean, y_std))
    plt.title("Y-Straight")
    plt.xlabel('Y-Straight in mm')
    plt.ylabel('number of counts')
    plt.grid()
    # plt.xlim([450,600])
    plt.show()

    # theta
    theta_mean = np.mean(theta)
    theta_var = np.var(theta)
    plt.hist(theta[np.newaxis].T, bins=15, density=True, color='g')
    theta_std = np.sqrt(theta_var)
    theta_left = np.linspace(theta_mean - 3 * theta_std, theta_mean + 3 * theta_std, 20)
    plt.plot(theta_left, stats.norm.pdf(theta_left, theta_mean, theta_std))
    plt.grid()
    plt.xlabel('angle in degrees')
    plt.ylabel('number of counts')
    plt.title("Angle")
    plt.show()

if __name__ == "__main__":
    x,y,theta = read_csv("Results.xlsx",sheet_number=2)
    x,y,theta = remove_outliers(x,0.01,0.01), remove_outliers(y,0.01,0.01),remove_outliers(theta,0.01,0.01)
    x, y, theta = run_pca(x),run_pca(y),run_pca(theta)
    draw_gaussian(x,y,theta)

