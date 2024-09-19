import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import datetime as datetime
import statsmodels.api as sm
from scipy.optimize import minimize

def random_weights(d,L,weight_type = "unif_B^d", seed = 0, nu = None):
    """
    Generate a matrix of random weights according to a specified distribution.

    Parameters:
    -----------
    d : int
        The dimensionality of each weight vector (number of rows in the output matrix).
    
    L : int
        The number of weight vectors to generate (number of columns in the output matrix).
    
    weight_type : str, optional, default="unif_B^d"
        The type of random weights to generate. Options include:
        - "gaussian": Standard normal distribution.
        - "unif_S^d-1": Uniformly distributed on the (d-1)-dimensional sphere S^(d-1).
        - "unif_B^d": Uniformly distributed inside the unit ball B^d_1.
        - "t-student": standard d-dimensional t-student distribution with nu degrees of freedom.
        - "unif_B^d+bias": Uniformly distributed inside the unit ball B^d_1, with uniform biases
                   over [-sqrt(d),sqrt(d)] (as R = L^{1/(2k-2\eps+1) ~ 1).
    
    seed : int, optional, default=0
        The seed for the random number generator, ensuring reproducibility.
    
    nu : int, optional, default = None
        degrees of freedom for the t-student distribution.


    Returns:
    --------
    W : numpy.ndarray
        A (d x L) matrix of random weights generated according to the specified distribution.
    xi : np.ndarray, optional
        A L-dimensional vector of random biases, if sample_biases is True.

    Raises:
    -------
    ValueError
        If an invalid `weight_type` is provided.

    Notes:
    ------
    - If `weight_type` is "unif_S^d-1", the vectors are uniformly distributed over the surface
      of the unit sphere S^(d-1), meaning they have a unit norm.
    - If `weight_type` is "unif_B^d", the vectors are uniformly distributed inside the unit ball
      B^d_1, which includes scaling by a random radial distance.
    """

    # set the seed
    np.random.seed(seed)

    # sample normal
    W = np.random.normal(size = (d,L))

    if weight_type == "gaussian":
        return W, 0
    elif weight_type == "unif_S^d-1":
        # Sample uniform over the Sphere S^{d-1}, i.e. we sample a random direction.
        W = (1/np.linalg.norm(W, axis = 0, ord = 2, keepdims = True)) * W
        return W, 0
    elif weight_type == "unif_B^d":
        # Sample uniform over the ball of radius one B^d_1.
        W = (1/np.linalg.norm(W, axis = 0, ord = 2, keepdims = True)) * W
        U = np.random.uniform(low = 0, high = 1, size = L)
        Rad = U ** (1/d)  
        W = Rad * W
        return W, 0
    elif (weight_type == "t-student") and (nu != None):
        chi_squared = np.random.chisquare(nu, size=L)
        W = W * np.sqrt(nu/chi_squared)
        return W, 0
    elif weight_type == "unif_B^d+bias":
        W = (1/np.linalg.norm(W, axis = 0, ord = 2, keepdims = True)) * W
        U = np.random.uniform(low = 0, high = 1, size = L)
        Rad = U ** (1/d)  
        W = Rad * W
        xi = np.random.uniform(low = -np.sqrt(d), high = np.sqrt(d), size = L)
        return [W , xi]
    else:
        raise ValueError("Invalid weight type.")
    

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# Function to compute the instruments and associated factors
def compute_X_R_F(results, char_cols, date,W, activation = "ReLU", normalize = False, winsorize = False, N = 1000, xi = 0):
    """
    Compute the instruments (X) and associated factors (F) based on the provided activation function
    and whether normalization is applied. The function takes in a specific date, a weight matrix (W),
    and the activation function to use.

    Parameters:
    -----------
    results: pd.DataFrame
        The dataframe with all the characteristics and stock returns.

    char_cols: list
        The list of columns of characteristics.

    date : str or pd.Timestamp
        The date for which the instruments and factors should be computed.
        
    W : np.ndarray
        The weight matrix used in the linear transformation before applying the activation function.

    activation : str, optional (default="ReLU")
        The activation function to apply. Options are:
        - "ReLU": Applies the Rectified Linear Unit function (ReLU).
        - "tanh": Applies the hyperbolic tangent function (tanh).
        - "sigmoid": Applies the sigmoid function.
        - "cos_sin": Applies the cosine and sine activations.
        An error is raised if an invalid activation function is provided.

    normalize : bool, optional (default=False)
        If True, the computed instruments (X) will be centered and normalized:
        - Centered: Mean-centered so that the average of each characteristic is zero.
        - Normalized: Scaled to have a unit one norm, meaning each column has a leverage of one dollar.

    winsorize : bool, optional (default=False)
        If True, the returns will be winsorized at 100% to remove the influence of outliers.
    
    N : int
        Normalizing constant for the factors, all factors get multiplied by these, can be set to average N_t.
    
    Returns:
    --------
    X : np.ndarray
        The computed instruments after applying the activation function and (optionally) normalization.

    R : np.ndarray
        The vector of adjusted returns (`R_e_adj`) for the specified date.

    F : np.ndarray
        The computed factors associated with the random features, obtained by multiplying 
        the transpose of X with R.

    Raises:
    -------
    ValueError:
        If an invalid activation function is specified.

    Notes:
    ------
    The adjusted excess returns (`R_e_adj`) is expected to be a column in the DataFrame `results`.

    Example:
    --------
    X, R, F = compute_X_R_F("2023-01-31", W, results, char_cols, activation="tanh", normalize=False)
    """
    Z_R = results[results["date"] == date]
    Z = Z_R[char_cols].to_numpy()
    R = Z_R["R_e_adj"].to_numpy()
    if activation == "ReLU":
        X = np.maximum(Z@W+xi,0) 
    elif activation == "tanh":
        X = np.tanh(Z@W+xi)
    elif activation == "sigmoid":
        X = sigmoid(Z@W+xi)
    elif activation == "cos_sin":
        W = W[:,:int(W.shape[1]/2)]
        X = np.concatenate((np.cos(Z @ W + xi), np.sin(Z @ W + xi)), axis=1)
    else:
        raise ValueError("Invalid activation.")
    N_t = R.shape[0]
    
    X = (N/N_t) * X
    if normalize:
        # center the derived characteristics
        X = X - np.mean(X, axis = 0, keepdims = True)
        # normalize so they have unit one norm (i.e. fixed leverage, one dollar exposure).
        X = (1/np.linalg.norm(X, axis = 0, ord = 1, keepdims = True)) * X 
        # X = (1/np.std(X, axis = 0, keepdims = True)) * X # alternative normalization
    
    if winsorize:
        R = np.minimum(R,1.0)
    # compute the factors associated with the random features
    F =  X.T @ R
    return X, R, F

# Main functions

def simple_split_SDF(F, W, dates_list, IS_size, L, results, char_cols, activation, normalize, gammas, xi = 0):
    """
    Computes optimal portfolio coefficients and out-of-sample returns using a split between training (IS) and test set (OOS).

    Parameters:
    - F: Factor matrix (number of factors x number of time points).
    - W: random weights.
    - dates_list: List of dates corresponding to the time points in F.
    - IS_size: Size of the in-sample (IS) period.
    - L: Number of factors/portfolio coefficients.
    - results: Dataframe of results.
    - char_cols: columns of characteristics in the results.
    - activation: Activation function.
    - normalize: Boolean indicating whether to normalize (False by default).
    - gammas: List of egularization parameters.

    Returns:
    - b_hat_split: Optimal portfolio coefficients for each gamma.
    - R_P_OOS_split: Out-of-sample portfolio returns for each gamma.
    - SR_OOS_split: Out-of-sample Sharpe ratios for each gamma.
    - R_P_split_max: Portfolio returns corresponding to the maximum Sharpe ratio.
    """

    # Split between training (IS) and test set (OOS)
    F_IS = F[:, :IS_size]
    F_OOS = F[:, IS_size:]
    T_OOS = F_OOS.shape[1]

    # Regularization parameters
    
    n_gamma = len(gammas)
    # Compute optimal portfolio coefficients
    b_hat_split = np.zeros((n_gamma, L))
    B = F_IS.T @ F_IS
    for i, gamma in enumerate(gammas):
        b_hat_split[i, :] = (1 / IS_size) * F_IS @ np.linalg.solve((1 / IS_size) * B + gamma * np.eye(IS_size), np.ones(IS_size))

    # Construct a portfolio with unit leverage (i.e., ||w_t||_1 = 1 for all times)
    R_P_OOS_split = np.zeros((n_gamma, T_OOS))
    for t, date in enumerate(dates_list[IS_size:]):
        X_t, R, _ = compute_X_R_F(results,char_cols, date, W, activation, normalize, xi = xi)
        v_t = X_t @ b_hat_split.T
        w_t = v_t / np.linalg.norm(v_t, axis=0, ord=1, keepdims=True)
        R_P_OOS_split[:, t] = w_t.T @ R

    # OOS Sharpe ratios

    # Choose the portfolio return with maximum Sharpe ratio
    SR_OOS_split = np.mean(R_P_OOS_split,axis = 1)/np.std(R_P_OOS_split,axis = 1) * np.sqrt(12)
    R_P_split_max = R_P_OOS_split[np.argmax(SR_OOS_split),:]

    return b_hat_split, R_P_OOS_split, SR_OOS_split, R_P_split_max

def rolling_window_SDF(F, W, dates_list, rolling_window_size, L, results,char_cols, activation, normalize, gammas, xi = 0):
    """
    Computes optimal portfolio coefficients and out-of-sample returns using a rolling window approach.

    Parameters:
    - F: Factor matrix (number of factors x number of time points).
    - W: Random weights.
    - dates_list: List of dates corresponding to the time points in F.
    - rolling_window_size: Size of the rolling window for the in-sample (IS) period.
    - L: Number of factors/portfolio coefficients.
    - results: Dataframe of results.
    - char_cols: columns of characteristics in the results.
    - activation: Activation function.
    - normalize: Boolean indicating whether to normalize (False by default).
    - gammas: List of regularization parameters.

    Returns:
    - R_P_OOS_rolling: Out-of-sample portfolio returns for each gamma.
    - SR_OOS_rolling: Out-of-sample Sharpe ratios for each gamma.
    - R_P_rolling_max: Portfolio returns corresponding to the maximum Sharpe ratio.
    """

    # Number of out-of-sample time points
    T = len(dates_list)
    n_gamma = len(gammas)

    # Initialize array to store out-of-sample portfolio returns
    R_P_rolling = np.zeros((n_gamma, T - rolling_window_size))

    # Iterate over the rolling windows
    for t, date in enumerate(dates_list[rolling_window_size:], start=rolling_window_size):
        F_IS = F[:, t - rolling_window_size:t]

        # Initialize array to store optimal portfolio coefficients
        b_hat_rolling = np.zeros((n_gamma, L))
        # Compute optimal portfolio coefficients for each gamma
        B = F_IS.T @ F_IS
        for i, gamma in enumerate(gammas):
            b_hat_rolling[i, :] = (1 / rolling_window_size) * F_IS @ np.linalg.solve((1 / rolling_window_size) * B + gamma * np.eye(rolling_window_size), np.ones(rolling_window_size))
    
        # unit leverage returns
        X_t, R, _ = compute_X_R_F(results,char_cols,date,W,activation,normalize, xi = xi)
        v_t = X_t @ b_hat_rolling.T
        w_t = v_t/np.linalg.norm(v_t, axis = 0, ord = 1)
        R_P_rolling[:,t-rolling_window_size] = w_t.T @ R

    # Compute OOS Sharpe ratios
    SR_rolling = np.mean(R_P_rolling, axis=1) / np.std(R_P_rolling, axis=1) * np.sqrt(12)
    R_P_rolling_max = R_P_rolling[np.argmax(SR_rolling),:]

    return R_P_rolling, SR_rolling, R_P_rolling_max


def different_number_factors_SDF(F, W, dates_list, rolling_window_size, d, multipliers, results,char_cols, activation, normalize, gammas, xi = 0):
    """
    Computes out-of-sample Sharpe ratios for different number of random features and gammas using a rolling window approach.

    Parameters:
    - F: Factor matrix (number of factors x number of time points).
    - W: Random weights.
    - dates_list: List of dates (datetimes) corresponding to the time points in F.
    - rolling_window_size: Size of the rolling window for the in-sample (IS) period.
    - d: number of basic characteristics.
    - multipliers: List of multipliers to determine the number of random features.
    - results: Dataframe of results.
    - char_cols: columns of characteristics in the results.
    - activation: Activation function.
    - normalize: Boolean indicating whether to normalize (False by default).
    - gammas: List of regularization parameters.
    - xi: optional, biases.

    Returns:
    - SR_OOS_all: Out-of-sample Sharpe ratios for each number of random features and gammas.
    - Norm_all: Out-of-sample norm of the approximate SDF.
    """
    T = len(dates_list)
    n_gamma = len(gammas)

    # Initialize array to store out-of-sample portfolio returns for each combination
    R_P_OOS_all = np.zeros((len(multipliers), n_gamma, T - rolling_window_size))
    Norm_OOS_all = np.zeros((len(multipliers), n_gamma, T - rolling_window_size))
    ones_gammas = np.ones((n_gamma))

    for t, date in enumerate(dates_list[rolling_window_size:], start=rolling_window_size):
        X, R, Fs = compute_X_R_F(results,char_cols,date,W,activation,normalize, xi = xi)
        for m_idx, multiplier in enumerate(multipliers):
            L_m = int(multiplier * d)
            F_IS = F[:L_m, t - rolling_window_size:t]
            b_hat = np.zeros((n_gamma, L_m))
            B = F_IS.T @ F_IS
            for i, gamma in enumerate(gammas):
                b_hat[i, :] = (1 / rolling_window_size) * F_IS @ np.linalg.solve((1 / rolling_window_size) * B + gamma * np.eye(rolling_window_size), np.ones(rolling_window_size))

            # Compute portfolio excess returns out of sample
            X_t = X[:,:L_m]
            F_t = Fs[:L_m]
            v_t = X_t @ b_hat.T
            w_t = v_t/np.linalg.norm(v_t, axis = 0, ord = 1)
            R_P_OOS_day = w_t.T @ R
            R_P_OOS_all[m_idx, :, t - rolling_window_size] = R_P_OOS_day
            prediction = b_hat @ F_t
            objective = (ones_gammas - prediction) ** 2
            Norm_OOS_all[m_idx, :, t - rolling_window_size] = objective
        if date.month == 12:
            print("Finished processing year:", date.year)

    # Compute OOS Sharpe ratios for each combination of factors and gammas
    SR_OOS_all = np.zeros((len(multipliers), n_gamma))
    Norm_all = np.zeros((len(multipliers), n_gamma))
    for m_idx in range(len(multipliers)):
        for g_idx in range(n_gamma):
            R_P_OOS_mg = R_P_OOS_all[m_idx, g_idx, :]
            SR_OOS_all[m_idx, g_idx] = np.mean(R_P_OOS_mg) / np.std(R_P_OOS_mg) * np.sqrt(12)
            Norm_all[m_idx, g_idx] = np.mean(Norm_OOS_all[m_idx, g_idx, :])
    return SR_OOS_all, Norm_all



def rolling_cv_SDF(F, W, dates_list, rolling_window_size, gammas, results, char_cols, activation, normalize, k_folds = 5, xi = 0):
    """
    Computes out-of-sample portfolio returns and Sharpe ratio using a rolling window approach with cross-validation 
    to select the optimal regularization parameter.

    Parameters:
    - F: Factor matrix (number of factors x number of time points).
    - W: Random weights.
    - dates_list: List of dates corresponding to the time points in F.
    - rolling_window_size: Size of the rolling window for the in-sample (IS) period.
    - gammas: List of regularization parameters.
    - results: Dataframe of results.
    - char_cols: columns of characteristics in the results.
    - activation: Activation function.
    - normalize: Boolean indicating whether to normalize (False by default).

    Returns:
    - R_P_OOS_cv: Out-of-sample portfolio returns.
    - SR_OOS_cv: Out-of-sample Sharpe ratio.
    - optimal_gammas: time series of gammas chosen by cross validation.
    """

    T = len(dates_list)
    n_gamma = len(gammas)
    # Initialize array to store out-of-sample portfolio returns
    R_P_OOS_cv = np.zeros(T - rolling_window_size)
    optimal_gammas = np.zeros(T - rolling_window_size)

    kf = KFold(n_splits=k_folds, shuffle=False)

    for t, date in enumerate(dates_list[rolling_window_size:], start=rolling_window_size):
        F_IS = F[:, t - rolling_window_size:t]

        # Every year we perform cross-validation to find the optimal gamma.
        if ((t - rolling_window_size) % 12 == 0):
            # Initialize variables to track the best gamma and its corresponding objective
            best_gamma = None

            # Store the average CV objective for each gamma
            sum_cv_objs = np.zeros(n_gamma)

            # Cross-validation loop
            for train_index, val_index in kf.split(F_IS.T):
                F_train, F_val = F_IS[:, train_index], F_IS[:, val_index]
                T_train = F_train.shape[1]

                # Compute B_train once for all gammas
                B_train = F_train.T @ F_train

                for idx, gamma in enumerate(gammas):
                    # Solve for b_hat using training data
                    b_hat_cv = (1 / T_train) * F_train @ np.linalg.solve((1 / T_train) * B_train + gamma * np.eye(T_train), np.ones(T_train))

                    # Compute the CV objective on validation data
                    ones_val = np.ones(F_val.shape[1])
                    prediction = b_hat_cv.T @ F_val
                    objective = np.mean((ones_val - prediction) ** 2)

                    # Accumulate the CV objective for averaging later
                    sum_cv_objs[idx] += objective

                # Average the objective across all folds
                avg_cv_objs = sum_cv_objs/k_folds

                # Find the gamma with the minimum average CV objective
                best_idx = np.argmin(avg_cv_objs)
                best_gamma = gammas[best_idx]

        # Store the best gamma for this window
        optimal_gammas[t - rolling_window_size] = best_gamma

        # Use the best gamma to compute b_hat and OOS returns
        B = F_IS.T @ F_IS
        b_hat_optimal = (1 / rolling_window_size) * F_IS @ np.linalg.solve((1 / rolling_window_size) * B + best_gamma * np.eye(rolling_window_size), np.ones(rolling_window_size))

        # Unit leverage returns
        X_t, R, _ = compute_X_R_F(results, char_cols, date, W, activation, normalize, xi = xi)
        v_t = X_t @ b_hat_optimal
        w_t = v_t / np.linalg.norm(v_t, axis=0, ord=1)
        R_P_OOS_cv[t - rolling_window_size] = w_t.T @ R

    # Compute OOS Sharpe ratios
    SR_OOS_cv = np.mean(R_P_OOS_cv) / np.std(R_P_OOS_cv) * np.sqrt(12)

    return R_P_OOS_cv, SR_OOS_cv, optimal_gammas
    
def convergence_SDF(F, W, dates_list, rolling_window_size, Ts, Ls, results,char_cols, activation, normalize, gammas, xi = 0):
    """
    Computes out-of-sample Sharpe ratios for different number of random features, rolling window sizes and gammas.

    Parameters:
    - F: Factor matrix (number of factors x number of time points).
    - W: Random weights.
    - dates_list: List of dates corresponding to the time points in F.
    - rolling_window_size: Size of the rolling window for the in-sample (IS) period.
    - Ts: list of sample sizes for training.
    - Ls: list of number of random features considered.
    - results: Dataframe of results.
    - char_cols: columns of characteristics in the results.
    - activation: Activation function.
    - normalize: Boolean indicating whether to normalize (False by default).
    - gammas: List of regularization parameters.
    - xi: optional, biases.

    Returns:
    - SR_OOS_all: Out-of-sample Sharpe ratios for each number of random features and gammas.
    - Norm_all: Out-of-sample norm of the approximate SDF.
    """
    T = len(dates_list)
    n_gamma = len(gammas)
    n_factors = len(Ls)

    # Initialize array to store out-of-sample portfolio returns for each combination
    R_P_OOS_all = np.zeros((n_factors, n_gamma, T - rolling_window_size))
    Norm_OOS_all = np.zeros((n_factors, n_gamma, T - rolling_window_size))
    ones_gammas = np.ones((n_gamma))
    for t, date in enumerate(dates_list[rolling_window_size:], start=rolling_window_size):
        X, R, Fs = compute_X_R_F(results, char_cols, date,W,activation,normalize, xi = xi)
        for m_idx, L_m in enumerate(Ls):
            rolling_size = Ts[m_idx]
            F_IS = F[:L_m, t - rolling_size:t]
            b_hat = np.zeros((n_gamma, L_m))
            B = F_IS.T @ F_IS
            for i, gamma in enumerate(gammas):
                b_hat[i, :] = (1 / rolling_size) * F_IS @ np.linalg.solve((1 / rolling_size) * B + gamma * np.eye(rolling_size), np.ones(rolling_size))

            # Compute portfolio excess returns out of sample
            X_t = X[:,:L_m]
            F_t = Fs[:L_m]
            v_t = X_t @ b_hat.T
            w_t = v_t/np.linalg.norm(v_t, axis = 0, ord = 1)
            R_P_OOS_day = w_t.T @ R
            R_P_OOS_all[m_idx, :, t - rolling_window_size] = R_P_OOS_day
            prediction = b_hat @ F_t
            objective = (ones_gammas - prediction) ** 2
            Norm_OOS_all[m_idx, :, t - rolling_window_size] = objective
        if date.month == 12:
            print("Finished processing year: ", date.year)

    # Compute OOS Sharpe ratios for each combination of factors and gammas
    SR_OOS_all = np.zeros((n_factors, n_gamma))
    Norm_all = np.zeros((n_factors, n_gamma))
    for m_idx in range(n_factors):
        for g_idx in range(n_gamma):
            R_P_OOS_mg = R_P_OOS_all[m_idx, g_idx, :]
            SR_OOS_all[m_idx, g_idx] = np.mean(R_P_OOS_mg) / np.std(R_P_OOS_mg) * np.sqrt(12)
            Norm_all[m_idx, g_idx] = np.mean(Norm_OOS_all[m_idx, g_idx, :])
    return SR_OOS_all, Norm_all

# Plotting functions

def plot_SR(gammas,SR,path):
    # Plot the out-of-sample Sharpe ratios
    plt.plot(gammas, SR)
    plt.xscale('log')
    plt.xlabel('$\gamma$')
    plt.ylabel('OOS $SR$')
    plt.title('OOS Sharpe ratio ($\gamma$)')
    plt.savefig(path, format='png', dpi=300)
    plt.show()

def plot_yoy_return_over_time(return_series,path):
    """
    Computes and plots the YoY return over time for a given series of monthly returns using log returns.
    
    Parameters:
    - return_series: pd.Series, monthly returns indexed by datetime.
    
    Returns:
    - yearly_compounded_returns: pd.Series, the YoY compounded returns.
    """
    # Ensure the index is in datetime format
    return_series.index = pd.to_datetime(return_series.index)

    # Calculate log returns
    log_returns = np.log(1 + return_series)

    # Calculate the rolling 12-month sum of log returns
    rolling_log_returns = log_returns.rolling(window=12).sum()

    # Convert back to regular returns
    yearly_compounded_returns = np.exp(rolling_log_returns) - 1

    # Convert to percentage
    yearly_compounded_returns = yearly_compounded_returns * 100
    yearly_compounded_returns =yearly_compounded_returns.dropna()

    # Plot the yearly compounded return over time
    plt.figure(figsize=(12, 6))
    plt.plot(yearly_compounded_returns)
    plt.xlabel('Date')
    plt.ylabel('YoY Return (%)')
    plt.title('YoY Return Over Time')

    # Set x-axis major ticks to show each year
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Set minor ticks to show every month
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()
    plt.xlim([yearly_compounded_returns.index[0], yearly_compounded_returns.index[-1]])

    plt.grid(True)
    plt.savefig(path, format='png', dpi=300)
    plt.show()

    return yearly_compounded_returns


def plot_cumulative_return(return_series,path):
    """
    Computes and plots the cumulative return over time for a given series of monthly returns.
    
    Parameters:
    - return_series: pd.Series, monthly returns indexed by datetime.
    
    Returns:
    - cumulative_returns: pd.Series, the cumulative returns.
    """
    # Ensure the index is in datetime format
    return_series.index = pd.to_datetime(return_series.index)
    
    # Calculate the cumulative return over time
    cumulative_returns = (1 + return_series).cumprod()

    # Convert to percentage if needed
    cumulative_returns = cumulative_returns * 100

    # Plot the cumulative return over time
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns.index, cumulative_returns)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.title('Cumulative Return Over Time')

    # Set x-axis major ticks to show each year
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Set minor ticks to show every month
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()
    plt.xlim([cumulative_returns.index[0], cumulative_returns.index[-1]])

    # Set y-axis to log scale and format the ticks as plain numbers
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.gca().yaxis.set_minor_formatter(mticker.ScalarFormatter())
    plt.gca().ticklabel_format(style='plain', axis='y')

    plt.grid(True)
    plt.savefig(path, format='png', dpi=300)
    plt.show()

    return cumulative_returns


def plot_different_number_factors(SR_OOS_all, gammas, multipliers,path):
    # Plot the out-of-sample Sharpe ratios as a color map
    plt.figure(figsize=(12, 8))

    # Plot with gamma on x-axis and L/d on y-axis, both in log scale
    plt.imshow(SR_OOS_all, aspect='auto', cmap='viridis',
            extent=[np.log10(gammas[0]), np.log10(gammas[-1]), np.log10(multipliers[-1]), np.log10(multipliers[0])])

    # Add colorbar
    plt.colorbar(label='OOS Sharpe Ratio')

    # Set x-axis to show log-scale of gamma
    plt.xlabel('$\gamma$')

    # Set y-axis label
    plt.ylabel('$L/d$')

    # Set plot title
    plt.title('OOS Sharpe ratio as a function of $\gamma$ and $L/d$')

    # Invert y-axis to have multipliers in the correct order
    plt.gca().invert_yaxis()

    # Set custom y-ticks to show the original multiplier values
    yticks = np.log10(multipliers)
    ytick_labels = [f'{m}' for m in multipliers]
    plt.yticks(ticks=yticks, labels=ytick_labels)

    # Reduce the number of x-ticks by selecting a subset of gamma values
    xticks = np.log10(gammas)[::4]  # Select every second gamma value for fewer ticks
    xtick_labels = [f'$10^{{{int(np.log10(g))}}}$' for g in gammas[::4]]  # Corresponding labels
    plt.xticks(ticks=xticks, labels=xtick_labels)
    plt.savefig(path, format='png', dpi=300)
    # Display the plot
    plt.show()

def plot_rolling_sharpe_ratio(return_series,n_years : int, path,mkt_series = None):
    """
    Computes and plots the 5-year rolling Sharpe ratio (annualized) for a given series of monthly returns.
    
    Parameters:
    - return_series: pd.Series, monthly returns indexed by datetime.
    - path: str, file path to save the plot.
    
    Returns:
    - rolling_sharpe_ratio: pd.Series, the 5-year rolling Sharpe ratio.
    """
    # Ensure the index is in datetime format
    return_series.index = pd.to_datetime(return_series.index)

    # Define the rolling window size for 5 years (5 years * 12 months)
    window_size = n_years * 12

    # Calculate the rolling mean and standard deviation
    rolling_mean = return_series.rolling(window=window_size).mean()
    rolling_std = return_series.rolling(window=window_size).std()

    # Calculate the rolling Sharpe ratio (annualized)
    rolling_sharpe_ratio = (rolling_mean / rolling_std) * np.sqrt(12)

    # Drop NaN values from the series
    rolling_sharpe_ratio = rolling_sharpe_ratio.dropna()

    # Plot the rolling Sharpe ratio over time
    plt.figure(figsize=(12, 4))
    plt.plot(rolling_sharpe_ratio, label = "MVE portfolio")
    plt.xlabel('Date')
    plt.ylabel(f"{n_years}-Year Rolling Sharpe Ratio (Annualized)")
    plt.title(f"{n_years}-Year Rolling Sharpe Ratio Over Time")

    if mkt_series is not None:
        return_index = return_series.index
        mkt_series = mkt_series.loc[return_index]
        rolling_mean_mkt = mkt_series.rolling(window=window_size).mean()
        rolling_std_mkt = mkt_series.rolling(window=window_size).std()
        rolling_sharpe_ratio_mkt = (rolling_mean_mkt / rolling_std_mkt) * np.sqrt(12)
        rolling_sharpe_ratio_mkt = rolling_sharpe_ratio_mkt.dropna()
        plt.plot(rolling_sharpe_ratio_mkt, label = "Market")
        plt.legend()


    # Set x-axis major ticks to show each year
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Set minor ticks to show every month
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()
    plt.xlim([rolling_sharpe_ratio.index[0], rolling_sharpe_ratio.index[-1]])

    plt.grid(True)
    plt.savefig(path, format='png', dpi=300)
    plt.show()

    return rolling_sharpe_ratio

def calculate_alpha_beta(portfolio_returns, market_returns,path):
    """
    Calculate the alpha, beta, and other related statistics for a portfolio.
    
    Parameters:
    - portfolio_returns (pd.DataFrame or pd.Series): Portfolio returns.
    - market_returns (pd.DataFrame or pd.Series): Market returns.

    Returns:
    - results (dict): A dictionary containing the calculated alpha, beta, 
                      and related statistics.
    """
    
    # Ensure the index is aligned between portfolio and market returns
    market_returns = market_returns.loc[portfolio_returns.index]
    
    # Add constant for the intercept
    X = sm.add_constant(market_returns)
    
    # Calculate leverage adjustment
    lev_adj = np.std(market_returns) / np.std(portfolio_returns)
    
    # Adjust portfolio returns by leverage adjustment
    y = pd.Series(portfolio_returns * lev_adj, index=portfolio_returns.index)
    
    # Perform the regression
    model = sm.OLS(y, X).fit()
    
    # Extract the monthly alpha (intercept)
    alpha_monthly = model.params['const']
    
    # Annualize the alpha, in percentage
    alpha_annualized = alpha_monthly * 12 * 100
    
    # Extract the standard error of the intercept (alpha)
    alpha_se = model.bse['const']
    z_score = alpha_monthly / alpha_se
    alpha_annualized_se = alpha_se * 12 * 100
    
    # Extract the beta (coefficient for the market excess return)
    beta = model.params['mktrf']
    beta_se = model.bse['mktrf']
    
    # Calculate the residuals (epsilon)
    residuals = model.resid
    
    # Standard deviation of residuals
    std_eps = np.std(residuals)
    
    # Compile the results into a dictionary
    results = {
        'Monthly Alpha': [alpha_monthly],
        'Monthly Alpha SE': [alpha_se],
        'Z-score': [z_score],
        'Beta': [beta],
        'Beta SE': [beta_se],
        'Annualized Alpha': [alpha_annualized],
        'Annualized Alpha SE': [alpha_annualized_se],
        'Std Dev of Residuals': [std_eps]
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv(path, index=False)
    
    print(f"Monthly Alpha: {alpha_monthly * 100:.2f}%")
    print(f"Monthly Alpha standard error: {alpha_se * 100:.2f}%")
    print(f"Annualized Alpha: {alpha_annualized:.2f}%")
    print(f"Annualized Alpha standard error: {alpha_annualized_se:.2f}%")
    print(f"Z-score: {z_score:.4f}")
    print(f"Beta: {beta:.2f}")
    print(f"Beta standard error: {beta_se:.4f}")
    print(f"Standard Deviation of Residuals: {std_eps * 100:.2f}%")

    return results

def calculate_fama_french_alpha_beta(portfolio_returns, market_returns, fama_french_factors, path):
    """
    Calculate the alpha, betas, and other related statistics using the Fama-French 5-factor model.
    
    Parameters:
    - portfolio_returns (pd.DataFrame or pd.Series): Portfolio returns.
    - market_returns (pd.DataFrame or pd.Series): Market returns.
    - fama_french_factors (pd.DataFrame): DataFrame containing the Fama-French factors 
                                          (e.g., "mktrf", "smb", "hml", "rmw", "cma").

    Returns:
    - results (dict): A dictionary containing the calculated alpha, betas, 
                      and related statistics.
    """
    
    # Ensure the index is aligned between portfolio returns, market returns, and Fama-French factors
    p_index = portfolio_returns.index
    market_returns = market_returns.loc[p_index]
    fama_french_factors = fama_french_factors.loc[p_index]

    # Select Fama-French 5 factors as independent variables
    X = fama_french_factors[["mktrf", "smb", "hml", "rmw", "cma"]]
    X = X.apply(pd.to_numeric, errors='coerce')

    # Add a constant for the intercept
    X = sm.add_constant(X)

    # Adjust leverage of the portfolio to match the standard deviation of the market excess return
    lev_adj = np.std(market_returns) / np.std(portfolio_returns)
    y = pd.Series(portfolio_returns * lev_adj, index=p_index)
    y = pd.to_numeric(y, errors='coerce')

    # Perform the regression
    model = sm.OLS(y, X).fit()

    # Extract the monthly alpha (intercept)
    alpha_monthly = model.params['const']

    # Annualize the alpha, in percentage
    alpha_annualized = alpha_monthly * 12 * 100

    # Extract the standard error of the intercept (alpha)
    alpha_se = model.bse['const']
    z_score = alpha_monthly / alpha_se
    alpha_annualized_se = alpha_se * 12 * 100

    # Extract the betas for the Fama-French factors
    betas = model.params[["mktrf", "smb", "hml", "rmw", "cma"]]

    # Calculate the residuals (epsilon)
    residuals = model.resid

    # Standard deviation of residuals
    std_eps = np.std(residuals)

    # Compile the results into a dictionary
    results = {
        'Monthly Alpha (%)': [alpha_monthly * 100],
        'Monthly Alpha SE (%)': [alpha_se * 100],
        'Annualized Alpha (%)': [alpha_annualized],
        'Annualized Alpha SE (%)': [alpha_annualized_se],
        'Z-score': [z_score],
        'Betas': [betas.to_dict()],
        'Std Dev of Residuals': [std_eps]
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv(path, index=False)

    print(f"Monthly Alpha: {alpha_monthly * 100:.2f}%")
    print(f"Monthly Alpha SE: {alpha_se * 100:.2f}%")
    print(f"Annualized Alpha: {alpha_annualized:.2f}%")
    print(f"Annualized Alpha standard error: {alpha_annualized_se:.2f}%")
    print(f"Z-score: {z_score:.4f}")
    print(f"Betas:")
    for factor, beta in betas.items():
        print(f"  {factor}: {beta:.4f}")
    print(f"Standard Deviation of Residuals: {std_eps * 100:.2f}%")
    
    return results
