import pandas as pd
import numpy as np
from PyALE import ale 
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import t
from typing import List,Dict,Tuple
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter  # noqa: E402
from skopt.space import Categorical,Real
from functools import partial
from skopt.plots import _cat_format

def cmds(D, k=2):
    """Classical multidimensional scaling

    Theory and code references:
    https://en.wikipedia.org/wiki/Multidimensional_scaling#Classical_multidimensional_scaling
    http://www.nervouscomputer.com/hfs/cmdscale-in-python/

    Arguments:
    D -- A squared matrix-like object (array, DataFrame, ....), usually a distance matrix
    """

    n = D.shape[0]
    if D.shape[0] != D.shape[1]:
        raise Exception("The matrix D should be squared")
    if k > (n - 1):
        raise Exception("k should be an integer <= D.shape[0] - 1")

    # (1) Set up the squared proximity matrix
    D_double = np.square(D)
    # (2) Apply double centering: using the centering matrix
    # centering matrix
    center_mat = np.eye(n) - np.ones((n, n)) / n
    # apply the centering
    B = -(1 / 2) * center_mat.dot(D_double).dot(center_mat)
    # (3) Determine the m largest eigenvalues
    # (where m is the number of dimensions desired for the output)
    # extract the eigenvalues
    eigenvals, eigenvecs = np.linalg.eigh(B)
    # sort descending
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    # (4) Now, X=eigenvecs.dot(eigen_sqrt_diag),
    # where eigen_sqrt_diag = diag(sqrt(eigenvals))
    eigen_sqrt_diag = np.diag(np.sqrt(eigenvals[0:k]))
    ret = eigenvecs[:, 0:k].dot(eigen_sqrt_diag)
    return ret

def order_groups(X, feature):
    """Assign an order to the values of a categorical feature.

    The function returns an order to the unique values in X[feature] according to
    their similarity based on the other features.
    The distance between two categories is the sum over the distances of each feature.

    Arguments:
    X -- A pandas DataFrame containing all the features to considering in the ordering
    (including the categorical feature to be ordered).
    feature -- String, the name of the column holding the categorical feature to be ordered.
    """

    features = X.columns
    # groups = X[feature].cat.categories.values
    groups = X[feature].unique()
    D_cumu = pd.DataFrame(0, index=groups, columns=groups)
    K = len(groups)
    for j in set(features) - set([feature]):
        D = pd.DataFrame(index=groups, columns=groups)
        # discrete/factor feature j
        # e.g. j = 'color'
        if (X[j].dtypes.name == "category") | (
            (len(X[j].unique()) <= 10) & ("float" not in X[j].dtypes.name)
        ):
            # counts and proportions of each value in j in each group in 'feature'
            cross_counts = pd.crosstab(X[feature], X[j])
            cross_props = cross_counts.div(np.sum(cross_counts, axis=1), axis=0)
            for i in range(K):
                group = groups[i]
                D_values = abs(cross_props - cross_props.loc[group]).sum(axis=1) / 2
                D.loc[group, :] = D_values
                D[group] = D_values
        else:
            # continuous feature j
            # e.g. j = 'length'
            # extract the 1/100 quantiles of the feature j
            seq = np.arange(0, 1, 1 / 100)
            q_X_j = X[j].quantile(seq).to_list()
            # get the ecdf (empiricial cumulative distribution function)
            # compute the function from the data points in each group
            X_ecdf = X.groupby(feature)[j].agg(ECDF)
            # apply each of the functions on the quantiles
            # i.e. for each quantile value get the probability that j will take
            # a value less than or equal to this value.
            q_ecdf = X_ecdf.apply(lambda x: x(q_X_j))
            for i in range(K):
                group = groups[i]
                D_values = q_ecdf.apply(lambda x: max(abs(x - q_ecdf[group])))
                D.loc[group, :] = D_values
                D[group] = D_values
        D_cumu = D_cumu + D
    # reduce the dimension of the cumulative distance matrix to 1
    D1D = cmds(D_cumu, 1).flatten()
    # order groups based on the values
    order_idx = D1D.argsort()
    groups_ordered = D_cumu.index[D1D.argsort()]
    return pd.Series(range(K), index=groups_ordered)


def compute_ALE(data,model,feat,space,samples,name,include_CI=False, C=0.95):
    d1 = pd.DataFrame(space.inverse_transform(samples),columns=[n for n in name])

    if d1[feat].dtype in ['int','float']:
        # data = data.drop(columns=feat)
        # data[feat] = d1[feat]  
        ale_eff = ale(X=data, model=model, feature=[feat],plot=False, grid_size=50, include_CI=True, C=0.95)
        return ale_eff
    else:
        ale_eff = ale(X=data, model=model, feature=[feat],plot=False, grid_size=50, include_CI=True, C=0.95)
        return ale_eff
        
        # if (data[feat].dtype.name != "category") or (not data[feat].cat.ordered):
        #     data[feat] = data[feat].astype(str)
        #     groups_order = order_groups(data, feat)
        #     groups = groups_order.index.values
        #     data[feat] = data[feat].astype(
        #         pd.api.types.CategoricalDtype(categories=groups, ordered=True)
        #     )

        # groups = data[feat].unique()
        # groups = groups.sort_values()
        # feature_codes = data[feat].cat.codes
        # groups_counts = data.groupby(feat).size()
        # groups_props = groups_counts / sum(groups_counts)

        # K = len(groups)

        # # create copies of the dataframe
        # X_plus = data.copy()
        # X_neg = data.copy()
        # # all groups except last one
        # last_group = groups[K - 1]
        # ind_plus = data[feat] != last_group
        # # all groups except first one
        # first_group = groups[0]
        # ind_neg = data[feat] != first_group
        # # replace once with one level up
        # X_plus.loc[ind_plus, feat] = groups[feature_codes[ind_plus] + 1]
        # # replace once with one level down
        # X_neg.loc[ind_neg, feat] = groups[feature_codes[ind_neg] - 1]
        # try:
        #     # predict with original and with the replaced values
        #     # encode the categorical feature
        #     y_hat = model.predict(data).ravel()

        #     # predict
        #     y_hat_plus = model.predict(data[ind_plus]).ravel()

        #     # predict
        #     y_hat_neg = model.predict(data[ind_neg]).ravel()
        # except Exception as ex:
        #     raise Exception(
        #         """There seems to be a problem when predicting with the model.
        #         Please check the following: 
        #             - Your model is fitted.
        #             - The list of predictors contains the names of all the features"""
        #         """ used for training the model.
        #             - The encoding function takes the raw feature and returns the"""
        #     """ right columns encoding it, including the case of a missing category.
        #     """
        #     )

        # # compute prediction difference
        # Delta_plus = y_hat_plus - y_hat[ind_plus]
        # Delta_neg = y_hat[ind_neg] - y_hat_neg

        # # compute the mean of the difference per group
        # delta_df = pd.concat(
        #     [
        #         pd.DataFrame(
        #             {"eff": Delta_plus, feat: groups[feature_codes[ind_plus] + 1]}
        #         ),
        #         pd.DataFrame({"eff": Delta_neg, feat: groups[feature_codes[ind_neg]]}),
        #     ]
        # )
        # res_df = delta_df.groupby([feat]).mean()
        # res_df["eff"] = res_df["eff"].cumsum()
        # res_df.loc[groups[0]] = 0
        # # sort the index (which is at this point an ordered categorical) as a safety measure
        # res_df = res_df.sort_index()
        # res_df["eff"] = res_df["eff"] - sum(res_df["eff"] * groups_props)
        # res_df["size"] = groups_counts
        # if include_CI:
        #     ci_est = delta_df.groupby([feat]).eff.agg(
        #         [("CI_estimate", lambda x: CI_estimate(x, C=C))]
        #     )
        #     lowerCI_name = "lowerCI_" + str(int(C * 100)) + "%"
        #     upperCI_name = "upperCI_" + str(int(C * 100)) + "%"
        #     res_df[lowerCI_name] = res_df[["eff"]].subtract(ci_est["CI_estimate"], axis=0)
        #     res_df[upperCI_name] = upperCI = res_df[["eff"]].add(
        #         ci_est["CI_estimate"], axis=0
        #     )
        # return res_df
    
def CI_estimate(x_vec, C=0.95):
    """Estimate the size of the confidence interval of a data sample.

    The confidence interval of the given data sample (x_vec) is
    [mean(x_vec) - returned value, mean(x_vec) + returned value].
    """
    alpha = 1 - C
    n = len(x_vec)
    stand_err = x_vec.std() / np.sqrt(n)
    critical_val = 1 - (alpha / 2)
    z_star = stand_err * t.ppf(critical_val, n - 1)
    return z_star


def plot_ALE(data,model,space,samples,name,plot_dims):

    n_dims = len(plot_dims)
    fig, ax = plt.subplots(n_dims, 1,
                           figsize=(2 * n_dims, 3 * n_dims))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.5, wspace=0.1)
    
    for i,feat in enumerate(name):
        index, dim = plot_dims[i]
        iscat = [isinstance(dim[1], Categorical) for dim in plot_dims]
        ax_ = ax[i]
        ale_eff = compute_ALE(data,model,feat,space,samples,name,include_CI=False, C=0.95)
    
        if not iscat[i]:
            #ax_.set_xscale('log')    
            sample = space.rvs(n_samples=51)#grid_size +1
            xi = space.transform(sample)
            xi[:,0] = ale_eff.index.values
            xi = space.inverse_transform(xi)
            ax_.plot(np.array(xi)[:,0].astype(float),ale_eff['eff'])
        else:
            ax_.errorbar(
            ale_eff.index.astype(str),
            ale_eff["eff"],
            yerr=0,
            capsize=3,
            marker="o",
            linestyle="dashed",
        )
            ax2 = ax_.twinx()
            ax2.set_ylabel("Size")
            ax2.bar(ale_eff.index.astype(str), ale_eff["size"], alpha=0.1, align="center")
            ax2.tick_params(axis="y")
            #ax2.set_title("1D ALE Plot - Discrete/Categorical")
            #fig.tight_layout()
        if not iscat[i]:
            low, high = dim.bounds
            ax_.set_xlim(low, high)
        if dim.prior == 'log-uniform':
            ax_.set_xscale('log')
        else:
            ax_.xaxis.set_major_locator(MaxNLocator(6, prune='both',
                                                            integer=iscat[i]))
        if iscat[i]:
            ax_.xaxis.set_major_formatter(FuncFormatter(
                            partial(_cat_format, dim)))
        ax_.set_ylabel("Effect on prediction (centered)")
        ax_.set_xlabel(feat)
    fig.suptitle('Accumulated Local Effects Plots for each Hyperparameter')
    plt.tight_layout()
    
    
    