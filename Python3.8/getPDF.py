# getPDF.py
#
# Copyright 2021 Ottawa Hospital Research Institute
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# Revision history:
#   Original Python version by Andre Carrington, 2021

def getPDF(X, kernel_type, bandwidth_method, quiet):
#
# Kernel density estimation using the Rosenblatt-Parzen estimator
#
# Ported from Matlab code written by André Carrington in 2017 at the University of Waterloo
#
# kernel_type:      'epanechnikov', 'gaussian', 'triangular', 'uniform'
# bandwidth_method: 'old','new'
# quiet:             0,1
#
# Based on formulas from:
#   - Racine - Non-parametric Econometrics: A Primer
#   - Stata - kdensity (help)
#   - other peer-reviewed papers
#
# The output of KDE and applications of a KDE:
#   - are sensitive to bin width
#   - are usually not sensitive to the choice of kernel
#   - may be sensitive to the domain for which the KDE is estimated
#
    import numpy        as np
    import scipy.stats  as stats
    import matplotlib.pyplot   as plt

    n           = len(X)
    mu          = float(np.mean(X))
    sigma       = float(np.std(X))
    minX        = mu   - 3*sigma
    maxX        = mu   + 3*sigma
    rangeX      = maxX - minX

    # if rangeX is zero then change it to [-1,+1]
    if rangeX == 0:
        rangeX = +1 - (-1)
        sigma = +1 / 3
    #endif

    if   bandwidth_method == 'old':
        # optimal bin_width for unimodal densities with any kernel,
        #   based on Integrated Mean Squared Error (at each point), MISE,
        #   for Gaussian kernel
        bin_width = 1.059*sigma*(n**(-1/5))
    elif bandwidth_method == 'new':
        # better (new) bin_width for bimodal or any density in general
        #   with any kernel, based on Interquartile rangeX instead of
        #   standard deviation, and based on a smaller multiplier as
        #   appropriate for the possibility of biomodal distributions
        m         = min(sigma, stats.iqr(X)/1.349) # formula from stata kdensity
        bin_width = (0.9*m)*(n**(-1/5))
    else:
        raise ValueError('Incorrect bandwidth_method specified')
    #endif

    # limit number of bins to 10,000
    if rangeX/bin_width > 10000:
        new_bin_width = rangeX/10000
        bin_width     = new_bin_width
        if not quiet:
            print(f'Modified bin_width from: {bin_width:9.04g} to {new_bin_width:9.04g}')
        #endif
    else:
        if not quiet:
            print(f'bin_width: {bin_width:9.04g}')
        #endif
    #endif

    Xc_cts      = np.arange(minX, maxX, bin_width)
    num_bins    = len(Xc_cts)
    Y           = np.zeros([num_bins, 1])

    # get KDE
    i = 0
    for x in Xc_cts:
        # Rosenblatt-Parzen kernel density estimate
        Y[i] = vectorized_KDE(kernel_type, x, X, n, bin_width)
        i    = i + 1
    #endif

    if not quiet:
        # plot
        fig1 = plt.figure()
        t    = f'Density estimate ({kernel_type} kernel, {bandwidth_method} bandwidth)'

        # plot KDE
        a = len(Xc_cts)
        b = len(Y)
        if a == b:
            plt.plot(Xc_cts, Y)
            plt.title(t)
        else:
            raise ValueError(f'Difference in plot vector sizes (Xc_cts, Y): {a}, {b}')
        #endif
    #endif
    return Xc_cts, Y
#enddef

def vectorized_KDE(kernel_type, x, X, n, h):
    k = (1/(n*h)) * sum( vectorized_RP_kernel(kernel_type, (X-x)/h) )
    return k
#enddef

def vectorized_RP_kernel(kernel_type, z):
    import numpy as np
    if not isinstance(z, np.ndarray):
        raise ValueError('Requires a numpy array as input')
    #endif
    if   kernel_type == 'epanechnikov':
        condn = np.abs(z) <= np.sqrt(5)
        k     = condn * ((3/4) * (1/np.sqrt(5)) * (1-(z**2)/5))
    elif kernel_type == 'uniform':
        condn = np.abs(z) <= 1
        k     = condn * 0.5
    elif kernel_type == 'triangular':
        condn = np.abs(z) <= 1
        k     = condn * (1-np.abs(z))
    elif kernel_type == 'gaussian':
        a     = np.sqrt(2*np.pi)
        k     = (1/a) * np.exp(-(z**2)/2)
        # note: in the context of KDE, the definition of a Gaussian
        #       kernel is not the standard definition of a Gaussian
        #       because KDE already includes expressions similar to the
        #       effect of mu and sigma, and thus mu and sigma are not
        #       included.  A full Gaussian kernel is different...
    #endif
    return k
#enddef
