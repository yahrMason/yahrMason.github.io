---
excerpt: "An exploration of GAMs 'from scratch'"
layout: "posts"
title: "Generalized Additive Models in Julia"
date: 2023-06-21
categories: "Bayes"
tags:
  - "julia"
  - "bayes"
---


## Introduction
The motivation to write this post comes from the [apparent need](https://discourse.julialang.org/t/generalized-additive-models-in-julia/10041/4) for a Julia package to fit [Generalized Additive Models](https://en.wikipedia.org/wiki/Generalized_additive_model), or GAMs.
I have found GAMs to be a very useful tool, and both R and Python have good implementations of these types of models in the [{mgcv}](https://cran.r-project.org/web/packages/mgcv/index.html) and [pyGAM](https://pygam.readthedocs.io/en/latest/) packages, respectively.
As a data scientist looking to learn more about Julia, I thought it would be a worthwhile exercise to unpack GAMs and try to build one in Julia. I am not yet at the competency where I could whip up a mgcv or pyGAM port to Julia, but I wanted to explore how these models are built, and how they could potentially be implemented in Julia.

### What are GAMs?
GAMs are tools that allow us to model an outcome variable as the sum of different functions over the input variables.
Typically, these functions are smooth function approximations called called splines, but this is not necessary for GAMs.
A common approach to building splines is to use [basis functions](https://en.wikipedia.org/wiki/Basis_function#:~:text=In%20mathematics%2C%20a%20basis%20function,linear%20combination%20of%20basis%20vectors.), which allow us to build arbitrarily "wiggly" curves to fit the data.
Smoothing splines help avoid over-fitting by regularizing the "wiggliness" of the curves (typically through penalities on the second derivative at the knots).

### My Learning Journey
My quest to figure out how to code up these models started at this [reply](https://discourse.julialang.org/t/generalized-additive-models-in-julia/10041/5?u=yahrmason) to the Julia Discourse thread linked above.
Fitting a mixed effects model using MCMC (aka a Bayesian Hierarchical Model)? That sure caught my attention. So I took a look at [the paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.2193) referenced in the thread. I quickly realized that if I wanted to build a GAM from scratch, I needed to find a resource that had code.
My search then lead me to [Tristan Mahr's FANTASTIC blog post](https://www.tjmahr.com/random-effects-penalized-splines-same-thing/) on penalized splines. Serious props to Tristan. His post does an excellent job of building the intuition of how mixed effects models related to penalized smoothing splines.
His post references the work of Simon Wood on Generalized Additive Models multiple times, so I figured it would behoove me to go right to the source and pick up a copy of [his textbook](https://www.taylorfrancis.com/books/mono/10.1201/9781315370279/generalized-additive-models-simon-wood) on the subject.
This text is a must read for anyone looking to implement GAMs from scratch. Most of my Julia code is adapted from his R code that builds these from scratch.

### What This Post is
The goal of this post is to demonstrate how to use Julia to fit penalized, smooth splines and additive models to data. It seemed to me that there were sparse resources freely available on the internet that demonstrated how to do this, so I am going to attempt to fill that void.
I am not going to go into great detail about what B-Splines are and how to fit them. There are so many great resources out there for that, and I don't want to re-invent the wheel. I have provided links to many of those resources in the below.

### Aknowledgements
This tutorial is heavily inspired by examples from Richard McElreath's [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/) text, specifically his spline examples in Chapter 4.
Max Lapan has developed and maintains [Julia translations](https://shmuma.github.io/rethinking-2ed-julia/) of the Statistical Rethinking codebase. His notes on chapter 4 were very helpful in the development of this tutorial.
[doggo dot jl](https://www.youtube.com/@doggodotjl) has developed a video [tutorial](https://www.youtube.com/watch?v=bXg2hGUF3X8) demonstrating how to build Bayesian Splines in Julia on their YouTube channel. It provides a really great introduction to the topic with great examples.
Simon Wood's text, [Generalized Additive Models: An Introduction with R](https://www.amazon.com/Generalized-Additive-Models-Introduction-Statistical/dp/1498728332), is the primary the primary resource I used for learning these topics. The code in this post has been adapted from the R code in the text.
Last but certainly not least, I was inspired to learn the concepts in this tutorial by reading Tristan Mahr's [blog post on smoothing splines](https://www.tjmahr.com/random-effects-penalized-splines-same-thing/).

## Imports

~~~~{.julia}
using Turing # for Bayes
using Random # for reproducibility
using FillArrays # for building models
using MCMCChains # for MCMC sampling
using Distributions # for building model
using DataFrames # for data
using RDatasets # for data
using StatisticalRethinking # for data
using StatsPlots # for plotting
using GLM # for fitting models
using Optim # for optimizing parameters
using BSplines # for splines
using LinearAlgebra # for matrix math

# set seed for reproducibility
Random.seed!(0);
~~~~~~~~~~~~~




## Data

The data set we will be using is the `cherry_blossoms` dataset, which can be found in Chapter 6 of [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/).
We will use the dataset to model the day of year of first blossom for cherry trees in Japan using a univariate smoothing spline (GAM with 1 predictor variable).
First, we will load the dataset from the `StatisticalRethinking` package. We will do some basic data cleaning and look at the first few rows.

~~~~{.julia}
# Import the cheery_blossoms dataset.
data = CSV.read(sr_datadir("cherry_blossoms.csv"), DataFrame);

# drop records that have missing day of year values
data = data[(data.doy .!= "NA"),:];

# convert day of year to numeric column
data[!,:doy] = parse.(Float64,data[!,:doy]);

# Show the first five rows of the dataset.
first(data, 5)
~~~~~~~~~~~~~

~~~~
5×5 DataFrame
 Row │ year   doy      temp     temp_upper  temp_lower
     │ Int64  Float64  String7  String7     String7
─────┼─────────────────────────────────────────────────
   1 │   812     92.0  NA       NA          NA
   2 │   815    105.0  NA       NA          NA
   3 │   831     96.0  NA       NA          NA
   4 │   851    108.0  7.38     12.1        2.66
   5 │   853    104.0  NA       NA          NA
~~~~





Our goal is to model the relationship between `doy` (Day of Year) and `year` variables.
Let's plot the relationship between these columns of the data.

~~~~{.julia}
# Create x and y variables
x = data.year;
y = data.doy;

# Make a function to make a scatter plot of the raw data
function PlotCherryData(; kwargs...)
    scatter(x,y,label = "Data"; kwargs...)
    xlabel!("Year")
    ylabel!("Day of First Blossom")
end;

# Plot the data
PlotCherryData()
~~~~~~~~~~~~~

![png](/assets/2023-06-21-gams-julia_files/2023-06-21-gams-julia_3_1.png)\ 




From the plot we can see what appears to be a non-linear relationship between the year and day of first blossom.
We will explore this relationship using B-Splines

## Building the Basis 

There are two parameters we need to define our basis with the `BSplines` package. These parameters are the number of knots (or breakpoints) and the order of the basis functions.
It should be noted that the order of a basis function is equal to the degree plus 1. For example, if we want to use cubic basis functions (degree 3), the order is equal to `3+1=4`.

For these data, we will use a cubic spline (order = 4). But how many knots should we choose? In this case, we will use 25 knots. The choice for using a relatively large number of knots will produce a rather "wiggly" fit.  In Chapter 4 section 2 of his text, Simon Wood recommends choosing a number of knots that is larger than needed and induce smoothing later.

The following code builds the knots at the quantiles of the independent variable and builds the basis.
We can simply call the `plot` function on the `basis` object to view the basis functions.

~~~~{.julia}
# Parameters to define basis
N_KNOTS = 15;
ORDER = 4; # third degree -> cubic basis

function QuantileBasis(x::AbstractVector, n_knots::Int64, order::Int64)

    # Build a list of the Knots (breakpoints)
    KnotsList = quantile(x, range(0, 1; length=n_knots));

    # Define the Basis Object
    Basis = BSplineBasis(order, KnotsList);

    return Basis
end

Basis = QuantileBasis(x, N_KNOTS, ORDER);

# Plot the Basis Functions
plot(Basis, title = "Basis Functions", legend = false)
xlabel!("Year")
~~~~~~~~~~~~~

![png](/assets/2023-06-21-gams-julia_files/2023-06-21-gams-julia_4_1.png)\ 




In order to fit our spline regression model, we need to build a design matrix from our basis object.
The following code builds our design matrix and plots the matrix using the `heatmap` function.

~~~~{.julia}
# Build a Matrix representation of the Basis
function BasisMatrix(Basis::BSplineBasis{Vector{Float64}}, x::AbstractVector)
    splines = vec(
        mapslices(
            x -> Spline(Basis,x), 
            diagm(ones(length(Basis))),
            dims=1
        )
    );
    X = hcat([s.(x) for s in splines]...)
    return X 
end

X = BasisMatrix(Basis, x);

# Plot the basis matrix using the heatmap function
heatmap(X, title = "Basis Design Matrix", yflip = true)
ylabel!("Observation Number")
xlabel!("Basis Function")
~~~~~~~~~~~~~

![png](/assets/2023-06-21-gams-julia_files/2023-06-21-gams-julia_5_1.png)\ 




Now that we have a design matrix built from our basis functions, we can fit our model.

## Fitting the Spline

The first model we will fit is a simple regression model using OLS. This model won't have a smoothing penalty applied, and will serve as a comparative baseline for the penalized smoothing splines that we fit later.
We can generate our spline function by multiplying the basis design matrix by the regression coefficients. We will construct our Bayesian model using Turing.

~~~~{.julia}
# Fit model with basic OLS
β_spline = coef(lm(X,y));
~~~~~~~~~~~~~




The plots below displays the resulting weighed basis functions, and the resulting spline fit.

~~~~{.julia}
# Plot Weighted Basis 
plot(x, β_spline .* Basis, title = "Weighted Basis Functions", legend = false)
ylabel!("Basis * Weight")
xlabel!("Year")
~~~~~~~~~~~~~

![png](/assets/2023-06-21-gams-julia_files/2023-06-21-gams-julia_7_1.png)\ 


~~~~{.julia}
# Plot Data and Spline
PlotCherryData(alpha = 0.20)
plot!(x, sum(β_spline .* Basis), color = :black, linewidth = 3, label = "Basic Spline")
~~~~~~~~~~~~~

![png](/assets/2023-06-21-gams-julia_files/2023-06-21-gams-julia_8_1.png)\ 




From the plot above, it can be seen the the resulting spline fit to the data is rather "wiggly" and not smooth.

## Making it Smooth

Smoothness is induced by enforcing a penalty on the second derivative at the knots.
The first thing we need to do is build an identity matrix with diagonal the length of the number of basis functions and twice taking the difference of that matrix.
To do this, we will first define a general function to take differences of matrices.

~~~~{.julia}
# General Function to take differences of Matrices
function diffm(A::AbstractVecOrMat, dims::Int64, differences::Int64)
    out = A
    for i in 1:differences
        out = Base.diff(out, dims=dims)
    end
    return out
end

# Difference Matrix
# Note we take the difference twice.
function DifferenceMatrix(Basis::BSplineBasis{Vector{Float64}})
    D = diffm(
        diagm(0 => ones(length(Basis))),
        1, # matrix dimension
        2  # number of differences
    )
    return D
end

D = DifferenceMatrix(Basis);
~~~~~~~~~~~~~




The `D` matrix represents the second derivative at the knots, or breakpoints, of our spline.
In order to induce smoothness in our function, we need to apply a penalty to these differences.
We can do this by augmenting our objective function to minimize from standard OLS to:

$$
\lvert y - X\beta \rvert^2 + \lambda \int_a^b[f^{\prime\prime}(x)]^2 dx
$$

It took me a minute to really grasp what the following code is doing when I first read it. But essentially what we can do is mulitply the lambda parameter by our penalty matrix, `D` and row-append it to our basis design matrix, `X`. We then need to append zeros to our outcome vector to make sure the rows are equal between the two.
What this allows us to do is use standard OLS to estimate the model coefficients with the lambda penalty applied. Typically when I see penalized regression models the parameters need to be estimated via some optimization algorithm. I just thought it was really cool to the clever use of OLS here.

~~~~{.julia}
# Define Penalty
λ = 1.0

# Augment Model Matrix and Outcome Data
function PenaltyMatrix(Basis::BSplineBasis{Vector{Float64}}, λ::Float64, x::AbstractVector, y::AbstractVector)

    X = BasisMatrix(Basis, x) # Basis Matrix
    D = DifferenceMatrix(Basis) # D penalty matrix
    Xp = vcat(X, sqrt(λ)*D) # augment model matrix with penalty
    yp = vcat(y, repeat([0],size(D)[1])) # augment data with penalty

    return Xp, yp
end

Xp, yp = PenaltyMatrix(Basis, λ, x, y);

β_penalized = coef(lm(Xp,yp));

# Plot Data and Spline
PlotCherryData(alpha = 0.20)
plot!(x, sum(β_spline .* Basis), color = :black, linewidth = 3, label = "Basic Spline")
plot!(x, sum(β_penalized .* Basis), color = :red, linewidth = 3, label = "Penalized Spline")
~~~~~~~~~~~~~

![png](/assets/2023-06-21-gams-julia_files/2023-06-21-gams-julia_10_1.png)\ 




From the plot above we can see that the red penalized spline with `λ = 1` is smoother than the black basic spline fit.
But how can we determine what the optimal value for lambda should be?

## Finding Optimal Penalty


~~~~{.julia}
# Function to calculate GCV for given penalty
function GCV(param::AbstractVector, Basis::BSplineBasis{Vector{Float64}}, x::AbstractVector, y::AbstractVector)
    n = length(Basis.breakpoints)
    Xp, yp = PenaltyMatrix(Basis, param[1], x, y)
    β = coef(lm(Xp,yp))
    H = Xp*inv(Xp'Xp)Xp' # hat matrix
    trF = sum(diag(H)[1:n])
    y_hat = Xp*β
    rss = sum((yp-y_hat)[1:n].^2) ## residual SS
    gcv = n*rss/(n-trF)^2
    return gcv
end

# Function that Optimized Lambda based on GCV
function OptimizeGCVLambda(Basis::BSplineBasis{Vector{Float64}}, x::AbstractVector, y::AbstractVector)

    # optimization bounds     
    lower = [0]
    upper = [Inf]
    initial_lambda = [1.0]

    # Run Optimization
    inner_optimizer = GradientDescent()
    res = Optim.optimize(
        lambda -> GCV(lambda, Basis, x, y), 
        lower, upper, initial_lambda, 
        Fminbox(inner_optimizer)
    )
    return Optim.minimizer(res)[1]
end

λ_opt = OptimizeGCVLambda(Basis, x, y);
~~~~~~~~~~~~~




Now that we have obtained an optimal value for lambda, we can build the penalized design matrix and fit our spline.

~~~~{.julia}
Xp_opt, yp_opt = PenaltyMatrix(Basis, λ_opt, x, y);

β_opt = coef(lm(Xp_opt,yp_opt));

PlotCherryData(alpha=0.2)
plot!(x, sum(β_spline .* Basis), color = :black, linewidth = 3, label = "Basic Spline")
plot!(x, sum(β_penalized .* Basis), color = :red, linewidth = 3, label = "Spline: λ = 1")
plot!(x, sum(β_opt .* Basis), color = :blue, linewidth = 3, label = "Spline: λ = $(round(λ_opt,digits=3))")
~~~~~~~~~~~~~

![png](/assets/2023-06-21-gams-julia_files/2023-06-21-gams-julia_12_1.png)\ 




It is clear that the blue spline, with penalty of λ = 72.198 is far smoother than the other splines we fit. That is exactly what we were looking for, but this is not the only avenue to obtain this result.

## Using a Bayesian Hierarchical Approach

Now for the fun stuff. In order to use a Bayesian approach, we need to re-parameterize our model.
Wood goes into detail on the theory of this parameterization on pages 173-174 of his text. The procedure involves some pretty low level linear algebra. 
To be truthful, I do not have a good intuition of what the matrix algebra is doing for us here. But I was able to figure out how to translate the R code into Julia.

For those looking for a more detailed explanation of this math, I will have to point you to Wood's [text](https://www.taylorfrancis.com/books/mono/10.1201/9781315370279/generalized-additive-models-simon-wood).

One thing to note on terminology, Bayesian Hierarchical are sometimes called "Random Effects" models, and models with no established hierarchy are referred to as "Fixed Effects" models.
Models with hierarchical and non-hierarchical variables are called "Mixed Effect" models (having both random and fixed variables). For the example below, I will adopt the terminology of "random" and "fixed" effects.

From this Julia code, the resulting `X_fe` matrix represents the fixed effects matrix and `Z_re` represents the random effect matrix.

~~~~{.julia}
# Function to build the Mixed Effect Matricies
function MixedEffectMatrix(Basis::BSplineBasis{Vector{Float64}}, x::AbstractVector)

    # Build Basis Matrix
    X = BasisMatrix(Basis, x)

    # Build difference Matrix
    D = DifferenceMatrix(Basis)

    ## Reparameterization
    D_pls = vcat(zeros(2,length(Basis)), D);
    D_pls[diagind(D_pls)] .= 1;
    XSD = (UpperTriangular(D_pls') \ X')';

    # Split to Fixed and Random Effects
    X_fe = XSD[:, 1:2]; # Fixed
    Z_re = XSD[:, 3:end]; # Random

    return X_fe, Z_re
end

X_fe, Z_re = MixedEffectMatrix(Basis, x);
~~~~~~~~~~~~~




We can now define our Turing model to fit the hierarchical smoothing spline model. This model function accepts the following arguments:

  - `X_fe` is the fixed effect matrix;
  - `Z_re` is the random effect matrix;
  - `y` is the element we want to predict;
  - `μ_int` is the center of the intercept prior;
  - `σ_prior` is the standard deviation of coefficient priors;
  - `err_prior` defines the exponential prior on likelihood error;
  - `re_prior` defines the exponential prior on random effect variance.

~~~~{.julia}
# Define Turing model
@model function SmoothSplineModel(X_fe, Z_re, y, μ_int, σ_prior, err_prior, re_prior)

    ## Priors 
    # Intercept
    α ~ Normal(μ_int, σ_prior)

    # Fixed Effects
    B ~ MvNormal(repeat([0],2), σ_prior)
    
    # Random Effects
    μ_b ~ Normal(0, σ_prior)
    σ_b ~ Exponential(re_prior)
    b ~ filldist(Normal(μ_b, σ_b), size(Z_re)[2])
    
    # error
    σ ~ Exponential(err_prior)

    ## Determined Variables
    μ = α .+ X_fe * B + Z_re * b
    
    ## Likelihood
    y ~ MvNormal(μ, σ)
end;
~~~~~~~~~~~~~




We will use MCMC to perform inference. The following code will use the No U-turn (NUTS) sampler to draw 1000 samples from the posterior. Typically it is advised to run multiple chains, see the [Turing docs](https://turing.ml/v0.22/docs/using-turing/guide#sampling-multiple-chains), but for this demo we will use just a single chain.
Turing has a lot of neat features, such as the ability to simply call `summarystats(chain)` to assess model convergence.

~~~~{.julia}
# Define Model Function
spline_mod = SmoothSplineModel(X_fe, Z_re, y, 100, 10, 5, 5);

# MCMC Sampling
chain_smooth = sample(spline_mod, NUTS(), 1_000);

# Summary
summarystats(chain_smooth)
~~~~~~~~~~~~~

~~~~
Summary Statistics
  parameters       mean       std   naive_se      mcse        ess      rhat
    ⋯
      Symbol    Float64   Float64    Float64   Float64    Float64   Float64
    ⋯

           α   100.7245    5.7284     0.1811    0.3041   404.0234    1.0007
    ⋯
        B[1]    -0.9087    5.8662     0.1855    0.3069   387.6929    0.9997
    ⋯
        B[2]     1.4121    5.6707     0.1793    0.2980   412.1859    1.0012
    ⋯
         μ_b    -0.4046    0.7199     0.0228    0.0276   631.3371    0.9994
    ⋯
         σ_b     2.4099    1.3013     0.0411    0.1285   109.4226    0.9999
    ⋯
        b[1]     0.0158    2.3636     0.0747    0.1158   374.9430    1.0038
    ⋯
        b[2]    -1.0021    1.8532     0.0586    0.0852   599.8406    1.0001
    ⋯
        b[3]    -2.0444    1.7129     0.0542    0.0960   362.6718    0.9998
    ⋯
        b[4]    -0.5341    1.6501     0.0522    0.0831   436.8299    1.0055
    ⋯
        b[5]     0.7402    1.7728     0.0561    0.0872   424.6244    1.0027
    ⋯
        b[6]     2.9408    2.3164     0.0732    0.1970   155.2558    0.9999
    ⋯
        b[7]    -2.5384    2.5569     0.0809    0.2331   130.3484    1.0000
    ⋯
        b[8]     0.4037    1.9761     0.0625    0.1504   178.5638    1.0003
    ⋯
        b[9]     0.1127    1.6944     0.0536    0.0893   339.1566    1.0008
    ⋯
       b[10]    -0.7577    1.7918     0.0567    0.0915   405.7169    1.0019
    ⋯
       b[11]     0.8570    2.0056     0.0634    0.1431   203.4626    1.0021
    ⋯
       b[12]    -2.1061    1.7970     0.0568    0.1329   173.6762    1.0053
    ⋯
      ⋮           ⋮          ⋮         ⋮          ⋮         ⋮          ⋮   
    ⋱
                                                     1 column and 4 rows om
itted
~~~~





We can then take the mean of the posterior for each parameter and build the spline predictions for each observation to compare with our previous models.

~~~~{.julia}
# Get posterior draws for relevant parameters
post_smooth = get(chain_smooth, [:α, :B, :b]);

# Compute Predictions
pred_smooth = [post_smooth[:α]...]' .+ X_fe * hcat(post_smooth[:B]...)' .+ Z_re * hcat(post_smooth[:b]...)';
~~~~~~~~~~~~~




The following plot compares the original spline model to the smoothing spline model that we just fit. As we can see, the smoothing spline fit looks far smoother despite having the exact same number of knots and basis functions as the original model.

~~~~{.julia}
# Plot Data and Splines
PlotCherryData(alpha=0.2)
plot!(x, sum(β_opt .* Basis), color = :blue, linewidth = 3, label = "Spline: λ = $(round(λ_opt,digits=3))")
plot!(x, mean(pred_smooth;dims=2), color = :green, linewidth = 3, label = "Spline: Bayes")
~~~~~~~~~~~~~

![png](/assets/2023-06-21-gams-julia_files/2023-06-21-gams-julia_17_1.png)\ 




Since our outcomes are so similar, what are we getting using the Bayesian model over optimizing the λ penalty? 
Well, one benefit is that we can use the model posterior to build the [posterior predictive]() distribution for the outcome.
This gives us a probability distribution over the outcome, which can be useful.

This is a way to take advantage of the full uncertainty quantification that a Bayes model gives us.
The following Julia code demonstrates how we can leverage the full posterior for predictions by building a partial dependence plot.

~~~~{.julia}
function PartialDepPlot(x,pred)

    # Sort Data
    x_order = sortperm(x)
    x = x[x_order]
    pred = pred[x_order,:]

    # Build HDI
    hdi = mapslices(x -> hpdi(x,alpha=0.05), pred; dims=2);

    # Build Partial Dependence Plot
    plt = plot(x, mean(pred;dims=2));
    plot!(
        plt,
        x, hdi[:,1], 
        fillrange = hdi[:,2],
        color = :green, fillcolor = :green,
        alpha = 0.25, fillalpha = 0.25,
        legend = false
    );
    return plt;
end

# Plot the HDI
PartialDepPlot(x, pred_smooth)
scatter!(x,y,color=:blue,xlabel="Year",ylabel="Day of First Blossom",legend=false,alpha=0.2)
~~~~~~~~~~~~~

![png](/assets/2023-06-21-gams-julia_files/2023-06-21-gams-julia_18_1.png)\ 




## More Variables: Building an Additive Model

Now that we have Julia code capable of building univariate smoothing splines, let's try our hand at building an additive model with two predictor variables.
This example uses the `trees` dataset from the `RDatasets` package. The data set has three variables, tree volume, girth, and height.
We will build an additive model that builds smooth splines over `Girth` and `Height` and predicts the tree `Volume`. 
This is the same data set that Simon Wood uses in section 4.3 of his text. But this example is going to differ in approach, it will use the Bayesian approach for our GAM model.

First let's load the data and plot the relationships.

~~~~{.julia}
# Load Data
trees = dataset("datasets", "trees");

# Define Variables
y = trees.Volume;
x1 = trees.Girth;
x2 = trees.Height;

# Plot the Data
plot(
    scatter(x1,y,legend=false,xlabel = "Tree Girth"),
    scatter(x2,y,legend=false,xlabel = "Tree Height"),
    ylabel = "Tree Volume"
)
~~~~~~~~~~~~~

![png](/assets/2023-06-21-gams-julia_files/2023-06-21-gams-julia_19_1.png)\ 




The relationships look like they could be fairly well modeled with a standard linear model. We will see how the smoothing splines approximate these relationships.

The following code loads the data set and performs the data-preprocessing steps from above.

~~~~{.julia}
# Build Basis Functions
Basis1 = QuantileBasis(x1, 10, 4);
Basis2 = QuantileBasis(x2, 10, 4);

# Build Mixed Effects Matrices
X_fe_1, Z_re_1 = MixedEffectMatrix(Basis1, x1);
X_fe_2, Z_re_2 = MixedEffectMatrix(Basis2, x2);

# Organize Matrices in Vectors
X_fe = [X_fe_1, X_fe_2];
Z_re = [Z_re_1, Z_re_2];
~~~~~~~~~~~~~




Next, we will define a Turing Model similar to the one above, but this model now builds two smoothing splines instead of one.
I am certain there is a more elegant "Julian" way to build this, but for now the verbose approach will suffice.

~~~~{.julia}
# Define Additive Model that uses two splines
@model function AdditiveModel(X_fe, Z_re, y, μ_int, σ_prior, err_prior, re_prior)

    ## Priors 
    # Intercept
    α ~ Normal(μ_int, σ_prior)

    # Fixed Effects
    ## Variable 1
    B1 ~ MvNormal(repeat([0],2), σ_prior)
    ## Variable 2
    B2 ~ MvNormal(repeat([0],2), σ_prior)
    
    # Random Effects
    ## Variable 1
    μ_b1 ~ Normal(0, σ_prior)
    σ_b1 ~ Exponential(re_prior)
    b1 ~ filldist(Normal(μ_b1, σ_b1), size(Z_re[1])[2])
    ## Variable 2
    μ_b2 ~ Normal(0, σ_prior)
    σ_b2 ~ Exponential(re_prior)
    b2 ~ filldist(Normal(μ_b2, σ_b2), size(Z_re[2])[2])
    
    # error
    σ ~ Exponential(err_prior)

    ## Determined Variables
    μ = α .+ X_fe[1] * B1 + X_fe[2] * B2 + Z_re[1] * b1 + Z_re[2] * b2
    
    ## Likelihood
    y ~ MvNormal(μ, σ)
end;
~~~~~~~~~~~~~




Next, we will instantiate our model, run MCMC to assess parameter posterior values, and build predictions.

~~~~{.julia}
# Define Model Function
add_mod = AdditiveModel(X_fe, Z_re, y, 30, 5, 1, 1);

# MCMC Chain
chain_add = sample(add_mod, NUTS(), 1_000);

# Get Posterior Draws
post_add = get(chain_add, [:α, :B1, :B2, :b1, :b2]);

# Compute Predictions
## Girth
pred_add_1 = X_fe_1 * hcat(post_add[:B1]...)' +  Z_re_1 * hcat(post_add[:b1]...)';
## Height
pred_add_2 = X_fe_2 * hcat(post_add[:B2]...)' +  Z_re_2 * hcat(post_add[:b2]...)';
## Volume
pred_add = [post_add[:α]...]' .+ (pred_add_1 + pred_add_2);
~~~~~~~~~~~~~




And inspect our partial dependence plots.

~~~~{.julia}
GirthPlot = PartialDepPlot(x1, pred_add_1);
xlabel!(GirthPlot, "Tree Girth");
HeightPlot = PartialDepPlot(x2, pred_add_2);
xlabel!(HeightPlot, "Tree Height");
plot(GirthPlot, HeightPlot, plot_title = "Partial Dependence Plots")
ylabel!("Tree Volume")
~~~~~~~~~~~~~

![png](/assets/2023-06-21-gams-julia_files/2023-06-21-gams-julia_23_1.png)\ 




Lastly, we can observe the relationship between actual and predicted tree volume.

~~~~{.julia}
# Compute R Squared
RSqrd = cor(y, mean(pred_add;dims=2))[1,1]^2

# Plot Actual v Predictions
scatter(
    y, mean(pred_add;dims=2), 
    legend = false,
    plot_title = "Predicted Volume v Actual Volume"
);
annotate!((20, 60, "R2: $(round(RSqrd,digits=3))"));
xlabel!("Tree Volume");
ylabel!("Predicted Volume")
~~~~~~~~~~~~~

![png](/assets/2023-06-21-gams-julia_files/2023-06-21-gams-julia_24_1.png)\ 




Obviously we would need to hold out some data to properly assess model accuracy.

The objective of the post was to build a GAM in Julia "from scratch". I am fairly happy with the results, but there is certainly room for improvement. If you made it this far, I want to thank you for your interest in this topic. I hope you found my post on smoothing splines useful.