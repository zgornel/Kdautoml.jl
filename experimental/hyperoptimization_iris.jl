using MLJ
using DataFrames
using Combinatorics
using MLJMultivariateStatsInterface
using MLJ.CategoricalArrays
using Revise
using Kdautoml

# Feature product stuff
function feature_product(data::DataFrame; C=2)
    n, m = size(data);
    cmbs = combinations(1:m, C);
    newdata = zeros(n, length(cmbs));
    for (i, c) in enumerate(cmbs)
        newdata[:,i] .= map(*, eachcol(data[:,c])...);
    end;
    prod_features = ["product_$(join(string.(c),'-'))" for c in cmbs];
    return hcat(data, DataFrame(newdata, prod_features), makeunique=true);
end;

if (!@isdefined FeatureProduct)
    mutable struct FeatureProduct <: MLJ.Static
        C::Int
    end
end
MLJ.transform(fp::FeatureProduct, _, df) = feature_product(df, C=fp.C);

## Classifier
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()

# Do the machine
data = MLJ.load_iris() |> DataFrame
X = Matrix(data[:,1:4])
y = Vector(data[:,5])

#pipe = (x->DataFrame(x, :auto)) |> Standardizer() |> FeatureSelector(features=[:x1, :x2], ignore=false) |> FeatureProduct(3) |> PCA(maxoutdim=3) |> tree
components_pipe = [(x->DataFrame(x, :auto)),  Standardizer(), FeatureSelector(features=[:x1, :x2, :x3, :x4], ignore=false) , FeatureProduct(3) , PCA(maxoutdim=3) , tree]
pipe = MLJ.Pipeline(components_pipe...)

## Regular usage
#mach = machine(pipe, X, MLJ.CategoricalArray(y)) |> fit!;
#ŷ = MLJ.predict(mach, X)

## Build ranges (the pipeline object needs to exist) <-- needs to be generated from KB!!!
hypers = [(idx=6, params=(name=:max_depth, lower=1, upper=15)),
          (idx=6, params=(name=:feature_importance, values=[:impurity, :split])),
          (idx=5, params=(name=:maxoutdim, lower=1, upper=4, scale=:linear))
         ]
r=Kdautoml.build_ranges(pipe, components_pipe, hypers)
@show r
## Hyperparameter optimization
tuned_pipe = TunedModel(pipe, ranges=r, resampling=Holdout(fraction_train=0.1, shuffle=true), measures=Accuracy(), tuning=Grid(resolution=15))
mach = machine(tuned_pipe, X, y) |> fit!
ŷ = MLJ.predict(mach, X);
