PIPESYNTHESIS_CONTAINER_NAME = "neo4j_pipesynthesis_kb"
FEATURESYNTHESIS_CONTAINER_NAME = "neo4j_featuresynthesis_kb"

# TODO: Make this parametric
get_neo4j_user() = get(ENV, "NEO4J_USER", "neo4j")
get_neo4j_pass() = get(ENV, "NEO4J_PASS", "test")
get_container(container_name) = chop(read(pipeline(`docker ps `,`grep "$container_name"`, `awk '{print $1}'`), String))

# Utility functions
function get_recursively(d, ks; splitter='/', default=nothing)
    keys = split(strip(isequal(splitter), ks), string(splitter))
    reduce((d, k)->get(d, k, default), keys, init=d)
end


#TODO: Check docker is installed, container valid, etc
function cypher_shell(container, user, pass, cypher_cmd; wait=true, output=false)
    @info "KB: Querying with $(length(IOBuffer(cypher_cmd).data)) bytes KB in container $container..."
    @debug "KB: Query:\n$cypher_cmd"
    full_cmd = `docker exec --interactive $container cypher-shell --format plain -u $user -p $pass $cypher_cmd`
    if !output
        run(full_cmd, wait=wait)
        return nothing
    else
        return read(full_cmd, String)
    end
    #if !isempty(result)
    #    d,h = readdlm(IOBuffer(result), ',', String, '\n', header=true)
    #    @show d, h
    #    return (header=h, data=d)
    #else
    #    return nothing
    #end
end


# Function that parses neo4j results into a Matrix{String}
parse_neo4j_result(result) = if !isempty(result)
        return String.(readdlm(IOBuffer(result), ',','\n';header=true)[1])
    else
        return String[]
    end

# Function that takes the memory of a generated program
# builds a pipeline, trains and executes it
# `memory` is a MutableNamedTuple with fields:
#   `pipeline::Vector` - contains pipeline transforms
#   `input <:Any ` - input data
#   `output <: Any`- output data
function build_and_run_ml_pipeline(memory; measures=Accuracy(), tuning=Grid(resolution=30))
    if  :pipeline in keys(memory) && :data in keys(memory)
        # assemble pipeline
        pipe = MLJ.Pipeline(memory.pipeline...)

        # extract actual data from sources
        _data = hasproperty(memory, :data) ? memory.data.data : nothing
        _targets = hasproperty(memory, :targets) ? memory.targets.data : nothing

        # build tunable model (if the case)
        tuned_pipe = if (typeof(pipe) <: Probabilistic || typeof(pipe) <: Deterministic) && !isempty(memory.hyperparameters)
                TunedModel(pipe,
                           ranges=build_ranges(pipe, memory.pipeline, memory.hyperparameters),
                           resampling=memory.split,
                           measures=measures,
                           tuning=tuning)
            else
                pipe
            end
        # make machine
        mach = if typeof(tuned_pipe) <: Static
                machine(tuned_pipe)
             elseif typeof(tuned_pipe) <: Unsupervised
                machine(tuned_pipe, memory.data)
             elseif typeof(tuned_pipe) <: Supervised
                machine(tuned_pipe, _data, coerce(df2vec(_targets), Multiclass))  # classification
             end

        # fit/train
        fit!(mach, verbosity=0)

        # run
        if typeof(pipe) <: Supervised
            return MLJ.predict(mach, _data)
        else
            return MLJ.transform(mach, _data)
        end
    end
end


#TODO(Corneliu): Implement a `build_machine` utility function that operates of memory only
#                (It can be called by the function above as well)

df2vec(s::MLJ.Source; kwargs...) = df2vec(s.data; kwargs...)

function df2vec(df::DataFrame; colidx=nothing) # must have a single column
    if colidx != nothing
        return df[!, colidx]
    else
        return getproperty(df, first(propertynames(df)))  # get first col
    end
end

# Returns a mapping between MLJ generated pipeline object
# property names and the indices of the components in the vectorr
# used to build the pipeline. Needed for hyperparameter tuning where
# we have hypeparams associated to indices (in the vector) and a pipe object
# with no clear mapping between hypeparams and pipe object properties
function pipe_prop_to_idx(components)
    d = Dict{Symbol, Int}()
    _names = Symbol[]
    for (i, c) in enumerate(components)
        _name=MLJ.MLJBase.generate_name!(c, _names, only=MLJ.Model)
        push!(d, _name=>i)
    end
    return d
end

## Build ranges for a given pipeline object in a format that can be passed
# to a TunedModel object for hyperparameter search
#
# Note: the pipeline object needs to exist so it can be wrapped.
# Format of hypers is:
#	[(idx=6, hyperparams=(name=:max_depth, lower=1, upper=15)),
#    (idx=6, hyperparams=(name=:feature_importance, values=[:impurity, :split])),
#    (idx=5, hyperparams=(name=:maxoutdim, lower=1, upper=4, scale=:linear))
#	]
function build_ranges(pipe, components, hypers)
    ranges = Vector{MLJ.MLJBase.ParamRange}()
    propidxs = pipe_prop_to_idx(components)
    for (compname, idx) in propidxs   # pipe element loop: iterates over pipe object property names and indices in pipe vector
        for (hidx, hyperparams) in hypers  # hyperparameter loop: iteratees of indices in pipe vector and corresponding hyperparameter
            if hidx == idx
                _r = if hasproperty(hyperparams, :lower) && hasproperty(hyperparams, :upper)
                        range(pipe, :($(compname).$(hyperparams.name)), lower=hyperparams.lower, upper=hyperparams.upper)
                     elseif hasproperty(hyperparams, :values)
                        range(pipe, :($(compname).$(hyperparams.name)), values=hyperparams.values)
                     else
                        nothing
                     end
                _r != nothing && push!(ranges, _r)
            end
        end
    end
    return ranges
end


feature_product(data::Tables.MatrixTable; C=2) = feature_product(DataFrame(data);C=C)

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


mutable struct FeatureProduct <: MLJ.Static
  C::Int
end

MLJ.transform(fp::FeatureProduct, _, df) = feature_product(df, C=fp.C);

# Do 2-class linear discriminant analysis: returns true is more than frac% of
# the samples in `data` are linearly separable considering classes from `targets`
function xor_linear_separability(data, targets, frac=0.9)
    classes = sort(unique(targets))
    yp = findall(==(classes[2]), targets)
    Xp = data[:, yp]
    yn = findall(==(classes[1]), targets)
    Xn = data[:, yn]
    try
        model = MultivariateStats.fit(LinearDiscriminant, Xp, Xn)
        pp = MultivariateStats.predict(model, Xp) .== ones(length(yp))
        pn = MultivariateStats.predict(model, Xn) .== zeros(length(yn))
        p = ((sum(pp)+sum(pn))/size(data,2))
        return p >= frac
    catch
        @warn "Failed to apply LDA linear separability"
        return false
    end
end

# Used by KernelPCA
function rbf_kernel(p)
    return (x,y)->exp(-(norm(x-y)^2)/(2p^2))
end

# Function that takes a categorical array and changes it to UnivatiateFinite
# for use with `log_loss`
change_into_UnivariateFinite(v::MLJ.CategoricalArrays.CategoricalVector) = begin
    vu = sort(unique(v));
    MLJ.UnivariateFinite(vu, [float.(v.==vi) for vi in vu], pool=v)
end

change_into_UnivariateFinite(v) = v  # if the input is


# Kernelizer
mutable struct Kernelizer <: MLJModelInterface.Unsupervised
    f::Function
end

function Kernelizer(;f=(x,y)->y'x)
    model   = Kernelizer(f)
    message = MLJModelInterface.clean!(model)
    isempty(message) || @warn message
    return model
end

function MLJModelInterface.clean!(m::Kernelizer)
    warning = ""
    a=rand(10);
    if !(m.f(a,a) isa Number)
        warning *= "Kernelizer function must return a Number"
    end
    return warning
end

function MLJModelInterface.fit(m::Kernelizer, verbosity::Int, X)
    return (X, m.f), nothing, ()
end

function MLJModelInterface.transform(m::Kernelizer, fitresult, Xnew)
    kernel_data, kernel_f = fitresult
    #Xmatrix = MLJBase.matrix(Xnew, transpose=true)
    #TODO: Improve this
    tmp = [kernel_f(xnew, xkernel) for xkernel in eachrow(MLJ.matrix(kernel_data)) for xnew in eachrow(MLJ.matrix(Xnew))]
    return DataFrame(MLJ.table(reshape(tmp, size(Xnew,1), size(kernel_data,1))))
end

Base.eachcol(m::Tables.MatrixTable) = eachcol(MLJ.matrix(m))

Base.size(m::Tables.MatrixTable, args...) = size(MLJ.matrix(m),args...)
