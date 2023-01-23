using Logging
global_logger(ConsoleLogger(stdout, Logging.Debug))
using Pkg
Pkg.activate(joinpath(dirname(@__FILE__), ".."))
using Kdautoml

kbpath = joinpath(dirname(@__FILE__), "../../../data/knowledge/pipe_synthesis.toml")
@info "Loading KB at $kbpath"
kb = Kdautoml.kb_load(kbpath)

# Declare and initialize program
pipes = Kdautoml.Pipelines(;backend=:Dagger)  # header is automaticall added

# Define transition clojure to provide kb and program
primed_transition = (args...)->Kdautoml.transition(args...; kb=kb, pipelines=pipes)

# Build pipelines
csvpath = joinpath(dirname(@__FILE__), "../../..","data/datasets/iris.csv")
dfs_args = ("\"$(joinpath(dirname(@__FILE__), "../../../data/knowledge/feature_synthesis.toml"))\"", 1, true)  # kb path, max_depth, calculate

components= [Kdautoml.LoadData((arguments=(true, "\"$(csvpath)\"", "','"), execute=true)),
             Kdautoml.PreprocessData((arguments=([1,2,3,4],), execute=true,)),
             #Kdautoml.DFS((arguments=dfs_args, execute=true)),  # can be transformation, generation or selection
             #Kdautoml.FeatureOperation((arguments=(), execute=true)),  # can be transformation, generation or selection
             #Kdautoml.FeatureOperation((arguments=(), execute=true)),  # can be transformation, generation or selection
             #Kdautoml.FeatureOperation((arguments=(), execute=true)),  # can be transformation, generation or selection
             Kdautoml.SelectModel((execute=true, preconditions=(:DataPrecondition, :PipelinePrecondition, :InputPrecondition))),
             Kdautoml.SplitCV((arguments=(3, true), execute=true)),
             Kdautoml.ModelData((execute=true,)),
             Kdautoml.EvalModel((arguments=(:accuracy,), execute=true,))
      ]

#Check first
@assert reduce(Kdautoml._transition, components; init=Kdautoml.NoData(nothing)) isa Kdautoml.End{Nothing}

endstate = reduce(primed_transition, components, init=Kdautoml.NoData(nothing))

# Execute statement by statement
#Kdautoml.execute_program!(pipes)
