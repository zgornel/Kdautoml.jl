using Logging
global_logger(ConsoleLogger(stdout, Logging.Info))
using Pkg
Pkg.activate(joinpath(dirname(@__FILE__), ".."))
using Kdautoml

BASE_PATH =joinpath(dirname(@__FILE__), "..") 
kbpath = joinpath(BASE_PATH, "data/knowledge/pipe_synthesis.toml")
@info "Loading KB at $kbpath"
kb = Kdautoml.kb_load(kbpath; kb_type=:neo4j, kb_flavour=:pipe_synthesis)

# Declare and initialize program
pipes = Kdautoml.Pipelines(;backend=:Dagger)  # header is automaticall added

# Define transition clojure to provide kb and program
primed_transition = (args...)->Kdautoml.transition(args...; kb=kb, pipelines=pipes)

# Build pipelines
csvpath = joinpath(BASE_PATH,"data/datasets/iris.csv")
dfs_args = (joinpath(BASE_PATH, "data/knowledge/feature_synthesis.toml"), 1, true, :neo4j, :feature_synthesis)  # kb path, max_depth, calculate, kb_type, kb_flavour

components= [Kdautoml.LoadData((arguments=(true, csvpath, ','), execute=true)),
             Kdautoml.PreprocessData((arguments=([1,2,3,4],), execute=true,)),
             Kdautoml.SelectModel((execute=true, preconditions=(:DataPrecondition, :PipelinePrecondition, :InputPrecondition))),
             Kdautoml.SplitCV((arguments=(3, true), execute=true)),
             Kdautoml.ModelData((execute=true,)),
             Kdautoml.EvalModel((arguments=(:accuracy,), execute=true,))
      ]

#Check first
@assert reduce(Kdautoml.ControlFlow._transition, components; init=Kdautoml.NoData(nothing)) isa Kdautoml.End{Nothing}

_logger = ConsoleLogger(stdout, Logging.Info)
with_logger(_logger) do
    endstate = reduce(primed_transition, components, init=Kdautoml.NoData(nothing))
end
# Execute statement by statement
#Kdautoml.execute_program!(pipes)
