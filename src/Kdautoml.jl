module Kdautoml
    using Reexport

    export DeepFeatureSynthesis,
           AbstractComponent,
           AbstractState,
           CodeNode,
           build,
           build_and_run_ml_pipeline,
           execute,
           paths,
           push!

    function __init__()
        # This bit is executed after module load
        # Declare container stuff
    end

    include("control.jl")     # CF - highest level, most abstract, has all top-level definitions
    include("program.jl")     # PE - second level, needs stuff defined previously
    include("ks.jl")          # KS - third level, needs methods defined in the previous two
    include("dfs/DeepFeatureSynthesis.jl") # second level also, defines KB query interface for features

end # module
