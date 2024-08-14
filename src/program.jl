@reexport module ProgramExecution

using AutoHashEquals
import AbstractTrees
import Base: push!, pop!
import ..ControlFlow

export AbstractPipelines, Pipelines, AbstractProgram, DaggerProgram, CodeNode, print_tree_debug
# TODO: Look over this interesting links
# • https://github.com/SciML/RuntimeGeneratedFunctions.jl (https://github.com/SciML/RuntimeGeneratedFunctions.jl)
# • https://domluna.github.io/JuliaFormatter.jl/stable/ (julia formatter)
# • https://github.com/jkrumbiegel/HotTest.jl/blob/main/src/HotTest.jl (useful stuff for wrapping code into modules)
#
# CodeNode - basic structure representing the node of the tree
@auto_hash_equals mutable struct CodeNode
    id::String
    name::String
    code
    children::Vector{CodeNode}
end


# id-less constructor
CodeNode(name::String, code, children::Vector{CodeNode}) = CodeNode(string(hash(rand())), name, code, children)

CodeNode(name::String, code) = CodeNode(string(hash(rand())), name, code, CodeNode[])

CodeNode(name::Symbol, args...) = CodeNode(string(name), args...)  # handle Symbol names


function AbstractTrees.children(n::CodeNode)
    isempty(n.children) && return ()
    return n.children
end

AbstractTrees.printnode(io::IO, n::CodeNode) = print(io, "•-[$(n.name)]")

printnode_debug(io::IO, n::CodeNode) = print(io, "•-[$(n.name); \"$(n.id)\"]")

print_tree_debug(io::IO, n::CodeNode; kwargs...) = AbstractTrees.print_tree(printnode_debug, io, n; kwargs...)
print_tree_debug(n::CodeNode; kwargs...) = print_tree_debug(stdout, n; kwargs...)

Base.show(io::IO, n::CodeNode) = begin
    nc = length(AbstractTrees.children(n))
    chname = nc <= 1 ? "child" : "children"
    if nc == 0
        print(io, "Leaf node \"$(n.name)\"")
    else
        nodenames = map(x->"\"$(x.name)\"", AbstractTrees.children(n))
        print(io, "CodeNode with $nc $chname [\"$(n.name)\"]•-[$(join(nodenames, ","))]")
    end
end

has_children(node) = !isempty(AbstractTrees.children(node))

function ControlFlow.paths(startnode, endnodeids)
    # Function that shrinks a Vector{CodeNode} to the first occurence
    # of a CodeNode that has `node` in its children (used to go back
    # in the tree if the vector is obtained through a pre-order traversal
    shrinkto = (stack, node)->begin
        cutidx = findfirst(n->in(node, AbstractTrees.children(n)), stack)
        !isnothing(cutidx) && (stack = stack[1:cutidx])
        return stack
    end

    ordering = collect(AbstractTrees.PreOrderDFS(startnode))
    PATHS = Vector{Pair{CodeNode, Vector{CodeNode}}}()
    startnode = first(ordering)
    tmp = [startnode]
    !has_children(startnode) && return Dict(startnode=>[startnode])
    for node in ordering[begin+1:end]
        #println("CodeNode=$(node.name), tmp=$(map(n->n.name, tmp)))")
        !in(node, AbstractTrees.children(last(tmp))) && (tmp = shrinkto(tmp, node))
        if node.id in endnodeids  # sought node
            #println("Returning paths for $(node.name)!")
            push!(PATHS, node=>vcat(tmp, node))
        else  # another node (if not leaf, track)
            has_children(node) && push!(tmp, node)
        end
    end
    return PATHS
end


function ControlFlow.prune!(root, leaf)
    @assert length(AbstractTrees.children(leaf)) == 0 "Deletion start node has to be a leaf"
    _, treepath = first(ControlFlow.paths(root, [leaf.id]))
    pos = -1
    for (i, node) in Iterators.reverse(enumerate(treepath))
        if length(AbstractTrees.children(node)) <= 1
            pos = i  # pos is the index of the last node that has 0 or 1 children
        else         # in a continous sequence
            break
        end
    end
    popped = []
    if pos > 1  # pos == 1 is the root
        for (j, node) in enumerate(treepath[pos-1].children)
            if node == treepath[pos]
                _pop = popat!(treepath[pos-1].children, j)
                push!(popped, _pop)
                break
            end
        end
    end
    pos == 1 && @error "Whole tree cannot be deleted"
    return popped
end


# Program - basic structure holding the code-state of a program
abstract type AbstractProgram end

struct DaggerProgram <: AbstractProgram
    name::String
    header::String
    segments::Vector{String}
end

DaggerProgram(;header=true) = begin
    name = string(hash(rand()), base=16)
    _header = ifelse(header, form_header(DaggerProgram, name), "")
    return DaggerProgram(name, _header, String[])
end


Base.show(io::IO, program::DaggerProgram) = begin
    print(io, "DaggerProgram with $(length(program.segments)) code segments.\n")
end


function form_header(::Type{DaggerProgram}, name)
    symmod = "Module_" * name
    _code = """module $symmod;\n
               using Logging;\n
               global_logger(ConsoleLogger(stdout, Logging.Debug));\n
               using TOML;\n
               using MutableNamedTuples;\n
               using Dagger;\n
               using DataFrames;\n
               using MLJ;\n
               using Kdautoml;\n
               using Kdautoml.DeepFeatureSynthesis;\n"""

    symnode = "v_"* string(hash(rand()), base=16)  # in the program, the symbol gets associated to output value
    _code *= "$symnode = Dagger.@par (x->x)(MutableNamedTuple(data=nothing, pipe_out=nothing, pipeline=[], hyperparameters=[]));\n"
end

last_var(program::DaggerProgram) = begin
    if !isempty(program.segments) 
        last_line = program.segments[findlast(startswith("v_"), program.segments)]
    else  # take variable from header
        last_line = last(filter(l->!startswith("v_",l), collect(eachline(IOBuffer(program.header)))))
    end
    return replace(first(split(last_line, "=")), r"\s"=>"")
end

form_footer(program::DaggerProgram) = begin
    lv = last_var(program)
    if !isnothing(lv)
        return "v_$(program.name) = collect($(last_var(program))); end"
    else
        return "end"
    end
end


function ControlFlow.execute(program::DaggerProgram)
    try
        @debug "PROGRAM_EXEC: Assembling program..."
        _prg = map([program.header, program.segments..., form_footer(program)]) do line
                replace(line, "\n"=>" ")
              end
        prg = strip(join(_prg, "\n"))
        eval(Meta.parse(prg))
        _mod = getproperty(@__MODULE__, Symbol("Module_", program.name))
        @info "PROGRAM_EXEC: Executed program (in $_mod)..."
        output_symbol = Symbol("v_$(program.name)")
        result = getproperty(_mod, output_symbol)
        return result
    catch e
        @warn "Something went wrong with partial execution: $e"
        return nothing
    end
end

function clear!(;current_mod=@__MODULE__)
    @debug "Clearning... (current module is $current_mod)"
    for m in filter(startswith("Module_"), string.(names(current_mod, all=true)))
    strmod = "$(current_mod).$m"
    mod = eval(Meta.parse(strmod))
          eval(Meta.parse("import $strmod"))
          for symb in names(mod, all=true)
              startswith(string(symb), "v_") && Core.eval(mod, :($symb=nothing))
          end
      end
      GC.gc()
end

#TODO: rename this function
Base.push!(program::DaggerProgram, node) = begin
    symnode = "v_"* string(hash(rand()), base=16)  # in the program, the symbol gets associated to output value
    func_code = hasproperty(node.code, :code) ? node.code.code : ""
    package = hasproperty(node.code, :package) ? node.code.package : nothing
    arguments = hasproperty(node.code, :arguments) ? node.code.arguments : ()
    hyperparameters = hasproperty(node.code, :hyperparameters) ? node.code.hyperparameters : nothing
    lv = last_var(program);
    # Build standardized function signature (for KB functions)
    #  - add last Dagger node as first argument of current call
    #  - add hyperparameters as second
    #  - the rest of the arguments from input follow ...
    lv != nothing && (arguments = (last_var(program), hyperparameters, arguments...))

    # Explicitly add a nothing as first argument (used for first ops in pipeline)
    # This requires in the KB a fixed signature of the form `foo(input, args...)`
    lv == nothing && (arguments = (nothing, arguments...))

    if package != nothing
        if package isa String
            push!(program.segments, "\nusing $package;\n")
        else
            for p in package
                push!(program.segments, "\nusing $p;\n")
            end
        end
    end
    func_code != nothing && push!(program.segments, "$symnode = Dagger.@par ($func_code)($( join(arguments,",") ));")
    return symnode
end

#TODO: rename or remove, not used
# To push a popped node:
# `julia> push!(prg, CodeNode("MyNode", (code=pop!(prg),)))`
Base.pop!(program::DaggerProgram) = begin
    code = pop!(program.segments)
    code = replace(code, r"v_[\w]+.=.Dagger\.@par.\([\s]*"=>""); # eliminate first part
    code = replace(code, r"\)\([\w\s,.;]+\);$"=>"");              # eliminate last part
end


# Program structure API;
# Contains symbolic structure of the program and results form executions
# indexed by CodeNode id's
abstract type AbstractPipes end

struct Pipelines{P<:AbstractProgram} <: AbstractPipes
    tree::CodeNode
    artifacts::Dict{String, Any}  # execution results based on node id
end

Pipelines{P}() where {P} = Pipelines{P}(CodeNode("root", nothing, CodeNode[]), Dict{String, Any}())

Pipelines(::P) where {P<:AbstractProgram} = Pipelines{P}()

Pipelines(;backend=:Dagger) = begin
    backend == :Dagger && return Pipelines{DaggerProgram}()
end

Base.show(io::IO, pipelines::Pipelines{P}) where {P<:AbstractProgram} = begin
    nl = 0
    for _ in AbstractTrees.Leaves(pipelines.tree)
        nl+= 1
    end
    print("Pipelines{$P} with $nl pipelines and $(length(pipelines.artifacts)) artifacts.")
end

function ControlFlow.build(nodes::Vector{CodeNode}, ::Pipelines{P}) where {P<:AbstractProgram}
    program = P(;header=true)
    for node in nodes
        node.name == "root" && continue
        push!(program, node)
    end
    return program
end

id2node(id, tree) = begin
    for node in (AbstractTrees.PreOrderDFS(tree))
        node.id == id && return node
    end
end

function ControlFlow.build(nodeid::String, pipelines)
    node = id2node(nodeid, pipelines.tree)
    for (leaf, path) in ControlFlow.paths(pipelines.tree, [nodeid])
        if leaf.id == nodeid
            return ControlFlow.build(path, pipelines)
        end
    end
end

end # module
