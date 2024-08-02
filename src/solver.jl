# What would be the general form ?
# s = (data used to build the query)
## Calculate preconditions for all relevant symbols (AST components)
## We assume Xᵢcontains ll symbols corresponding to individual AST subcomponents (i.e. X₁ for scalar getters, X₂ for tensor getters etc.)
# preconds = Dict([xᵢ => map(x->Base.invokelatest(Meta.parse(code))(s), preconditions) for xᵢ in Xᵢ)
#
## Preconditions for individual AST subcomponents
# [xᵢ == false for xᵢin [x₁...xₘ] if prod(preconds[xᵢ]) == false]
# [xᵢ ∈ 0:1 for xᵢin [x₁...xₘ] if prod(preconds[xᵢ]) == true]
# sum(xᵢ for xᵢin [x₁...xₘ]) == 1 # for necessary subcomponents
# sum(xᵢ for xᵢ in [x₁...xₙ] ==0  # for unnecessary subcomponents
#
## Conditions that apply to a whole class of specific subcomponents (i.e. for all conditions, q1="data values need to be discrete" == true)
# [xᵢ== q for xᵢin [x₁...xₘ]]
#
## Feature type and input type conditions
# z == qₚ # z = true is bias associated to it (i.e. qₚ==isa_Entytfeature(feature_type)) is true
# ~qₛ*z == z # condition that associates z (feature type variable) to a condition on the input (i.e. feature_type isa EntityFeature == ~isa(input, EntityFeature))
#            # as long as z==false, qₛ does not matter; when z==true,
@reexport module KSSolver

using ...Random
using ...DelimitedFiles
using ...LinearAlgebra  # for `norm` in `rbf_kernel` (utils.jl)
using ...DataStructures
using ...MLJ
using ...DataFrames
using ...MultivariateStats
using ...ConstraintSolver
using ...MacroTools
using ...Kdautoml  # needed for KB-defined precondition code in feature synthesis
                  # that when executed, references `Kdautoml`

import ..KnowledgeSystem  # to extent API


const CS = ConstraintSolver

const CSMODEL = CS.Model(CS.optimizer_with_attributes(CS.Optimizer,
                    "time_limit"=>1000,
                    "all_solutions"=>true,
                    "all_optimal_solutions"=>true))


function clean_variables!(model)
    var_names = keys(CS.JuMP.object_dictionary(model))
    var_refs = Iterators.flatten(values(CS.JuMP.object_dictionary(model)))
    for vr in var_refs
       CS.JuMP.delete(model, vr)
    end
    for vn in var_names
       CS.JuMP.unregister(model, vn)
    end
end

function clean_constraints!(model)
    for ct in CS.JuMP.list_of_constraint_types(model)
        for c in CS.JuMP.all_constraints(model, ct...)
            cn = Symbol(CS.JuMP.name(c))
            CS.JuMP.delete(model, c)
            CS.JuMP.unregister(model, cn)
        end
    end
end


"""
Creates a CSP (constraint satisfaction problem) from the data returned from the kb (datakb, see format below) and solves it.

# Example of data format expected for building the CSP problem:
 
 datakb = Dict(:components => Dict(:component_1 => [# func code => vector of preconds code
                                                    (name = "x1", code=:(x->x), preconditions=[:(x->true), :(x->true)]),
                                                    (name = "x2", code=:(x->x^2), preconditions=[:(x->false), :(x->true), (x->false)])
                                                   ],
                                   :component_2 => [(name = "x3", code=:(x->x^3), preconditions=[:(x->true), :(x->false)]),
                                                    (name = "x4", code=:(x->x^4), preconditions=[:(x->false)]),
                                                    (name = "x5", code=:(x->x^5), preconditions=[:(x->true)]),
                                                    (name = "x6", code=:(x->x^6), preconditions=[:(x->true)])
                                                   ],
                                    ),
 			  )
"""
function KnowledgeSystem.solve_csp(datakb, state)
    namemapping = Dict{Symbol, Dict{String, CS.VariableRef}}()
    sol = []
    if !isempty(get(datakb, :components, Dict()))
        # Clean model from previous variables and constraints
        clean_constraints!(CSMODEL)
        clean_variables!(CSMODEL)
        # Map datakb components to variables and retain mapping
        for (comp, funcs) in get(datakb, :components, Dict())
            k = randstring(5)
            n = length(funcs)
            # Note: the statements below are evaluated as there was no way to interpolate
            #       dynamically new variable names into the macros @variable and @constraint
            xvars = gensym()
            @eval KSSolver xxv = CS.@variable(CSMODEL, $xvars[1:$n], Bin)  # create variables f₁∈ 0:1, f₂∈ 0:1 ...
            @eval KSSolver CS.@constraint(CSMODEL, sum($xvars) == 1)
            push!(namemapping, comp=>Dict(funcs[i].name=>xxv[i] for i in 1:n))
            for (i, ((name, code, hyperparameters, package, preconditions), _)) in enumerate(zip(funcs, xxv))
                for ps in preconditions  # eval preconditions for each component and add constraints
                    _f = eval(Meta.parse(strip(ps.code)))
                    _fc = Base.invokelatest(_f, (ps.args...))
                    pv = Base.invokelatest(_fc, state)
                    @info "CS: Executed precondition $(ps.name) => $pv"
                    @eval KSSolver CS.@constraint(CSMODEL, $xvars[$i] == $xvars[$i] * $pv)
                end
            end
        end

        # Solve
        CS.optimize!(CSMODEL);
        sol = [ Dict(k => CS.JuMP.value(k; result=i) for k in Iterators.flatten(values(CSMODEL.obj_dict)))
                    for i in 1:CS.JuMP.result_count(CSMODEL)
              ];  # Array of Dicts of length number of solutions (each Dict a solution)
    end
    return namemapping, sol
end


function KnowledgeSystem.add_data_to_csp_solution(datakb, namemapping, solutions, components)
    rnm = Dict{Symbol, Dict{CS.VariableRef,String}}()
    for (k, v) in namemapping
        push!(rnm, k=>Dict(w=>l for (l,w) in v))
    end
    sols = []
    for s in solutions
        _sol = []
        for (var, bval) in s
            if bval == 1
                for component in components
                    if haskey(rnm[component], var)
                        _fname = rnm[component][var]
                        _fdata= first(filter(x->x.name==_fname, datakb[:components][component]))
                        push!(_sol, component=>(name=_fdata.name, code=_fdata.code, hyperparameters=_fdata.hyperparameters, package=_fdata.package))
                    end
                end
            end
        end
        push!(sols, _sol)
    end
    return sols
end

end  # module
