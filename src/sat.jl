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

¬(x::Num) = 1-x
¬(x::Bool) = ~x


function _replace_symbol(expr, what, val)
    MacroTools.postwalk(expr) do ex
        if what == :x && @capture(ex, x) return :($val)
        elseif what == :y && @capture(ex, y) return :($val)
        elseif what == :p && @capture(ex, p) return :($val)
        elseif what == :z && @capture(ex, z) return :($val)
        else return ex end
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
 			   :component_rules => [("x1", "x3")=>:(x==¬y), ("x1", "x4")=>:(x==¬y)],
 			   :preconditions => [(name="p1", template=:(z == true; z==¬p), code=:(x->begin @info "precond for feature==false"; false; end)),
 			  				      (name="p2", template=:(z == true; z==p), code=:(x->begin @info "precond2 for feature==true"; true; end))]
 			  )
"""
function solve_csp(datakb, state)
    constraints = []
    namemapping = Dict{Symbol, Dict{String, Num}}()

    # Map datakb components to variables and retain mapping
    for (comp, funcs) in get(datakb, :components, Dict())
        varsymb = gensym()
        n = length(funcs)
        vars = Num.(Symbolics.variable.(varsymb, 1:n))  # create variables f₁, f₂ ...
        for v in vars
            push!(constraints, v ∈ 0:1)
        end
        push!(constraints, sum(v for v in vars) ==1)
        push!(namemapping, comp=>Dict(funcs[i].name=>vars[i] for i in 1:n))

        # Evaluate preconditions for each function
        preconds = []
        for ((name, code, hyperparameters, package, preconditions), fsymb) in zip(funcs, vars)
            for ps in preconditions
                #ps.code != nothing && begin
                #    @info "Importing $(ps.package)"
                #    eval("import $(ps.package)")
                #end
                _f = eval(Meta.parse(strip(ps.code)))
                _fc = Base.invokelatest(_f, (ps.args...))
                pv = Base.invokelatest(_fc, state)
                @info "KB(SAT): Executed precondition $(ps.name) => $pv"
                push!(constraints, fsymb == fsymb * pv)
            end
        end
    end

    # Process component rules
    for (nodes, templ) in get(datakb, :component_rules, [])
        _rule = templ
        for (comp, v) in datakb[:components]
            for (i, w) in enumerate(v)
                w.name == nodes[1] && (_rule = _replace_symbol(_rule, :x, namemapping[comp][w.name]))
                w.name == nodes[2] && (_rule = _replace_symbol(_rule, :y, namemapping[comp][w.name]))
            end
        end
        push!(constraints, eval(_rule))
    end

    # Process feature preconditions
    F_SYMB = gensym()
    _fs = Num(Symbolics.variable(F_SYMB))
    push!(constraints, _fs ∈ 0:1)
    push!(constraints, _fs == true)
    local ntempl
    for (name, templ, code) in get(datakb, :preconditions, [])
        _data = ""  # FAKE
        pv = Base.invokelatest(eval(code), _data)
        ntempl = _replace_symbol(templ, :p, pv)
        ntempl = _replace_symbol(ntempl, :z, Num(Symbolics.variable(F_SYMB)))
        push!(constraints, eval(ntempl))
    end

    # Solve
    pb = DiscreteCSP(constraints);
    sol = all_solutions(pb);
    return namemapping, constraints, sol
end


function add_data_to_csp_solution(datakb, namemapping, solutions, props)
    rnm = Dict{Symbol, Dict{Num,String}}()
    for (k, v) in namemapping
        push!(rnm, k=>Dict(w=>l for (l,w) in v))
    end
    sols = []
    for s in solutions
        _sol = []
        for (var, bval) in s
            if bval == 1
                for prop in props
                    if haskey(rnm[prop], var)
                        _fname = rnm[prop][var]
                        _fdata= first(filter(x->x.name==_fname, datakb[:components][prop]))
                        push!(_sol, prop=>(name=_fdata.name, code=_fdata.code, hyperparameters=_fdata.hyperparameters, package=_fdata.package))
                    end
                end
            end
        end
        push!(sols, _sol)
    end
    return sols
end
