"""
Function that calculates features based on:
    J. M. Kanter and K. Veeramachaneni, "Deep feature synthesis: Towards automating data science endeavors,"
    2015 IEEE International Conference on Data Science and Advanced Analytics (DSAA), 2015, pp. 1-10,
    doi: 10.1109/DSAA.2015.7344858.

The input parameters are
    • current - an entity node
    • entities - all entities and their links i.e a graph of entity nodes
    • visited - a list of visited nodes
    • features - back, direct and entity ffeatures; a Dict with :rfeatures, :dfeatures and :efeatures as keys, AST vectors as values
"""
function deep_feature_synthesis!(features, current, entities, visited, max_depth=1; kb=nothing, calculate=false)
    max_depth < 0 && return
    visited = union(visited, current)
    bl = backward_links(entities, current)
    fl = direct_links(entities, current)
    @info "Node $current: visited=$visited, $bl, $fl, $(length(bl)) backward links, $(length(fl)) forward links"
    build_identity_features!(features, entities, current, current, max_depth; kb, calculate)
    for e in bl
        deep_feature_synthesis!(features, e, entities, visited, max_depth-1; kb, calculate)
        build_backward_features!(features, entities, current, e, max_depth; kb, calculate)
    end
    for e in fl
        if e ∈ visited
            println("$e already visited, skipping...")
            continue
        end
        deep_feature_synthesis!(features, e, entities, visited, max_depth-1; kb, calculate)
        build_direct_features!(features, entities, current, e, max_depth; kb, calculate)
    end
    build_entity_features!(features, entities, current, current, max_depth; kb, calculate)
end


function deep_feature_synthesis(current, entities, visited, max_depth=1; kb=nothing, calculate=false)
    features = Dict{Symbol, AbstractFeature}()
    deep_feature_synthesis!(features, current, entities, visited, max_depth; kb, calculate)
    return features
end


"""
Deep feature synthesis function for a single table (DataFrame) input.
"""
function deep_feature_synthesis(df::DataFrame,
                                max_depth=1;
                                df_name=Symbol(:df_, gensym()),
                                kb=nothing,
                                calculate=false)
    mg = MetaDiGraph()
    add_vertices!(mg, 1)
    set_props!(mg, 1, Dict(:df_name=>df_name, :df=>df))
    return deep_feature_synthesis(1, mg, [], max_depth; kb, calculate)
end


"""
Function that transforms a feature object to a DataFrame.
"""
function to_df(features)
    #Helper functions
    feat_name(k, v::IdentityFeature) = Symbol(:I_, k)
    feat_name(k, v::EntityFeature) = Symbol(:E_, k)
    feat_name(k, v::DirectFeature) = Symbol(:D_, k)
    feat_name(k, v::ReverseFeature) = Symbol(:R_, k)

    # Return in one go
    featkeys = collect(keys(features))
    skeysidx = sortperm(featkeys)  # idexes of sorted keys
    return DataFrame(reduce(hcat, (features[featkeys[i]].values for i in skeysidx)), :auto), Dict()
    #
    #TODO: Make *sure* that feature names in the resulting DataFrame are generated consistently
    #      between different runs with the same input; the scheme below is buggy and naming of
    #      features is inconsistent across runs
    #
    ###_tmp = [feat_name(i, features[featkeys[i]]) => features[featkeys[i]].values
    ###        for i in skeysidx]
    ###return DataFrame(_tmp), Dict(k=>i for (i,k) in enumerate(map(first, _tmp)))
end


backward_links(g, v) = setdiff(neighborhood(g, v, 1.0; dir=:in), [v])  # what references node v
direct_links(g, v) = setdiff(neighborhood(g, v, 1.0; dir=:out), [v])  # what node v references


build_identity_features!(args...; kwargs...) = build_features!(IdentityFeature, args...; kwargs...)
build_entity_features!(args...; kwargs...) = build_features!(EntityFeature, args...; kwargs...)
build_direct_features!(args...; kwargs...) = build_features!(DirectFeature, args...; kwargs...)
build_backward_features!(args...; kwargs...) = build_features!(ReverseFeature, args...; kwargs...)


print_info(::Type{IdentityFeature}, depth, df_name, link) = @info "[Depth=$depth] Building ifeatures for $df_name..."
print_info(::Type{EntityFeature}, depth, df_name, link) = @info "[Depth=$depth] Building efeatures for $df_name..."
print_info(::Type{ReverseFeature}, depth, df_name, link) = @info "[Depth=$depth] Building rfeatures for $df_name using link $link..."
print_info(::Type{DirectFeature}, depth, df_name, link) = @info "[Depth=$depth] Building dfeatures for $df_name using link $link..."


_get_link(::Type{ReverseFeature}, g, dst, src) = begin
    cols = get_prop(g, src, dst, :link)
    TableLink((_get_df_name(g, src), cols[1]), (_get_df_name(g, dst), cols[2]))
end

_get_link(::Type{DirectFeature}, g, src, dst) = begin
    cols = get_prop(g, src, dst, :link)
    TableLink((_get_df_name(g, src), cols[1]), (_get_df_name(g, dst), cols[2]))
end

_get_link(::Type, g, src, dst) = nothing


_get_df(g, ::Nothing) = nothing

_get_df(g, node::Int) = get_prop(g, node, :df)

_get_df(g, dfname::Symbol) = begin
    for (k,v) in g.vprops
        v[:df_name] == dfname && return get(v, :df, nothing)
    end
end

_get_df_name(g, node) = get_prop(g, node, :df_name)
_get_df_name(g, ::Nothing) = nothing

###fetch_df_and_name(g, node::Int) = begin
###    return get_prop(g, node, :df), get_prop(g, node, :df_name)
###end

###fetch_df_and_name(g, ::Nothing) = (nothing, nothing)


function build_features!(feature_type, features, g, current_table, link_table, depth; kb=nothing, calculate=false, use_vectors=true)
    link= _get_link(feature_type, g, current_table, link_table)
    print_info(feature_type, depth, _get_df_name(g, current_table), link)
    data = (features=features,
            graph=g,
            current_table=_get_df_name(g, current_table),  # always the write table as well
            link=link)
    _build_features!(feature_type, data, kb; use_vectors, calculate, depth)
    nothing
end


# Returns an interator over the features
_feature_filter(table) = p->p[2].table==table && isa(p[2], AbstractFeature)
_get_features(feature_type, data, table) = ((k, v) for (k,v) in Iterators.filter(_feature_filter(table), data.features))


name_eltype_iterator(features::Base.Generator) = ((k, eltype(typeof(v))) for (k,v) in features)

name_eltype_iterator(df::DataFrame) = begin
    desc = describe(df)
    zip(getproperty(desc, :variable), getproperty(desc, :eltype))
end

# Returns the table from which input data/features is read
get_read_table(::Type{IdentityFeature}, data) = data.current_table
get_read_table(::Type{EntityFeature}, data) = data.current_table
get_read_table(::Type{DirectFeature}, data) = data.link.dst[1]  # != data.current_table
get_read_table(::Type{ReverseFeature}, data) = data.link.src[1]  # != data.current_table

# Returns the table from which input data/features is read
get_write_table(::Type{<:AbstractFeature}, data) = data.current_table

# Returns all the features (df/iterator over features) of a table
get_read_data(ft::Type{IdentityFeature}, data) = _get_df(data.graph, get_read_table(ft, data))
get_read_data(ft::Type{<:AbstractFeature}, data) = _get_features(ft, data, get_read_table(ft, data))


# Returns a common structure used to both obtain feature AST's and calculate features
get_fs_state_data(ft::Type{IdentityFeature}, data, feature_name, feature_eltype, use_vectors) =
    (feature_type = ft,
	 read_table = get_read_table(ft, data),
     write_table = get_write_table(ft, data),
     input_feature = feature_name,
     input_eltype = feature_eltype,
     use_vectors = use_vectors,
     input_data = get_read_data(ft, data),
     agg_column = nothing,
     mask_column = nothing)

get_fs_state_data(ft::Type{EntityFeature}, data, feature_name, feature_eltype, use_vectors) =
    (feature_type = ft,
	 read_table = get_read_table(ft, data),
     write_table = get_write_table(ft, data),
     input_feature = feature_name,
     input_eltype = feature_eltype,
     use_vectors = use_vectors,
     input_data = data.features,
     agg_column = nothing,
     mask_column = nothing)

get_fs_state_data(ft::Type{DirectFeature}, data, feature_name, feature_eltype, use_vectors) =
    (feature_type = ft,
	 read_table = get_read_table(ft, data),
     write_table = get_write_table(ft, data),
     input_feature = feature_name,
     input_eltype = feature_eltype,
     use_vectors = use_vectors,
     input_data = data.features,
     agg_column = __find_identity_feature(data.features, get_read_table(ft, data), data.link.dst[2]),    # column that contains unique values in refereced table
     mask_column = __find_identity_feature(data.features, get_write_table(ft, data), data.link.src[2]))  # corresponding column in referencing table

get_fs_state_data(ft::Type{ReverseFeature}, data, feature_name, feature_eltype, use_vectors) =
    (feature_type = ft,
	 read_table = get_read_table(ft, data),
     write_table = get_write_table(ft, data),
     input_feature = feature_name,
     input_eltype = feature_eltype,
     use_vectors = use_vectors,
     input_data = data.features,
     agg_column = __find_identity_feature(data.features, get_write_table(ft, data), data.link.dst[2]),  # column that contains unique values in refereced table
     mask_column = __find_identity_feature(data.features, get_read_table(ft, data), data.link.src[2]))  # corresponding column in referencing table


__find_identity_feature(features, table, column) = begin
    for (k, v) in features
        try
            # for `df.column` format in getter code  use `r"df\.(?<col>\w+)*\[idx\]"` in matcher below
            if match(r"\(df\[!, Symbol\(\"(?<col>\w+)\"\)\]\)\[idx\]", string(v.ast))[:col] == string(column) &&
               v.table == table &&
               v isa IdentityFeature
                    return k
            end
        catch
            nothing
        end
    end
end


# features is a Dict{Symbol, AbstractFeature}
function _build_features!(feature_type, data, kb; use_vectors=true, calculate=false, depth=0)
    n_input_features = 0
    n_max_features = 0
    n_calculated_features = 0
    tmpfeatures = Dict{Symbol, DeepFeatureSynthesis.AbstractFeature}()

    for (fname, feltype) in name_eltype_iterator(get_read_data(feature_type, data))
        fs_state = get_fs_state_data(feature_type, data, fname, feltype, use_vectors)
        kb_result = ControlFlow.kb_query(kb, fs_state)
        comps = [FeatureComponents(;r...) for r in kb_result]
        base_ast = build_generic_feature_ast(fs_state.feature_type; use_vectors=fs_state.use_vectors)

        # 3. Loop through feature components  make features
        skipped = 0
        fvalues = nothing
        n_input_features += 1
        for (i, comp) in enumerate(comps)
            n_max_features += 1
            f_ast = MacroTools.striplines(materialize(base_ast, comp))                             # • create actual feature AST

            # Calculate feature
            fvalues, new_feltype = calculate_feature(f_ast, fs_state; calculate=calculate)         # • executes feature function
            feature_name = Symbol(hash(fname)+hash(f_ast)+hash(feltype))  # • create feature name
            n_calculated_features +=1

            # Create feature object and push it in the structure of temporary features
            fobject = build_feature_struct(fs_state.feature_type, new_feltype, fs_state.write_table, f_ast, fvalues)
            if !haskey(data.features, feature_name) && !hasvalue(data.features, fobject) && !hasvalue(tmpfeatures, fobject)
                # A stricter check based exclusively on whether the feature values are already
                # present among the other features;
                if !hasvalue(data.features, fvalues) && !hasvalue(tmpfeatures, fvalues)
                    push!(tmpfeatures, feature_name=>fobject)
                else
                    @debug "OOPS: Values of the feature found"
                end
            else
                @debug "OOPS: Identical feature found"
            end
        end
    end
    println("* Added features: $n_input_features inputs, $(length(tmpfeatures))/$n_max_features outputs")
    merge!(data.features, tmpfeatures)
    nothing
end


function calculate_feature(f_ast, fs_state; calculate=true)
    _infer_feature_eltype(f_ast, data, ::Type{IdentityFeature}) = begin
		f = eval(f_ast)
		typeof(Base.invokelatest(f, data, 1))  # run for a first sample
	end

    _infer_feature_eltype(f_ast, data, other_feature_type) = begin
        f = eval(f_ast)
        Core.Compiler.return_type(f, (typeof(data), Int))
    end

    _length(::Type{<:AbstractFeature}, fs_state) = begin
        data_size(data::Dict, col) = length(data[col].values)
        data_size(data::DataFrame, col) = size(data, 1)
        return data_size(fs_state.input_data, fs_state.input_feature)
    end
    _length(::Type{DirectFeature}, fs_state) = length(fs_state.input_data[fs_state.mask_column].values)
    _length(::Type{ReverseFeature}, fs_state) = length(fs_state.input_data[fs_state.agg_column].values)

    if calculate
		f = eval(f_ast)
        try
            v = [Base.invokelatest(f, fs_state.input_data, idx) for idx in 1:_length(fs_state.feature_type, fs_state)]  # apply function over rows
            ft = eltype(v)
            print(".")
		    return v, eltype(v)
        catch e
            @warn "Failed to compute feature $e"
            return [nothing for _ in 1:_length(fs_state.feature_type, fs_state)], Nothing
        end
	else
		v = [nothing for _ in 1:_length(fs_state.feature_type, fs_state)]
		ft = try
				_infer_feature_eltype(f_ast, fs_state.input_data, fs_state.feature_type)
			catch e
				@warn "OOps could not infer type! $e"
				Nothing
			end
		return v, ft
	end
end


# Check if a value exists in the values of a structure
hasvalue(features::Dict, feature) = begin
    for (_, f) in features
        f == feature && return true
    end
    return false
end

hasvalue(features::Dict, values::AbstractVector) = any(f.values == values for (_, f) in features)

# Methods for building feature names
build_feature_name(f_ast, fs_state) = _build_feature_name(fs_state.feature_type, fs_state.input_feature, fs_state.input_eltype, f_ast)
_build_feature_name(::Type{IdentityFeature}, input_name, input_type, f_ast) = return Symbol(:I_, input_name, "_", hash(f_ast) + hash(input_name))
_build_feature_name(::Type{EntityFeature}, input_name, input_type, f_ast) = return Symbol(:E_, hash(f_ast) + hash(input_name))
_build_feature_name(::Type{DirectFeature}, input_name, input_type, f_ast) = return Symbol(:D_, hash(f_ast) + hash(input_name))
_build_feature_name(::Type{ReverseFeature}, input_name, input_type, f_ast) = return Symbol(:R_, hash(f_ast) + hash(input_name))

# Methods for obtaining feature structs
build_feature_struct(ft, T, table, f_ast, vals) = ft{T}(table, f_ast, vals)
