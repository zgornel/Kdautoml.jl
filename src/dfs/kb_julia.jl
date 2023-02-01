# ScalarGetter_factory(ft::Type{<:AbstractFeature}, fdata) = begin
#     fdata.input_data isa DataFrame && return [:((df, idx)->df.$(fdata.input_feature)[idx])]  # The scalar getter for any feature just gets a value in a column
#     fdata.input_data isa Dict && return [:((d, idx)->d[Symbol($(string(fdata.input_feature)))].values[idx])]
#     return []
# end
# 
# TensorGetter_factory(ft::Type{<:AbstractFeature}, fdata) = begin
#     fdata.input_data isa DataFrame && return [:(df->df.$(fdata.input_feature))]
#     fdata.input_data isa Dict  && return  [:(d->d[Symbol($(string(fdata.input_feature)))].values)]
#     return []
# end
# 
# 
# TensorConditioner_factory(ft::Type{<:IdentityFeature}, fdata) = [:(x->true)]
# TensorConditioner_factory(ft::Type{<:EntityFeature}, fdata) = [:(x->true)]
# # Because the conditions for Direct and Reverse features are not functions,
# # the exact argument name i.e. DATA_SYMBOL value used in the AST
# # has to be used in order to correctly pass the input feature into the condition
# TensorConditioner_factory(ft::Type{<:DirectFeature}, fdata) = [:($DATA_SYMBOL[Symbol($(string(fdata.agg_column)))].values .==
#                                                                  $DATA_SYMBOL[Symbol($(string(fdata.mask_column)))].values[idx])]
# TensorConditioner_factory(ft::Type{<:ReverseFeature}, fdata) = [:($DATA_SYMBOL[Symbol($(string(fdata.agg_column)))].values[idx] .==
#                                                                   $DATA_SYMBOL[Symbol($(string(fdata.mask_column)))].values)]
# 
# TensorReducer_factory(ft::Type{<:AbstractFeature}, fdata) = begin
#     fdata.input_eltype <: Number && return [:(x->$f(x)) for f in [sum, mean]]
#     fdata.input_eltype <: Union{Symbol, <:AbstractString} && return [:(x->$f(x)) for f in [length]]
#     return []
# end
# 
# ScalarReducer_factory(ft::Type{<:AbstractFeature}, fdata) = begin
#     (fdata.input_eltype <: Number && fdata.use_vectors == false) && return [:(x->x^2), :(x->x^3)]
#     (fdata.input_eltype <: Number && fdata.use_vectors == true) && return [:((x,y)->x/(y+eps())), :((x,y)->x/(y+eps())*100)]
#     (fdata.input_eltype <: Union{Symbol, <:AbstractString} && fdata.use_vectors == true) && return [:((x,y)->length(string(x))/(y+eps())), :((x,y)->length(string(x))/(y+eps())*100)]
#     (fdata.input_eltype <: Union{Symbol, <:AbstractString} && fdata.use_vectors==false) && return [:(x->length(string(x)))]
#     return []
# end
# 
# TensorFilter_factory(ft, fdata) = begin
#     return [:((c,x)->filter(c,x))]
# end
# 
# 
# FACTORIES = [ScalarReducer_factory, ScalarGetter_factory, TensorReducer_factory,
#             TensorConditioner_factory, TensorFilter_factory, TensorGetter_factory]
# 
