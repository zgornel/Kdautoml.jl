using DelimitedFiles, Random
using DataFrames, Debugger
using Kdautoml
df = DataFrame(Matrix(readdlm("../../data/datasets/xor_10x10.tsv", header=true)[1])[:,1:2], :auto)
kbpath = joinpath("../../data/knowledge/feature_synthesis_xor_usecase.toml")
kb = Kdautoml.kb_load(kbpath)

#features = Dict{Symbol, Kdautoml.DeepFeatureSynthesis.AbstractFeature}()
@enter ff=Kdautoml.DeepFeatureSynthesis.deep_feature_synthesis(df, 3; kb=kb, calculate=true);
#ff=kb["resources"]["code"]["components"]["ScalarReducer_from_Tensor_Number"]["code"] |>strip|>Meta.parse |> eval |> f->Base.invokelatest(f, ((agg_column=:x,mask_column=:y)))
#eval(first(ff))(df,2)
