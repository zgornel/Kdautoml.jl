struct FSMTransitionError <: Exception end


### State types
abstract type AbstractState end

mutable struct NoData{T} <: AbstractState
    metadata::T  # inital state
end

mutable struct Data{T} <: AbstractState
    metadata::T  # data characteristics
end

mutable struct ModelableData{T} <: AbstractState
    metadata::T  # data characteristics
end

mutable struct Model{T} <: AbstractState
    metadata::T  # model characteristics
end

mutable struct End{T} <: AbstractState
    metadata::T  # error evaluation results
end

### Component types
abstract type AbstractComponent end

struct LoadData{T} <: AbstractComponent
    metadata::T  # information about the data
end

struct PreprocessData{T} <: AbstractComponent
    metadata::T  # how/what to preprocess form the data
end

struct SplitData{T} <: AbstractComponent
    metadata::T  # how to split the data
end

struct SplitHoldout{T} <: AbstractComponent
    metadata::T  # how to split the data
end

struct SplitCV{T} <: AbstractComponent
    metadata::T  # how to split the data
end

struct SplitStratifiedCV{T} <: AbstractComponent
    metadata::T  # how to split the data
end

struct FeatureOperation{T} <: AbstractComponent
    metadata::T  # a generic feature operation
end

struct FeatureGeneration{T} <: AbstractComponent
    metadata::T  # a generic feature operation
end

struct DimensionalityReduction{T} <: AbstractComponent
    metadata::T  # a generic feature operation
end

struct DFS{T} <: AbstractComponent
    metadata::T  # the deep feature synthesis feature generator
end

struct ProductFeatures{T} <: AbstractComponent
    metadata::T  # a generic feature operation
end

struct FeatureSelection{T} <: AbstractComponent
    metadata::T  # information on subsets
end

struct RandomFeatureSelection{T} <: AbstractComponent
    metadata::T  # information on subsets
end

struct DirectFeatureSelection{T} <: AbstractComponent
    metadata::T  # information on subsets
end

struct SelectModel{T} <: AbstractComponent
    metadata::T  # model selector
end

struct ModelData{T} <: AbstractComponent
    metadata::T  # model type, properties
end

struct EvalModel{T} <: AbstractComponent
    metadata::T  # how to evaluate the model
end

struct EvalClassifier{T} <: AbstractComponent
    metadata::T  # how to evaluate the model
end


### State transitions
_transition(::AbstractState, ::AbstractComponent; init=nothing) = throw(FSMTransitionError)

_transition(state::NoData, component::LoadData; init=nothing) = ControlFlow.Data(init)

_transition(state::ControlFlow.Data, component::PreprocessData; init=nothing) = ControlFlow.Data(init)

_transition(state::ControlFlow.Data, component::SelectModel; init=nothing) = ModelableData(init)

_transition(state::ControlFlow.Data, component::SplitData; init=nothing) = ControlFlow.Data(init)
_transition(state::ControlFlow.Data, component::SplitHoldout; init=nothing) = ControlFlow.Data(init)
_transition(state::ControlFlow.Data, component::SplitCV; init=nothing) = ControlFlow.Data(init)
_transition(state::ControlFlow.Data, component::SplitStratifiedCV; init=nothing) = ControlFlow.Data(init)

_transition(state::ControlFlow.Data, component::FeatureSelection; init=nothing) = ControlFlow.Data(init)
_transition(state::ControlFlow.Data, component::RandomFeatureSelection; init=nothing) = ControlFlow.Data(init)
_transition(state::ControlFlow.Data, component::DirectFeatureSelection; init=nothing) = ControlFlow.Data(init)

_transition(state::ControlFlow.Data, component::FeatureOperation; init=nothing) = ControlFlow.Data(init)
_transition(state::ControlFlow.Data, component::FeatureGeneration; init=nothing) = ControlFlow.Data(init)
_transition(state::ControlFlow.Data, component::DimensionalityReduction; init=nothing) = ControlFlow.Data(init)
_transition(state::ControlFlow.Data, component::ProductFeatures; init=nothing) = ControlFlow.Data(init)
_transition(state::ControlFlow.Data, component::DFS; init=nothing) = ControlFlow.Data(init)

_transition(state::ModelableData, component::FeatureSelection; init=nothing) = ModelableData(init)
_transition(state::ModelableData, component::RandomFeatureSelection; init=nothing) = ModelableData(init)
_transition(state::ModelableData, component::DirectFeatureSelection; init=nothing) = ModelableData(init)

_transition(state::ModelableData, component::FeatureOperation; init=nothing) = ModelableData(init)
_transition(state::ModelableData, component::FeatureGeneration; init=nothing) = ModelableData(init)
_transition(state::ModelableData, component::DimensionalityReduction; init=nothing) = ModelableData(init)
_transition(state::ModelableData, component::DFS; init=nothing) = ModelableData(init)
_transition(state::ModelableData, component::ProductFeatures; init=nothing) = ModelableData(init)


_transition(state::ModelableData, component::SplitData; init=nothing) = ModelableData(init)
_transition(state::ModelableData, component::SplitHoldout; init=nothing) = ModelableData(init)
_transition(state::ModelableData, component::SplitCV; init=nothing) = ModelableData(init)
_transition(state::ModelableData, component::SplitStratifiedCV; init=nothing) = ModelableData(init)

_transition(state::ModelableData, component::ModelData; init=nothing) = Model(init)

_transition(state::Model, component::EvalModel; init=nothing) = begin
	End(init)
end
