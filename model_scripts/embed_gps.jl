using FileIO
using CSV
using DataFrames
using Statistics
using StatsPlots
using BSON: @save, @load #BSON is the Julia equivalent of pickle
using HDF5
using Distances

using CUDA 
using Flux
using ProgressMeter

using Flux, Statistics
using Flux.Data: DataLoader
using Flux:flatten
using Flux.Losses: mse
using Base: @kwdef
using CUDA
using Metrics: r2_score, adjusted_r2_score

function load_hdf_embed_checkpoint(hdf_filepath)
    h5open(hdf_filepath, "r") do file
        article_ids = read(file["article_ids"])
        targets = read(file["targets"])
        #global text_subtext_embeds = read(file["text_subtext_embeddings"])
        #global subtext_embeds_attn_masks = read(file["subtext_embeddings_attention_masks"])
        text_embeds = read(file["text_embeddings"])

        #@show typeof(article_ids), size(article_ids)
        #@show typeof(targets), size(targets)
        #@show typeof(text_subtext_embeds), size(text_subtext_embeds)
        #@show typeof(subtext_embeds_attn_masks), size(subtext_embeds_attn_masks)
        #@show typeof(text_embeds), size(text_embeds)

        return article_ids, targets, text_embeds
    end
end

function load_predict_ready_data(predict_feature, input_type, embed_type, batch_size, device, return_loaders)
    #predict_feature, input_type, embed_type
    #base_folder = "./embed_checkpoints/" * predict_feature * "/" * input_type * "/"
    base_folder = "S:\\Sync\\University\\2023_MRP_1\\MRP1_WorkDir\\model_scripts\\embed_checkpoints\\" * predict_feature * "\\" * input_type * "\\"
    predict_feature = predict_feature == "sentiment" ? "_senti" : ""
    #predict_feature = input_type == "subtextLevel" ? predict_feature * "_subtextLevel" : predict_feature
    exchange_model = embed_type == "xlnet" ? true : false
    embed_type = embed_type == "xlnet" ? "zero" : embed_type
    if input_type == "headline"
        train_data_filepath = base_folder * "train$(predict_feature)_distilroberta_$(embed_type)_embeddings.hdf5"
    elseif input_type == "body"
        train_data_filepath = base_folder * "train$(predict_feature)_distilroberta_body_$(embed_type)_embeddings.hdf5"
    elseif input_type == "subtextLevel"
        train_data_filepath = base_folder * "train$(predict_feature)_subtextLevel_distilroberta_$(embed_type)_embeddings.hdf5"
    end

    if exchange_model
        train_data_filepath = replace(train_data_filepath, "distilroberta" => "xlnet")
    end


    #train_data_filepath = "./embed_checkpoints/engagement/train_roberta_zero_embeddings.hdf5"

    article_ids, targets, text_embeds = load_hdf_embed_checkpoint(train_data_filepath)    
    embed_size = size(text_embeds, 1)
    X_train = Float32.(text_embeds) |> device
    y_train = Float32.(targets) |> device

    test_data_filepath = replace(train_data_filepath, "train" => "test")
    article_ids, targets, text_embeds = load_hdf_embed_checkpoint(test_data_filepath)
    X_val = Float32.(text_embeds) |> device
    y_val = Float32.(targets) |> device

    #Normalize arrays such that each column has a mean of 0 and a standard deviation of 1
    X_train = (X_train .- mean(X_train, dims=1)) ./ std(X_train, dims=1)
    X_val = (X_val .- mean(X_val, dims=1)) ./ std(X_val, dims=1)

    if return_loaders
        return DataLoader((X_train, y_train), batchsize=batch_size, shuffle=true), DataLoader((X_val, y_val), batchsize=8, shuffle=false), embed_size
    else
        return X_train, y_train, X_val, y_val, embed_size
    end
end


using Statistics, GLM

function variance_inflation_factor(X)
    vif_dict = Dict()
    for i in 1:size(X, 1)
        y = convert(Vector{Float64}, X[i, :])
        X_rest = X[1:end .!= i, :]

        # Add an intercept (constant term) to X_rest
        X_rest = vcat(ones(1, size(X_rest, 2)), X_rest)

        # Perform linear regression of y on X_rest
        model = lm(X_rest', y)

        r2 = GLM.r2(model)
        vif = 1 / (1 - r2)

        vif_dict[i] = vif
    end
    return vif_dict
end




X_train, y_train, X_val, y_val = load_predict_ready_data("engagement", 
                                                        "body", #"subtextLevel", "headline", "body"
                                                        "zero", #"SimCE-Head-Body",# "xlnet", #"SimCE-Head-Body", "CosSim" # xlnet
                                                        32, cpu, false)
y_train = y_train[1, :]
y_val = y_val[1, :] 
#Convert data fully to Float64
X_train = Float64.(X_train)
y_train = Float64.(y_train)
X_val = Float64.(X_val)
y_val = Float64.(y_val)

@show size(X_train), size(y_train)

using GaussianProcesses
using Random
using MLUtils

Random.seed!(230)

mZero = MeanConst(mean(y_train))                   #Zero mean function
#mZero = MeanLin(fill(1.0, size(X_train, 1)))                   #Zero mean function
#degree = 2
#mZero = MeanPoly(fill(0.5, size(X_train, 1),4))                   #Zero mean function

#kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)
# Only Matern 1/2, 3/2 and 5/2 are implementable

kern = Matern(5/2,0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

logObsNoise = -0.1                        # log standard deviation of observation noise (this is optional)
#gp = GP(X_train,y_train,mZero,kern,logObsNoise)
#
#likelihood = GaussLik(logObsNoise)         # Gaussian likelihood with log standard deviation of logObsNoise
bin_threshold = mean(y_train) 
#y_train = convert(Vector{Bool}, y_train .> bin_threshold) #Convert to binary
#y_val = convert(Vector{Bool}, y_val .> bin_threshold) #Convert to binary

gp = GP(X_train,y_train,mZero,kern,logObsNoise)
using Optim
for i in 1:2
    @show i
    optimize!(gp; method=GradientDescent())   # Optimise the hyperparameters
end
#optimize!(gp; method=ConjugateGradient())   # Optimise the hyperparameters

μ, σ² = predict_y(gp, X_val)
@show μ
@show sqrt.(σ²)

#Print RMSE and Adjusted r2
@show rmse = sqrt(mean((y_val .- μ).^2))
@show r2 = r2_score(y_val, μ)
@show adjusted_r2 = adjusted_r2_score(y_val, μ, size(X_val, 1))

pythonplot() 

#plt = scatter(X_val[1, :], μ, xlabel="Hidden dimension 1", ylabel="Predicted Engagement", title="Predicted Engagement")
#display(plt)

plt = scatter(y_val, μ, xlabel="True Engagement", ylabel="Predicted Engagement", title="Validation set (25 articles)", legend=false, dpi=300)
plot!(plt, [0, 1], [0, 1], color=:black)
ylims!(0, 1)
xlims!(0, 1)
savefig(plt, "engagement_prediction_true.png")
display(plt)

#Plot residual histogram
plt = histogram(abs.(y_val .- μ), xlabel="Absolute prediction residual", ylabel="Frequency", legend=false, dpi=300, bins=0:0.01:0.3,  margin=2Plots.mm)
xticks!(0:0.02:0.3)
savefig(plt, "engagement_prediction_residual.png")

