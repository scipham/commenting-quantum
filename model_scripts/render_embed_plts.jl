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
import Random
using CUDA

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




X_train, y_train, X_val, y_val = load_predict_ready_data("sentiment", 
                                                        "body", #"subtextLevel", "headline", "body"
                                                        "CosSim", #"SimCE-Head-Body",# "xlnet", #"SimCE-Head-Body", "CosSim" # xlnet
                                                       32, cpu, false)

#For engagement: body input gives simce advantage while headline gives cossim (label-based!) advantage
# for subtextLevel no large differences between zero and the other mdoes

#For sentiment: 
#Same story as for engagment for subtextLevel

#For sentiment: headline input goes very high (especially with simce) while body input is strong advantage for cossim
# ------ Training set ------

#pythonplot()

#Sum of std of each embedding feature:
@show std(X_train, dims=1)
@show mean(std(X_train, dims=1))

X_train = cat(X_train, X_val, dims=2)

pair_cos_sim = pairwise(cosine_dist, X_train, dims=2)
#Plot pair_cos_sim as heatmap
pair_cos_sim = 1.0 .- pair_cos_sim 

#################### If we import pre-computed pairwise cosine similarities ##########
using NPZ

using LinearAlgebra

function reconstruct_symmetric(v::Vector{T}) where T
    n = isqrt(2*length(v))
    if n * (n + 1) ÷ 2 != length(v)
        throw(ArgumentError("The length of the vector is not valid for a symmetric matrix."))
    end

    #Add diagonal:
    diag = 1.0

    M = Matrix{T}(undef, n, n)
    k = 1
    for i in 1:n
        for j in i:n
            if i == j
                M[i, j] = diag
            else
                M[i, j] = v[k]
                M[j, i] = v[k]
                k += 1
            end
        end
    end
    return M
end

# Example
v = npzread("S:\\Sync\\University\\2023_MRP_1\\MRP1_WorkDir\\model_scripts\\embed_checkpoints\\label_collection.npy")
M = reconstruct_symmetric(v)

@show size(M)


pair_cos_sim = M
####################
meas_label = "Similarity labels - Train set" #"Cosine Similarity - Trunc. Body" #"Similarity labels - Train set" #"Cosine Similarity - Trunc. Body"
prefixed_meas_label ="\n\nSimilarity labels - Train set" #"Cosine Similarity - Trunc. Body" "\n\nCosine Similarity - Trunc. Body" #"\n\nSimilarity labels - Train set" #"Cosine Similarity - Trunc. Body"
fnt_size = 12
train_hm = heatmap(pair_cos_sim, xticks=1:10:1000, yticks=1:10:1000, size=(810,600), margin=12Plots.mm,  colorbar_title=prefixed_meas_label, tickfontsize=fnt_size, colorbar_tickfontsize=fnt_size, colorbar_titlefontsize=fnt_size, dpi=300)

display(train_hm)
#Save plot to file
savefig(train_hm, "hm_pair_cos_sim_trunc_body.png")

#Pair_cos_sim histogram of values
histplt = histogram(reshape(pair_cos_sim,:), normalize=:probability, legend=false, xlabel=meas_label, size=(750,500), xguidefontsize=12, tickfontsize=12, dpi=300, bins=70)
xlims!(-1.05, 1.05)
ylims!(0, 0.1)
display(histplt)
#Save plot to file
savefig(histplt, "hist_pair_cos_sim_trunc_body.png")


