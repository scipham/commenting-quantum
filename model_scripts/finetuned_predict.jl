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
using HyperTuning

function load_hdf_embed_checkpoint(hdf_filepath)
    h5open(hdf_filepath, "r") do file
        article_ids = read(file["article_ids"])
        targets = read(file["targets"])
        #global text_subtext_embeds = read(file["text_subtext_embeddings"])
        #global subtext_embeds_attn_masks = read(file["subtext_embeddings_attention_masks"])
        text_embeds = read(file["text_embeddings"])

        @show typeof(article_ids), size(article_ids)
        @show typeof(targets), size(targets)
        #@show typeof(text_subtext_embeds), size(text_subtext_embeds)
        #@show typeof(subtext_embeds_attn_masks), size(subtext_embeds_attn_masks)
        @show typeof(text_embeds), size(text_embeds)

        return article_ids, targets, text_embeds
    end
end

function load_predict_ready_data(predict_feature, input_type, embed_type, batch_size, device)
    #predict_feature, input_type, embed_type
    base_folder = "./model_scripts/embed_checkpoints/" * predict_feature * "/" * input_type * "/"
    predict_feature = predict_feature == "sentiment" ? "_senti" : ""
    if input_type == "headline"
        train_data_filepath = base_folder * "train$(predict_feature)_distilroberta_$(embed_type)_embeddings.hdf5"
    elseif input_type == "body"
        train_data_filepath = base_folder * "train$(predict_feature)_distilroberta_body_$(embed_type)_embeddings.hdf5"
    end
    article_ids, targets, text_embeds = load_hdf_embed_checkpoint(train_data_filepath)    
    embed_size = size(text_embeds, 1)
    X_train = Float32.(text_embeds) |> device
    y_train = Float32.(targets) |> device

    test_data_filepath = replace(train_data_filepath, "train" => "test")
    article_ids, targets, text_embeds = load_hdf_embed_checkpoint(test_data_filepath)
    X_val = Float32.(text_embeds) |> device
    y_val = Float32.(targets) |> device

    return DataLoader((X_train, y_train), batchsize=batch_size, shuffle=true), DataLoader((X_val, y_val), batchsize=batch_size, shuffle=false), embed_size
end


# Define the loss function (mean squared error for regression)

rmse(y_model, y_true) = Flux.sqrt(Flux.mse(y_model, y_true))
mse_loss(y_model, y_true) = Flux.mse(y_model, y_true)


# L2 Regularization term
function regularization(model)
  return sum(x -> sum(abs2, x), Flux.params(model))
end

function dispersion(y_model, y_true)
    return -mean(abs.(y_model .- y_true) * abs.(y_model' .- y_model))
end

function eval_error(loader, model, device)
    cum_rmse = 0
    ntot = 0
    for (x, y) in loader
        ŷ = model(x)
        cum_rmse += rmse(ŷ, y)
        ntot += size(x)[end]
    end
    return cum_rmse/ntot
end


function objective(trial)
    @show trial

    # fix seed for the RNG
    seed = get_seed(trial)
    Random.seed!(seed)

    # activate CUDA if possible
    device = CUDA.functional() ? gpu : cpu
    # Create test and train dataloaders
    batch_size = 16
    predict_feature = "engagement" #, "sentiment"
    input_type = "headline" #, "body"
    embed_type = "zero" #, "SimCE-Head-Body", "CosSim"
    train_loader, val_loader, embed_size = load_predict_ready_data(predict_feature, input_type, embed_type, batch_size, device)

    # get suggested hyperparameters
    activation = relu
    @suggest n_dense in trial
    @suggest hidden_nodes in trial

    # Create the model with dense layers (fully connected)
    layers = []
    n_input = embed_size #Becomes hidden size after first dense layer
    for n in hidden_nodes[1:n_dense]
        push!(layers, Dense(n_input, n, activation))
        n_input = n
    end
    push!(layers, Dense(n_input, 1))
    model = Chain(layers) |> device
    # model parameters
    ps = Flux.params(model)  

    # hyperparameters for the optimizer
    @suggest η in trial
    @suggest λ in trial

    # Instantiate the optimizer
    opt = AdamW(η)

    mse_loss_with_regular(y_model, y_true) = mse_loss(y_model, y_true) + λ * regularization(model) + dispersion(y_model, y_true)
    loss_function = mse_loss_with_regular

    # Training
    @suggest n_epochs in trial
    #number of training epochs
    
    reg_error = 1.0
    train_losses = Float64[]
    val_losses = Float64[]
    @showprogress 5 "Epochs ... " for epoch in 1:n_epochs
        for epoch in 1:n_epochs
            total_train_loss = 0.0
            count = 0 
            for (x, y) in train_loader
                gs = gradient(Flux.params(model)) do
                    ŷ = model(x)
                    loss = loss_function(ŷ, y)
                    total_train_loss += loss
                    count += 1
                    return loss
                end
                Flux.Optimise.update!(opt, ps, gs)
            end
            push!(train_losses, total_train_loss / count)
            
            # Validation
            total_val_loss = 0.0
            count = 0
            for (x, y) in val_loader
                ŷ = model(x)
                loss = loss_function(ŷ, y)
                total_val_loss += loss
                count += 1
            end
            push!(val_losses, total_val_loss / count)


            # Compute intermediate accuracy error
            reg_error = eval_error(val_loader, model, device)
            # report value to pruner
            report_value!(trial, reg_error)
            # check if pruning is necessary
            should_prune(trial) && (return)
        end
    end
    # if accuracy is over 90%, then trials is considered as feasible
    reg_error < 0.001 && report_success!(trial)
    # return objective function value
    return reg_error
end


function main() 
    # maximum and minimum number of dense layers
    MIN_DENSE = 2
    MAX_DENSE = 5

    scenario = Scenario(### hyperparameters
                        # data filepath
                        # ["headline", "body"]
                        # ["zero", "CosSim", "SimCE-Head-Body"]
                        # learning rates
                        η = (1e-9..1e-2),
                        # regularization parameter
                        λ = (0.001..0.1),
                        # activation functions
                        #activation = [leakyrelu, relu],
                        # loss functions
                        #loss = [mse, logitcrossentropy],
                        # number of dense layers
                        n_dense = MIN_DENSE:MAX_DENSE,
                        # number of neurons for each dense layer
                        hidden_nodes = Bounds(fill(4, MAX_DENSE), fill(128, MAX_DENSE)),
                        # number of epochs
                        n_epochs = (10..100),
                        ### Common settings
                        pruner= MedianPruner(start_after = 5#=trials=#, prune_after = 10#=epochs=#),
                        verbose = true, # show the log
                        max_trials = 4, # maximum number of hyperparameters computed
                    )


    display(scenario)

    # minimize accuracy error
    HyperTuning.optimize(objective, scenario) 
end

main()