from parameters import args_parser
import torch
import numpy as np
import random
from model_indep import GraphHSA as module_arch
import pandas as pd
import pickle
from sklearn import metrics
import os
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
import warnings
import math
from collections import Counter

warnings.filterwarnings("ignore")


def obtain_metrics(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true).astype(int)

    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        print(
            "Warning: Only one class present in y_true.")
        acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred.round().astype(int))
        pre, rec, f1, auc, aupr = 0.0, 0.0, 0.0, 0.5, 0.5
        return [aupr, auc, f1, acc, pre, rec]

    acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred.round().astype(int))
    pre = metrics.precision_score(y_true=y_true, y_pred=y_pred.round().astype(int), zero_division=0)
    rec = metrics.recall_score(y_true=y_true, y_pred=y_pred.round().astype(int), zero_division=0)
    auc = metrics.roc_auc_score(y_true=y_true, y_score=y_pred)
    aupr = metrics.average_precision_score(y_true=y_true, y_score=y_pred)
    f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred.round().astype(int), zero_division=0)
    return [aupr, auc, f1, acc, pre, rec]


def _get_feed_dict(data, feature_index, drug_neighbor_set, symptom_neighbor_set, edge_index, n_hop, device):
    symptoms = data[:, feature_index['symptom']]
    drugs = data[:, feature_index['drug']]
    symptoms_neighbors, drugs_neighbors = [], []
    model_n_hop = n_hop

    if not drug_neighbor_set or not symptom_neighbor_set:
        raise ValueError("Drug or Symptom neighbor set is empty.")
    try:
        drug_set_depth = len(next(iter(drug_neighbor_set.values())))
        sym_set_depth = len(next(iter(symptom_neighbor_set.values())))
    except StopIteration:
        raise ValueError("Neighbor set appears valid but contains no actual entries.")
    except TypeError as e:
        print(f"Error accessing neighbor set depth: {e}");
        raise
    if model_n_hop > drug_set_depth: raise ValueError(
        f"Model requires n_hop={model_n_hop}, but drug_neighbor_set only has depth {drug_set_depth}.")
    if model_n_hop > sym_set_depth: raise ValueError(
        f"Model requires n_hop={model_n_hop}, but symptom_neighbor_set only has depth {sym_set_depth}.")

    for hop in range(model_n_hop):
        try:
            drugs_neighbors_batch = [drug_neighbor_set[d][hop] for d in drugs.numpy()]
            symptoms_neighbors_batch = [symptom_neighbor_set[s][hop] for s in symptoms.numpy()]
            drugs_neighbors.append(torch.LongTensor(drugs_neighbors_batch).to(device))
            symptoms_neighbors.append(torch.LongTensor(symptoms_neighbors_batch).to(device))
        except KeyError as e:
            print(f"Error: Missing key {e} in neighbor set.");
            raise
        except Exception as e:
            print(f"Unexpected error creating neighbor tensors at hop {hop}: {e}");
            raise

    drug_feat = torch.LongTensor(list(drug_neighbor_set.keys())).to(device)
    sym_feat = torch.LongTensor(list(symptom_neighbor_set.keys())).to(device)

    return symptoms.to(device), drugs.to(
        device), sym_feat, drug_feat, symptoms_neighbors, drugs_neighbors, torch.LongTensor(edge_index.T).to(device)


def evaluate_model(model, data_loader, feature_index, drug_neighbor_set, symptom_neighbor_set, edge_index, device):
    model.eval()
    all_preds = []
    all_targets = []

    if len(data_loader) == 0:
        print("Warning: evaluate_model received an empty data loader.")
        return [0.0] * 6

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            try:
                target_device = target.to(device)
                output, _, _, _, _, _, _ = model(*_get_feed_dict(data, feature_index, drug_neighbor_set,
                                                                 symptom_neighbor_set, edge_index, model.n_hop, device))
                pred = torch.sigmoid(output)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
            except Exception as e:
                print(f"Error during evaluation batch {batch_idx}: {e}")
                continue

    if not all_preds:
        print("Warning: Evaluation completed but no predictions were generated (all batches might have failed).")
        return [0.0] * 6

    all_targets_array = np.array(all_targets).squeeze()
    return obtain_metrics(np.array(all_preds), all_targets_array)


if __name__ == '__main__':
    args = args_parser()
    random.seed(args.seed);
    torch.manual_seed(args.seed);
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True;
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    print("Loading data...")
    try:
        with open('../data/node_num_dict.pickle', "rb") as f:
            node_num_dict = pickle.load(f)
        with open('../data/feature_index.pickle', "rb") as f:
            feature_index = pickle.load(f)
        with open('../data/symptom_neighbor_set.pickle', "rb") as f:
            symptom_neighbor_set = pickle.load(f)
        with open('../data/drug_neighbor_set.pickle', "rb") as f:
            drug_neighbor_set = pickle.load(f)
        with open('../data/node_map_dict.pickle', "rb") as f:
            node_map_dict = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading essential data file: {e}.");
        exit()
    except Exception as e:
        print(f"Error during pickle file loading: {e}");
        exit()

    dic_ids_sym = {};
    dic_ids_drug = {};
    dic_ids_pro = {}
    for k, v in node_map_dict.items():
        if isinstance(k, str) and 'C' in k: dic_ids_sym[v] = k
        if isinstance(k, str) and 'U' in k: dic_ids_drug[v] = k
        if isinstance(k, int): dic_ids_pro[v] = k

    try:
        full_train_loader = torch.load(args.data_dir + '/train_data_loader.pth', weights_only=False)
        test_data_loader = torch.load(args.data_dir + '/test_data_loader.pth', weights_only=False)
        edge_index_df = pd.read_csv(args.data_dir + '/train_edge_index.txt', sep=',')
        edge_index = edge_index_df.values.astype(int)
    except FileNotFoundError as e:
        print(f"Error loading data loader or edge index file: {e}.");
        exit()
    except Exception as e:
        print(f"Error during data loader/edge index loading: {e}");
        exit()

    try:
        edge_index = edge_index[edge_index[:, -1] == 1][:, :-1]
        edge_index[:, 1] = edge_index[:, 1] + node_num_dict.get('drug', 0)
    except IndexError as e:
        print(f"Error processing edge_index array: {e}. Shape incorrect.");
        exit()
    except KeyError as e:
        print(f"Error processing edge_index: Missing key {e} in node_num_dict.");
        exit()

    print("Initializing and loading the pre-trained model...")
    try:
        model = module_arch(protein_num=node_num_dict.get('protein', 0),
                            symptom_num=node_num_dict.get('symptom', 0),
                            drug_num=node_num_dict.get('drug', 0),
                            emb_dim=args.emb_dim, n_hop=args.n_hop_model,
                            l1_decay=args.l1_decay).to(device)
    except KeyError as e:
        print(f"Error initializing model: Missing key {e} in node_num_dict.");
        exit()

    model_path = os.path.join(args.save_dir, 'best_model_indep.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model file not found at {model_path}. Please run main_indep.py first.")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=True)
        print(f"Successfully loaded pre-trained model from {model_path}")
    except Exception as e:
        print(f"Error loading pre-trained model weights: {e}")
        exit()

    print("\nEvaluating Pre-trained Model on the Test Set (Before Fine-tuning)")
    metrics_test_initial = evaluate_model(model, test_data_loader, feature_index, drug_neighbor_set,
                                          symptom_neighbor_set, edge_index, device)
    aupr_test_initial, auc_test_initial = metrics_test_initial[0], metrics_test_initial[1]
    print(f"Initial Test Performance: AUPR={aupr_test_initial:.4f}, AUC={auc_test_initial:.4f}")

    print("\nStarting Causal Invariant Fine-tuning...")

    FINETUNE_EPOCHS = 60
    LEARNING_RATE = 5e-6
    IRM_LAMBDA = 1e-5
    PATIENCE = 30

    BATCH_SIZE = full_train_loader.batch_size if full_train_loader.batch_size is not None else 64
    VALIDATION_SPLIT = 0.1
    not_improved_count = 0

    full_train_dataset = full_train_loader.dataset
    if len(full_train_dataset) < 10:
        print("Warning: Full training dataset too small to split for validation. Skipping fine-tuning.")
        final_model_path = model_path
    else:
        val_size = max(1, int(len(full_train_dataset) * VALIDATION_SPLIT))
        train_size = len(full_train_dataset) - val_size
        if train_size <= 0:
            print("Warning: Not enough samples for training after split. Skipping fine-tuning.")
            final_model_path = model_path
        else:
            train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size],
                                                    generator=torch.Generator().manual_seed(args.seed))
            print(f"Split training data into {len(train_subset)} for fine-tuning and {len(val_subset)} for validation.")
            finetune_valid_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

            if len(train_subset) < 2:
                print("Warning: Not enough fine-tuning samples to create 2 IRM environments. Skipping fine-tuning.")
                final_model_path = model_path
            else:
                print("Generating environments based on Node Degree (Popularity) for Balanced Split...")

                temp_loader = DataLoader(train_subset, batch_size=len(train_subset), shuffle=False)
                all_data_tensor, _ = next(iter(temp_loader))

                train_drug_indices = all_data_tensor[:, feature_index['drug']].cpu().numpy()
                train_symptom_indices = all_data_tensor[:, feature_index['symptom']].cpu().numpy()

                drug_counts = Counter(train_drug_indices)
                sym_counts = Counter(train_symptom_indices)

                sample_scores = []
                for i in range(len(train_drug_indices)):
                    d_id = train_drug_indices[i]
                    s_id = train_symptom_indices[i]
                    score = drug_counts[d_id] + sym_counts[s_id]
                    sample_scores.append(score)

                sample_scores = np.array(sample_scores)
                sorted_indices_local = np.argsort(sample_scores)
                split_point = len(sorted_indices_local) // 2

                env1_indices = sorted_indices_local[:split_point].tolist()
                env2_indices = sorted_indices_local[split_point:].tolist()

                print(
                    f"Degree-based Split Successful (Median Cut) - Env1 (Tail): {len(env1_indices)}, Env2 (Head): {len(env2_indices)}")


                if not env1_indices or not env2_indices:
                    print("Warning: Could not create two non-empty IRM environments. Skipping fine-tuning.")
                    final_model_path = model_path
                else:
                    env1_loader = DataLoader(Subset(train_subset, env1_indices), batch_size=BATCH_SIZE, shuffle=True)
                    env2_loader = DataLoader(Subset(train_subset, env2_indices), batch_size=BATCH_SIZE, shuffle=True)

                    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                    loss_fn = torch.nn.BCEWithLogitsLoss()

                    metrics_valid_baseline = evaluate_model(model, finetune_valid_loader, feature_index,
                                                            drug_neighbor_set,
                                                            symptom_neighbor_set, edge_index, device)
                    best_valid_metric = metrics_valid_baseline[0] + metrics_valid_baseline[1]
                    print(
                        f"Baseline performance on new validation set: AUPR={metrics_valid_baseline[0]:.4f}, AUC={metrics_valid_baseline[1]:.4f}")

                    fine_tuning_successful = False

                    optimized_model_path = os.path.join(args.save_dir, 'best_model_DELTA_GA_optimized.pth')
                    force_save_path = os.path.join(args.save_dir, 'best_model_DELTA_GA_forced.pth')

                    if os.path.exists(optimized_model_path): os.remove(optimized_model_path)

                    for epoch in range(1, FINETUNE_EPOCHS + 1):
                        model.train()
                        iter1, iter2 = iter(env1_loader), iter(env2_loader)
                        epoch_total_loss = 0.0
                        batch_count = 0

                        while True:
                            try:
                                data1, target1 = next(iter1)
                                data2, target2 = next(iter2)
                                target1 = target1.float().squeeze(-1).to(device)
                                target2 = target2.float().squeeze(-1).to(device)
                                optimizer.zero_grad()

                                logits1, _, _, _, _, _, _ = model(
                                    *_get_feed_dict(data1, feature_index, drug_neighbor_set, symptom_neighbor_set,
                                                    edge_index, model.n_hop, device))
                                loss1 = loss_fn(logits1, target1)
                                logits2, _, _, _, _, _, _ = model(
                                    *_get_feed_dict(data2, feature_index, drug_neighbor_set, symptom_neighbor_set,
                                                    edge_index, model.n_hop, device))
                                loss2 = loss_fn(logits2, target2)

                                dummy_w = torch.tensor(1.0, device=device).requires_grad_()
                                grad1 = \
                                torch.autograd.grad(loss_fn(logits1 * dummy_w, target1), [dummy_w], create_graph=True)[
                                    0]
                                grad2 = \
                                torch.autograd.grad(loss_fn(logits2 * dummy_w, target2), [dummy_w], create_graph=True)[
                                    0]
                                penalty = grad1.pow(2) + grad2.pow(2)

                                total_loss = loss1 + loss2 + IRM_LAMBDA * penalty
                                total_loss.backward()
                                optimizer.step()
                                epoch_total_loss += total_loss.item()
                                batch_count += 1
                            except StopIteration:
                                break
                            except Exception as e:
                                print(f"Error during fine-tuning batch in epoch {epoch}: {e}")
                                continue

                        if batch_count == 0:
                            continue

                        metrics_valid = evaluate_model(model, finetune_valid_loader, feature_index, drug_neighbor_set,
                                                       symptom_neighbor_set, edge_index, device)
                        aupr_valid, auc_valid = metrics_valid[0], metrics_valid[1]
                        current_valid_metric = aupr_valid + auc_valid

                        avg_epoch_loss = epoch_total_loss / batch_count
                        print(
                            f"Epoch {epoch}/{FINETUNE_EPOCHS} | Avg Loss: {avg_epoch_loss:.4f} | Val AUPR: {aupr_valid:.4f} | Val AUC: {auc_valid:.4f}")


                        if current_valid_metric > best_valid_metric:
                            best_valid_metric = current_valid_metric
                            not_improved_count = 0
                            torch.save(model.state_dict(), optimized_model_path)
                            print(f"  -> Val Improved! Saving OPTIMIZED model.")
                        else:
                            not_improved_count += 1
                            print(f"  -> Val not improved. Count: {not_improved_count}/{PATIENCE}")

                        torch.save(model.state_dict(), force_save_path)

                        if not_improved_count >= PATIENCE:
                            print(f"\nEarly stopping at Epoch {epoch}.")
                            break

                    if os.path.exists(optimized_model_path):
                        final_model_path = optimized_model_path
                        print(f"\n[Success] Found Optimized Model (Best Val Performance). Using it for testing.")
                    elif os.path.exists(force_save_path):
                        final_model_path = force_save_path
                        print(f"\n[Warning] No Val improvement. Using Forced Model (Last Epoch) for testing.")
                    else:
                        final_model_path = model_path
                        print(f"[Failure] Fine-tuning failed completely. Reverting to pre-trained.")

    print("\n--- Starting final evaluation on the unseen test set (Using selected model) ---")

    if not os.path.exists(final_model_path):
        print(f"Error: Final model path '{final_model_path}' does not exist. Cannot evaluate.")
        exit()

    print(f"Loading final model for evaluation from: {final_model_path}")
    try:
        model.load_state_dict(torch.load(final_model_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"Error loading final model weights: {e}")
        exit()
    model.eval()

    test_preds, test_targets, test_embeds = [], [], []
    all_att_syms_batches, all_att_drugs_batches = [], []
    symptoms_neighborss_batches, drugs_neighborss_batches = [], []
    symptomss, drugss = [], []

    if len(test_data_loader) == 0:
        print("Warning: Test data loader is empty. Cannot perform final evaluation.")
        results_test_final = [0.0] * 6
    else:
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_data_loader):
                try:
                    target = target.to(device)
                    feed_dict_args = _get_feed_dict(data, feature_index, drug_neighbor_set, symptom_neighbor_set,
                                                    edge_index, model.n_hop, device)

                    output, embeds, _, _, _, att_sym_list, att_drug_list = model(*feed_dict_args)

                    pred = torch.sigmoid(output)
                    test_preds.extend(pred.cpu().numpy())
                    test_targets.extend(target.cpu().numpy())
                    test_embeds.extend(embeds.cpu().numpy())

                    if att_sym_list: all_att_syms_batches.append(att_sym_list[0].cpu().numpy())
                    if att_drug_list: all_att_drugs_batches.append(att_drug_list[0].cpu().numpy())

                    symptoms_batch = data[:, feature_index['symptom']]
                    drugs_batch = data[:, feature_index['drug']]
                    symptomss.extend(symptoms_batch.cpu().numpy())
                    drugss.extend(drugs_batch.cpu().numpy())

                    original_s_neighbors = feed_dict_args[4]
                    original_d_neighbors = feed_dict_args[5]

                    if model.n_hop > 0:
                        if original_s_neighbors: symptoms_neighborss_batches.append(
                            original_s_neighbors[0].cpu().numpy())
                        if original_d_neighbors: drugs_neighborss_batches.append(original_d_neighbors[0].cpu().numpy())


                except Exception as e:
                    print(f"Error during final evaluation batch {batch_idx}: {e}")
                    continue

        if not test_preds:
            print("Final evaluation failed. No predictions were generated.")
            results_test_final = [0.0] * 6
        else:
            test_targets_array = np.array(test_targets).squeeze()
            results_test_final = obtain_metrics(np.array(test_preds), test_targets_array)

    print(f"\n==> Final Test Performance: AUPR={results_test_final[0]:.4f}, AUC={results_test_final[1]:.4f} <==")
    print(f"\n--- Saving final results and attention/neighbor files ---")
    try:
        preds_out_file = args.save_dir + '/preds_indep_DET_GA.txt'
        embeds_out_file = args.save_dir + '/embeds_indep_DET_GA.txt'
        results_out_file = args.save_dir + '/results_indep_DET_GA.txt'
        att_sym_out_file = args.save_dir + '/att_symptom_DET_GA.txt'
        att_drug_out_file = args.save_dir + '/att_drug_DET_GA.txt'
        sym_neighbors_out_file = args.save_dir + '/symptoms_neighbors_DET_GA.txt'
        drug_neighbors_out_file = args.save_dir + '/drugs_neighbors_DET_GA.txt'
        sym_index_out_file = args.save_dir + '/symptoms_index_DET_GA.txt'
        drug_index_out_file = args.save_dir + '/drugs_index_DET_GA.txt'
        sym_neighbors_name_out_file = args.save_dir + '/symptoms_neighbors_name_DET_GA.txt'
        drug_neighbors_name_out_file = args.save_dir + '/drugs_neighbors_name_DET_GA.txt'
        sym_name_out_file = args.save_dir + '/symptoms_name_DET_GA.txt'
        drug_name_out_file = args.save_dir + '/drugs_name_DET_GA.txt'

        np.savetxt(preds_out_file,
                   np.hstack((np.array(test_preds).reshape((-1, 1)), np.array(test_targets).reshape((-1, 1)))))
        np.savetxt(embeds_out_file, np.array(test_embeds))
        np.savetxt(results_out_file, np.array(results_test_final))

        if all_att_syms_batches:
            all_att_syms = np.concatenate(all_att_syms_batches, axis=0)
            np.savetxt(att_sym_out_file, all_att_syms)
            print(f"Saved {os.path.basename(att_sym_out_file)} with shape: {all_att_syms.shape}")
        else:
            print(f"No data to save for {os.path.basename(att_sym_out_file)}")

        if all_att_drugs_batches:
            all_att_drugs = np.concatenate(all_att_drugs_batches, axis=0)
            np.savetxt(att_drug_out_file, all_att_drugs)
            print(f"Saved {os.path.basename(att_drug_out_file)} with shape: {all_att_drugs.shape}")
        else:
            print(f"No data to save for {os.path.basename(att_drug_out_file)}")

        if symptoms_neighborss_batches:
            all_symptoms_neighbors = np.concatenate(symptoms_neighborss_batches, axis=0)
            np.savetxt(sym_neighbors_out_file, all_symptoms_neighbors)
            print(f"Saved {os.path.basename(sym_neighbors_out_file)} with shape: {all_symptoms_neighbors.shape}")
        else:
            print(f"No data to save for {os.path.basename(sym_neighbors_out_file)}")

        if drugs_neighborss_batches:
            all_drugs_neighbors = np.concatenate(drugs_neighborss_batches, axis=0)
            np.savetxt(drug_neighbors_out_file, all_drugs_neighbors)
            print(f"Saved {os.path.basename(drug_neighbors_out_file)} with shape: {all_drugs_neighbors.shape}")
        else:
            print(f"No data to save for {os.path.basename(drug_neighbors_out_file)}")

        if symptomss:
            np.savetxt(sym_index_out_file, np.array(symptomss))
        else:
            print(f"No data to save for {os.path.basename(sym_index_out_file)}")

        if drugss:
            np.savetxt(drug_index_out_file, np.array(drugss))
        else:
            print(f"No data to save for {os.path.basename(drug_index_out_file)}")

        print("\n--- Converting IDs to names using in-memory data ---")
        all_symptoms_arr = np.array(symptomss)
        all_drugs_arr = np.array(drugss)
        if all_drugs_arr.size > 0:
            symptoms_neighbors2, drugs_neighbors2, symptoms2, drugs2 = [], [], [], []
            num_samples = len(all_drugs_arr)

            for i in range(num_samples):
                drug_id = int(all_drugs_arr[i])
                sym_id = int(all_symptoms_arr[i])
                drugs2.append(dic_ids_drug.get(drug_id, f"UNKNOWN_DRUG_{drug_id}"))
                symptoms2.append(dic_ids_sym.get(sym_id, f"UNKNOWN_SYMP_{sym_id}"))

                if 'all_symptoms_neighbors' in locals() and i < all_symptoms_neighbors.shape[0]:
                    s_neigh_names = [dic_ids_pro.get(int(k), f"UNKNOWN_PRO_{int(k)}") for k in all_symptoms_neighbors[i]
                                     if not np.isnan(k)]
                    symptoms_neighbors2.append(s_neigh_names)
                else:
                    symptoms_neighbors2.append([])

                if 'all_drugs_neighbors' in locals() and i < all_drugs_neighbors.shape[0]:
                    d_neigh_names = [dic_ids_pro.get(int(k), f"UNKNOWN_PRO_{int(k)}") for k in all_drugs_neighbors[i] if
                                     not np.isnan(k)]
                    drugs_neighbors2.append(d_neigh_names)
                else:
                    drugs_neighbors2.append([])


            def save_list_of_lists(filename, data):
                with open(filename, 'w', encoding='utf-8') as f:
                    for line_list in data:
                        f.write(' '.join(map(str, line_list)) + '\n')


            save_list_of_lists(sym_neighbors_name_out_file, symptoms_neighbors2)
            save_list_of_lists(drug_neighbors_name_out_file, drugs_neighbors2)
            with open(sym_name_out_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(symptoms2))
            with open(drug_name_out_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(drugs2))
            print("Name files saved.")
        else:
            print("No index data available to generate name files.")

    except Exception as e:
        print(f"Error saving results files or converting names: {e}")

    print(f"\nDetailed results saved to: {args.save_dir}")
