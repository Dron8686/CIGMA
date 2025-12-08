from parameters import args_parser
import torch
import numpy as np
import random
from model_cv import GraphHSA as module_arch
import pandas as pd
import pickle
from sklearn import metrics
import os
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
import warnings
import math

warnings.filterwarnings("ignore")


def obtain_metrics(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true).astype(int)

    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        print(
            "Warning: Only one class present in y_true. AUC/AUPR/Precision/Recall/F1 metrics may be undefined or misleading.")
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
                output, _, _, _, _ = model(*_get_feed_dict(data, feature_index, drug_neighbor_set,
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
    # --- 再现性设置 ---
    random.seed(args.seed);
    torch.manual_seed(args.seed);
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True;
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    if not os.path.exists(args.save_dir):
        print(f"Error: Save directory '{args.save_dir}' not found. Please run main_cv.py first.")
        exit()

    print("Loading common data files...")
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

    try:
        test_data_loader = torch.load(args.data_dir + '/test_data_loader.pth', weights_only=False)
        print(f"Successfully loaded global test set from {args.data_dir}/test_data_loader.pth")
    except FileNotFoundError as e:
        print(f"Error loading global test set: {e}. Did save_train_val_test_data.py run?");
        exit()

    print(f"\nStarting Causal Invariant Fine-tuning for 5 Folds in {args.save_dir}...")

    FINETUNE_EPOCHS = 50
    LEARNING_RATE = 0.0001
    IRM_LAMBDA = 0.5
    VALIDATION_SPLIT = 0.1
    PATIENCE = 10

    final_model_paths_for_folds = []

    try:
        model = module_arch(protein_num=node_num_dict.get('protein', 0),
                            symptom_num=node_num_dict.get('symptom', 0),
                            drug_num=node_num_dict.get('drug', 0),
                            emb_dim=args.emb_dim, n_hop=args.n_hop_model,
                            l1_decay=args.l1_decay).to(device)
    except KeyError as e:
        print(f"Error initializing model: Missing key {e} in node_num_dict.");
        exit()

    for fold_id in range(5):
        print(f"\n--- Processing Fold {fold_id}/4 ---")

        model_path = os.path.join(args.save_dir, f'best_model_cv_fold_{fold_id}.pth')
        if not os.path.exists(model_path):
            print(f"Error: Pre-trained model file not found at {model_path}. Skipping fold {fold_id}.")
            final_model_paths_for_folds.append(None)
            continue

        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print(f"Successfully loaded pre-trained model for fold {fold_id} from {model_path}")
        except Exception as e:
            print(f"Error loading pre-trained model weights for fold {fold_id}: {e}");
            exit()

        try:
            fold_train_loader = torch.load(args.data_dir + f'/train_data_loader_fold_{fold_id}.pth', weights_only=False)
            edge_index_df = pd.read_csv(args.data_dir + f'/train_edge_index_fold_{fold_id}.txt', sep=',')
            edge_index = edge_index_df.values.astype(int)
        except FileNotFoundError as e:
            print(f"Error loading data for fold {fold_id}: {e}. Skipping fold.");
            final_model_paths_for_folds.append(model_path)
            continue

        try:
            edge_index = edge_index[edge_index[:, -1] == 1][:, :-1]
            edge_index[:, 1] = edge_index[:, 1] + node_num_dict.get('drug', 0)
        except Exception as e:
            print(f"Error processing edge_index for fold {fold_id}: {e}. Skipping fold.");
            final_model_paths_for_folds.append(model_path)
            continue

        fold_train_dataset = fold_train_loader.dataset
        BATCH_SIZE = fold_train_loader.batch_size if fold_train_loader.batch_size is not None else 64
        not_improved_count = 0

        if len(fold_train_dataset) < 10:
            print(f"Warning: Fold {fold_id} training dataset too small to split. Skipping fine-tuning for this fold.")
            final_model_paths_for_folds.append(model_path)
            continue

        val_size = max(1, int(len(fold_train_dataset) * VALIDATION_SPLIT))
        train_size = len(fold_train_dataset) - val_size

        if train_size <= 0:
            print(f"Warning: Not enough samples for training in fold {fold_id} after split. Skipping fine-tuning.")
            final_model_paths_for_folds.append(model_path)
            continue

        train_subset, val_subset = random_split(fold_train_dataset, [train_size, val_size],
                                                generator=torch.Generator().manual_seed(args.seed))
        finetune_valid_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

        if len(train_subset) < 2:
            print(f"Warning: Not enough fine-tuning samples in fold {fold_id} for 2 IRM envs. Skipping fine-tuning.")
            final_model_paths_for_folds.append(model_path)
            continue

        env_indices = list(range(len(train_subset)))
        random.shuffle(env_indices)
        split_point = len(env_indices) // 2
        env1_indices, env2_indices = env_indices[:split_point], env_indices[split_point:]

        if not env1_indices or not env2_indices:
            print(f"Warning: Could not create two non-empty IRM envs for fold {fold_id}. Skipping fine-tuning.")
            final_model_paths_for_folds.append(model_path)
            continue

        env1_loader = DataLoader(Subset(train_subset, env1_indices), batch_size=BATCH_SIZE, shuffle=True)
        env2_loader = DataLoader(Subset(train_subset, env2_indices), batch_size=BATCH_SIZE, shuffle=True)
        print(
            f"Fold {fold_id}: Created {len(train_subset)} training, {len(val_subset)} validation, {len(env1_indices)}/{len(env2_indices)} IRM envs.")

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        metrics_valid_baseline = evaluate_model(model, finetune_valid_loader, feature_index,
                                                drug_neighbor_set,
                                                symptom_neighbor_set, edge_index, device)
        best_valid_metric = metrics_valid_baseline[0] + metrics_valid_baseline[1]  # AUPR + AUC
        print(
            f"Fold {fold_id} Baseline on new val set: AUPR={metrics_valid_baseline[0]:.4f}, AUC={metrics_valid_baseline[1]:.4f}")

        fine_tuning_successful = False
        causal_model_path_fold = os.path.join(args.save_dir, f'best_model_DELTA_GA_fold_{fold_id}.pth')

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

                    logits1, _, _, _, _ = model(
                        *_get_feed_dict(data1, feature_index, drug_neighbor_set, symptom_neighbor_set,
                                        edge_index, model.n_hop, device))
                    loss1 = loss_fn(logits1, target1)

                    logits2, _, _, _, _ = model(
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
                    print(f"Error during fine-tuning batch in epoch {epoch} (fold {fold_id}): {e}")
                    continue

            if batch_count == 0:
                print(f"Epoch {epoch} (Fold {fold_id}): No successful batches. Skipping validation.")
                continue

            metrics_valid = evaluate_model(model, finetune_valid_loader, feature_index, drug_neighbor_set,
                                           symptom_neighbor_set, edge_index, device)
            aupr_valid, auc_valid = metrics_valid[0], metrics_valid[1]
            current_valid_metric = aupr_valid + auc_valid
            avg_epoch_loss = epoch_total_loss / batch_count
            print(
                f"F{fold_id} E{epoch}/{FINETUNE_EPOCHS} | Avg Loss: {avg_epoch_loss:.4f} | Val AUPR: {aupr_valid:.4f} | Val AUC: {auc_valid:.4f}")

            if current_valid_metric > best_valid_metric:
                best_valid_metric = current_valid_metric
                try:
                    torch.save(model.state_dict(), causal_model_path_fold)
                    print(
                        f"  -> Fold {fold_id} model improved to {best_valid_metric:.4f}. Model saved.")
                    fine_tuning_successful = True
                except Exception as e:
                    print(f"  -> Error saving fine-tuned model for fold {fold_id}: {e}")
                not_improved_count = 0
            else:
                not_improved_count += 1

            if not_improved_count >= PATIENCE:
                print(
                    f"\nEarly stopping at Epoch {epoch} for fold {fold_id}.")
                break

        if fine_tuning_successful:
            final_model_paths_for_folds.append(causal_model_path_fold)
            print(f"Fold {fold_id} fine-tuning finished. Using {os.path.basename(causal_model_path_fold)}.")
        else:
            final_model_paths_for_folds.append(model_path)  # reverts to pre-trained
            print(f"Fold {fold_id} fine-tuning finished. No improvement, reverting to {os.path.basename(model_path)}.")

    print("\n--- Starting final evaluation on the unseen global test set ---")

    all_fold_metrics = []

    if len(test_data_loader) == 0:
        print("Error: Global test data loader is empty. Cannot perform final evaluation.")
        exit()

    global_edge_index_path = os.path.join('../data', 'all_edge_index.txt')
    print(f"Loading global edge index for evaluation from {global_edge_index_path}...")
    try:
        edge_index_df_global_eval = pd.read_csv(global_edge_index_path, sep=',')
        edge_index_global_eval = edge_index_df_global_eval.values.astype(int)

        edge_index_global_eval = edge_index_global_eval[edge_index_global_eval[:, -1] == 1][:, :-1]
        edge_index_global_eval[:, 1] = edge_index_global_eval[:, 1] + node_num_dict.get('drug', 0)

        print(f"Successfully loaded and processed global edge index (all_edge_index.txt).")
    except FileNotFoundError:
        print(f"Error: '{global_edge_index_path}' not found.")
        print("Please ensure save_train_val_test_data.py has been run successfully.")
        exit()
    except Exception as e:
        print(f"Error processing '{global_edge_index_path}': {e}. Exiting.")
        exit()

    for fold_id, final_model_path in enumerate(final_model_paths_for_folds):
        if final_model_path is None:
            print(f"Skipping evaluation for fold {fold_id} as model path is missing.")
            all_fold_metrics.append([0.0] * 6)
            continue

        print(f"Evaluating Fold {fold_id} using model: {os.path.basename(final_model_path)}")
        try:
            model.load_state_dict(torch.load(final_model_path, map_location=device, weights_only=True))
        except Exception as e:
            print(f"Error loading final model weights for fold {fold_id}: {e}. Skipping.")
            all_fold_metrics.append([0.0] * 6)
            continue

        fold_metrics = evaluate_model(model, test_data_loader, feature_index, drug_neighbor_set,
                                      symptom_neighbor_set, edge_index_global_eval, device)

        all_fold_metrics.append(fold_metrics)
        print(f"==> Fold {fold_id} Test Performance: AUPR={fold_metrics[0]:.4f}, AUC={fold_metrics[1]:.4f} <==")

    all_fold_metrics_np = np.array(all_fold_metrics)
    avg_metrics = np.mean(all_fold_metrics_np, axis=0)
    std_metrics = np.std(all_fold_metrics_np, axis=0)

    print("\n--- Final 5-Fold Average Performance on Global Test Set (After Fine-tuning) ---")
    print(f"AUPR: {avg_metrics[0]:.4f} +/- {std_metrics[0]:.4f}")
    print(f"AUC:  {avg_metrics[1]:.4f} +/- {std_metrics[1]:.4f}")
    print(f"F1:   {avg_metrics[2]:.4f} +/- {std_metrics[2]:.4f}")
    print(f"Acc:  {avg_metrics[3]:.4f} +/- {std_metrics[3]:.4f}")

    try:
        results_out_file = args.save_dir + '/results_GA_CV_folds.txt'
        results_avg_out_file = args.save_dir + '/results_GA_CV_AVG.txt'

        np.savetxt(results_out_file, all_fold_metrics_np, header="AUPR AUC F1 Acc Pre Rec", comments="")

        avg_std_metrics = np.vstack((avg_metrics, std_metrics))
        np.savetxt(results_avg_out_file, avg_std_metrics, header="AUPR AUC F1 Acc Pre Rec",
                   comments="Row 1: Avg, Row 2: Std\n")

        print(f"\nDetailed 5-fold results saved to: {results_out_file}")
        print(f"Average results saved to: {results_avg_out_file}")

    except Exception as e:
        print(f"Error saving results files: {e}")

    print("\nGA_cv.py run finished.")