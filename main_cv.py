from parameters import args_parser
import torch
import numpy as np
import random
from model_cv import GraphHSA as module_arch
from torch import optim
import pandas as pd
import pickle
from sklearn import metrics
import os

def obtain_metrics(y_pred, y_true):
    acc=metrics.accuracy_score(y_pred=y_pred.round(), y_true=y_true)
    pre=metrics.precision_score(y_pred=y_pred.round(), y_true=y_true)
    rec=metrics.recall_score(y_pred=y_pred.round(), y_true=y_true)
    auc=metrics.roc_auc_score(y_score=y_pred, y_true=y_true)
    aupr=metrics.average_precision_score(y_score=y_pred, y_true=y_true)
    f1=metrics.f1_score(y_pred=y_pred.round(), y_true=y_true)
    return [aupr,auc,f1,acc,pre,rec]

args = args_parser()
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def _get_feed_dict(data,feature_index,drug_neighbor_set,symptom_neighbor_set,edge_index,n_hop):
    symptoms = data[:, feature_index['symptom']]
    drugs = data[:, feature_index['drug']]
    symptoms_neighbors, drugs_neighbors = [], []
    for hop in range(n_hop):
        drugs_neighbors.append(torch.LongTensor([drug_neighbor_set[d][hop] \
                                                      for d in drugs.numpy()]).to(device))
        symptoms_neighbors.append(torch.LongTensor([symptom_neighbor_set[s][hop] \
                                                   for s in symptoms.numpy()]).to(device))
    drug_feat=torch.LongTensor(list(drug_neighbor_set.keys())).to(device)
    sym_feat=torch.LongTensor(list(symptom_neighbor_set.keys())).to(device)

    return symptoms.to(device), drugs.to(device),sym_feat, drug_feat, symptoms_neighbors, drugs_neighbors,torch.LongTensor(edge_index.T).to(device)

if __name__ == '__main__':
    if not os.path.exists(args.save_dir+'/'):
        os.makedirs(args.save_dir+'/')
    with open('../data/node_num_dict.pickle', "rb") as f:
        node_num_dict = pickle.load(f)
    with open('../data/feature_index.pickle', "rb") as f:
        feature_index = pickle.load(f)
    with open('../data/symptom_neighbor_set.pickle', "rb") as f:
        symptom_neighbor_set = pickle.load(f)
    with open('../data/drug_neighbor_set.pickle', "rb") as f:
        drug_neighbor_set = pickle.load(f)
    #print(node_num_dict,feature_index,symptom_neighbor_set.keys(),drug_neighbor_set.keys())
    results_valid=[]
    for fold_id in range(5):
        train_data_loader = torch.load(args.data_dir+'/train_data_loader_fold_'+str(fold_id)+'.pth')
        valid_data_loader = torch.load(args.data_dir+'/valid_data_loader_fold_'+str(fold_id)+'.pth')
        edge_index=pd.read_csv(args.data_dir+'/train_edge_index_fold_'+str(fold_id)+'.txt',sep=',').values.astype(int)
        edge_index=edge_index[edge_index[:,-1]==1][:,:-1]
        edge_index[:,1]=edge_index[:,1]+node_num_dict['drug']

        model = module_arch(protein_num=node_num_dict['protein'],
                            symptom_num=node_num_dict['symptom'],
                            drug_num=node_num_dict['drug'],
                            emb_dim=args.emb_dim,
                            n_hop=args.n_hop_model,
                            l1_decay=args.l1_decay).to(device)

        criterion = torch.nn.BCEWithLogitsLoss()#getattr(module_loss, config['loss'])
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4, amsgrad=True)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        not_improved_count = 0
        best_val_acc = 0.0
        for epoch in range(args.epochs):
            model.train()
            train_loss=0.0
            train_preds = []
            train_targets = []
            for batch_idx, (data, target) in enumerate(train_data_loader):
                target = target.to(device)
                output, embeds_all, multi_loss, output_pro, output_gcn = model(*_get_feed_dict(data,feature_index,drug_neighbor_set,symptom_neighbor_set,edge_index,model.n_hop))
                if args.lambda_cl==0:
                    dgi_loss=0
                else:
                    dgi_loss=model.constrative_loss(embeds_all,target)
                if args.lambda_pred==0 or args.embed_type!='pro_gcn':
                    pred_loss=0
                else:
                    pred_loss=criterion(output_pro, target.squeeze()) + criterion(output_gcn, target.squeeze())/2
                loss_tra = criterion(output, target.squeeze()) + args.lambda_cl*dgi_loss + args.lambda_sim*multi_loss + args.lambda_pred*pred_loss
                optimizer.zero_grad()
                loss_tra.backward()
                optimizer.step()
                train_loss += loss_tra.item()

                with torch.no_grad():
                    y_pred = torch.sigmoid(output)
                    train_preds.extend(y_pred.cpu().detach().numpy())
                    train_targets.extend(target.cpu().detach().numpy())
            #print(train_preds, train_targets)
            cv_tra=obtain_metrics(np.array(train_preds), np.array(train_targets))
            avg_train_loss = train_loss / len(train_data_loader)
            print(f'Epoch {epoch}, Train Loss: {avg_train_loss}, Metrics: {cv_tra}')

            if valid_data_loader:
                model.eval()
                with torch.no_grad():
                    valid_loss = 0.0
                    valid_preds = []
                    valid_targets = []
                    for batch_idx, (data, target) in enumerate(valid_data_loader):
                        target = target.to(device)
                        output, embeds_all, multi_loss, output_pro, output_gcn = model(*_get_feed_dict(data,feature_index,drug_neighbor_set,symptom_neighbor_set,edge_index,model.n_hop))
                        if args.lambda_cl==0:
                            dgi_loss=0
                        else:
                            dgi_loss=model.constrative_loss(embeds_all,target)
                        if args.lambda_pred==0 or args.embed_type!='pro_gcn':
                            pred_loss=0
                        else:
                            pred_loss=criterion(output_pro, target.squeeze()) + criterion(output_gcn, target.squeeze())/2
                        loss_val = criterion(output, target.squeeze()) + args.lambda_cl*dgi_loss + args.lambda_sim*multi_loss + args.lambda_pred*pred_loss
                        valid_loss += loss_val.item()

                        pred = torch.sigmoid(output)
                        valid_preds.extend(pred.cpu().detach().numpy())
                        valid_targets.extend(target.cpu().detach().numpy())
                        #print((data, target),output,pred)
                cv_valid=obtain_metrics(np.array(valid_preds), np.array(valid_targets))
                avg_valid_loss = valid_loss / len(valid_data_loader)
                print(f'Epoch {epoch}, Valid Loss: {avg_valid_loss}, Metrics: {cv_valid}')

                if cv_valid[3] > best_val_acc:
                    best_val_acc = cv_valid[3]
                    not_improved_count = 0
                    torch.save(model.state_dict(), args.save_dir+'best_model_cv_fold_'+str(fold_id)+'.pth')
                    print(f"Epoch {epoch}: Best model saved with ACC: {best_val_acc:.4f}")
                else:
                    not_improved_count += 1

            if lr_scheduler is not None:
                lr_scheduler.step()
            if not_improved_count >= args.early_stop:
                print(f"Early stopping at Epoch {epoch} for fold {fold_id}. Best ACC: {best_val_acc:.4f}")
                break

        #### valid
        print(f"Loading best model for fold {fold_id}")
        model.load_state_dict(torch.load(args.save_dir+'best_model_cv_fold_'+str(fold_id)+'.pth'))
        model.eval()

        valid_preds = []
        valid_targets = []

        with torch.no_grad():
            for data, target in valid_data_loader:
                target = target.to(device)
                output, _, _, _, _ = model(*_get_feed_dict(data, feature_index, drug_neighbor_set, symptom_neighbor_set,edge_index, model.n_hop))
                pred = torch.sigmoid(output)
                valid_preds.extend(pred.cpu().numpy())
                valid_targets.extend(target.cpu().numpy())
        results_valid.append(obtain_metrics(np.array(valid_preds), np.array(valid_targets)))
        np.savetxt(args.save_dir+'preds_cv_fold_'+str(fold_id)+'.txt', np.hstack((np.array(valid_preds).reshape((-1,1)), np.array(valid_targets).reshape((-1,1)))))

results_valid2=np.average(np.array(results_valid),axis=0)
np.savetxt(args.save_dir+'results_cv.txt', np.vstack((np.array(results_valid), results_valid2.reshape((1,-1)))))
#print(results_valid2)

        
            
        
    
    
    
