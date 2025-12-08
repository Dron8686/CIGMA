from parameters import args_parser
import torch
import numpy as np
import random 
from model_indep import GraphHSA as module_arch  
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
        
    with open('../data/node_map_dict.pickle', "rb") as f:  
        node_map_dict = pickle.load(f)
    dic_ids_sym={}
    dic_ids_drug={}
    dic_ids_pro={}
    for k, v in node_map_dict.items():
        if isinstance(k, str) and 'C' in k:
            dic_ids_sym[v]=k
        if isinstance(k, str) and 'U' in k:
            dic_ids_drug[v]=k
        if isinstance(k, int):
            dic_ids_pro[v]=k
            
    train_data_loader = torch.load(args.data_dir+'/train_data_loader.pth')
    test_data_loader = torch.load(args.data_dir+'/test_data_loader.pth')
    edge_index=pd.read_csv(args.data_dir+'/train_edge_index.txt',sep=',').values.astype(int)
    edge_index=edge_index[edge_index[:,-1]==1][:,:-1]
    edge_index[:,1]=edge_index[:,1]+node_num_dict['drug']

    model = module_arch(protein_num=node_num_dict['protein'],
                        symptom_num=node_num_dict['symptom'],
                        drug_num=node_num_dict['drug'],
                        emb_dim=args.emb_dim,
                        n_hop=args.n_hop_model,
                        l1_decay=args.l1_decay).to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4, amsgrad=True)         
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    not_improved_count = 0
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train() 
        train_loss=0.0
        train_preds = []
        train_targets = []
        for batch_idx, (data, target) in enumerate(train_data_loader): 
            target = target.to(device)
            output, embeds_all, multi_loss, output_pro, output_gcn, att_sym, att_drug = model(*_get_feed_dict(data,feature_index,drug_neighbor_set,symptom_neighbor_set,edge_index,model.n_hop)) 
            
            if args.lambda_cl==0:
                dgi_loss=0
            else:
                dgi_loss=model.constrative_loss(embeds_all,target)
            if args.lambda_pred==0:
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
        cv_tra=obtain_metrics(np.array(train_preds), np.array(train_targets))
        avg_train_loss = train_loss / len(train_data_loader)
        print(f'Epoch {epoch}, Train Loss: {avg_train_loss}, Metrics: {cv_tra}')
    
        if cv_tra[3] > best_acc:
            best_acc = cv_tra[3]
            not_improved_count = 0
            torch.save(model.state_dict(), args.save_dir+'best_model_indep.pth')
            print(f"Epoch {epoch}: Best model saved with ACC: {best_acc:.4f}")
        else:
            not_improved_count += 1
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        if not_improved_count >= args.early_stop:
            print(f"Early stopping at Epoch {epoch}. Best ACC: {best_acc:.4f}")
            break

    print("Loading best model")
    model.load_state_dict(torch.load(args.save_dir+'best_model_indep.pth'))
    model.eval()
    
    test_preds = []
    test_targets = []
    test_embeds = []
    att_syms = []
    att_drugs = []
    symptoms_neighborss = []
    drugs_neighborss = []
    symptomss = []
    drugss = []
    with torch.no_grad():
        for data, target in test_data_loader:
            target = target.to(device)
            output, embeds, _, _, _, att_sym, att_drug = model(*_get_feed_dict(data, feature_index, drug_neighbor_set, symptom_neighbor_set,edge_index, model.n_hop))
            
            
            symptoms = data[:, feature_index['symptom']]
            drugs = data[:, feature_index['drug']]
    
            symptoms_neighbors, drugs_neighbors = [], []
            for hop in range(model.n_hop):
                drugs_neighbors.append(torch.LongTensor([drug_neighbor_set[d][hop] \
                                                              for d in drugs.numpy()]).to(device))
                symptoms_neighbors.append(torch.LongTensor([symptom_neighbor_set[s][hop] \
                                                           for s in symptoms.numpy()]).to(device))
            

            pred = torch.sigmoid(output)
            test_preds.extend(pred.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
            test_embeds.extend(embeds.cpu().numpy())
            
            att_syms+=att_sym
            att_drugs+=att_drug
            symptomss.extend(symptoms.cpu().numpy())
            drugss.extend(drugs.cpu().numpy())
            
            symptoms_neighborss+=symptoms_neighbors
            drugs_neighborss+=drugs_neighbors
            
    results_test=obtain_metrics(np.array(test_preds), np.array(test_targets))
    np.savetxt(args.save_dir+'preds_indep.txt', np.hstack((np.array(test_preds).reshape((-1,1)), np.array(test_targets).reshape((-1,1)))))
    np.savetxt(args.save_dir+'embeds_indep.txt', test_embeds)
    np.savetxt(args.save_dir+'results_indep.txt', np.array(results_test))
    np.savetxt(args.save_dir+'att_symptom.txt', att_syms[0].cpu().numpy())
    np.savetxt(args.save_dir+'att_drug.txt', att_drugs[0].cpu().numpy())
    np.savetxt(args.save_dir+'symptoms_neighbors.txt', symptoms_neighborss[0].cpu().numpy())
    np.savetxt(args.save_dir+'drugs_neighbors.txt', drugs_neighborss[0].cpu().numpy())
    np.savetxt(args.save_dir+'symptoms_index.txt', symptomss)
    np.savetxt(args.save_dir+'drugs_index.txt', drugss)
    
    symptoms_neighbors=np.loadtxt(args.save_dir+'/symptoms_neighbors.txt')
    drugs_neighbors=np.loadtxt(args.save_dir+'/drugs_neighbors.txt')
    symptoms=np.loadtxt(args.save_dir+'/symptoms_index.txt')
    drugs=np.loadtxt(args.save_dir+'/drugs_index.txt')
    
    symptoms_neighbors2=[]
    drugs_neighbors2=[]
    symptoms2=[]
    drugs2=[]
    for i in range(len(drugs)):
        drugs2.append(dic_ids_drug[drugs[i]])
        symptoms2.append(dic_ids_sym[symptoms[i]])
        symptoms_neighbors2.append([dic_ids_pro[k] for k in symptoms_neighbors[i]])
        drugs_neighbors2.append([dic_ids_pro[k] for k in drugs_neighbors[i]])
       
    np.savetxt(args.save_dir+'symptoms_neighbors_name.txt', np.array(symptoms_neighbors2))
    np.savetxt(args.save_dir+'drugs_neighbors_name.txt', np.array(drugs_neighbors2))
    with open(args.save_dir+'symptoms_name.txt','w') as f:
        f.write('\n'.join(symptoms2))
    with open(args.save_dir+'drugs_name.txt','w') as f:
        f.write('\n'.join(drugs2))
    
    print(results_test)
    
    
    
