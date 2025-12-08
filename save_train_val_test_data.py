from parameters import args_parser
import collections  
import torch
import random
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
import pandas as pd
import os
import torch.utils.data as Data
import pickle
import networkx as nx
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader

args = args_parser()
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

def get_neighbor_set(items, item_target_dict, n_hop, n_memory):
    print('constructing neighbor set ...')
    neighbor_set = collections.defaultdict(list)
    for item in items:
        for hop in range(n_hop):
            if hop == 0:
                replace = len(item_target_dict[item]) < n_memory
                target_list = list(np.random.choice(item_target_dict[item], 
                                                    size=n_memory, replace=replace))
            else:
                origin_nodes = neighbor_set[item][-1]
                neighbors = []
                for node in origin_nodes:
                    neighbors += graph.neighbors(node)
                replace = len(neighbors) < n_memory
                target_list = list(np.random.choice(neighbors, size=n_memory, replace=replace))
            
            neighbor_set[item].append(target_list)
    return neighbor_set

def obtain_data():
    drug_symptom_df = pd.read_csv('../data/dsi_1v1.csv')
    ppi_df = pd.read_excel(os.path.join('../data/ppi.xlsx'), engine='openpyxl')
    spi_df = pd.read_csv(os.path.join('../data/sym_pro.csv'))
    dpi_df = pd.read_csv(os.path.join('../data/drug_pro_delcolumn_ppi.csv'))
    protein_node = list(set(ppi_df['proteinA']) | set(ppi_df['proteinB']))
    symptom_node = list(set(spi_df['symptom']))
    drug_node = list(set(dpi_df['drug']))
    node_num_dict = {'protein': len(protein_node), 'symptom': len(symptom_node), 'drug': len(drug_node)}
    
    node_map_dict = {protein_node[idx]:idx for idx in range(len(protein_node))}  
    node_map_dict.update({symptom_node[idx]:idx for idx in range(len(symptom_node))})
    node_map_dict.update({drug_node[idx]:idx for idx in range(len(drug_node))})
    
    with open(os.path.join('../data/node_map_dict.pickle'), 'wb') as f:
        pickle.dump(node_map_dict, f)

    print('undirected graph')
    print('# proteins: {0}, # drugs: {1}, # symptoms: {2}'.format(
            len(protein_node), len(drug_node), len(symptom_node)))
    print('# protein-protein interactions: {0}, # drug-protein associations: {1}, # symptom-protein associations: {2}'.format(
        len(ppi_df), len(dpi_df), len(spi_df)))
    
    ppi_df['proteinA'] = ppi_df['proteinA'].map(node_map_dict)
    ppi_df['proteinB'] = ppi_df['proteinB'].map(node_map_dict)
    ppi_df = ppi_df[['proteinA', 'proteinB']]

    spi_df['symptom'] = spi_df['symptom'].map(node_map_dict)
    spi_df['protein'] = spi_df['protein'].map(node_map_dict)
    spi_df = spi_df[['symptom', 'protein']]

    dpi_df['drug'] = dpi_df['drug'].map(node_map_dict)
    dpi_df['protein'] = dpi_df['protein'].map(node_map_dict)
    dpi_df = dpi_df[['drug', 'protein']]

    drug_symptom_df['drug'] = drug_symptom_df['drug'].map(node_map_dict)
    drug_symptom_df['symptom'] = drug_symptom_df['symptom'].map(node_map_dict)
    
    drug_symptom_df.to_csv(os.path.join('../data/drug_symptom_process.csv'), index=False)
    drug_symptom_df = drug_symptom_df[['drug','symptom','indication']]
    feature_index = {'symptom': 1, 'drug': 0}
    
    drug_symptom_df = drug_symptom_df.sample(frac=1, random_state=args.seed)
    feature = torch.from_numpy(drug_symptom_df.to_numpy())
    label = torch.from_numpy(drug_symptom_df[['indication']].to_numpy())
    feature = feature.type(torch.LongTensor)
    label = label.type(torch.FloatTensor)
    dataset = Data.TensorDataset(feature, label)
    return ppi_df,spi_df,dpi_df,dataset,feature_index,node_num_dict,drug_symptom_df

def warm_start(dataset):
    if not os.path.exists('../data/warm_start/'):
        os.makedirs('../data/warm_start/')
    idx_full = np.arange(len(dataset))
    np.random.shuffle(idx_full)
    
    test_idx  = idx_full[0:int(len(dataset) * 0.1)]
    test_sampler = SubsetRandomSampler(test_idx)
    train_all_idx  = idx_full[int(len(dataset) * 0.1):]
    inters=dsi_df.loc[train_all_idx.tolist()]
    inters.to_csv('../data/warm_start/train_edge_index.txt',index=False)
    train_all_sampler = SubsetRandomSampler(train_all_idx)
    
    init_kwargs = {
        'dataset': dataset,
        'batch_size': args.batch_size,
        'collate_fn': default_collate,
        'num_workers': args.num_workers
    }
    test_data_loader = DataLoader(sampler=test_sampler, **init_kwargs)
    torch.save(test_data_loader,'../data/warm_start/test_data_loader.pth')
    
    train_all_data_loader = DataLoader(sampler=train_all_sampler, **init_kwargs)
    torch.save(train_all_data_loader,'../data/warm_start/train_data_loader.pth')
    
    sss = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold_id, (train_index, valid_index) in enumerate(sss.split(train_all_idx)):
        train_idx = np.array(train_all_idx)[train_index]
        valid_idx = np.array(train_all_idx)[valid_index]
        
        inters=dsi_df.loc[train_idx.tolist()]
        inters.to_csv('../data/warm_start/train_edge_index_fold_'+str(fold_id)+'.txt',index=False)
        
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        train_data_loader = DataLoader(sampler=train_sampler, **init_kwargs)
        valid_data_loader = DataLoader(sampler=valid_sampler, **init_kwargs)
        
        torch.save(train_data_loader,'../data/warm_start/train_data_loader_fold_'+str(fold_id)+'.pth')
        torch.save(valid_data_loader,'../data/warm_start/valid_data_loader_fold_'+str(fold_id)+'.pth')

def cold_starts(dataset):
    if not os.path.exists('../data/cold_starts/'):
        os.makedirs('../data/cold_starts/')
    
    idx_full = np.arange(len(dataset))
    np.random.shuffle(idx_full)
    test_idx  = idx_full[0:int(len(dataset) * 0.05)]
    test_sampler = SubsetRandomSampler(test_idx)
    
    drugs=list(set(dsi_df.loc[test_idx].loc[:,'drug'].values.tolist()))
    symptoms=list(set(dsi_df.loc[test_idx].loc[:,'symptom'].values.tolist()))
        
    train_all_idx  = dsi_df[~dsi_df['drug'].isin(drugs)][~dsi_df['symptom'].isin(symptoms)].index.tolist()
    inters=dsi_df.loc[train_all_idx]
    inters.to_csv('../data/cold_starts/train_edge_index.txt',index=False)
    train_all_sampler = SubsetRandomSampler(train_all_idx)
    
    init_kwargs = {
        'dataset': dataset,
        'batch_size': args.batch_size,
        'collate_fn': default_collate,
        'num_workers': args.num_workers
    }
    test_data_loader = DataLoader(sampler=test_sampler, **init_kwargs)
    torch.save(test_data_loader,'../data/cold_starts/test_data_loader.pth')
    
    train_all_data_loader = DataLoader(sampler=train_all_sampler, **init_kwargs)
    torch.save(train_all_data_loader,'../data/cold_starts/train_data_loader.pth')

    sss = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold_id, (train_index, valid_index) in enumerate(sss.split(train_all_idx)):
        train_idx = np.array(train_all_idx)[train_index]
        valid_idx = np.array(train_all_idx)[valid_index]
        inters=dsi_df.loc[train_idx.tolist()]
        inters.to_csv('../data/cold_starts/train_edge_index_fold_'+str(fold_id)+'.txt',index=False)
        
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        train_data_loader = DataLoader(sampler=train_sampler, **init_kwargs)
        valid_data_loader = DataLoader(sampler=valid_sampler, **init_kwargs)
        
        torch.save(train_data_loader,'../data/cold_starts/train_data_loader_fold_'+str(fold_id)+'.pth')
        torch.save(valid_data_loader,'../data/cold_starts/valid_data_loader_fold_'+str(fold_id)+'.pth')
        print(list(set(dsi_df.loc[train_idx.tolist()].loc[:,'symptom'].values.tolist()) & set(symptoms)))
        print(list(set(dsi_df.loc[valid_idx.tolist()].loc[:,'symptom'].values.tolist()) & set(symptoms)))
        print(list(set(dsi_df.loc[train_idx.tolist()].loc[:,'drug'].values.tolist()) & set(drugs)))
        print(list(set(dsi_df.loc[valid_idx.tolist()].loc[:,'drug'].values.tolist()) & set(drugs)))
        
def cold_start_herb(dataset):
    if not os.path.exists('../data/cold_start_herb/'):
        os.makedirs('../data/cold_start_herb/')

    drugs=list(set(dsi_df.loc[:,'drug'].values.tolist()))
    np.random.shuffle(drugs)
    test_drugs  = drugs[0:int(len(drugs) * 0.1)]
    
    test_idx=dsi_df[dsi_df['drug'].isin(test_drugs)].index.tolist()
    test_sampler = SubsetRandomSampler(test_idx)
    train_all_idx  = dsi_df[~dsi_df['drug'].isin(test_drugs)].index.tolist()
    inters=dsi_df.loc[train_all_idx]
    inters.to_csv('../data/cold_start_herb/train_edge_index.txt',index=False)
    train_all_sampler = SubsetRandomSampler(train_all_idx)
    
    init_kwargs = {
        'dataset': dataset,
        'batch_size': args.batch_size,
        'collate_fn': default_collate,
        'num_workers': args.num_workers
    }
    test_data_loader = DataLoader(sampler=test_sampler, **init_kwargs)
    torch.save(test_data_loader,'../data/cold_start_herb/test_data_loader.pth')
    
    train_all_data_loader = DataLoader(sampler=train_all_sampler, **init_kwargs)
    torch.save(train_all_data_loader,'../data/cold_start_herb/train_data_loader.pth')

    sss = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold_id, (train_index, valid_index) in enumerate(sss.split(train_all_idx)):
        train_idx = np.array(train_all_idx)[train_index]
        valid_idx = np.array(train_all_idx)[valid_index]
        inters=dsi_df.loc[train_idx.tolist()]
        inters.to_csv('../data/cold_start_herb/train_edge_index_fold_'+str(fold_id)+'.txt',index=False)
        
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        train_data_loader = DataLoader(sampler=train_sampler, **init_kwargs)
        valid_data_loader = DataLoader(sampler=valid_sampler, **init_kwargs)
        
        torch.save(train_data_loader,'../data/cold_start_herb/train_data_loader_fold_'+str(fold_id)+'.pth')
        torch.save(valid_data_loader,'../data/cold_start_herb/valid_data_loader_fold_'+str(fold_id)+'.pth')
        print(list(set(dsi_df.loc[train_idx.tolist()].loc[:,'drug'].values.tolist()) & set(test_drugs)))
        print(list(set(dsi_df.loc[valid_idx.tolist()].loc[:,'drug'].values.tolist()) & set(test_drugs)))
         

def cold_start_symptom(dataset):
    if not os.path.exists('../data/cold_start_symptom/'):
        os.makedirs('../data/cold_start_symptom/')

    drugs=list(set(dsi_df.loc[:,'symptom'].values.tolist()))
    np.random.shuffle(drugs)
    test_drugs  = drugs[0:int(len(drugs) * 0.1)]
    
    test_idx=dsi_df[dsi_df['symptom'].isin(test_drugs)].index.tolist()
    test_sampler = SubsetRandomSampler(test_idx)
    train_all_idx  = dsi_df[~dsi_df['symptom'].isin(test_drugs)].index.tolist()
    inters=dsi_df.loc[train_all_idx]
    inters.to_csv('../data/cold_start_symptom/train_edge_index.txt',index=False)
    train_all_sampler = SubsetRandomSampler(train_all_idx)
    
    init_kwargs = {
        'dataset': dataset,
        'batch_size': args.batch_size,
        'collate_fn': default_collate,
        'num_workers': args.num_workers
    }
    test_data_loader = DataLoader(sampler=test_sampler, **init_kwargs)
    torch.save(test_data_loader,'../data/cold_start_symptom/test_data_loader.pth')
    
    train_all_data_loader = DataLoader(sampler=train_all_sampler, **init_kwargs)
    torch.save(train_all_data_loader,'../data/cold_start_symptom/train_data_loader.pth')
    
    sss = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold_id, (train_index, valid_index) in enumerate(sss.split(train_all_idx)):
        train_idx = np.array(train_all_idx)[train_index]
        valid_idx = np.array(train_all_idx)[valid_index]
        inters=dsi_df.loc[train_idx.tolist()]
        inters.to_csv('../data/cold_start_symptom/train_edge_index_fold_'+str(fold_id)+'.txt',index=False)
        
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        train_data_loader = DataLoader(sampler=train_sampler, **init_kwargs)
        valid_data_loader = DataLoader(sampler=valid_sampler, **init_kwargs)
        
        torch.save(train_data_loader,'../data/cold_start_symptom/train_data_loader_fold_'+str(fold_id)+'.pth')
        torch.save(valid_data_loader,'../data/cold_start_symptom/valid_data_loader_fold_'+str(fold_id)+'.pth')
        print(list(set(dsi_df.loc[train_idx.tolist()].loc[:,'symptom'].values.tolist()) & set(test_drugs)))
        print(list(set(dsi_df.loc[valid_idx.tolist()].loc[:,'symptom'].values.tolist()) & set(test_drugs)))
    
        
if __name__ == '__main__':
    ppi_df,spi_df,dpi_df,dataset,feature_index,node_num_dict,dsi_df=obtain_data()
    dsi_df.to_csv('../data/all_edge_index.txt',index=False)
    
    with open(os.path.join('../data/node_num_dict.pickle'), 'wb') as f:
        pickle.dump(node_num_dict, f)
    with open(os.path.join('../data/feature_index.pickle'), 'wb') as f:
        pickle.dump(feature_index, f)

    tuples = [tuple(x) for x in ppi_df.values]
    graph = nx.Graph()
    graph.add_edges_from(tuples)
    
    # 1-symptom
    symptom_protein_dict = collections.defaultdict(list)
    symptom_list = list(set(spi_df['symptom'])) 
    for symptom in symptom_list:
        symptom_df = spi_df[spi_df['symptom']==symptom]  
        target = list(set(symptom_df['protein']))
        symptom_protein_dict[symptom] = target
    # 2-drug
    drug_protein_dict = collections.defaultdict(list)
    drug_list = list(set(dpi_df['drug']))
    for drug in drug_list:
        drug_df = dpi_df[dpi_df['drug']==drug]
        target = list(set(drug_df['protein']))
        drug_protein_dict[drug] = target
        
    symptoms = list(symptom_protein_dict.keys())
    drugs = list(drug_protein_dict.keys())
    symptom_neighbor_set = get_neighbor_set(symptoms, symptom_protein_dict, 
                                            args.n_hop_data, args.n_memory)
    drug_neighbor_set = get_neighbor_set(drugs, drug_protein_dict, 
                                            args.n_hop_data, args.n_memory)

    with open(os.path.join('../data/symptom_neighbor_set.pickle'), 'wb') as f:
        pickle.dump(symptom_neighbor_set, f)
    with open(os.path.join('../data/drug_neighbor_set.pickle'), 'wb') as f:
        pickle.dump(drug_neighbor_set, f)
    
    warm_start(dataset)
    cold_start_herb(dataset)
    cold_start_symptom(dataset)
    cold_starts(dataset)

    

    
    
    
    
    
    
    
    
    
    