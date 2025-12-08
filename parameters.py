import argparse
def args_parser():
    parser = argparse.ArgumentParser(description='Models')
    #parser.add_argument('--fold_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2025, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--save_dir', type=str, default='results/', help='output_file_path')
    parser.add_argument('--early_stop', type=int, default=10, help='early_stop.')
    
    ####model setting
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--l1_decay', type=float, default=1e-6, help='l1_decay.')
    parser.add_argument('--emb_dim', type=int, default=64, help='emb_dim.')
    parser.add_argument('--n_hop_model', type=int, default=1, help='select hop neighbors in model.')
    
    ####data setting
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
    parser.add_argument('--n_hop_data', type=int, default=1, help='select hop neighbors in data collection.')
    parser.add_argument('--data_dir', type=str, default='../data/', help='data_dir')
    parser.add_argument('--n_memory', type=int, default=128, help='n_memory.')
    parser.add_argument('--num_workers', type=int, default=2, help='num_workers.')     
        
    ####parameters    
    parser.add_argument('--lambda_pred', type=float, default=0.6, help='weight of prediction loss.')    
    parser.add_argument('--lambda_cl', type=float, default=1.0, help='weight of contrastive loss.')    
    parser.add_argument('--lambda_sim', type=float, default=0.8, help='weight of similarity loss.')    
    
    parser.add_argument('--embed_type', type=str, default='pro_gcn', help='Node embeddings')   ###init,pro,pro_gcn,gcn 
    parser.add_argument('--gcn_layer', type=int, default=1, help='GCN layers')
    parser.add_argument('--GCN_types', type=str, default='GCN', help='GCN_types')

    args = parser.parse_args()
    return args