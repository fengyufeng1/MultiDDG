import os
import argparse
import pandas as pd
from torch.autograd import Variable
from sklearn.model_selection import KFold
from model import *
import json
import random
from scipy.stats import pearsonr
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import copy
from tqdm import tqdm

# Seed
SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)


# Path
File_path = "./"
Dataset_Path = "./Dataset/"

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda:0")

def train_one_epoch(model, data_loader):
    epoch_loss_train = 0.0
    n = 0
    train_pred, train_true = [], []
    for data in data_loader:
        model.optimizer.zero_grad()
        _, _, labels, node, graph, wt_node, wt_graph, atom, edge, wt_atom, wt_edge = data
        
        node = Variable(node.cuda())
        wt_node = Variable(wt_node.cuda())
        graph = Variable(graph.cuda())
        wt_graph = Variable(wt_graph.cuda())
        y_true = Variable(labels.cuda())
        atom = Variable(atom.cuda())
        wt_atom = Variable(wt_atom.cuda())
        edge = Variable(edge.cuda())
        wt_edge = Variable(wt_edge.cuda())
        
        node = torch.squeeze(node)
        wt_node = torch.squeeze(wt_node)
        graph  = torch.squeeze(graph)
        wt_graph  = torch.squeeze(wt_graph)
        y_true = torch.squeeze(y_true)
        atom = torch.squeeze(atom)
        wt_atom = torch.squeeze(wt_atom)
        edge  = torch.squeeze(edge)
        wt_edge  = torch.squeeze(wt_edge)

        y_pred = model(node, graph, wt_node, wt_graph, atom, edge, wt_atom, wt_edge)
 
        y_true = y_true.float()
        y_pred = y_pred.float()
        
        # calculate loss
        loss = model.criterion(y_pred, y_true)   
        # backward gradient
        loss.backward()

        # update all parameters
        model.optimizer.step()
        # model.scheduler.step()

        train_pred.append(y_pred.item())
        train_true.append(y_true.item())
        epoch_loss_train += loss.item()
        n += 1

    epoch_loss_train_avg = epoch_loss_train / n
    
    return epoch_loss_train_avg


def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}

    for data in data_loader:
        with torch.no_grad():
            PDB_names, WT_PDB_names, labels, node, graph, wt_node, wt_graph, atom, edge, wt_atom, wt_edge = data
 
            node = Variable(node.cuda())
            wt_node = Variable(wt_node.cuda())
            graph = Variable(graph.cuda())
            wt_graph = Variable(wt_graph.cuda())
            y_true = Variable(labels.cuda())
            atom = Variable(atom.cuda())
            wt_atom = Variable(wt_atom.cuda())
            edge = Variable(edge.cuda())
            wt_edge = Variable(wt_edge.cuda())
            
            node = torch.squeeze(node)
            wt_node = torch.squeeze(wt_node)
            graph  = torch.squeeze(graph)
            wt_graph  = torch.squeeze(wt_graph)
            y_true = torch.squeeze(y_true)
            atom = torch.squeeze(atom)
            wt_atom = torch.squeeze(wt_atom)
            edge  = torch.squeeze(edge)
            wt_edge  = torch.squeeze(wt_edge)
    
            y_pred = model(node, graph, wt_node, wt_graph, atom, edge, wt_atom, wt_edge)

            loss = model.criterion(y_pred, y_true)
            
            valid_pred.append(y_pred.item()) 
            valid_true.append(y_true.item())

            epoch_loss += loss.item()
            n += 1

    epoch_loss_avg = epoch_loss / n

    return epoch_loss_avg, valid_true, valid_pred, pred_dict


def analysis(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    r, p = pearsonr(y_true, y_pred)
    results = {
        'mae':mae,
        'mse':mse,
        'RMSE':rmse,
        'r2':r2,
        'R': r,
        'P-value':p
    }
    if np.isnan(r):
        print(y_true, y_pred)
        
    return results


def train(feature_path, model_path, datasets_name, model, train_dataframe, valid_dataframe, fold = 0):
    train_loader = DataLoader(dataset=ProDataset(feature_path, train_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=ProDataset(feature_path, valid_dataframe), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    datasets_num = datasets_name.split('_')[1]
    loss_train_list, loss_val_list = [], []
    early_epoch = 6
    stop_threshold = 0
    best_epoch = -1
    best_val_loss = 10000
    best_rmse = float('inf') 
    best_model = None

    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()

        epoch_loss_train_avg = train_one_epoch(model, train_loader)
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred, _ = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred)
        loss_train_list.append(epoch_loss_train_avg)

        print("Train loss: ", epoch_loss_train_avg)
        print("Train RMSE: ", result_train['RMSE'])
        print("Train R2", result_train['r2'])
        print("Train R", result_train['R'])

        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred, _ = evaluate(model, valid_loader)
        result_valid = analysis(valid_true, valid_pred)
        loss_val_list.append(epoch_loss_valid_avg)

        print("Valid loss: ", epoch_loss_valid_avg)
        print("Valid RMSE: ", result_valid['RMSE'])
        print("Valid R2", result_valid['r2'])
        print("Valid R", result_valid['R'])

        # if  epoch_loss_valid_avg < best_val_loss:
        #     best_epoch = epoch + 1
        #     best_val_loss = epoch_loss_valid_avg
        #     best_model = copy.deepcopy(model) 

        if  epoch_loss_valid_avg < best_val_loss:
            best_epoch = epoch + 1
            best_val_loss = epoch_loss_valid_avg
            best_model = copy.deepcopy(model)       
            stop_threshold = 0
        else:
            stop_threshold += 1
            if stop_threshold > early_epoch:
                break
    
    torch.save(best_model.state_dict(), os.path.join(model_path, 'Fold'+str(fold)+'_best_model.pkl'))

    # np.save(File_path + f"Blind_test/Train_loss/{datasets_num}_loss/cv_Fold/{datasets_name}_Fold_{str(fold)}_train_loss_2.npy", np.array(loss_train_list))
    # np.save(File_path + f"Blind_test/Train_loss/{datasets_num}_loss/cv_Fold/{datasets_name}_Fold_{str(fold)}_val_loss_2.npy", np.array(loss_val_list))
            
    return best_epoch, best_model

def cross_validation(feature_path, model_path, datasets_name, all_dataframe, fold_number):

    PDB_names = all_dataframe['ID'].values
    WT_PDB_names = all_dataframe['WT_ID'].values
    labels = all_dataframe['label'].values
    kfold = KFold(n_splits = fold_number, shuffle = True)
    fold = 0
    best_epochs, best_models_states = [], []

    for train_index, valid_index in kfold.split(PDB_names, labels):
        print("\n\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Train on ------------")

        model = GraphPPIS(C_INPUT_DIM, C_HIDDEN_DIM, G_HEADS, G_INPUT_DIM, G_HIDDEN_DIM, A_HEADS, A_INPUT_DIM, A_HIDDEN_DIM, DROPOUT)
        model.cuda()

        best_epoch, best_model = train(feature_path, model_path, datasets_name, model, train_dataframe, valid_dataframe, fold + 1)
        best_epochs.append(str(best_epoch))
        
        best_models_states.append(best_model.state_dict())
        fold += 1

    print("\n\nBest epoch: " + " ".join(best_epochs))

    return round(sum([int(epoch) for epoch in best_epochs]) / fold_number), best_models_states

def ensemble_models(model_class, models_states, *args, **kwargs):
    """
    Create a weight-averaged model
    """
    # Initialize a new model based on the state dictionary of the first model
    ensemble_model = model_class(*args, **kwargs)
    ensemble_model.cuda()

    param_keys = list(models_states[0].keys())
    
    for key in param_keys:
        avg_param = torch.mean(torch.stack([state_dict[key] for state_dict in models_states]), dim=0)
        ensemble_model.state_dict()[key].copy_(avg_param)
    
    return ensemble_model

def ensemble_best_models(model_path, datasets_name, best_models_states):
    """
    Cross-validation-based integrated modeling of optimal model state dictionaries
    """
    ensemble_model = ensemble_models(GraphPPIS, best_models_states, C_INPUT_DIM, C_HIDDEN_DIM, G_HEADS, G_INPUT_DIM, G_HIDDEN_DIM, A_HEADS, A_INPUT_DIM, A_HIDDEN_DIM, DROPOUT)
    torch.save(ensemble_model.state_dict(), os.path.join(model_path, f"{datasets_name}_Ensembled_Model.pkl"))
    print("Ensembled model saved using the best models from cross-validation.")

def train_full_model(feature_path, model_path, datasets_name, all_dataframe, aver_epoch):
    datasets_num = datasets_name.split('_')[1]
    loss_train_list, loss_val_list = [], []
    print("\n\nTraining a full model using all training data...\n")
    
    model = GraphPPIS(C_INPUT_DIM, C_HIDDEN_DIM, G_HEADS, G_INPUT_DIM, G_HIDDEN_DIM, A_HEADS, A_INPUT_DIM, A_HIDDEN_DIM, DROPOUT)
    model.load_state_dict(torch.load(os.path.join(model_path, f"{datasets_name}_Ensembled_Model.pkl")))
    model.cuda()
    
    train_loader = DataLoader(dataset=ProDataset(feature_path, all_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    best_epoch = -1
    best_train_loss = 10000
    best_rmse = float('inf') 
    best_model = None
    
    for epoch in range(NUMBER_EPOCHS):
        
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()
        epoch_loss_train_avg = train_one_epoch(model, train_loader)
        
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred, _ = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred)

        loss_train_list.append(epoch_loss_train_avg)

        print("Train loss: ", epoch_loss_train_avg)
        print("Train RMSE: ", result_train['RMSE'])
        print("Train R", result_train['R'])

        if epoch + 1 in [aver_epoch, NUMBER_EPOCHS]:
            torch.save(model.state_dict(), os.path.join(model_path, datasets_name+'_Full_model_{}.pkl'.format(epoch + 1)))

        if epoch_loss_train_avg < best_train_loss:
            best_train_loss = epoch_loss_train_avg
            best_model = copy.deepcopy(model)

    torch.save(best_model.state_dict(), os.path.join(model_path, datasets_name+'_Full_model_best.pkl'))
 
    # np.save(File_path + f"Blind_test/Train_loss/{datasets_num}_loss/{datasets_name}_full_train_loss.npy", np.array(loss_train_list))

    print("Model saved")

def main(datasets_list, feature_path, model_path):
    print(datasets_list)

    for datasets in datasets_list:
        datasets_name = os.path.splitext(datasets)[0]
    
        print('\n\n', datasets + ':')

        IDs, WT_IDs, labels = [], [], [] 
        with open(Dataset_Path + datasets, "rb") as f:
            Train_data = pickle.load(f)                  
        for ID in Train_data:
            IDs.append(ID)
            item = Train_data[ID]
            WT_IDs.append(item[0])
            labels.append(item[1])   
        train_dic = {"ID": IDs, "WT_ID": WT_IDs, "label": labels}
        train_dataframe = pd.DataFrame(train_dic)

        aver_epoch, best_models_states = cross_validation(feature_path, model_path, datasets_name, train_dataframe, fold_number = 5)
        ensemble_best_models(model_path, datasets_name, best_models_states)
        train_full_model(feature_path, model_path, datasets_name, train_dataframe, aver_epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some datasets.')
    parser.add_argument('datasets', metavar='N', type=str, nargs='+',
                        help='the dataset names')
    parser.add_argument('--feature_path', dest='feature_path', default='./Feature/Feature_2M/',
                        help='path to the feature directory (default: ./Feature/Feature_2M/)')
    parser.add_argument('--model_path', dest='model_path', default='../Train_Model/M1340_Model/',
                        help='path to the model directory (default: ../Train_Model/M1340_Model/)')

    args = parser.parse_args()

    main(args.datasets, args.feature_path, args.model_path)

