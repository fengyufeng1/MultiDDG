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

# Path
File_path = "./"
Dataset_Path = "./Dataset/"
Model_Path = "./CV_STRUC/cv_struc_model/"

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


def train(feature_path, datasets_name, model, train_dataframe, valid_dataframe, fold = 0):
    train_loader = DataLoader(dataset=ProDataset(feature_path, train_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=ProDataset(feature_path, valid_dataframe), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    loss_train_list, loss_val_list = [], []
    best_rmse = 10000
    best_val_loss = 10000
    best_epoch = -1
    best_rmse = float('inf') 
    best_r = float('inf') 


    for epoch in range(NUMBER_EPOCHS):
        model.train()

        epoch_loss_train_avg = train_one_epoch(model, train_loader)
        loss_train_list.append(epoch_loss_train_avg)

        # _, train_true, train_pred, _ = evaluate(model, train_loader)
        # result_train = analysis(train_true, train_pred)
        # print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        # print("Train loss: ", epoch_loss_train_avg)
        # print("Train RMSE: ", result_train['RMSE'])
        # print("Train R", result_train['R'])

        epoch_loss_valid_avg, valid_true, valid_pred, _ = evaluate(model, valid_loader)
        result_valid = analysis(valid_true, valid_pred)
     
        # print(f"========== Evaluate Valid set, epoch:{str(epoch + 1)} ==========")
        # print("Valid loss: ", epoch_loss_valid_avg)
        # print("Valid RMSE: ", result_valid['RMSE'])
        # print("Valid R", result_valid['R'])

        if  epoch_loss_valid_avg < best_val_loss:
            best_epoch = epoch + 1
            best_val_loss = epoch_loss_valid_avg
            best_rmse, best_r = result_valid['RMSE'], result_valid['R']
            torch.save(model.state_dict(), os.path.join(Model_Path, datasets_name+'_Fold'+str(fold)+'_best_model.pkl'))

    return best_epoch, valid_true, valid_pred, best_rmse, best_r


def cross_validation(feature_path, datasets_name, all_dataframe):

    PDB_names = all_dataframe['ID'].values
    WT_PDB_names = all_dataframe['WT_ID'].values
    labels = all_dataframe['label'].values
    fold = 0
    best_epochs = []
    all_RMSE_list, all_R_list, all_y_true, all_y_pred = [], [], [], []

    # S1131,S4169,S8338,S645
    f = open(File_path + f'CV_STRUC/cv_struc_precluster/divided-{datasets_name}.pkl', 'rb')
    divid = pickle.load(f)
    f.close()
    split_folds = []
    for key in divid.keys():
        split_folds.append((divid[key][0], divid[key][1]))

    for train_index, valid_index in split_folds:

        print("\n\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Train on ------------")

        model = GraphPPIS(C_INPUT_DIM, C_HIDDEN_DIM, G_HEADS, G_INPUT_DIM, G_HIDDEN_DIM, A_HEADS, A_INPUT_DIM, A_HIDDEN_DIM, DROPOUT)
        model.cuda()

        best_epoch, y_true, y_pred, rmse, r = train(feature_path, datasets_name, model, train_dataframe, valid_dataframe, fold + 1)

        # RMSE and R values for each fold
        print(f"\nFold{str(fold + 1)}: \nRMSE:{rmse} \nR:{r}")

        best_epochs.append(str(best_epoch))
        all_RMSE_list.append(rmse)
        all_R_list.append(r)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        fold += 1

    # Average of all Folds
    print("\n\nRMSE mean:", np.mean(all_RMSE_list))
    print("R mean:",np.mean(all_R_list))

    # whole RMSE and R value
    rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    r, p = pearsonr(all_y_true, all_y_pred)

    print("\n\nwhole RMSE mean:", rmse)
    print("whole R mean:", r)

    # np.save(File_path+f'CV_STRUC/cv_struc_eval_list/{datasets_name}_RMSE.npy', all_RMSE_list)
    # np.save(File_path+f'CV_STRUC/cv_struc_eval_list/{datasets_name}_R.npy', all_R_list)
    # np.save(File_path+f'CV_STRUC/cv_struc_eval_list/{datasets_name}_all_y_true.npy', all_y_true)
    # np.save(File_path+f'CV_STRUC/cv_struc_eval_list/{datasets_name}_all_y_pred.npy', all_y_pred)

    return

def main(datasets_list, feature_path):
    print(datasets_list)
    for datasets in datasets_list:
        datasets_name = os.path.splitext(datasets)[0]
        print('\n\n', datasets + ':')

        with open(Dataset_Path + datasets, "rb") as f:
            Train_data = pickle.load(f)

        IDs, WT_IDs, labels = [], [], []        
        for ID in Train_data:
            IDs.append(ID)
            item = Train_data[ID]
            WT_IDs.append(item[0])
            labels.append(item[1])

        train_dic = {"ID": IDs, "WT_ID": WT_IDs, "label": labels}
        train_dataframe = pd.DataFrame(train_dic)
        cross_validation(feature_path, datasets_name, train_dataframe)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some datasets.')
    parser.add_argument('datasets', metavar='N', type=str, nargs='+',
                        help='the dataset names')
    parser.add_argument('--feature_path', dest='feature_path', default='./Feature/Feature_2M/',
                        help='path to the feature directory (default: ./Feature/Feature_2M/)')

    args = parser.parse_args()

    main(args.datasets, args.feature_path)
