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
Model_Path = "./CV_ONE/cv_one_model/"

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

def train(feature_path, WT_name, datasets_name, datasets_num, model, train_dataframe, valid_dataframe, fold = 0):
    train_loader = DataLoader(dataset=ProDataset(feature_path, train_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=ProDataset(feature_path, valid_dataframe), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    loss_train_list, loss_val_list = [], []
    early_epoch = 10
    stop_threshold = 0
    best_epoch = -1
    best_val_loss = 10000
    best_model = None

    for epoch in range(NUMBER_EPOCHS):
        model.train()

        epoch_loss_train_avg = train_one_epoch(model, train_loader)
        loss_train_list.append(epoch_loss_train_avg)

        _, train_true, train_pred, _ = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred)
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        print("Train loss: ", epoch_loss_train_avg)
        print("Train RMSE: ", result_train['RMSE'])
        print("Train R", result_train['R'])

        epoch_loss_valid_avg, valid_true, valid_pred, _ = evaluate(model, valid_loader)
        loss_val_list.append(epoch_loss_valid_avg)

        # if  epoch_loss_valid_avg < best_val_loss:
        #     best_epoch = epoch + 1
        #     best_val_loss = epoch_loss_valid_avg
        #     torch.save(model.state_dict(), os.path.join(Model_Path+f"{datasets_num}_model/", WT_name+'_'+datasets_name+'_Fold'+str(fold)+'_best_model.pkl'))

        if  epoch_loss_valid_avg < best_val_loss:
            best_epoch = epoch + 1
            best_val_loss = epoch_loss_valid_avg
            # torch.save(model.state_dict(), os.path.join(Model_Path+f"{datasets_num}_model/", WT_name+'_'+datasets_name+'_Fold'+str(fold)+'_best_model.pkl')) 
            stop_threshold = 0
        else:
            stop_threshold += 1
            if epoch > 10 and stop_threshold > early_epoch:
                break

    print(f"\n\ntrain epoch is {epoch + 1}")  

    return best_epoch, valid_true, valid_pred


def main(datasets_lists, feature_path):

    for datasets in datasets_lists:
        datasets_name = os.path.splitext(datasets)[0]
        datasets_num = datasets_name.split('_')[1]

        # M1707,S4169,S4191,S8338
        WT_csv = File_path + f"CV_ONE/cv_one_wt_csv/WT_{datasets_num}.csv"
        WT_PDB_lists = (pd.read_csv(WT_csv))['WT_name'].tolist()
        
        print('\n\n', datasets + ':')

        IDs, WT_IDs, labels, only_one_lists = [], [], [], []
        best_epochs, all_RMSE_list, all_R_list, all_y_true, all_y_pred = [], [], [], [], []
        
        # get dataframe
        with open(Dataset_Path + datasets, "rb") as f:
            Train_data = pickle.load(f)        
        for ID in Train_data:
            IDs.append(ID)
            item = Train_data[ID]
            WT_IDs.append(item[0])
            labels.append(item[1])
        ALL_dic = {"ID": IDs, "WT_ID": WT_IDs, "label": labels}
        ALL_dataframe = pd.DataFrame(ALL_dic)
        
        # Perform the leave-one-structure-out cross-validation
        fold = 0
        for WT_name in WT_PDB_lists[:]:
            
            print(f"\n\n----- Train on Fold{str(fold+1)}: {WT_name} -----")

            valid_dataframe = pd.DataFrame()
            train_dataframe = pd.DataFrame()
            for index,row in ALL_dataframe.iterrows():
                parts = row['ID'].split('_')
                if len(parts)>6:
                    mask_name = "WT2_" + parts[1]+'_'+parts[2]+'_'+parts[3]+'_'+parts[4]
                else:
                    mask_name = "WT2_" + parts[1]+'_'+parts[2]+'_'+parts[3]
                
                if mask_name == WT_name:
                    valid_dataframe = pd.concat([valid_dataframe, row.to_frame().T], ignore_index=True)
                else:
                    train_dataframe = pd.concat([train_dataframe, row.to_frame().T], ignore_index=True)
            
            # The number of mutations is 1
            if len(valid_dataframe) == 1:  
                print(f"train_list is {len(train_dataframe)}, valid_list is {len(valid_dataframe)}")

                model = GraphPPIS(C_INPUT_DIM, C_HIDDEN_DIM, G_HEADS, G_INPUT_DIM, G_HIDDEN_DIM, A_HEADS, A_INPUT_DIM, A_HIDDEN_DIM, DROPOUT)
                model.cuda()

                # preload
                train_loader = DataLoader(dataset=ProDataset(feature_path, train_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
                valid_loader = DataLoader(dataset=ProDataset(feature_path, valid_dataframe), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
                model.load_state_dict(torch.load(Model_Path+f"{datasets_num}_model/"+WT_name+"_"+datasets_name+"_Fold"+str(fold+1)+"_best_model.pkl"))
                epoch_loss_valid_avg, y_true, y_pred, _ = evaluate(model, valid_loader)
                # preload

                # # train
                # best_epoch, y_true, y_pred = train(feature_path, WT_name, datasets_name, datasets_num, model, train_dataframe, valid_dataframe, fold + 1)
                # # train

                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)

                fold += 1
                only_one_lists.append(WT_name)
                print(f"\nThe number of mutations in {WT_name} was 1.")              
        
            else:
                print(f"train_list is {len(train_dataframe)}, valid_list is {len(valid_dataframe)}")

                model = GraphPPIS(C_INPUT_DIM, C_HIDDEN_DIM, G_HEADS, G_INPUT_DIM, G_HIDDEN_DIM, A_HEADS, A_INPUT_DIM, A_HIDDEN_DIM, DROPOUT)
                model.cuda()

                # preload
                train_loader = DataLoader(dataset=ProDataset(feature_path, train_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
                valid_loader = DataLoader(dataset=ProDataset(feature_path, valid_dataframe), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)               
                model.load_state_dict(torch.load(Model_Path+f"{datasets_num}_model/"+WT_name+"_"+datasets_name+"_Fold"+str(fold+1)+"_best_model.pkl"))
                epoch_loss_valid_avg, y_true, y_pred, _ = evaluate(model, valid_loader)
                # preload

                # # train
                # best_epoch, y_true, y_pred = train(feature_path, WT_name, datasets_name, datasets_num, model, train_dataframe, valid_dataframe, fold + 1)
                # # train

                result_valid = analysis(y_true, y_pred)
                               
                # RMSE and R values for each fold
                rmse = result_valid['RMSE']
                r = result_valid['R']
                print(f"\nRMSE:{rmse},  R:{r}")

                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)

                fold += 1
        
        # whole RMSE and R value
        rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
        r, p = pearsonr(all_y_true, all_y_pred)

        print("\n\nwhole RMSE mean:", rmse)
        print("whole R mean:", r)

        # np.save(File_path+f"CV_ONE/cv_one_eval_list/{datasets_name}_all_y_true.npy", all_y_true)
        # np.save(File_path+f"CV_ONE/cv_one_eval_list/{datasets_name}_all_y_pred.npy", all_y_pred)
        
    print(f"\n\nThe number of mutations to 1 is {len(only_one_lists)}.")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some datasets.')
    parser.add_argument('datasets', metavar='N', type=str, nargs='+',
                        help='the dataset names')
    parser.add_argument('--feature_path', dest='feature_path', default='./Feature/Feature_2M/',
                        help='path to the feature directory (default: ./Feature/Feature_2M/)')


    args = parser.parse_args()

    main(args.datasets, args.feature_path)