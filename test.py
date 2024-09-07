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
from scipy.stats import kendalltau
from scipy.stats import spearmanr

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
    kendall_correlation, kendall_p_value = kendalltau(y_true, y_pred)
    Spearman_correlation, Spearman_p_value = spearmanr(y_true, y_pred)
    results = {
        'mae':mae,
        'mse':mse,
        'RMSE':rmse,
        'r2':r2,
        'R': r,
        'P-value':p,
        'Kendalls_tau': kendall_correlation,
        'Spearmans_Rank': Spearman_correlation,
    }

    return results


def test(dataset_name, feature_path, model_path, test_dataframe):
    test_loader = DataLoader(dataset=ProDataset(feature_path, test_dataframe), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    for model_name in sorted(os.listdir(model_path)):
        print(model_name)

        model = GraphPPIS(C_INPUT_DIM, C_HIDDEN_DIM, G_HEADS, G_INPUT_DIM, G_HIDDEN_DIM, A_HEADS, A_INPUT_DIM, A_HIDDEN_DIM, DROPOUT)
        model.cuda()

        model.load_state_dict(torch.load(model_path + model_name, map_location=torch.device('cpu'))) # map_location='cuda:1' 

        epoch_loss_test_avg, test_true, test_pred, pred_dict = evaluate(model, test_loader)

        result_test = analysis(test_true, test_pred)
        
        # True and Pred
        # np.save(File_path + f"Blind_test/True-Pred_list/{dataset_name}_true_list.npy", np.array(test_true))
        # np.save(File_path + f"Blind_test/True-Pred_list/{dataset_name}_pred_list.npy", np.array(test_pred))
        
        print("========== Evaluate Test set ==========")
        print("Test loss: ", epoch_loss_test_avg)
        print("Test MAE: ", result_test['mae'])
        print("Test RMSE: ", result_test['RMSE'])
        print("Test R",  result_test['R'])
        print("Test Kendall's tau correlation",  result_test['Kendalls_tau'])
        print("Spearman's Rank Correlation",  result_test['Spearmans_Rank'])
        print()


def main(Test_datasets_list, feature_path, model_path):
    print(Test_datasets_list)
    
    for dataset in Test_datasets_list:
        print(f"\n-------------Evaluate GraphPPIS on {dataset}-------------\n\n")
        dataset_name = os.path.splitext(dataset)[0]

        IDs, WT_IDs, labels = [], [], []   
        with open(Dataset_Path + dataset, "rb") as f:
            Test_data = pickle.load(f)                 
        for ID in Test_data:
            IDs.append(ID)
            item = Test_data[ID]
            WT_IDs.append(item[0])
            labels.append(item[1])
        test_dic = {"ID": IDs, "WT_ID": WT_IDs, "label": labels}
        test_dataframe = pd.DataFrame(test_dic)
        test(dataset_name, feature_path, model_path, test_dataframe)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some datasets.')
    parser.add_argument('datasets', metavar='N', type=str, nargs='+',
                        help='the test dataset names')
    parser.add_argument('--feature_path', dest='feature_path', default='./Feature/Feature_10A/',
                        help='path to the feature directory (default: ./Feature/Feature_10A/)')
    parser.add_argument('--model_path', dest='model_path', default='../Train_Model/M1340_Model_2/',
                        help='path to the model directory (default: ../Train_Model/M1340_Model_2/)')

    args = parser.parse_args()

    main(args.datasets, args.feature_path, args.model_path)

