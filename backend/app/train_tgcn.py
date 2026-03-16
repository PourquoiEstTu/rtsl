import logging
import os
from sys import exit

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset

import utils
from configs import Config
from tgcn_model import GCN_muti_att
from sign_dataset import Sign_Dataset
from train_utils import train, validation

import itertools
from copy import deepcopy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# def run(split_file, pose_data_root, configs, save_model_to=None):
#     epochs = configs.max_epochs
#     log_interval = configs.log_interval
#     num_samples = configs.num_samples
#     hidden_size = configs.hidden_size
#     drop_p = configs.drop_p
#     num_stages = configs.num_stages

#     # setup dataset
#     train_dataset = Sign_Dataset(index_file_path=split_file, split=['train', 'val'], pose_root=pose_data_root,
#                                  img_transforms=None, video_transforms=None, num_samples=num_samples)

#     train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
#                                                     shuffle=True)

#     val_dataset = Sign_Dataset(index_file_path=split_file, split='test', pose_root=pose_data_root,
#                                img_transforms=None, video_transforms=None,
#                                num_samples=num_samples,
#                                sample_strategy='k_copies')
#     val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
#                                                   shuffle=True)

#     logging.info('\n'.join(['Class labels are: '] + [(str(i) + ' - ' + label) for i, label in
#                                                      enumerate(train_dataset.label_encoder.classes_)]))

#     # REPLACING gpu WITH cpu: 
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # setup the model
#     model = GCN_muti_att(input_feature=num_samples*2, hidden_feature=,
#                          num_class=len(train_dataset.label_encoder.classes_), p_dropout=drop_p, num_stage=num_stages).to(device)

#     # setup training parameters, learning rate, optimizer, scheduler
#     lr = configs.init_lr
#     # optimizer = optim.SGD(vgg_gru.parameters(), lr=lr, momentum=0.00001)
#     optimizer = optim.Adam(model.parameters(), lr=lr, eps=configs.adam_eps, weight_decay=configs.adam_weight_decay)

#     # record training process
#     epoch_train_losses = []
#     epoch_train_scores = []
#     epoch_val_losses = []
#     epoch_val_scores = []

#     best_test_acc = 0
#     # start training
#     for epoch in range(int(epochs)):
#         # train, test model

#         print('start training.')
#         train_losses, train_scores, train_gts, train_preds = train(log_interval, model,
#                                                                    train_data_loader, optimizer, epoch)
#         print('start testing.')
#         val_loss, val_score, val_gts, val_preds, incorrect_samples = validation(model,
#                                                                                 val_data_loader, epoch,
#                                                                                 save_to=save_model_to)
#         # print('start testing.')
#         # val_loss, val_score, val_gts, val_preds, incorrect_samples = validation(model,
#         #                                                                         val_data_loader, epoch,
#         #                                                                         save_to=save_model_to)

#         logging.info('========================\nEpoch: {} Average loss: {:.4f}'.format(epoch, val_loss))
#         logging.info('Top-1 acc: {:.4f}'.format(100 * val_score[0]))
#         logging.info('Top-3 acc: {:.4f}'.format(100 * val_score[1]))
#         logging.info('Top-5 acc: {:.4f}'.format(100 * val_score[2]))
#         logging.info('Top-10 acc: {:.4f}'.format(100 * val_score[3]))
#         logging.info('Top-30 acc: {:.4f}'.format(100 * val_score[4]))
#         logging.debug('mislabelled val. instances: ' + str(incorrect_samples))

#         # save results
#         epoch_train_losses.append(train_losses)
#         epoch_train_scores.append(train_scores)
#         epoch_val_losses.append(val_loss)
#         epoch_val_scores.append(val_score[0])

#         # save all train test results
#         np.save('outputs/epoch_training_losses.npy', np.array(epoch_train_losses))
#         np.save('outputs/epoch_training_scores.npy', np.array(epoch_train_scores))
#         np.save('outputs/epoch_test_loss.npy', np.array(epoch_val_losses))
#         np.save('outputs/epoch_test_score.npy', np.array(epoch_val_scores))

#         if val_score[0] > best_test_acc:
#             best_test_acc = val_score[0]
#             best_epoch_num = epoch

#             torch.save(model.state_dict(), f"splits/{subset.replace('asl','')}/{best_epoch_num}_{best_test_acc}.pth")
#             # print("Saved model!")
#             # exit(0)

#     # why below error???

#     utils.plot_curves()

#     class_names = train_dataset.label_encoder.classes_
#     utils.plot_confusion_matrix(train_gts, train_preds, classes=class_names, normalize=False,
#                                 save_to='outputs/train-conf-mat')
#     utils.plot_confusion_matrix(val_gts, val_preds, classes=class_names, normalize=False, save_to='outputs/val-conf-mat')

def run(split_file, pose_data_root, configs, save_model_to=None):
    epochs = configs.max_epochs
    log_interval = configs.log_interval
    num_samples = configs.num_samples
    hidden_size = configs.hidden_size
    drop_p = configs.drop_p
    num_stages = configs.num_stages

    # setup dataset
    train_dataset = Sign_Dataset(index_file_path=split_file, split=['train', 'val'], pose_root=pose_data_root,
                                 img_transforms=None, video_transforms=None, num_samples=num_samples)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size, shuffle=True)

    val_dataset = Sign_Dataset(index_file_path=split_file, split='test', pose_root=pose_data_root,
                               img_transforms=None, video_transforms=None,
                               num_samples=num_samples, sample_strategy='k_copies')
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size, shuffle=True)

    logging.info('\n'.join(['Class labels are: '] + [(str(i) + ' - ' + label) for i, label in
                                                     enumerate(train_dataset.label_encoder.classes_)]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN_muti_att(input_feature=num_samples*2, hidden_feature=hidden_size,
                         num_class=len(train_dataset.label_encoder.classes_),
                         p_dropout=drop_p, num_stage=num_stages).to(device)

    lr = configs.init_lr
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=configs.adam_eps, weight_decay=configs.adam_weight_decay)

    # early stopping config
    patience = 15          # stop if no improvement for this many epochs
    min_delta = 0.001      # minimum improvement to count as improvement
    early_stop_counter = 0
    best_test_acc = 0

    epoch_train_losses, epoch_train_scores = [], []
    epoch_val_losses,   epoch_val_scores   = [], []

    for epoch in range(int(epochs)):
        print('start training.')
        train_losses, train_scores, train_gts, train_preds = train(log_interval, model,
                                                                   train_data_loader, optimizer, epoch)
        print('start testing.')
        val_loss, val_score, val_gts, val_preds, incorrect_samples = validation(model,
                                                                                val_data_loader, epoch,
                                                                                save_to=save_model_to)

        logging.info('========================\nEpoch: {} Average loss: {:.4f}'.format(epoch, val_loss))
        logging.info('Top-1 acc: {:.4f}'.format(100 * val_score[0]))
        logging.info('Top-3 acc: {:.4f}'.format(100 * val_score[1]))
        logging.info('Top-5 acc: {:.4f}'.format(100 * val_score[2]))
        logging.info('Top-10 acc: {:.4f}'.format(100 * val_score[3]))
        logging.info('Top-30 acc: {:.4f}'.format(100 * val_score[4]))
        logging.debug('mislabelled val. instances: ' + str(incorrect_samples))

        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_val_losses.append(val_loss)
        epoch_val_scores.append(val_score[0])

        np.save('outputs/epoch_training_losses.npy', np.array(epoch_train_losses))
        np.save('outputs/epoch_training_scores.npy', np.array(epoch_train_scores))
        np.save('outputs/epoch_test_loss.npy',       np.array(epoch_val_losses))
        np.save('outputs/epoch_test_score.npy',      np.array(epoch_val_scores))

        # save best model + early stopping check
        if val_score[0] > best_test_acc + min_delta:
            best_test_acc = val_score[0]
            best_epoch_num = epoch
            early_stop_counter = 0  # reset counter on improvement

            torch.save(model.state_dict(), f"splits/{subset.replace('asl','')}/{best_epoch_num}_{best_test_acc:.4f}.pth")
            print(f"New best model saved: epoch={best_epoch_num}, val_acc={best_test_acc:.4f}")
        else:
            early_stop_counter += 1
            print(f"No improvement. Early stop counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best val acc: {best_test_acc:.4f}")
            logging.info(f"Early stopping triggered at epoch {epoch}. Best val acc: {best_test_acc:.4f}")
            break

    utils.plot_curves()

    class_names = train_dataset.label_encoder.classes_
    utils.plot_confusion_matrix(train_gts, train_preds, classes=class_names, normalize=False,
                                save_to='outputs/train-conf-mat')
    utils.plot_confusion_matrix(val_gts, val_preds, classes=class_names, normalize=False,
                                save_to='outputs/val-conf-mat')

# grid search for hyperparameters
def grid_search(split_file, pose_data_root, config_file):
    
    # define hyperparameter grid
    param_grid = {
        'hidden_size':    [64, 128, 256],
        'num_stages':     [2, 4, 8, 20],
        'drop_p':         [0.3, 0.5, 0.7],
        'init_lr':        [0.001, 0.0001],
        'weight_decay':   [0, 1e-4, 1e-3],
        'batch_size':     [32, 64],
    }

    # fixed params  
    max_epochs = 50
    patience   = 10
    min_delta  = 0.001
    num_samples = 50

    # generate all combinations
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    total = len(combinations)
    print(f"Total combinations: {total}")

    results = []
    best_overall_acc  = 0
    best_overall_params = None

    for run_idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n[{run_idx+1}/{total}] Testing: {params}")

        try:
            # setup datasets
            train_dataset = Sign_Dataset(
                index_file_path=split_file, split=['train', 'val'],
                pose_root=pose_data_root, img_transforms=None,
                video_transforms=None, num_samples=num_samples
            )
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset, batch_size=params['batch_size'], shuffle=True
            )
            val_dataset = Sign_Dataset(
                index_file_path=split_file, split='test',
                pose_root=pose_data_root, img_transforms=None,
                video_transforms=None, num_samples=num_samples,
                sample_strategy='k_copies'
            )
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset, batch_size=params['batch_size'], shuffle=False
            )

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = GCN_muti_att(
                input_feature=num_samples * 2,
                hidden_feature=params['hidden_size'],
                num_class=len(train_dataset.label_encoder.classes_),
                p_dropout=params['drop_p'],
                num_stage=params['num_stages']
            ).to(device)

            optimizer = optim.Adam(
                model.parameters(),
                lr=params['init_lr'],
                eps=1e-8,
                weight_decay=params['weight_decay']
            )

            # training loop with early stopping
            best_acc      = 0
            es_counter    = 0
            best_epoch    = 0

            for epoch in range(max_epochs):
                train(10, model, train_loader, optimizer, epoch)
                val_loss, val_score, _, _, _ = validation(model, val_loader, epoch, save_to=None)

                if val_score[0] > best_acc + min_delta:
                    best_acc    = val_score[0]
                    best_epoch  = epoch
                    es_counter  = 0
                else:
                    es_counter += 1

                if es_counter >= patience:
                    print(f"  Early stop at epoch {epoch}, best acc: {best_acc:.4f}")
                    break

            results.append({**params, 'best_acc': best_acc, 'best_epoch': best_epoch})
            print(f"  Result: val_acc={best_acc:.4f} at epoch {best_epoch}")

            if best_acc > best_overall_acc:
                best_overall_acc    = best_acc
                best_overall_params = {**params, 'best_epoch': best_epoch}
                print(f"  *** New best overall: {best_overall_acc:.4f} ***")

        except Exception as e:
            print(f"  Failed: {e}")
            results.append({**params, 'best_acc': 0, 'best_epoch': -1, 'error': str(e)})
            continue

    # sort and display results
    results.sort(key=lambda x: x['best_acc'], reverse=True)

    print("\n" + "="*60)
    print("GRID SEARCH COMPLETE")
    print("="*60)
    print(f"Best params: {best_overall_params}")
    print(f"Best val acc: {best_overall_acc:.4f}")
    print("\nTop 10 results:")
    for r in results[:10]:
        print(f"  acc={r['best_acc']:.4f} | {r}")

    # save results to csv
    import csv
    with open('outputs/grid_search_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("\nFull results saved to outputs/grid_search_results.csv")

    return best_overall_params, results


if __name__ == "__main__":
    root        = '/u50/chandd9/capstone/rtsl/backend/app/'
    subset      = 'asl100'
    split_file  = "/u50/chandd9/capstone/rtsl/backend/app/asl_citizen/asl_citizens100.json"
    pose_data_root = "/u50/chandd9/downloads/asl_cit_pt"
    config_file = os.path.join(root, 'config.ini')

    best_params, all_results = grid_search(split_file, pose_data_root, config_file)
    print(f"\nBest hyperparameters found: {best_params}")


# if __name__ == "__main__":
#     root = '/u50/chandd9/capstone/rtsl/backend/app/' # My path of the project root directory

#     subset = 'asl100' # using asl100 subset first.

#     # split_file = os.path.join(root, 'splits/{}.json'.format(subset))
#     split_file = "/u50/chandd9/capstone/rtsl/backend/app/asl_citizen/asl_citizens100.json"
#     # pose_data_root = "/u50/quyumr/archive/asl-live-tl-features"
#     pose_data_root = "/u50/chandd9/downloads/asl_cit_pt"
#     config_file = os.path.join(root, 'config.ini')
#     configs = Config(config_file)

#     # print('outputs/{}.log'.format(os.path.basename(config_file)[:-4]))
#     logging.basicConfig(filename='outputs/{}.log'.format(os.path.basename(config_file)[:-4]), level=logging.DEBUG, filemode='w+')

#     logging.info('Calling main.run()')
#     run(split_file=split_file, configs=configs, pose_data_root=pose_data_root)
#     logging.info('Finished main.run()')
#     utils.plot_curves()
