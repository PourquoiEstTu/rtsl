import logging
import os
# from sys import exit

import numpy as np
import torch
import torch.optim as optim
# from torch.utils.data import Dataset

import utils
from configs import Config
from tgcn_model import GCN_muti_att
from sign_dataset import Sign_Dataset
from train_utils import train, validation

import itertools
import matplotlib.pyplot as plt
import csv

from sklearn.metrics import recall_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def normal_run(split_file, pose_data_root, configs, save_model_to=None, labels_to_include=None):
    epochs = configs.max_epochs
    log_interval = configs.log_interval
    num_samples = configs.num_samples
    hidden_size = configs.hidden_size
    drop_p = configs.drop_p
    num_stages = configs.num_stages
    patience = configs.patience
    min_delta = configs.min_delta

    # setup dataset
    train_dataset = Sign_Dataset(index_file_path=split_file, split=['train', 'val'], pose_root=pose_data_root,
                                 img_transforms=None, video_transforms=None, num_samples=num_samples,
                                 labels_to_include=labels_to_include)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                                    shuffle=True)

    val_dataset = Sign_Dataset(index_file_path=split_file, split='test', pose_root=pose_data_root,
                               img_transforms=None, video_transforms=None,
                               num_samples=num_samples, sample_strategy='seq',
                               labels_to_include=labels_to_include)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                                  shuffle=True)

    test_dataset = Sign_Dataset(index_file_path=split_file, split='test', pose_root=pose_data_root,
                                img_transforms=None, video_transforms=None,
                                num_samples=num_samples, sample_strategy='seq',
                                labels_to_include=labels_to_include)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    if labels_to_include:
        print(f"Filtering to {len(labels_to_include)} labels: {labels_to_include}")

    logging.info('\n'.join(['Class labels are: '] + [(str(i) + ' - ' + label) for i, label in
                                                     enumerate(train_dataset.label_encoder.classes_)]))

    # REPLACING gpu WITH cpu: 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # setup the model
    model = GCN_muti_att(input_feature=num_samples*2, hidden_feature=hidden_size,
                         num_class=len(train_dataset.label_encoder.classes_), p_dropout=drop_p, num_stage=num_stages).to(device)

    # setup training parameters, learning rate, optimizer, scheduler
    lr = configs.init_lr
    # optimizer = optim.SGD(vgg_gru.parameters(), lr=lr, momentum=0.00001)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=configs.adam_eps, weight_decay=configs.adam_weight_decay)

    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_val_losses = []
    epoch_val_scores = []

    best_test_acc = 0
    
    # early stopping config
    early_stop_counter = 0
    best_test_acc = 0
    
    # start training
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

            torch.save(model.state_dict(), f"outputs/{configs.subset.replace('asl','')}/{best_epoch_num}_{best_test_acc:.4f}.pth")
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
    utils.plot_confusion_matrix(val_gts, val_preds, classes=class_names, normalize=False, save_to='outputs/val-conf-mat')

    # print test results
    best_model_path = f"outputs/{configs.subset.replace('asl','')}/{best_epoch_num}_{best_test_acc:.4f}.pth"
    model.load_state_dict(torch.load(best_model_path, weights_only=False))
    print(f"Loaded best model from: {best_model_path}")

    print("\n-- evaluation (benchmark on test) --")
    testloader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=configs.batch_size, shuffle=False
    )
    _, scores, _, _, _ = validation(model, testloader, best_epoch_num, save_to=None)

    print(f"Top-1  acc: {100 * scores[0]:.2f}%")
    print(f"Top-3  acc: {100 * scores[1]:.2f}%")
    print(f"Top-5  acc: {100 * scores[2]:.2f}%")
    print(f"Top-10 acc: {100 * scores[3]:.2f}%")
    print(f"Top-30 acc: {100 * scores[4]:.2f}%") # wow top 30 does great!

    # calculate recall
    recall_m = recall_score(val_gts, val_preds, average='macro')
    print(f"Macro-average recall: {recall_m:.4f}")

    recall_w = recall_score(val_gts, val_preds, average='weighted')
    print(f"Weighted-average recall: {recall_w:.4f}")

# grid search for hyperparameters
#     param_grid = {
#         'hidden_size':    [64, 128, 256],
#         'num_stages':     [2, 4, 8, 20],
#         'drop_p':         [0.3, 0.5, 0.7],
#         'init_lr':        [0.001, 0.0001],
#         'weight_decay':   [0, 1e-4, 1e-3],
#         'batch_size':     [16, 32, 64],
#     }

def grid_search(split_file, pose_data_root, config_file, n_runs=1):
    
    param_grid = {
        'hidden_size':  [64, 128, 256],
        'num_stages':   [2, 4, 8, 20],
        'drop_p':       [0.3, 0.5, 0.7],
        'weight_decay': [0, 1e-4, 1e-3],
    }

    # fixed params
    max_epochs  = 100
    patience    = 15
    min_delta   = 0.001
    num_samples = 50
    batch_size  = 32
    init_lr     = 0.001

    keys         = list(param_grid.keys())
    values       = list(param_grid.values())
    combinations = list(itertools.product(*values))
    total        = len(combinations)
    print(f"Total combinations: {total}, runs per combo: {n_runs}, total runs: {total * n_runs}")

    results = []
    best_overall_acc    = 0
    best_overall_params = None

    for run_idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n[{run_idx+1}/{total}] Testing: {params}")

        run_accs = []

        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}")
            try:
                train_dataset = Sign_Dataset(
                    index_file_path=split_file, split=['train', 'val'],
                    pose_root=pose_data_root, img_transforms=None,
                    video_transforms=None, num_samples=num_samples
                )
                train_loader = torch.utils.data.DataLoader(
                    dataset=train_dataset, batch_size=batch_size, shuffle=True
                )
                val_dataset = Sign_Dataset(
                    index_file_path=split_file, split='test',
                    pose_root=pose_data_root, img_transforms=None,
                    video_transforms=None, num_samples=num_samples,
                    sample_strategy='k_copies'
                )
                val_loader = torch.utils.data.DataLoader(
                    dataset=val_dataset, batch_size=batch_size, shuffle=False
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
                    lr=init_lr,
                    eps=1e-8,
                    weight_decay=params['weight_decay']
                )

                best_acc   = 0
                es_counter = 0
                best_epoch = 0

                for epoch in range(max_epochs):
                    train(10, model, train_loader, optimizer, epoch)
                    val_loss, val_score, _, _, _ = validation(model, val_loader, epoch, save_to=None)

                    if val_score[0] > best_acc + min_delta:
                        best_acc   = val_score[0]
                        best_epoch = epoch
                        es_counter = 0
                    else:
                        es_counter += 1

                    if es_counter >= patience:
                        print(f"    Early stop at epoch {epoch}, best acc: {best_acc:.4f}")
                        break

                run_accs.append(best_acc)
                print(f"    Run {run+1} best acc: {best_acc:.4f}")

            except Exception as e:
                print(f"    Run {run+1} failed: {e}")
                continue

        if run_accs:
            avg_acc  = np.mean(run_accs)
            std_acc  = np.std(run_accs)
            best_acc = max(run_accs)
        else:
            avg_acc = std_acc = best_acc = 0

        print(f"  Combo result: avg={avg_acc:.4f} std={std_acc:.4f} best={best_acc:.4f}")

        results.append({
            **params,
            'avg_acc':  avg_acc,
            'std_acc':  std_acc,
            'best_acc': best_acc,
            'all_runs': run_accs
        })

        if avg_acc > best_overall_acc:
            best_overall_acc    = avg_acc
            best_overall_params = {**params, 'avg_acc': avg_acc, 'std_acc': std_acc}
            print(f"  *** New best overall avg: {best_overall_acc:.4f} ***")

    # sort by avg_acc
    results.sort(key=lambda x: x['avg_acc'], reverse=True)

    print("\n" + "="*60)
    print("GRID SEARCH COMPLETE")
    print("="*60)
    print(f"Best params: {best_overall_params}")
    print(f"Best avg val acc: {best_overall_acc:.4f}")
    print("\nTop 10 results:")
    for r in results[:10]:
        print(f"  avg={r['avg_acc']:.4f} std={r['std_acc']:.4f} best={r['best_acc']:.4f} | {r}")

    # save to csv (exclude all_runs details for cleaner csv)
    csv_results = [{k: v for k, v in r.items() if k != 'all_runs'} for r in results]
    with open('outputs/grid_search_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
        writer.writeheader()
        writer.writerows(csv_results)
    print("\nFull results saved to outputs/grid_search_results.csv")

    return best_overall_params, results

def train_and_plot_subsets(subset_configs):
    """
    subset_configs: dict mapping subset size to (split_file, config) tuples
    e.g. {
        100:  ('path/to/asl100.json',  config_100),
        300:  ('path/to/asl300.json',  config_300),
        1000: ('path/to/asl1000.json', config_1000),
        2000: ('path/to/asl2000.json', config_2000),
    }
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_val_scores = {}

    for subset_size, (split_file, configs) in subset_configs.items():
        print(f"\n{'='*60}")
        print(f"Training ASL-{subset_size}")
        print(f"{'='*60}")

        num_samples = configs.num_samples
        hidden_size = configs.hidden_size
        drop_p      = configs.drop_p
        num_stages  = configs.num_stages
        max_epochs  = configs.max_epochs
        patience    = configs.patience

        if not os.path.isfile(split_file):
            print(f"Split file not found: {split_file}, skipping.")
            continue

        if subset_size == 100:
            train_dataset = Sign_Dataset(
                index_file_path=split_file, split=['train', 'val'],
                pose_root="/u50/chandd9/downloads/asl_cit_pt", img_transforms=None,
                video_transforms=None, num_samples=num_samples
            )
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset, batch_size=configs.batch_size,
                shuffle=True, num_workers=4, pin_memory=True
            )
            val_dataset = Sign_Dataset(
                index_file_path=split_file, split='test',
                pose_root="/u50/chandd9/downloads/asl_cit_pt", img_transforms=None,
                video_transforms=None, num_samples=num_samples                
            )
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset, batch_size=configs.batch_size,
                shuffle=False, num_workers=4, pin_memory=True
            )
        else:
            train_dataset = Sign_Dataset(
                index_file_path=split_file, split=['train', 'val'],
                pose_root="/u50/chandd9/downloads/tgcn_data", img_transforms=None,
                video_transforms=None, num_samples=num_samples
            )
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset, batch_size=configs.batch_size,
                shuffle=True, num_workers=4, pin_memory=True
            )
            val_dataset = Sign_Dataset(
                index_file_path=split_file, split='test',
                pose_root="/u50/chandd9/downloads/tgcn_data", img_transforms=None,
                video_transforms=None, num_samples=num_samples
            )
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset, batch_size=configs.batch_size,
                shuffle=False, num_workers=4, pin_memory=True
            )         

        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Classes: {len(train_dataset.label_encoder.classes_)}")
        print(f"Config: hidden={hidden_size}, stages={num_stages}, drop={drop_p}, lr={configs.init_lr}, batch={configs.batch_size}")

        model = GCN_muti_att(
            input_feature=num_samples * 2,
            hidden_feature=hidden_size,
            num_class=len(train_dataset.label_encoder.classes_),
            p_dropout=drop_p,
            num_stage=num_stages
        ).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=configs.init_lr,
            eps=configs.adam_eps,
            weight_decay=configs.adam_weight_decay
        )

        val_scores = []
        best_acc   = 0
        es_counter = 0

        for epoch in range(max_epochs):
            train(configs.log_interval, model, train_loader, optimizer, epoch)
            _, val_score, _, _, _ = validation(model, val_loader, epoch, save_to=None)

            val_scores.append(val_score[0])
            # scheduler.step(val_score[0])

            if val_score[0] > best_acc + configs.min_delta:
                best_acc   = val_score[0]
                es_counter = 0
            else:
                es_counter += 1

            if es_counter >= patience:
                print(f"Early stopping at epoch {epoch}, best acc: {best_acc:.4f}")
                val_scores.extend([best_acc] * (max_epochs - len(val_scores)))
                break

            print(f"ASL-{subset_size} Epoch {epoch+1}/{max_epochs}: val_acc={val_score[0]:.4f}")

        all_val_scores[subset_size] = val_scores
        np.save(f'outputs/val_scores_asl{subset_size}.npy', np.array(val_scores))

    # --- plot ---
    plt.figure(figsize=(12, 7))
    colors = {100: 'blue', 300: 'orange', 1000: 'green', 2000: 'red'}

    for subset_size, scores in all_val_scores.items():
        epochs     = list(range(1, len(scores) + 1))
        best_epoch = int(np.argmax(scores))
        best_score = max(scores) * 100

        plt.plot(epochs, [s * 100 for s in scores],
                 label=f'ASL-{subset_size}',
                 color=colors.get(subset_size, None),
                 linewidth=2)

    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Validation Accuracy (%)', fontsize=13)
    plt.title('Validation Accuracy vs Epochs by Vocabulary Size', fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/subset_comparison.png', dpi=150)
    plt.show()
    print("Plot saved to outputs/subset_comparison.png")

    return all_val_scores

# generate a graph for training curves of all subsets
# if __name__ == "__main__":
#     root           = '/u50/chandd9/capstone/rtsl/backend/app/'
#     pose_data_root = "/u50/chandd9/downloads/tgcn_npy"

#     # load a config per subset
#     config_100  = Config(os.path.join(root, 'config2.ini'))
#     config_300  = Config(os.path.join(root, 'config.ini'))
#     config_1000 = Config(os.path.join(root, 'config.ini'))
#     config_2000 = Config(os.path.join(root, 'config.ini'))

#     subset_configs = {
#         100:  (os.path.join(root, 'asl_citizen/asl_citizens100.json'),  config_100),
#         300:  (os.path.join(root, 'wlasl/asl300.json'),  config_300),
#         1000: (os.path.join(root, 'wlasl/asl1000.json'), config_1000),
#         2000: (os.path.join(root, 'wlasl/asl2000.json'), config_2000),
#     }

#     train_and_plot_subsets(
#         subset_configs=subset_configs
#     )

# Test run
if __name__ == "__main__":
    config_file = 'configs/config3.ini'
    configs = Config(config_file)
    root = configs.root

    logging.basicConfig(filename='outputs/{}.log'.format(os.path.basename(config_file)[:-4]), level=logging.DEBUG, filemode='w+')

    logging.info('Calling main.run()')
    normal_run(split_file=configs.split_file, configs=configs, pose_data_root=configs.pose_data_root, labels_to_include=["YES", "NO", "HELP", "EAT", "DRINK", "WANT", "FINISH", "GO", "WHAT", "WHO"])
    logging.info('Finished main.run()')