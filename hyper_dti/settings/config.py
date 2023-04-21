
import time
import numpy as np


def get_configs(parser):

    # Parse commandline arguments
    parser.add_argument("--data_dir", default='hyper_dti/data/', type=str, help='Path to data directory.')
    parser.add_argument("--name", default='run', type=str, help='Experiment name.')
    parser.add_argument("--wandb_username", default='none', type=str)
    parser.add_argument("--seed", default=None, type=int)

    # Dataset
    parser.add_argument("--dataset", default='Lenselink', type=str, choices=['Lenselink', 'KIBA', 'Davis'])
    parser.add_argument("--subset", default=False, action='store_true',
                        help='Take subset of full data for debugging. Only for Lenselink dataset.')
    parser.add_argument("--split", default='lpo', type=str, help='Splitting strategy.',
                        choices=['random',
                                 'temporal', 'leave-drug-out', 'ldo', 'leave-drug-cluster-out', 'ldco', 'lcco', 'cold-drug',
                                 'leave-target-out', 'lto', 'lpo', 'leave-protein-out', 'cold-target'
                                 'leave-drug-target-out', 'ldto', 'cold'])
    parser.add_argument("--test_fold", default=None, type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help='Fixed fold to use for testing during cross-validation predefined benchmarks.')
    parser.add_argument("--standardize_target", default=False, action='store_true')
    parser.add_argument("--standardize_drug", default=False, action='store_true')

    # Evaluation
    parser.add_argument("--loss_function", default='MAE', type=str, choices=['BCE', 'MSE', 'MAE'])
    parser.add_argument("--raw_reg_labels", default=False, action='store_true',
                        help='Turns off normalization of regression labels.')
    parser.add_argument("--test", default=False, action='store_true', help='Only test accompanied checkpointed model.')
    parser.add_argument("--checkpoint", default='', type=str,
                        help='Name of experiment for which model checkpoint to test.')
    parser.add_argument("--transfer", default=False, action='store_true',
                        help='Option for transferring checkpointed model trained on Lenselink dataset to test sets of '
                             'either of the other datasets.')
    parser.add_argument("--checkpoint_metric", default='Loss', type=str, choices=['Loss', 'MCC', 'last'],
                        help='Which metric checkpointed to use for eval.')                  # TODO integrate in trainer
    parser.add_argument("--cross_validate", default=False, action='store_true',
                        help='Cross-validate on all 10-folds of the given split (re-runs if split == temporal).')
    parser.add_argument("--folds_list", default=None, type=str)

    # Standard training arguments
    parser.add_argument("--epochs", default=1000, type=int, help='Number of epochs.')
    parser.add_argument("--patience", default=100, type=int,
                        help='Number of epochs without improvement in loss before cancelling training.')
    parser.add_argument("--batch_size", "-bs", default=32, type=int, help='Number of samples per batch.')
    parser.add_argument("--main_batch_size", default=32, type=int,
                        help='Mini-batching, number of drug compounds per batch.')
    parser.add_argument("--oversampling", default=32, type=int,
                        help='How many compounds to oversample if not enough are available for a given target.')
    parser.add_argument("--learning_rate", "-lr", default=0.0001, type=float)
    parser.add_argument("--scheduler", default='ReduceLROnPlateau', type=str, choices=['None', 'ReduceLROnPlateau'])
    parser.add_argument("--lr_patience", default=30, type=int)
    parser.add_argument("--lr_decay", default=0.5, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--momentum", default=0.8, type=float)
    parser.add_argument("--num_workers", default=0, type=int)

    parser.add_argument("--architecture", default='HyperPCM', type=str, help='Model architecture.',
                        choices=['DeepPCM', 'HyperPCM'])                                 # TODO add tabular baselines?
    parser.add_argument("--init", default='pwi', type=str, help='How to initialize HyperNetwork parameters.',
                        choices=['default', 'manual', 'pwi', 'manual_pwi'])
    parser.add_argument("--norm", default=None, type=str, choices=['learned'],
                        help='Normalization technique.')
    parser.add_argument("--selu", default=False, action='store_true', help='SeLU activation in HyperNet.')
    # HyperNetwork module
    parser.add_argument("--fcn_hidden_dim", default=256, type=int,
                        help='Number of hidden channels in HyperNetwork layers.')
    parser.add_argument("--fcn_layers", default=1, type=int, help='Number of layers in HyperNetwork.')
    # Classification module
    parser.add_argument("--cls_hidden_dim", default=1024, type=int,
                        help='Number of hidden channels in main CLS layers.')
    parser.add_argument("--cls_layers", default=1, type=int, help='Number of layers in main CLS.')

    # Context module
    parser.add_argument("--hopfield_QK_dim", default=512, type=int,
                        help='Query-Key dimension in Hopfield module.')
    parser.add_argument("--hopfield_heads", default=8, type=int,
                        help='Number of heads in Hopfield module.')
    parser.add_argument("--hopfield_beta", default=0.044194173824159216, type=float,
                        help='Beta, scaling in Hopfield module.')
    parser.add_argument("--hopfield_dropout", default=0.5, type=float,
                        help='Dropout in Hopfield module.')
    parser.add_argument("--hopfield_skip", default=False, action='store_true')
    parser.add_argument("--drug_context", default=False, action='store_true')
    parser.add_argument("--target_context", default=False, action='store_true')
    parser.add_argument("--remove_batch", default=False, action='store_true',
                        help='Removes current batch from training set for context module.')

    # Encoders
    parser.add_argument("--drug_encoder", default='CDDD', type=str, help='Drug encoder.',
                        choices=['MolBert', 'CDDD'])
    parser.add_argument("--target_encoder", default='SeqVec', type=str, help='Target encoder.',
                        choices=['UniRep', 'SeqVec', 'ProtBert', 'ProtT5'])

    args = parser.parse_args()

    # Collected configurations
    config = vars(args)

    # Adding non-optional configurations
    config['name'] = f'{args.split}/{args.name}_{time.strftime("%H%M")}'
    if config['seed'] is None or config['seed'] == -1:
        config['seed'] = np.random.randint(1, 10000)
    if args.test_fold is not None:
        valid = args.test_fold - 1 if args.test_fold != 0 else 9
        config['folds'] = {'valid': valid, 'test': args.test_fold}
    else:
        config['folds'] = None

    return config

