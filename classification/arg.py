import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, type=str, required=True, help="Path to .jsonl file containing dataset")
    parser.add_argument("--mode", default="scrub", type=str, required=True, help="which train mode: 'train-forget' or 'train-new'")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_unlearn", action="store_true", help="Whether to run unlearning.")
    parser.add_argument("--model_path", default=None, type=str, required=False, help="path to model file")
    parser.add_argument("--new_model_path", default=None, type=str, required=False, help="path to model file")
    parser.add_argument("--forget_model_path", default=None, type=str, required=False, help="path to model file")
    parser.add_argument("--save_path", type=str, required=False, help="path to save the model")
    parser.add_argument("--sample_ratio", default=None, type=float, required=False, help="subsample part of training data")
    parser.add_argument('--file_removals', type=str, default=None, help='file storing indices for sample removals')
    parser.add_argument('--file_as_new', type=str, default=None, help='file storing indices for those as new samples')

    parser.add_argument("--test_metric", default="f1", type=str, required=False, help="'f1' or 'kl', if kl, need two models")
    parser.add_argument("--test_data", default="test", type=str, required=False, help="'test' or 'forget' or 'remain'")

    parser.add_argument("--seed", default=0xDEADBEEF, type=int, required=False,
                        help="seed for random number generation, default 0xDEADBEEF")
    parser.add_argument("--max_seq_len", default=128, type=int, required=False,
                        help="maximum sequence length in transformer, default 128")

    parser.add_argument("--batch_size", default=256, type=int, required=False, help="training batch size, default 8")
    parser.add_argument("--max_update", default=50000, type=int, required=False,
                        help="maximum update numbers")
    parser.add_argument("--max_epoch", default=1, type=int, required=False,
                        help="maximum epoch numbers")
    parser.add_argument("--inner_step", default=10, type=int, required=False,
                        help="update every how many batches in residual data")
    parser.add_argument("--print_loss", default=500, type=int, required=False,
                        help="print loss every update numbers")
    parser.add_argument("--eval_update", default=1000, type=int, required=False,
                        help="eval every update numbers")
    parser.add_argument("--save_update", default=10000, type=int, required=False,
                        help="save model after eval")
    parser.add_argument("--weight_decay", default=0.01, type=float, required=False,
                        help="AdamW weight decay, default 0.01")
    parser.add_argument("--learning_rate", default=5e-5, type=float, required=False,
                        help="AdamW learning rate, default 5e-5")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, required=False, help="AdamW epsilon, default 1e-8")
    parser.add_argument("--warmup_steps", default=0, type=int, required=False,
                        help="Warmup steps for learning rate schedule, default 0")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, required=False,
                        help="max norm for gradient clipping, default 1.0")
    parser.add_argument("--retain_loss_ratio", default=0.1, type=float, required=False, help="add to retain loss")

    args, _ = parser.parse_known_args()
    return args, parser

