import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.",
                        )
    parser.add_argument("--unlearn_model_dir", default=None, type=str,
                        help="The model directory of the unlearned model, for evaluation.",
                        )
    parser.add_argument("--train_file", default=None, type=str, help="The input training file.")
    parser.add_argument("--dev_file", default=None, type=str, help="The input evaluation file.")
    parser.add_argument("--test_file", default=None, type=str, help="The input testing file.")

    parser.add_argument("--model_checkpoint",
                        default="opus_mt_de_en", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models, opus_mt_de_en or bart-base",
                        )
    parser.add_argument("--tokenizer_checkpoint",
                        default="opus_mt_de_en", type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models, opus_mt_de_en or bart-base",
                        )
    parser.add_argument("--not_reset_model", action="store_true")
    parser.add_argument("--use_bart_init", action="store_true")
    parser.add_argument("--bart_freeze_update", default=2000, type=int)

    parser.add_argument("--max_input_length", default=512, type=int)
    parser.add_argument("--max_target_length", default=512, type=int)

    parser.add_argument("--source", default="de", type=str)
    parser.add_argument("--target", default="en", type=str)

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_evaluate", action="store_true", help="Whether to evaluate result differences between 2 models.")

    # Other parameters
    parser.add_argument("--learning_rate", default=5e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--update_freq", default=8, type=int, help="Accumulate gradients before updating.")
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--dropout", default=0.3, type=float)
    parser.add_argument("--beam", default=5, type=int)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--adam_beta1", default=0.9, type=float,
                        help="Epsilon for Adam optimizer."
                        )
    parser.add_argument("--adam_beta2", default=0.98, type=float,
                        help="Epsilon for Adam optimizer."
                        )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer."
                        )
    parser.add_argument("--warmup_steps", default=4000, type=int,
                        help="num steps of training to perform linear learning rate warmup for."
                        )
    parser.add_argument("--weight_decay", default=0.0001, type=float,
                        help="Weight decay if we apply some."
                        )
    parser.add_argument("--lr_schedule", default="inverse_sqrt", type=str,
                        help="lr scheduler for optimizer."
                        )
    args, _ = parser.parse_known_args()
    return args, parser

