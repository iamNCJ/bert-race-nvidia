import time
import logging
import os
import random
from tqdm import tqdm
import numpy as np
import torch
from apex import amp
from apex.optimizers import FusedAdam
from apex.parallel import DistributedDataParallel as DDP

from data.RACEDataModule import RACEDataModule
from modeling import BertForMultipleChoice
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from utils import is_main_process
from helpers.GradientClipper import GradientClipper
from helpers.arg_parser import get_parser

from tensorboardX import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x

def main():
    ete_start = time.time()
    parser = get_parser()
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    if is_main_process():
        logger.info("device: {} ({}), n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, torch.cuda.get_device_name(0), n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    os.makedirs(args.output_dir, exist_ok=True)

    dm = RACEDataModule(
        model_name_or_path='./model/bert-large-uncased',
        datasets_loader='./data/RACELocalLoader.py',
        train_batch_size=args.train_batch_size,
        max_seq_length=args.max_seq_length,
        num_workers=8,
        num_preprocess_processes=64,
        use_sentence_selection=False,
        best_k_sentences=5,
    )
    dm.setup()
    train_examples = dm.dataset['train']
    num_train_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = BertForMultipleChoice.from_pretrained(args.bert_model,
                                                  cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                      args.local_rank),
                                                  num_choices=4)
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    optimizer = FusedAdam(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        bias_correction=False,
    )
    model, optimizer = amp.initialize(
        model,
        optimizers=optimizer,
        opt_level="O2",
        keep_batchnorm_fp32=False,
        loss_scale="dynamic" if args.loss_scale == 0 else args.loss_scale,
    )
    model = DDP(model)

    global_step = 0
    train_start = time.time()
    writer = SummaryWriter(os.path.join(args.output_dir, "asc001"))
    if args.do_train:
        train_dataloader = dm.train_dataloader()
        if is_main_process():
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_steps)
        model.train()
        gradClipper = GradientClipper(max_grad_norm=1.0)
        for ep in range(int(args.num_train_epochs)):
            tr_loss = 0
            train_iter = tqdm(train_dataloader, disable=False) if is_main_process() else train_dataloader
            if is_main_process():
                train_iter.set_description("Trianing Epoch: {}/{}".format(ep + 1, int(args.num_train_epochs)))
            for step, batch in enumerate(train_iter):
                loss = model(
                    input_ids=batch['input_ids'].to(device).reshape(batch['input_ids'].shape[0], 4, -1),
                    token_type_ids=batch['token_type_ids'].to(device).reshape(batch['token_type_ids'].shape[0], 4, -1),
                    attention_mask=batch['attention_mask'].to(device).reshape(batch['attention_mask'].shape[0], 4, -1),
                    labels=batch['label'].to(device),
                )
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                loss.backward()
                gradClipper.step(amp.master_params(optimizer))
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                if is_main_process():
                    train_iter.set_postfix(loss=loss.item())
                writer.add_scalar('loss', loss.item(), global_step=global_step)

    finish_time = time.time()
    writer.close()
    # Save a trained model
    if is_main_process():
        logger.info("ete_time: {}, training_time: {}".format(finish_time - ete_start, finish_time - train_start))
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)


if __name__ == "__main__":
    main()
