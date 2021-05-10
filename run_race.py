from __future__ import absolute_import, division, print_function

import pickle
import argparse
import logging
import os
import random
import wget
import json
import time

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import modeling
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear
from schedulers import LinearWarmUpScheduler
from apex import amp
from sklearn.metrics import matthews_corrcoef, f1_score
from utils import (is_main_process, mkdir_by_main_process, format_step,
                   get_world_size)

from helpers.arg_parser import parse_args


torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def init_optimizer_and_amp(model, learning_rate, loss_scale, warmup_proportion,
                           num_train_optimization_steps, use_fp16):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.01
        },
        {
            'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        },
    ]
    optimizer, scheduler = None, None
    if use_fp16:
        logger.info("using fp16")
        try:
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from "
                              "https://www.github.com/nvidia/apex to use "
                              "distributed and fp16 training.")

        if num_train_optimization_steps is not None:
            optimizer = FusedAdam(
                optimizer_grouped_parameters,
                lr=learning_rate,
                bias_correction=False,
            )
        amp_inits = amp.initialize(
            model,
            optimizers=optimizer,
            opt_level="O2",
            keep_batchnorm_fp32=False,
            loss_scale="dynamic" if loss_scale == 0 else loss_scale,
        )
        model, optimizer = (amp_inits
                            if num_train_optimization_steps is not None else
                            (amp_inits, None))
        if num_train_optimization_steps is not None:
            scheduler = LinearWarmUpScheduler(
                optimizer,
                warmup=warmup_proportion,
                total_steps=num_train_optimization_steps,
            )
    else:
        logger.info("using fp32")
        if num_train_optimization_steps is not None:
            optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=learning_rate,
                warmup=warmup_proportion,
                t_total=num_train_optimization_steps,
            )
    return model, optimizer, scheduler


def dump_predictions(path, label_map, preds, examples):
    label_rmap = {label_idx: label for label, label_idx in label_map.items()}
    predictions = {
        example.guid: label_rmap[preds[i]] for i, example in enumerate(examples)
    }
    with open(path, "w") as writer:
        json.dump(predictions, writer)


def main(args):
    args.fp16 = args.fp16 or args.amp
    if args.server_ip and args.server_port:
        # Distant debugging - see
        # https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        logger.info("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port),
            redirect_output=True,
        )
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of
        # sychronizing nodes/GPUs.
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, "
                "16-bits training: {}".format(
                    device,
                    n_gpu,
                    bool(args.local_rank != -1),
                    args.fp16,
                ))

    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or "
                         "`do_predict` must be True.")

    if is_main_process():
        if (os.path.exists(args.output_dir) and os.listdir(args.output_dir) and
                args.do_train):
            logger.warning("Output directory ({}) already exists and is not "
                           "empty.".format(args.output_dir))
    mkdir_by_main_process(args.output_dir)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                             args.gradient_accumulation_steps))
    if args.gradient_accumulation_steps > args.train_batch_size:
        raise ValueError("gradient_accumulation_steps ({}) cannot be larger "
                         "train_batch_size ({}) - there cannot be a fraction "
                         "of one sample.".format(
                             args.gradient_accumulation_steps,
                             args.train_batch_size,
                         ))
    args.train_batch_size = (args.train_batch_size //
                             args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


    #tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    tokenizer = BertTokenizer(
        args.vocab_file,
        do_lower_case=args.do_lower_case,
        max_len=512,
    )  # for bert large

    num_train_optimization_steps = None
    if args.do_train:
        train_features = get_train_features(
            args.data_dir,
            args.bert_model,
            args.max_seq_length,
            args.do_lower_case,
            args.local_rank,
            args.train_batch_size,
            args.gradient_accumulation_steps,
            args.num_train_epochs,
            tokenizer,
            processor,
        )
        num_train_optimization_steps = int(
            len(train_features) / args.train_batch_size /
            args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = (num_train_optimization_steps //
                                            torch.distributed.get_world_size())

    # Prepare model
    config = modeling.BertConfig.from_json_file(args.config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = modeling.BertForSequenceClassification(
        config,
        num_labels=num_labels,
    )
    logger.info("USING CHECKPOINT from {}".format(args.init_checkpoint))
    model.load_state_dict(
        torch.load(args.init_checkpoint, map_location='cpu')["model"],
        strict=False,
    )
    logger.info("USED CHECKPOINT from {}".format(args.init_checkpoint))

    model.to(device)
    # Prepare optimizer
    model, optimizer, scheduler = init_optimizer_and_amp(
        model,
        args.learning_rate,
        args.loss_scale,
        args.warmup_proportion,
        num_train_optimization_steps,
        args.fp16,
    )

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from "
                              "https://www.github.com/nvidia/apex to use "
                              "distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    loss_fct = torch.nn.CrossEntropyLoss()

    results = {}
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        train_data = gen_tensor_dataset(train_features)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
        )

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        latency_train = 0.0
        nb_tr_examples = 0
        model.train()
        tic_train = time.perf_counter()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss, nb_tr_steps = 0, 0
            for step, batch in enumerate(
                    tqdm(train_dataloader, desc="Iteration")):
                if args.max_steps > 0 and global_step > args.max_steps:
                    break
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = model(input_ids, segment_ids, input_mask)
                loss = loss_fct(
                    logits.view(-1, num_labels),
                    label_ids.view(-1),
                )
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up for BERT
                        # which FusedAdam doesn't do
                        scheduler.step()

                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
        latency_train = time.perf_counter() - tic_train
        tr_loss = tr_loss / nb_tr_steps
        results.update({
            'global_step':
                global_step,
            'train:loss':
                tr_loss,
            'train:latency':
                latency_train,
            'train:num_samples_per_gpu':
                nb_tr_examples,
            'train:num_steps':
                nb_tr_steps,
            'train:throughput':
                get_world_size() * nb_tr_examples / latency_train,
        })
        if is_main_process() and not args.skip_checkpoint:
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(
                {"model": model_to_save.state_dict()},
                os.path.join(args.output_dir, modeling.WEIGHTS_NAME),
            )
            with open(
                    os.path.join(args.output_dir, modeling.CONFIG_NAME),
                    'w',
            ) as f:
                f.write(model_to_save.config.to_json_string())

    if (args.do_eval or args.do_predict) and is_main_process():
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features, label_map = convert_examples_to_features(
            eval_examples,
            processor.get_labels(),
            args.max_seq_length,
            tokenizer,
        )
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_data = gen_tensor_dataset(eval_features)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size,
        )

        model.eval()
        preds = None
        out_label_ids = None
        eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        cuda_events = [(torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True))
                       for _ in range(len(eval_dataloader))]
        for i, (input_ids, input_mask, segment_ids, label_ids) in tqdm(
                enumerate(eval_dataloader),
                desc="Evaluating",
        ):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                cuda_events[i][0].record()
                logits = model(input_ids, segment_ids, input_mask)
                cuda_events[i][1].record()
                if args.do_eval:
                    eval_loss += loss_fct(
                        logits.view(-1, num_labels),
                        label_ids.view(-1),
                    ).mean().item()

            nb_eval_steps += 1
            nb_eval_examples += input_ids.size(0)
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = label_ids.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids,
                    label_ids.detach().cpu().numpy(),
                    axis=0,
                )
        torch.cuda.synchronize()
        eval_latencies = [
            event_start.elapsed_time(event_end)
            for event_start, event_end in cuda_events
        ]
        eval_latencies = list(sorted(eval_latencies))

        def infer_latency_sli(threshold):
            index = int(len(eval_latencies) * threshold) - 1
            index = min(max(index, 0), len(eval_latencies) - 1)
            return eval_latencies[index]

        eval_throughput = (args.eval_batch_size /
                           (np.mean(eval_latencies) / 1000))

        results.update({
            'eval:num_samples_per_gpu': nb_eval_examples,
            'eval:num_steps': nb_eval_steps,
            'infer:latency(ms):50%': infer_latency_sli(0.5),
            'infer:latency(ms):90%': infer_latency_sli(0.9),
            'infer:latency(ms):95%': infer_latency_sli(0.95),
            'infer:latency(ms):99%': infer_latency_sli(0.99),
            'infer:latency(ms):100%': infer_latency_sli(1.0),
            'infer:latency(ms):avg': np.mean(eval_latencies),
            'infer:latency(ms):std': np.std(eval_latencies),
            'infer:latency(ms):sum': np.sum(eval_latencies),
            'infer:throughput(samples/s):avg': eval_throughput,
        })
        preds = np.argmax(preds, axis=1)
        if args.do_predict:
            dump_predictions(
                os.path.join(args.output_dir, 'predictions.json'),
                label_map,
                preds,
                eval_examples,
            )
        if args.do_eval:
            results['eval:loss'] = eval_loss / nb_eval_steps
            eval_result = compute_metrics(args.task_name, preds, out_label_ids)
            results.update(eval_result)

    if is_main_process():
        logger.info("***** Results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        with open(os.path.join(args.output_dir, "results.txt"), "w") as writer:
            json.dump(results, writer)
    return results


if __name__ == "__main__":
    main(parse_args())
