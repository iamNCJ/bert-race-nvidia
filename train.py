# dm = datamodule()
# init dm
# model = bertForRace()
# init model, load pretrained weights

# for i in epochs:
#  # train
#   for batch in dm:
#     loss = model(batch)
#     optimizer.step()
#  # val
#   model.eval(batch in val)
# ...

import time
import logging
import random
from tqdm import tqdm, trange

from data.RACEDataModule import RACEDataModule
from model.BertForRace import BertForRace
from apex import amp
from utils import GradientClipper
from tensorboardX import SummaryWriter

import torch
import torch.nn

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    # attributes
    local_rank =
    no_cuda = 
    seed = 
    num_epochs = 
    train_batch_size = 

    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    if is_main_process():
        logger.info("device: {} ({}), n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, torch.cuda.get_device_name(0), n_gpu, bool(args.local_rank != -1), args.fp16))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    dm = RACEDataModule(
        model_name_or_path='./model/bert-large-uncased',
        datasets_loader='./data/RACELocalLoader.py',
        train_batch_size=32,
        max_seq_length=128,
        num_workers=8,
        num_preprocess_processes=96,
        use_sentence_selection=True,
        best_k_sentences=5,
    )

    model = BertForRace(
        pretrained_model='./model/bert-large-uncased',
        learning_rate=2e-5,
        num_train_epochs=20,
        train_batch_size=32,
        train_all=True,
        use_bert_adam=True,
    ).setup("fit")

    model.model.to(device)
    model.model.train()

    optimizer, scheduler = model.configure_optimizers()
    optimizer = optimizer[0]
    scheduler = scheduler['scheduler']

    grad_clipper = GradientClipper(1.0)

    train_start = time.time()
    writer = SummaryWriter(os.path.join(args.output_dir, "ascxx"))

    if is_main_process():
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

    train_dataloader = dm.train_dataloader()

    for epoch in range(num_epochs):
        tr_loss = 0
        train_iter = tqdm(train_dataloader, disable=False) if is_main_process() else train_dataloader
        if is_main_process():
            train_iter.set_description("Trianing Epoch: {}/{}".format(ep+1, int(args.num_train_epochs)))
        for step, batch in enumerate(train_iter):
            loss, correct = model.compute(batch)
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            tr_loss += loss.item()
            loss.backward()

            grad_clipper.step(amp.master_params(optimizer))

            scheduler.step()

            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar('loss', loss.item(), global_step=global_step)

    finish_time = time.time()
    writer.close()
    # Save a trained model
    if is_main_process():
        logger.info("ete_time: {}, training_time: {}".format(finish_time-ete_start, finish_time-train_start))
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
