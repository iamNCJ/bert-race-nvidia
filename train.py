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

from data.RACEDataModule import RACEDataModule
from model.BertForRace import BertForRace
from apex import amp

import torch
import torch.nn

class GradientClipper:
    """
    Clips gradient norm of an iterable of parameters.
    """
    def __init__(self, max_grad_norm):
        self.max_norm = max_grad_norm
        if multi_tensor_applier.available:
            import amp_C
            self._overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
            self.multi_tensor_scale = amp_C.multi_tensor_scale
        else:
            raise RuntimeError('Gradient clipping requires cuda extensions')

    def step(self, parameters):
        l = [p.grad for p in parameters if p.grad is not None]
        total_norm, _ = multi_tensor_applier(self.multi_tensor_l2norm, self._overflow_buf, [l], False)
        total_norm = total_norm.item()
        if (total_norm == float('inf')): return
        clip_coef = self.max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            multi_tensor_applier(self.multi_tensor_scale, self._overflow_buf, [l, l], clip_coef)

if __name__ == '__main__':
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

    optimizer, scheduler = model.configure_optimizers()
    optimizer = optimizer[0]
    scheduler = scheduler['scheduler']

    model.model.train()
    grad_clipper = GradientClipper(1.0)
    num_epochs = 6

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dm):
            loss = model.training_step(batch, batch_idx)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            grad_clipper.step(amp.master_params(optimizer))

            scheduler.step()

            optimizer.step()
            optimizer.zero_grad()
