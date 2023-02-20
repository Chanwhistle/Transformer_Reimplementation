import os, sys, time
import logging
import torch
import numpy as np
from torch import nn, optim
from config import *
from models.build_model import build_model
from Dataloader import *
from utils import get_bleu_score
from torchtext.vocab import *

DATASET = CustomDataset()

def train(model, data_loader, optimizer, criterion, epoch, checkpoint_dir):
    model.train()
    epoch_loss = 0
    
    spp = sp.SentencePieceProcessor()
    vocab_file = "./Tokenizer/vocab/de_32000/de_32000.model"
    spp.load(vocab_file)
    
    def itos(x): 
        xs = x.tolist()
        indexs = [x for x in xs if x not in DATASET.specials.values()]
        tokens = spp.DecodeIdsWithCheck(indexs)
        return tokens
    
    for idx, (src, trg) in enumerate(tqdm(data_loader)):
        
        src = src.to(model.device)
        trg = trg.to(model.device)
        trg_x = trg[:, :-1]
        trg_y = trg[:, 1:]
        optimizer.zero_grad()
        output, _ = model(src, trg_x)
                
        y_hat = torch.stack([out.max(dim=1)[1] for out in output], dim=0)
        y_gt = trg_y.contiguous()
        y_gt = y_gt.to(device=DEVICE, dtype=torch.long) # GPU로 data 복사 및 dtype 변경
        '''
        y_gt는 long tensor로 들어가고 y_hat은 float.64로 들어가야 함
        '''
        # pred = [out.max(dim=1)[1] for out in output]
        pred_str = list(map(itos, y_hat))
        gt = list(map(itos, y_gt))
        print(pred_str)
        print(gt)
        
        import pdb;pdb.set_trace()
        
        loss = criterion(y_hat, y_gt)
        import pdb;pdb.set_trace()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
    num_samples = idx + 1

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f"{epoch:04d}.pt")
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                   }, checkpoint_file)

    return epoch_loss / num_samples


def evaluate(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0
    
    total_bleu = []
        
    with torch.no_grad():
        for idx, (src, trg) in enumerate(tqdm(data_loader)):
            
            src = src.to(model.device)
            trg = trg.to(model.device)
            trg_x = trg[:, :-1]
            trg_y = trg[:, 1:]  
            
            output, _ = model(src, trg_x)
            
            y_hat = output.contiguous().view(-1, output.shape[-1])
            y_gt = trg_y.contiguous().view(-1)
            y_gt = y_gt.to(device=DEVICE, dtype=torch.long)  # GPU로 data 복사 및 dtype 변경
            loss = criterion(y_hat, y_gt)

            epoch_loss += loss.item()
            score = get_bleu_score(output, trg_y, DATASET.specials)
            total_bleu.append(score)
        num_samples = idx + 1


    loss_avr = epoch_loss / num_samples
    bleu_score = sum(total_bleu) / len(total_bleu)
    return loss_avr, bleu_score


def main():
    model = build_model(len(DATASET.vocab_src), len(DATASET.vocab_trg), device=DEVICE, drop_prob=DROPOUT_RATE)

    def initialize_weights(model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.kaiming_uniform_(model.weight.data)

    model.apply(initialize_weights)

    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

    criterion = nn.CrossEntropyLoss(ignore_index=DATASET.pad_idx)

    train_iter, dev_iter, test_iter = DATASET.get_iter(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    for epoch in range(N_EPOCH):
        logging.info(f"* * * * * epoch: {epoch:02} * * * * *")
        train_loss = train(model, train_iter, optimizer, criterion, epoch, CHECKPOINT_DIR)
        logging.info(f"train_loss: {train_loss:.5f}")
        dev_loss, bleu_score  = evaluate(model, dev_iter, criterion)
        print(dev_loss)
        if epoch > WARM_UP_STEP:
            scheduler.step(dev_loss)
        logging.info(f"dev_loss: {dev_loss:.5f}, bleu_score: {bleu_score:.5f}")

    test_loss, bleu_score = evaluate(model, test_iter, criterion)
    logging.info(f"test_loss: {test_loss:.5f}, bleu_score: {bleu_score:.5f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    logging.basicConfig(level=logging.INFO)
    main()