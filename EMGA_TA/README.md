![Alt text](/images/EMGA.png)


nohup python -u -m torch.distributed.launch --nproc_per_node=8 src_online/train_distributed.py --cuda --pre_train ./experiment/M_29.50dB_0.34.pth >training.log 2>&1 &
