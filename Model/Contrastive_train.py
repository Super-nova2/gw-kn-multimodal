from data_loader import create_training_dataloader
from model import *
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import h5py
import os
import datetime
import argparse
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

def train(args):

    float32_mode = args.float32
    if float32_mode:
        print("Training in Float32 precision.")
    else:
        print("Training in Mixed Precision (Float16) mode.")

    # --- 1. Setup Device & Config ---
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Create a unique log directory using timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if float32_mode:
        log_dir = os.path.join(os.path.dirname(args.ckpt_path), "tb_logs", f"run_fp32_{timestamp}")
    else:
        log_dir = os.path.join(os.path.dirname(args.ckpt_path), "tb_logs", f"run_mix_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging started at: {log_dir}")
    
    # --- 2. Data Preparation ---
    if args.steps_per_epoch is not None:
        steps_per_epoch = args.steps_per_epoch
    else:
        # Calculate correct steps_per_epoch based on optical data volume
        with h5py.File(args.data_path, 'r') as f:
            total_optical = f['events/optical_data/values'].shape[0]
        
        steps_per_epoch = total_optical // args.batch_size
        print(f"Dataset Size: {total_optical} | Steps/Epoch: {steps_per_epoch}")

    train_loader = create_training_dataloader(
        h5_path=args.data_path,
        batch_size=args.batch_size,
        steps_per_epoch=steps_per_epoch,
        num_workers=args.num_workers
    )
    
    # --- 3. Model Initialization ---
    model = GWOpticalContrastiveModel(
        gw_scalar_dim=7,
        gw_skymap_channels=7, # Assuming robust preprocessing output
        optical_input_dim=6,
        enc_dim=128,
        proj_dim=256
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    if not float32_mode:
        # Mixed Precision Scaler
        scaler = torch.amp.GradScaler(device=device.type)
    
    # --- 4. Training Loop ---
    model.train()

    global_step = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        # Tqdm progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch_data in enumerate(pbar):
            # Unpack data
            gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices = [x.to(device) for x in batch_data]
            if torch.isnan(opt_v).any() or torch.isinf(opt_v).any():
                print("NaN or Inf detected in optical values. Skipping batch.")
                continue
            elif torch.isnan(opt_err).any() or torch.isinf(opt_err).any():
                print("NaN or Inf detected in optical errors. Skipping batch.")
                continue
            
            # Create Reference Time Query (Learned queries need t_ref)
            # For mTAN, we typically query at the same observed times OR fixed grid.
            # Here we query at observed times for reconstruction/encoding.
            # (Note: In pure contrastive learning, we just encode the sequence)
            # Assuming Encoder implementation uses fixed reference points internally 
            # or we pass t_obs as t_ref to get representations at specific points.
            # BUT: OpticalEncoderWithCLS usually expects t_ref to generate the queries.
            # Simple strategy: Use linspace 0-1 as reference time (normalized)
            B = gw_s.size(0)
            N_ref = 64 # Number of reference points
            if float32_mode:
                opt_ref_t = torch.linspace(-0.3, 0.6, N_ref, dtype=torch.float32).unsqueeze(0).repeat(B, 1).to(device)
            else:
                opt_ref_t = torch.linspace(-0.3, 0.6, N_ref).unsqueeze(0).repeat(B, 1).to(device)
            
            optimizer.zero_grad()
            
            # Mixed Precision Forward
            if not float32_mode:
                with torch.amp.autocast(device_type=device.type):
                    loss, logits = model(
                        gw_s, gw_m, 
                        opt_t, opt_v, opt_ref_t, 
                        opt_mask, opt_err, opt_coords,
                        gw_indices, mask=True
                    )
                    # Backward
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                loss, logits = model(
                    gw_s, gw_m, 
                    opt_t, opt_v, opt_ref_t, 
                    opt_mask, opt_err, opt_coords,
                    gw_indices, mask=True
                )
                # Backward
                loss.backward()
                optimizer.step()
            
            # Logging
            loss_val = loss.item()
            epoch_loss += loss_val
            
            # TensorBoard Logging
            if batch_idx % 10 == 0:  # Log every 10 batches for smoother graphs
                # Calculate accuracy
                preds = torch.argmax(logits, dim=1)
                targets = torch.arange(B, device=device)
                acc = (preds == targets).float().mean().item()
                current_temp = model.log_temp.exp().item()
                
                # Write to TensorBoard
                writer.add_scalar('Train/Batch_Loss', loss_val, global_step)
                writer.add_scalar('Train/Batch_Accuracy', acc, global_step)
                writer.add_scalar('Train/Temperature', current_temp, global_step)
                writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)

            if batch_idx % 100 == 0:
                 pbar.set_postfix({
                    'Loss': f"{loss_val:.4f}", 
                    'Acc': f"{acc:.2f}",
                    'Temp': f"{current_temp:.2f}"
                })
            
            global_step += 1
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")

        writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
        
        # Save Checkpoint
        if float32_mode:
            checkpoint_path = os.path.join(args.ckpt_path, "fp32",f"checkpoint_fp32_epoch_{epoch+1}.pth")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        else:
            checkpoint_path = os.path.join(args.ckpt_path, "mix", f"checkpoint_mix_epoch_{epoch+1}.pth")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
    
    writer.close()


if __name__ == "__main__":
    """
    Example usage:
    python ML+GW+KN/Model/Contrastive_train.py --data_path data/LSST_KN_BNS/combined_dataset.h5 --epochs 2 --batch_size 32 --steps_per_epoch 10 --ckpt_path data/model/checkpoints
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="training_data.h5")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32) # Adjust based on VRAM
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--float32", action='store_true', help="Use Float32 precision for training")
    
    args = parser.parse_args()
    
    if os.path.exists(args.data_path):
        os.makedirs(args.ckpt_path, exist_ok=True)
        train(args)
    else:
        print("Data file not found.")