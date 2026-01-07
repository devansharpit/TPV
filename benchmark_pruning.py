'''

pretrained cifar models: https://github.com/chenyaofo/pytorch-cifar-models


python benchmark_pruning.py --model resnet56_cifar10 \
    --repeats 5 --N_batchs 50 --global_pruning  --pruning_ratio 0.9 \
    --iterative_steps 18 \
    --run_criteria "['Jacobian', 'JBR', 'L1', 'Random', 'BN Scale', 'FPGM', 'WHC', 'Taylor']" \
    --save_dir resnet56_cifar10

python benchmark_pruning.py --model vgg19_bn_cifar100 \
    --repeats 5 --N_batchs 50 --global_pruning  --pruning_ratio 0.9 \
    --iterative_steps 18 \
    --run_criteria "['Jacobian', 'JBR', 'L1', 'Random', 'BN Scale', 'FPGM', 'WHC', 'Taylor']" \
    --save_dir vgg19_bn_cifar100

python benchmark_pruning.py --model resnet50_imagenet \
    --repeats 5 --N_batchs 50 --global_pruning  --pruning_ratio 0.5 \
    --iterative_steps 18 \
    --run_criteria "['Jacobian', 'JBR', 'L1', 'Random', 'BN Scale', 'FPGM', 'WHC', 'Taylor']" \
    --save_dir resnet50_imagenet_bs256 --resume

python benchmark_pruning.py --model mobilenet_v2_imagenet \
    --repeats 5 --N_batchs 50 --global_pruning  --pruning_ratio 0.5 \
    --iterative_steps 18 \
    --run_criteria "['Jacobian', 'JBR', 'L1', 'Random', 'BN Scale', 'FPGM', 'WHC', 'Taylor']" \
    --save_dir mobilenet_v2_imagenet

'''
import copy
import gc
import logging
import sys, os
import time
import registry

import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import presets
from torchvision.transforms.functional import InterpolationMode


import torch_pruning as tp
import os
from tqdm import tqdm
from Selfmake_Importance import GroupJacobianImportance_accumulate, WHCImportance
from Selfmake_Importance import GroupJBRImportance_accumulate
from imagenet_dataloader import create_imagenet_dataloaders
from plot_benchmark_results import plot_results

import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='pruning')

parser.add_argument('--model', metavar='ARCH',  default='resnet18',help='path to dataset')
parser.add_argument('--save_dir', type=str, default='', help='Folder to save checkpoints and log.')

parser.add_argument('--run_criteria', type=str, default='', help='select which criteria for running; if '', run all; if ["L1"] for example, run "L1" only')

parser.add_argument('--global_pruning', action='store_true', help='global pruning or local pruning; default is local pruning')
parser.add_argument('--N_batchs', type=int, default=-1, help='how many batchs to use for importance estimation; if -1, use all')
parser.add_argument('--group_reduction', type=str, choices=['mean', 'first', 'max'], default='sum')
parser.add_argument('--normalizer', type=str, choices=['mean','max', 'sum', 'standarization', 'None'], default='None')

parser.add_argument('--repeats', type=int, default=5, help='how many times for repeat experiments')
parser.add_argument('--pruning_ratio', type=float, default=0.9)
parser.add_argument('--iterative_steps', type=int, default=18)
parser.add_argument('--bnrecal', action='store_true', help='whether to recalibrate batch norm statistics')

parser.add_argument('--resume', action='store_true', help='Resume from existing results.pth if available')

args = parser.parse_args()

def recalibrate_bn(
    model: nn.Module,
    train_loader,
    device: torch.device,
    max_batches: int = 200,
):
    """
    Recalculate BatchNorm running statistics after pruning.

    We:
      - put the model in train() mode so BN layers update running_mean/var
      - run a few batches forward with no_grad
      - then restore eval() mode

    For ResNet-18 on CIFAR-10 (no dropout), this is safe: the only thing
    that changes between train/eval is BN behavior.
    """
    model.train()

    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            if max_batches is not None and i >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            _ = model(images)

    model.eval()

def validate_model(model, val_loader):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss += torch.nn.functional.cross_entropy(outputs, labels, reduction='sum').item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return correct / len(val_loader.dataset), loss / len(val_loader.dataset)



if __name__ == "__main__":
    data_root = "./data"
    if args.model == 'resnet18_cifar100':
        dataset = 'cifar100'
        num_classes = 100
    elif args.model == 'vgg19_bn_cifar100':
        dataset = 'cifar100'
        num_classes = 100
    elif args.model == 'resnet56_cifar100':
        dataset = 'cifar100'
        num_classes = 100
    elif args.model == 'resnet18_cifar10':
        dataset = 'cifar10'
        num_classes = 10
    elif args.model == 'resnet56_cifar10':
        dataset = 'cifar10'
        num_classes = 10
    elif args.model == 'resnet50_imagenet':
        dataset = 'imagenet'
        num_classes = 1000
    elif args.model == 'mobilenet_v2_imagenet':
        dataset = 'imagenet'
        num_classes = 1000
    else:
        raise NotImplementedError
    
    ####### load data
    if dataset in ['cifar10', 'cifar100']:
        batch_size = 128
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        train_transform = transforms.Compose(
            [transforms.ToTensor(),  transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])

        if dataset=='cifar10':
            train_data = datasets.CIFAR10(data_root, train=True, transform=train_transform, download=False)
            test_data = datasets.CIFAR10(data_root, train=False, transform=test_transform, download=False)        
        elif dataset=='cifar100':
            train_data = datasets.CIFAR100(data_root, train=True, transform=train_transform, download=False)
            test_data = datasets.CIFAR100(data_root, train=False, transform=test_transform, download=False)         
        else:
            raise NotImplementedError

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=8, persistent_workers=True)
        val_loader = torch.utils.data.DataLoader(test_data, batch_size=500, shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=8, persistent_workers=True)    
        example_inputs = torch.randn(1, 3, 32, 32).cuda()

    elif dataset=='imagenet':
        batch_size = 256 
        # Create data loaders
        train_loader, val_loader, class_to_idx = create_imagenet_dataloaders(
            train_batch_size=batch_size,
            val_batch_size=32,
        )
        example_inputs = torch.randn(1, 3, 224, 224).cuda()
        # For JBR: need train_data reference (lazy-loads images, doesn't store in memory)
        train_data = train_loader.dataset

    else:
        raise NotImplementedError

        
    ###### prepare hyper-parameters
    N_batchs = args.N_batchs if args.N_batchs!=-1 else len(train_loader)
    print('N_batchs for importance estimation:', N_batchs)
    global_pruning = args.global_pruning
    network = args.model
    normalizer=None if args.normalizer=='None' else args.normalizer
    group_reduction=None  if args.group_reduction=='None' else args.group_reduction
    
    repeats = args.repeats
    pruning_ratio = args.pruning_ratio
    iterative_steps = args.iterative_steps # remove 5% filter each time by default
    
    save_dir = f'./results/benchmark_importance/{args.save_dir}'
    if global_pruning:
        save_dir += '_Global'
    else:
        save_dir += '_Local'        
    os.makedirs(save_dir, exist_ok=True)
    
    logger = logging.getLogger(save_dir+'/log.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s- %(message)s',
                        handlers=[
                            logging.FileHandler(save_dir+'/log.log'),
                            logging.StreamHandler()
                        ])

    logger = logging.getLogger()

    pretrained=False
    # example_inputs = torch.randn(1, 3, 32, 32).cuda()
        
    model = registry.get_model(args.model, num_classes=num_classes, pretrained=pretrained, target_dataset=dataset).cuda()
  
    model.eval()

    ### Importance criteria
    imp_dict = {
        # data-free
        'Random': tp.importance.RandomImportance(), 
        'L1': tp.importance.GroupMagnitudeImportance(p=1, group_reduction=group_reduction, normalizer=normalizer),
        'FPGM': tp.importance.FPGMImportance(group_reduction=group_reduction, normalizer=normalizer),
        'BN Scale': tp.importance.BNScaleImportance(group_reduction=group_reduction, normalizer=normalizer),
        'WHC': WHCImportance(group_reduction=group_reduction, normalizer=normalizer),      
        
        # data-driven
        'Taylor': tp.importance.TaylorImportance(group_reduction=group_reduction, normalizer=normalizer), 
        'JBR': GroupJBRImportance_accumulate(group_reduction=group_reduction, normalizer=normalizer),
        'Jacobian': GroupJacobianImportance_accumulate(group_reduction=group_reduction, normalizer=normalizer), 
        'Hessian': tp.importance.HessianImportance(group_reduction=group_reduction, normalizer=normalizer),
    }
    
    colors = {
        'WHC': 'C0',         # Blue
        'L1': 'C1',          # Orange
        'FPGM': 'C2',        # Green
        'BN Scale': 'C7',    # Gray
        'Random': 'C4',      # Purple
        
        'Taylor': 'C5',      # Brown
        'Hessian': 'C6',     # Pink
        'JBR': 'C10',        # Cyan
        'Jacobian': 'C3',    # Red
    }


    time_record = {}
    params_record = {}
    macs_record = {}
    train_loss_record = {}
    train_acc_record = {}
    val_loss_record = {}
    val_acc_record = {}
    
    # Resume from existing results if --resume flag is set
    results_path = f'{save_dir}/results.pth'
    if args.resume and os.path.exists(results_path):
        logger.info(f'Resuming from existing results: {results_path}')
        existing_data = torch.load(results_path, weights_only=False)
        params_record = existing_data.get('params_record', {})
        macs_record = existing_data.get('macs_record', {})
        train_loss_record = existing_data.get('train_loss_record', {})
        train_acc_record = existing_data.get('train_acc_record', {})
        val_loss_record = existing_data.get('val_loss_record', {})
        val_acc_record = existing_data.get('val_acc_record', {})
        time_record = existing_data.get('time_record', {})  # backward compatible
        logger.info(f'Loaded results for: {list(params_record.keys())}')
    

    
    
    # print the number of parameters and MACs of the original model
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    logger.info(f"Original Model - MAC(B): {base_macs/1e+9:.2f}, #Params: {base_nparams:.2f}")

    

    base_train_acc, base_train_loss = 0, 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.bnrecal:
        recalibrate_bn(model, train_loader, device, max_batches=50)
    base_val_acc, base_val_loss = validate_model(model, val_loader) 
    logger.info(f"MACs: {base_macs/base_macs:.2f}, MAC(B): {base_macs/1e+9:.2f}, #Params: {base_nparams/base_nparams:.2f}, Train_Acc: {base_train_acc:.4f}, train_Loss: {base_train_loss:.4f}, Val_Acc: {base_val_acc:.4f}, Val_Loss: {base_val_loss:.4f}")

    

    middle_name = f'Gr{group_reduction}_No{normalizer}_Nbatchs{N_batchs}'
    if global_pruning:
        middle_name += '_Global'
    else:
        middle_name += '_Local'
    
    for imp_name, imp in imp_dict.items():
        # Skip if already completed (resume mode)
        if args.resume and imp_name in params_record and len(params_record[imp_name]) >= repeats:
            logger.info(f'Skipping {imp_name} - already completed in previous run')
            continue
        
        if args.run_criteria=='':
            pass
        else:
            if imp_name not in eval(args.run_criteria):
                continue
        
        evaluating_time = 0
        overall_start_time = time.time()
        for repeat_ in range(repeats):
            
            
            # determined criteria do not need multiple tests
            if not ('Taylor' in imp_name or 'Hessian' in imp_name or 'Jacobian' in imp_name or 'JBR' in imp_name or 'Random' in imp_name) and (imp_name in params_record and len(params_record[imp_name])>=1):
                continue
            
            
            if imp_name in val_acc_record:
                print(f' (already done {len(val_acc_record[imp_name])} repeats)')
                if len(val_acc_record[imp_name])>=repeats:
                    print(f' (skip this repeat)')
                    continue
                repeat = len(val_acc_record[imp_name])
            else:
                print(f' (not done yet)')
                repeat = repeat_
            logger.info('='*50+imp_name+' repeat'+str(repeat)+'='*50 )# +
            if imp_name not in params_record:
                train_loss_record[imp_name] = [[]]
                train_acc_record[imp_name] = [[]]
                val_loss_record[imp_name] = [[]]
                val_acc_record[imp_name] = [[]]        
                params_record[imp_name] = [[]]
                macs_record[imp_name] = [[]]
            else:
                train_loss_record[imp_name].append([])
                train_acc_record[imp_name].append([])
                val_loss_record[imp_name].append([])
                val_acc_record[imp_name].append([])       
                params_record[imp_name].append([])
                macs_record[imp_name].append([])
                
            model=None
            torch.cuda.empty_cache() 
            
            model = registry.get_model(args.model, num_classes=num_classes, pretrained=pretrained, target_dataset=dataset).cuda()
 
            model.eval()
            torch.cuda.empty_cache() 
            
            # Reset importance object's internal state for this repeat
            if hasattr(imp, 'zero_grad'):
                imp.zero_grad()
            if hasattr(imp, 'zero_score'):
                imp.zero_score()
            
            if 'resnet' in args.model:
                ignored_layers = [model.fc]  # DO NOT prune the final classifier!
            else:
                ignored_layers = [model.classifier]
            
            pruner = tp.pruner.MetaPruner(
                model,
                example_inputs,
                iterative_steps=iterative_steps,
                importance=imp,
                pruning_ratio=pruning_ratio, 
                ignored_layers=ignored_layers,
                global_pruning=global_pruning,
            )

            # ------------------------------------------------------------
            # For JBR: freeze pseudo-labels from the *unpruned* model
            # ------------------------------------------------------------
            jbr_dataset = None  # Will store a TensorDataset of (imgs, teacher_preds)
            jbr_loader = None
            if isinstance(imp, GroupJBRImportance_accumulate):
                # Use shuffled loader to get random samples for this repeat
                # NOTE: persistent_workers=False to allow proper cleanup between repeats
                jbr_loader_init = torch.utils.data.DataLoader(
                    train_data,
                    batch_size=batch_size,
                    shuffle=True,  # Different random samples per repeat
                    num_workers=16,
                    pin_memory=True,
                    prefetch_factor=8,
                    persistent_workers=False,  # Allow workers to be cleaned up
                )

                all_imgs = []
                all_preds = []
                model.eval()
                cnt = 0
                with torch.no_grad():
                    for k, (imgs, lbls) in enumerate(jbr_loader_init):
                        imgs_cuda = imgs.cuda()
                        logits = model(imgs_cuda)  # unpruned model logits
                        preds = logits.argmax(dim=1).cpu()  # pseudo-labels y*
                        probs = logits.softmax(dim=1)
                        # keep preds whose max prob is >0.9
                        max_probs, _ = probs.max(dim=1)
                        mask = (max_probs > 0.9).cpu()
                        imgs = imgs[mask]
                        preds = preds[mask]
                        # Store both images and predictions together
                        all_imgs.append(imgs)
                        all_preds.append(preds)
                        cnt += imgs.shape[0]
                        if cnt >= batch_size * N_batchs:
                            break
                        # Free intermediate GPU tensors
                        del imgs_cuda, logits, probs, max_probs, mask
                
                # Explicitly delete the loader and force cleanup
                del jbr_loader_init
                torch.cuda.empty_cache()
                gc.collect()
                
                # Concatenate all samples into tensors and create a TensorDataset
                all_imgs = torch.cat(all_imgs, dim=0)[:batch_size * N_batchs]
                all_preds = torch.cat(all_preds, dim=0)[:batch_size * N_batchs]
                jbr_dataset = torch.utils.data.TensorDataset(all_imgs, all_preds)
                
                # Clear the lists to free memory
                del all_imgs, all_preds
                torch.cuda.empty_cache()
                
                # Create a fresh DataLoader with shuffle=True for true shuffling
                # This re-batches samples differently at each pruning iteration
                jbr_loader = torch.utils.data.DataLoader(
                    jbr_dataset,
                    batch_size=batch_size,
                    shuffle=True,  # True shuffling: different samples in each batch
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=False,  # Allow cleanup between repeats
                )


            logger.info(f"MACs: {base_macs/base_macs:.2f}, MAC(B): {base_macs/1e+9:.2f}, #Params: {base_nparams/base_nparams:.2f}, Train_Acc: {base_train_acc:.4f}, train_Loss: {base_train_loss:.4f}, Val_Acc: {base_val_acc:.4f}, Val_Loss: {base_val_loss:.4f}")

            params_record[imp_name][repeat].append(base_nparams)
            train_loss_record[imp_name][repeat].append(base_train_loss)
            train_acc_record[imp_name][repeat].append(base_train_acc)
            val_loss_record[imp_name][repeat].append(base_val_loss)
            val_acc_record[imp_name][repeat].append(base_val_acc)    
            macs_record[imp_name][repeat].append(base_macs)

            for i in range(iterative_steps):
                model.eval()
                
                evaluating_start_time = time.time()
                if isinstance(imp, tp.importance.HessianImportance):
                    imp.zero_grad() # clear accumulated gradients before each pruning step
                    for k, (imgs, lbls) in enumerate(train_loader):
                        if k>=N_batchs: break
                        imgs = imgs.cuda()
                        lbls = lbls.cuda()
                        output = model(imgs) 
                        # compute loss for each sample
                        loss = torch.nn.functional.cross_entropy(output, lbls, reduction='none')
                        '''
                        Note, the code from torch_pruning wrongly clear the gradients of the model here
                        so we remove the following line 
                        '''
                        # imp.zero_grad() # clear accumulated gradients
                        for l in loss:
                            model.zero_grad() # clear gradients
                            l.backward(retain_graph=True) # single-sample gradient
                            imp.accumulate_grad(model) # accumulate g^2  
                        torch.cuda.empty_cache() # in case CUDA OUT OF MEMORY                    
                            
                elif isinstance(imp, tp.importance.TaylorImportance):
                    model.zero_grad() # clear accumulated gradients before each pruning step
                    for k, (imgs, lbls) in enumerate(train_loader):
                        if k>=N_batchs: break
                        imgs = imgs.cuda()
                        lbls = lbls.cuda()
                        output = model(imgs)
                        loss = torch.nn.functional.cross_entropy(output, lbls)
                        loss.backward() 
                    torch.cuda.empty_cache()  # in case CUDA OUT OF MEMORY
                
                elif isinstance(imp, GroupJBRImportance_accumulate):
                    # JBR: accumulate J^T v over N_batchs and then build scores,
                    # using *fixed* pseudo-labels from the unpruned model.

                    imp.zero_grad()
                    imp.zero_score()

                    for imgs, y_star in jbr_loader:
                        imgs = imgs.cuda()
                        y_star = y_star.cuda()  # Fixed pseudo-labels from the unpruned model

                        logits = model(imgs)                      # current (possibly pruned) model
                        loss = torch.nn.functional.cross_entropy(logits, y_star)
                        model.zero_grad(set_to_none=True)
                        loss.backward()
                        imp.accumulate_grad(model)

                    imp.accumulate_score(model)
                    torch.cuda.empty_cache()

                elif imp_name == 'Jacobian':
                    imp.zero_score()
                    imp.zero_grad() # clear accumulated gradients
                    for k, (imgs, lbls) in enumerate(train_loader):
                        if k>=N_batchs: break
                        imgs = imgs.cuda()
                        lbls = lbls.cuda()
                        output = model(imgs) 
                        loss = torch.nn.functional.cross_entropy(output, lbls)
                        model.zero_grad() # clear gradients
                        loss.backward()
                        imp.accumulate_grad(model) # accumulate Jacobian  
                        if (k+1)%50==0 or k==N_batchs-1:
                            imp.accumulate_score(model)
                            torch.cuda.empty_cache()  # in case CUDA OUT OF MEMORY

                pruner.step()
                evaluating_time += time.time()-evaluating_start_time

                model.eval()
                macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
                if args.bnrecal:
                    recalibrate_bn(model, train_loader, device, max_batches=50)
                val_acc, val_loss = validate_model(model, val_loader)
                if dataset!='imagenet':
                    train_acc, train_loss = validate_model(model, train_loader)
                else:
                    train_acc, train_loss = 0, 0
                logger.info(f"MACs: {base_macs/macs:.2f}, MAC(B): {macs/1e+9:.2f}, #Params: {nparams/base_nparams:.2f}, Train_Acc: {train_acc:.4f}, Train_Loss: {train_loss:.4f}, Val_Acc: {val_acc:.4f}, Val_Loss: {val_loss:.4f}")
                params_record[imp_name][repeat].append(nparams)
                train_loss_record[imp_name][repeat].append(train_loss)
                train_acc_record[imp_name][repeat].append(train_acc)
                val_loss_record[imp_name][repeat].append(val_loss)
                val_acc_record[imp_name][repeat].append(val_acc)
                macs_record[imp_name][repeat].append(macs)

            # Cleanup at end of repeat to prevent memory accumulation
            if jbr_loader is not None:
                del jbr_loader
            if jbr_dataset is not None:
                del jbr_dataset
            del pruner
            del model
            model = None
            
            # Force garbage collection and clear CUDA cache
            gc.collect()
            torch.cuda.empty_cache()
            
            # Save results after each repeat
            print(f'Saving results after repeat {repeat}...')
            torch.save({'iterative_steps': iterative_steps, 'pruning_ratio':pruning_ratio, 'N_batchs':N_batchs, 'batch_size':batch_size, \
                        'params_record':params_record, 'macs_record':macs_record, 'train_loss_record':train_loss_record,\
                        'train_acc_record':train_acc_record, 'val_acc_record':val_acc_record, 'val_loss_record':val_loss_record,\
                        'time_record':time_record},\
                        f'{save_dir}/results.pth')
        
        time_record[imp_name] = [(time.time() - overall_start_time)/repeats/iterative_steps, evaluating_time/repeats/iterative_steps]
        logger.info(f'{imp_name} average overall time (including evaluation): {time_record[imp_name][0]}; average evaluating time: {time_record[imp_name][1]}')
        
        

        ######################### save #########################
        print('Saving results...')
        torch.save({'iterative_steps': iterative_steps, 'pruning_ratio':pruning_ratio, 'N_batchs':N_batchs, 'batch_size':batch_size, \
                    'params_record':params_record, 'macs_record':macs_record, 'train_loss_record':train_loss_record,\
                    'train_acc_record':train_acc_record, 'val_acc_record':val_acc_record, 'val_loss_record':val_loss_record,\
                    'time_record':time_record},\
                    f'{save_dir}/results.pth') # a_record_{network}_{middle_name}

       
    ######################### draw #########################




    label_fontsize = 30
    tick_fontsize  = 25
    title_fontsize  = 25
    legend_fontsize  = 15
    bar_legend_fontsize = 18

    
    results_path = f'./results/benchmark_importance/{args.save_dir}_Global'
    dataset_ = 'cifar' if 'cifar' in dataset else 'imagenet'
    plot_results(
        results_path=results_path,
        save_dir=results_path,
        dataset=dataset_,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        title_fontsize=title_fontsize,
        legend_fontsize=legend_fontsize,
        bar_legend_fontsize=bar_legend_fontsize
    )
