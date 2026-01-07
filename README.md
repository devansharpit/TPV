# TPV
Official repository for the paper "TPV: Parameter Perturbations Through the Lens of Test Prediction Variance"

# Files and Commands:

## Fig 1 & 2 (TPV stability on synthetic data)
python tpv_trace_synth_universal_scatter.py

## Fig 3 (TPV stability on CIFAR-10-- vary network width)
python tpv_cifar_universal_scatter_vary_w.py --dataset c10 --savefile tpv_cifar10_width_sweep

## Fig 11 (TPV stability on CIFAR-100-- vary network width)
python tpv_cifar_universal_scatter_vary_w.py --dataset c100 --savefile tpv_cifar100_width_sweep

## Fig 4 (TPV stability on CIFAR-10-- vary number of training samples)
python tpv_cifar_universal_scatter_vary_n_train.py --dataset c10 --savefile tpv_cifar10_vary_n_train_results

## Fig 12 (TPV stability on CIFAR-100-- vary number of training samples)
python tpv_cifar_universal_scatter_vary_n_train.py --dataset c100 --savefile tpv_cifar100_vary_n_train_results


## Fig 5 & 7 & 14 (Synthetic data Label Noise Experiment)
python tpv_label_noise.py

## Fig 8 (Cifar-100 Label Noise Experiment)
python tpv_label_noise_cifar.py --dataset cifar100

## Fig 13 (Cifar-10 Label Noise Experiment)
python tpv_label_noise_cifar.py --dataset cifar10

## Note for ImageNet Experiments
For all ImageNet experiments, please update the function `create_imagenet_dataloaders` in `imagenet_dataloader.py`

## Fig 6 (ImageNet Label Noise Experiment)

python tpv_label_noise_imagenet.py


## Fig 9, 10, 15 & 16 (Pruning Experiments)

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

