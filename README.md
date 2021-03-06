# Finding Lymphocytes

This repository represent the code usedduring the experiments for my master thesis.

## Counting lymphocytes with density maps

The idea is to phrase the tassk of counting lymphocytes as the a segmentation task where we try to reproduce a particular kind of target: a density maps.



# Density map object count and localization

# Dens map single gpu

python dens_count.py   
    -g 0    # gpu to use, gpu id
    -f False   # freeze encoder layer
    -r /home/papa/ly_decount/C_count_dens_map/experiments/dens_count_se_resnet50_imagenet_ep_120_bs_16_2020-11-12T13:18:13.581819/last.pth
    # optional, if we want to resume from checkpoint   
    -e se_resnet50   # model encoder architecture (see pytorch segmentation models)


-e : ["resnet50", "se_resnet50",
# Dens map distributed


### launch experiment with efficient b0 on 2 gpu


python dens_count_parallel.py -g 0,1 -en efficientnet-b0 -f True -s 6 -bs 52

### experiment with efficientnet b4 on 2 gpu


python dens_count_parallel.py -g 2,3 -en efficientnet-b4 -f True -s 6 -bs 52


python dens_count_parallel.py -g 0,1 -en efficientnet-b3 -f True -s 5 -bs 32

### efficientnet b3 full open

python dens_count_parallel.py -g 2,3 -en efficientnet-b3 -f False -s 5 -bs 16 -r /home/papa/ly_decount/C_count_dens_map/experiments/dens_count_efficientnet-b3_imagenet_ep_240_bs_18_resume_2020-11-17T02:39:07.243330/last.pth -lre 1.0

## efficientnet b3 frozen from imagenet and lower sigma and lr scheduler

python dens_count_parallel.py -g 2,3 -en efficientnet-b3 -f True -s 4 -bs 32

python dens_count_parallel.py -f True -lrf 1e-2 -g 2,3 -en efficientnet-b3 -s 5 -bs 32 -o ranger -nt efficientnetb3_from_scratch_with_plateau_scheduler_and_rangerlars_as_optimizer

## heavy augmentation
python dens_count_parallel.py -g 0,1 -en efficientnet-b3 -f False -s 5 -bs 12 -lr 1e-2 -lre 0.1 -lrf 0.5 -sh 0.5 -nt try_with_heav_color_augmentation_lr_scheduler_and_high_lr_start