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
