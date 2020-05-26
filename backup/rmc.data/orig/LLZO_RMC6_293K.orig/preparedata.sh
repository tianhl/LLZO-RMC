cp LLZO_RMC6_293K.rmc6f LLZO_RMC6_293K_orig.rmc6f
data2config -delete -one -rmc6f -vacancy [Li1 0.304] LLZO_RMC6_293K.rmc6f
data2config -delete -one -rmc6f -vacancy [Li2 0.507] LLZO_RMC6_293K_new.rmc6f
cp LLZO_RMC6_293K_new_new.rmc6f LLZO_RMC6_293K.rmc6f
