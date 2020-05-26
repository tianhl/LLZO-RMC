cp LLZO_RMC6_P1100K.rmc6f LLZO_RMC6_P1100K_orig.rmc6f
data2config -delete -one -rmc6f -vacancy [Li1 0.286] LLZO_RMC6_P1100K.rmc6f
data2config -delete -one -rmc6f -vacancy [Li2 0.512] LLZO_RMC6_P1100K_new.rmc6f
cp LLZO_RMC6_P1100K_new_new.rmc6f LLZO_RMC6_P1100K.rmc6f
