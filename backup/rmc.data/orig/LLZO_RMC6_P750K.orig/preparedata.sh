cp LLZO_RMC6_P750K.rmc6f LLZO_RMC6_P750K_orig.rmc6f
data2config -delete -one -rmc6f -vacancy [Li1 0.184] LLZO_RMC6_P750K.rmc6f
data2config -delete -one -rmc6f -vacancy [Li2 0.537] LLZO_RMC6_P750K_new.rmc6f
cp LLZO_RMC6_P750K_new_new.rmc6f LLZO_RMC6_P750K.rmc6f
