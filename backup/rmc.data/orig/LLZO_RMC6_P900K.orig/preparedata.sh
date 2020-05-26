cp LLZO_RMC6_P900K.rmc6f LLZO_RMC6_P900K_orig.rmc6f
data2config -delete -one -rmc6f -vacancy [Li1 0.172] LLZO_RMC6_P900K.rmc6f
data2config -delete -one -rmc6f -vacancy [Li2 0.540] LLZO_RMC6_P900K_new.rmc6f
cp LLZO_RMC6_P900K_new_new.rmc6f LLZO_RMC6_P900K.rmc6f
