cp LLZO_RMC6_600K.rmc6f LLZO_RMC6_600K_orig.rmc6f
data2config -delete -one -rmc6f -vacancy [Li1 0.238] LLZO_RMC6_600K.rmc6f
data2config -delete -one -rmc6f -vacancy [Li2 0.524] LLZO_RMC6_600K_new.rmc6f
cp LLZO_RMC6_600K_new_new.rmc6f LLZO_RMC6_600K.rmc6f
