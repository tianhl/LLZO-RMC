reset
set xlabel 'Q [1/{\305}]'
set ylabel 'Diff. CS [barns/sr/atom]'
set style line 1 lt 1ps 0 lc 1
set style line 2 lt 1ps 0 lc 2
set style line 3 lt 1ps 0 lc 3
x=0
y=0
i=-1
plot \
'i15-1-18937_tth_det2_0.dofr' u 1:((column(2)+0.0)+0.0) notitle w l ls 1, \
'/home/tianhl/workarea/LLZO/i15-1/ee19378-1/Gudrun_PDF/01-177/i15-1-18936_tth_det2_0.dofr' u 1:((column(2)+0.0)+0.0) notitle w l ls 2, \
'/home/tianhl/workarea/LLZO/i15-1/ee19378-1/Gudrun_PDF/01-027/i15-1-18935_tth_det2_0.dofr' u 1:((column(2)+0.0)+0.0) notitle w l ls 3
