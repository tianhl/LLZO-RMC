#!/bin/bash
#SBATCH --mail-user=tianhl@ihep.ac.cn
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --partition=NeuDA
#SBATCH --ntasks-per-node=1
#SBATCH --time=7-00:00:00 # 7 days
#SBATCH --output=mpi_job_slurm.log
date
echo $SLURM_JOB_NODELIST
pwd
/csns_workspace/SSG/tianhl/opt/dl-poly/dl_poly_4.09/execute/DLPOLY.Z
/csns_workspace/SSG/tianhl//opt/correlationfunc/correlation_Li
/csns_workspace/SSG/tianhl//opt/correlationfunc/correlation_O
/csns_workspace/SSG/tianhl//opt/correlationfunc/correlation_La
/csns_workspace/SSG/tianhl//opt/correlationfunc/correlation_Zr
/csns_workspace/SSG/tianhl/opt/RMCProfile/RMCProfile_package/exe/data2config -one -rmc6f -pdf 50 REVCON
#time srun -n 24 ./VanRod ./VanRod.in
date
