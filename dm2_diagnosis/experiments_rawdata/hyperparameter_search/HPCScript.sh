
#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J RaghavsJob
### -- ask for number of cores (default: 1) -- 
#BSUB -n 10
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=2GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 10GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s161227@student.dtu.dk
### -- send notification at start -- 
#BSUB -B
### -- send notification at completion -- 
#BSUB -N
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out
#BSUB -e Error_%J.err

# here follow the commands you want to execute 

module load python3/3.6.2

#pip3 install --user torch torchvision

#pip3 install --user pandas

#module load scipy/0.19.1-python-3.6.2

#module load matplotlib/2.0.2-python-3.6.2

#pip3 install --user scikit-learn

#pip3 install --user skorch

#pip3 install --upgrade --user pip

echo ------------------------
cp ../../dataset/ .
ls
echo -----------------------

python3 hyperparameter_search.py
