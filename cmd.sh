# you can change cmd.sh depending on what type of run you are using.
# If you have no runing system and want to run on a local machine, you
# can change all instances 'run.pl' to run.pl (but be careful and run
# commands one by one: most recipes will exhaust the memory on your
# machine).  run.pl works with GridEngine (qsub).  slurm.pl works
# with slurm.  Different runs are configured differently, with different
# run names and different ways of specifying things like memory;
# to account for these differences you can create and edit the file
# conf/run.conf to match your run's configuration.  Search for
# conf/run.conf in http://kaldi-asr.org/doc/run.html for more information,
# or search for the string 'default_config' in utils/run.pl or utils/slurm.pl.

run="CPU_run"
gpu_run="GPU_run"
export train_cmd="run.pl -q $run -l ram_free=1.5G,mem_free=1.5G,io=2.0"
export decode_cmd="run.pl -q $run -l ram_free=2.5G,mem_free=2.5G,io=2.0"
export cuda_cmd="run.pl -q $gpu_run -l gpu=1,io=2.0"

