#!/bin/bash
cd /home/xiaorong/workstation/tf-dnn-wsj5
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
dnn.sh 
EOF
) >sge.log
time1=`date +"%s"`
 ( dnn.sh  ) 2>>sge.log >>sge.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>sge.log
echo '#' Finished at `date` with status $ret >>sge.log
[ $ret -eq 137 ] && exit 100;
touch ./q/sync/done.4989
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o ./q/sge.log -q GPU_QUEUE -l gpu=1,io=2.0 -l h=compute-0-3    /home/xiaorong/workstation/tf-dnn-wsj5/./q/sge.sh >>./q/sge.log 2>&1
