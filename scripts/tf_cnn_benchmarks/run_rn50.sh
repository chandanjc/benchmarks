# $NGRAPHWORK points to full path for ngraph-stack root dir.
# Within ngraph-stack, ngraph-models, tf-models, etc. should be placed.

# Set the MPI_NUM_PROCS variable to decide number of MPI processes, hence,
# number of backends to use.
export MPI_NUM_PROCS=4

# Set the PER_CHIP_BATCH_SIZE. For multichip, each chip will train with
# this batch size.
export PER_CHIP_BATCH_SIZE=256

export OMP_NUM_THREADS=32

mpirun -np $MPI_NUM_PROCS -x LD_LIBRARY_PATH -H localhost -x OMP_NUM_THREADS -x PYTHONPATH -cpus-per-proc 8 -map-by socket --oversubscribe --report-bindings \
python ./tf_cnn_benchmarks.py \
--model=resnet50 \
--batch_size=$PER_CHIP_BATCH_SIZE \
--data_format NHWC \
--display_every=1 \
--mkl=true \
--datasets_use_prefetch=False \
--kmp_blocktime=1 \
--kmp_affinity=granularity=fine,compact,1,0 \
--data_name=imagenet \
--num_intra_threads 32 \
--num_inter_threads 32 \
--horovod_device cpu \
--variable_update=horovod \
--num_batches=1 \
--num_warmup_batches=0 \
--distortions=False \
--print_training_accuracy=1 \
--summary_verbosity=2 \
--save_summaries_steps=1 \
--batchnorm_persistent=False \
--optimizer=momentum
#--num_batches=29714 \
#--optimizer=momentum \
#--num_batches=29714 \
#--num_batches=59428 \
#--num_batches=14857 \  1 batch, 4 chips
#--model=official_resnet50 \ #TODO
#--data_dir=/dataset/TF_ImageNet_latest \
