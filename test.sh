MODEL=$HOME/llm/models/8x7b_instruct/
FIXLEN=1024
LEN=480

# specify number of OMP threads
# export OMP_NUM_THREADS=32

# enable thread binding and print out info on thread affinity
export OMP_DISPLAY_ENV=true
export OMP_DISPLAY_AFFINITY=true
export OMP_AFFINITY_FORMAT="Thread Affinity: %0.3L %.8n %.15{thread_affinity} %.12H"
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# python -u v0_cpu_experts/solo_gpu_model_profile.py --model-path $MODEL --prompt "to be, or not to be" --max-tokens $FIXLEN  $@
python -u $HOME/MoE-CPU-GPU-Collaboration-Inference/src/mixtral.py --prompt 'implement a red-black tree using C++ and show me the  example usage' --model-path $MODEL --max-tokens $FIXLEN  $@
