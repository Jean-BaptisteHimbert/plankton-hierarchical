# (submit.sh)
#!/bin/bash -l

echo "Parameters name is: $1"

rm ~/.bashrc_$1

# List of nodes in the cluster
nod=$(cat $OAR_FILE_NODES| sort -u)
nodes=($(echo $nod | tr "\n" " "))

echo "List of nodes is: $nod"

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

find_available_port() {
    local port=50000
    while lsof -Pi :$port -sTCP:LISTEN -t >/dev/null; do
        ((port++))
    done
    echo "$port"
}

# Calculate the world size, corresponding to the number of GPUS of each node times the number of nodes
master_addr=($(hostname -I))
master_port=$(find_available_port)
num_gpus=$(nvidia-smi --list-gpus | wc -l)
num_nodes=${#nodes[@]}
world_size=$((num_gpus*num_nodes))

echo "Master IP adress is: $master_addr"
echo "Using port number: $master_port"
echo "Num_gpus is: $num_gpus"
echo "Num_nodes is: $num_nodes"
echo "World size is: $world_size"


# Set the variables on the node with NODE_RANK 0

export MASTER_PORT=$master_port    
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=GRAPH
export PYTHONFAULTHANDLER=1
export NODE_RANK=0
export WORLD_SIZE=$world_size


# Set the environment variables on each node
for ((i=1; i<$num_nodes; i++)); do
  node="${nodes[$i]}"
  rm ~/.bashrc_$1_$i

  ssh "${nodes[$i]}" "echo 'export MASTER_PORT=$master_port' >> ~/.bashrc_$1_$i"
  ssh "${nodes[$i]}" "echo 'export MASTER_ADDR=$master_addr' >> ~/.bashrc_$1_$i"
  ssh "${nodes[$i]}" "echo 'export WORLD_SIZE=$world_size' >> ~/.bashrc_$1_$i"
  ssh "${nodes[$i]}" "echo 'export NODE_RANK=$i' >> ~/.bashrc_$1_$i"
  ssh "${nodes[$i]}" "echo 'export NCCL_DEBUG=INFO' >> ~/.bashrc_$1_$i"
  ssh "${nodes[$i]}" "echo 'export NCCL_DEBUG_SUBSYS=GRAPH' >> ~/.bashrc_$1_$i"
  ssh "${nodes[$i]}" "echo 'export PYTHONFAULTHANDLER=1' >> ~/.bashrc_$1_$i"    
done

# Installing all the librairies and dependencies

echo "INFO: Installing dependencies on node 0"
# pip install -r requirements.txt
for ((i=1; i<$num_nodes; i++)); do
  node="${nodes[$i]}"
  echo "INFO: Installing dependencies on node $i"
  # ssh "$node" "source ~/.bashrc_$1_$i && pip install -r $PWD/requirements.txt" &
done

wait

# Launching the training script in each

echo "INFO: Lauching script on node 0"
python TrainModelBACNN.py --param-name $1 --num-gpus $num_gpus --num-nodes $num_nodes &
for ((i=1; i<$num_nodes; i++)); do
  node="${nodes[$i]}"
  echo "INFO: Lauching script on node $i"
  ssh "$node" "source ~/.bashrc_$1_$i && python $PWD/TrainModelBACNN.py --param-name $1 --num-gpus $num_gpus --num-nodes $num_nodes" &
done

wait