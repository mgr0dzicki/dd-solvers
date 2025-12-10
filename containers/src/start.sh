#!/bin/bash
set -e

echo "Pod started"

# Set up the environment

echo "Exporting environment variables..."
printenv | grep -E '^RUNPOD_|^PATH=|^_=' | \
    awk -F = '{ print "export " $1 "=\"" $2 "\"" }' >> /etc/rp_environment
echo 'source /etc/rp_environment' >> ~/.bashrc
env | egrep -v "^(HOME=|USER=|MAIL=|LC_ALL=|LS_COLORS=|LANG=|HOSTNAME=|PWD=|TERM=|SHLVL=|LANGUAGE=|_=)" \
    >> /etc/environment

# Set up SSH if PUBLIC_KEY is provided

if [[ $PUBLIC_KEY ]]; then
    echo "Setting up SSH..."
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 -R ~/.ssh

    generate_ssh_key() {
        local key_type=$1
        local key_path=$2

        if [ ! -f "$key_path" ]; then
            ssh-keygen -t "$key_type" -f "$key_path" -q -N ''
            echo "$key_type key fingerprint:"
            ssh-keygen -lf "${key_path}.pub"
        fi
    }

    generate_ssh_key rsa /etc/ssh/ssh_host_rsa_key
    generate_ssh_key dsa /etc/ssh/ssh_host_dsa_key
    generate_ssh_key ecdsa /etc/ssh/ssh_host_ecdsa_key
    generate_ssh_key ed25519 /etc/ssh/ssh_host_ed25519_key

    service ssh start

    echo "SSH host keys:"
    for key in /etc/ssh/*.pub; do
        echo "Key: $key"
        ssh-keygen -lf $key
    done
fi

# Set up Jupyter Lab if JUPYTER_PASSWORD is provided

if [[ $JUPYTER_PASSWORD ]]; then
    echo "Starting Jupyter Lab..."
    mkdir -p /workspace && \
    cd / && \
    nohup jupyter lab \
        --allow-root \
        --no-browser \
        --port=8888 \
        --ip=* \
        --FileContentsManager.delete_to_trash=False \
        --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' \
        --ServerApp.token=$JUPYTER_PASSWORD \
        --ServerApp.allow_origin=* \
        --ServerApp.preferred_dir=/workspace \
        &> /jupyter.log &
    echo "Jupyter Lab started"
fi

# Apply MPI workaround

echo "Applying mpi4py workaround..."
apt remove -y python3-mpi4py
apt install -y fenicsx

# Keep the container running

echo "Setup finished, pod is ready to use."
sleep infinity
