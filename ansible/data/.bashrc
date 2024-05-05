if [ -f /etc/bash_completion ] && ! shopt -oq posix; then
    . /etc/bash_completion
fi
alias k='kubectl'
source <(kubectl completion bash | sed 's/kubectl/k/g')

