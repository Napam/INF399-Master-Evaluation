[ -z "$PS1" ] && return

export PS1="\[\e[33m\]\u@\[\e[1;32m\]IoU\[\e[36m\] \w \[\e[33m\]>\[\e[0m\] "
export TERM=xterm-256color
alias grep="grep --color=auto"
alias ls="ls --color=auto"

echo -e "\e[1;32m"
cat<<EOF
|\  |   /\   |\ |  |   /\   ------    ___  --  |  |
| \ |  /--\  |/ |--|  /--\     |       |  |  | |  |
|  \| /    \ |  |  | /    \    |      _|_  --  |__|
EOF
echo -e "\e[0;33m"

# Turn off colors
echo -e "\e[0m"
