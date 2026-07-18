source $HOME/.venvs/supermlx/bin/activate
sudo sysctl iogpu.wired_limit_mb=21100 && sudo purge
nice -n 19 python SuperMLX.py
