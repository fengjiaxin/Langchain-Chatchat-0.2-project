# linux环境下 nohup 运行服务
mkdir server_log
nohup python startup.py -a > ./server_log/server.out 2>&1&