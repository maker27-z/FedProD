import subprocess
import itertools

# 定义参数组合
algos = ["FedProD"]
d_alphas = [0.7] #[0.7, 10]
alpha_beta_pairs =[(0.1, 0.3)]    #[(0.1, 0.1)]#(0.1,0.1), (0.1, 0.3),

#lams = [0.12, 0.25, 0.5, 0.75, 1.0]
# 遍历所有组合
for algo, d_alpha, (alpha_param, beta_param) in itertools.product(algos, d_alphas, alpha_beta_pairs):
# for algo, d_alpha, lam in itertools.product(algos, d_alphas, lams):
    # 构造命令
    flag = 1 if algo == "FedProD" else 0  # 设置dp标志，FedProD为1，FedProto为0
    cmd = (
        f"python system/main.py -data cicids2017 -go test -did 0 -ncl 15 -m ResNet10 "
        f"-lbs 1024 -lr 0.02 -algo {algo} -M 10 -nc 5 -dev cuda -eg 1 -gr 50 -wp 10 -T 2 -lam 0.25 "
        f"-dp {flag} -glrt 0.01 -d_alpha {d_alpha} -alpha_param {alpha_param} -beta_param {beta_param}"
        # f"-dp {flag} -glrt 0.01 -d_alpha {d_alpha} -seed 0 -isP 0 -isN 0 -th 1.2"
    )
    
    print(f"\n{'=' * 80}")
    print(f"Running: {cmd}")
    print(f"{'=' * 80}\n")
    
    # 执行命令
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        continue  # 如果出错，继续下一个组合
