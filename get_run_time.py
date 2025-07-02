import wandb
import numpy as np

api = wandb.Api()

# 替换成你的 run 路径
run = api.run("jiangjohn116/Search-R1/fuck9e9c")

# 获取指定 metric 的历史记录
history = run.history(keys=["timing_s/step"])
print(history["timing_s/step"])
# 计算总耗时（单位：秒）
total_seconds = np.nansum(history["timing_s/step"][40:])
print(f"总训练耗时（timing_s/step 累加）：{total_seconds:.2f} 秒 = {total_seconds/3600:.2f} 小时")
