import yaml

GPU_PACKAGES = [
    "nvidia-", "triton", "deepspeed", "xformers", "flash-attn", "bitsandbytes", "apex"
]

def is_gpu_related(pkg):
    return any(pkg.lower().startswith(prefix) for prefix in GPU_PACKAGES)

with open("stt310_env_clean.yml", "r") as f:
    env = yaml.safe_load(f)

new_pip = []
for dep in env["dependencies"]:
    if isinstance(dep, dict) and "pip" in dep:
        for pkg in dep["pip"]:
            if not is_gpu_related(pkg):
                new_pip.append(pkg)

# Ghi file mới
env["dependencies"] = [
    d for d in env["dependencies"] if not isinstance(d, dict)
] + [{"pip": new_pip}]

with open("stt310_env_gpu_stripped.yml", "w") as f:
    yaml.dump(env, f, default_flow_style=False)

print("✅ File đã được làm sạch GPU packages: stt310_env_gpu_stripped.yml")
