import yaml

REMOVE_LIST = [
    "_libgcc_mutex", "_openmp_mutex", "ld_impl_linux-64", "libgcc-ng", "libgcc",
    "libgomp", "libnsl", "libuuid", "libxcrypt", "libzlib", "ncurses",
    "readline", "tk", "tzdata", "setuptools", "wheel", "openssl",
    "libsqlite", "liblzma", "ca-certificates"
]

def is_removable(pkg):
    if isinstance(pkg, str):
        return any(pkg.startswith(x + "=") or pkg == x for x in REMOVE_LIST)
    return False

with open("stt310_env.yml", "r") as f:
    env = yaml.safe_load(f)

# Lọc lại conda deps
deps = env.get("dependencies", [])
conda_pkgs = []
pip_pkgs = []

for dep in deps:
    if isinstance(dep, dict) and "pip" in dep:
        pip_pkgs.extend(dep["pip"])
    elif not is_removable(dep):
        conda_pkgs.append(dep)

# Tạo YAML mới
clean_env = {
    "name": env.get("name", "stt310"),
    "channels": env.get("channels", ["conda-forge"]),
    "dependencies": conda_pkgs + [{"pip": pip_pkgs}],
}

with open("stt310_env_clean.yml", "w") as f:
    yaml.dump(clean_env, f, default_flow_style=False)

print("✅ File mới đã được lưu: stt310_env_clean.yml")
