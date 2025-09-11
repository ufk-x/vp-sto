#!/bin/bash

# =============================================================================
# VP-STO 本地安装和验证完整解决方案
# =============================================================================

echo "=== VP-STO 完整安装解决方案 ==="

# 1. 激活正确的conda环境
echo "1. 激活vpsto环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vpsto

if [ "$CONDA_DEFAULT_ENV" != "vpsto" ]; then
    echo "   ✗ 请先运行: conda activate vpsto"
    exit 1
fi

echo "   ✓ 当前环境: $CONDA_DEFAULT_ENV"

# 2. 完全卸载所有vpsto相关包
echo "2. 完全卸载vpsto..."
pip uninstall vpsto -y 2>/dev/null || true
pip uninstall vpsto -y --force 2>/dev/null || true

# 3. 清理所有缓存
echo "3. 清理缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
python -m pip cache purge 2>/dev/null || true

# 4. 验证本地文件
echo "4. 验证本地文件..."
if ! grep -q "def minimize" vpsto/vpsto.py; then
    echo "   ✗ vpsto/vpsto.py中没有发现minimize方法!"
    echo "   请确保文件包含correct version"
    exit 1
fi
echo "   ✓ 本地文件包含minimize方法"

# 5. 使用开发模式安装
echo "5. 安装本地vpsto包(开发模式)..."
pip install -e . --no-cache-dir --force-reinstall

# 6. 验证安装
echo "6. 验证安装..."
python -c "
import sys
print('   Python路径包含:', [p for p in sys.path if 'vp-sto' in p])

try:
    from vpsto.vpsto import VPSTO, VPSTOOptions
    import inspect
    
    # 检查文件位置
    vpsto_file = inspect.getfile(VPSTO)
    print('   VPSTO源文件:', vpsto_file)
    
    # 检查方法
    test_opts = VPSTOOptions(2)
    test_vpsto = VPSTO(test_opts)
    methods = [m for m in dir(test_vpsto) if not m.startswith('_')]
    print('   可用方法:', methods)
    
    if 'minimize' in methods:
        print('   ✓ minimize方法可用')
    else:
        print('   ✗ minimize方法不可用')
        exit(1)
        
    if 'set_initial_guess' in methods:
        print('   ✓ set_initial_guess方法可用')
    else:
        print('   ✗ set_initial_guess方法不可用')
        
    # 检查minimize方法签名
    sig = inspect.signature(test_vpsto.minimize)
    print('   minimize签名:', sig)
    
except Exception as e:
    print('   ✗ 验证失败:', str(e))
    exit(1)
"

echo "=== 安装和验证完成 ==="
echo ""
echo "现在可以:"
echo "1. 重启Jupyter notebook kernel"
echo "2. 使用 from vpsto.vpsto import VPSTO, VPSTOOptions"
echo "3. 调用 vpsto_instance.minimize(...)"
