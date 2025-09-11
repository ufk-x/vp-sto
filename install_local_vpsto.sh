#!/bin/bash

# VP-STO 本地安装脚本
# 确保使用本地的minimize版本而不是已安装的版本

echo "=== VP-STO 本地安装脚本 ==="

# 0. 激活vpsto环境
echo "0. 激活vpsto conda环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vpsto

# 检查当前环境
echo "   当前环境: $CONDA_DEFAULT_ENV"
if [ "$CONDA_DEFAULT_ENV" != "vpsto" ]; then
    echo "   ✗ 未在vpsto环境中，请手动运行: conda activate vpsto"
    exit 1
fi

# 1. 清理已安装的包
echo "1. 卸载现有的vpsto包..."
pip uninstall vpsto -y

# 2. 清理Python缓存
echo "2. 清理Python缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# 3. 清理构建目录
echo "3. 清理构建目录..."
rm -rf build/ dist/ *.egg-info/

# 4. 检查vpsto.py文件是否包含minimize方法
echo "4. 检查vpsto.py文件..."
if grep -q "def minimize" vpsto/vpsto.py; then
    echo "   ✓ 发现minimize方法"
else
    echo "   ✗ 未发现minimize方法，请检查vpsto/vpsto.py文件"
    exit 1
fi

# 5. 重新安装本地包
echo "5. 安装本地vpsto包..."
pip install -e .

# 6. 验证安装
echo "6. 验证安装..."
python -c "
try:
    from vpsto.vpsto import VPSTO
    methods = [m for m in dir(VPSTO) if not m.startswith('_')]
    print('   可用方法:', methods)
    if 'minimize' in methods:
        print('   ✓ minimize方法可用')
    else:
        print('   ✗ minimize方法不可用')
        exit(1)
    import vpsto
    print('   包位置:', vpsto.__file__)
except Exception as e:
    print('   ✗ 导入失败:', e)
    exit(1)
"

echo "=== 安装完成 ==="
