import os
import sys
import winreg as reg
# Get path of current working directory and python.exe
cwd = os.getcwd()
python_exe = sys.executable
source_root = cwd[:cwd.rfind("\\")]
commands_root = source_root + "\\cmd"

# add current path to path env variable
directory_to_add = source_root
path_key_path = r'SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment\\'
path_key = reg.OpenKey(reg.HKEY_LOCAL_MACHINE, path_key_path)
try:
    current_path, _ = reg.QueryValueEx(path_key, "PYTHONPATH")
    path_key.Close()
    new_path = f"{current_path}{os.pathsep}{directory_to_add}"
    path_key = reg.OpenKey(reg.HKEY_LOCAL_MACHINE, path_key_path, 0, reg.KEY_SET_VALUE)
    if directory_to_add not in current_path:
        reg.SetValueEx(path_key, "PYTHONPATH", 0, reg.REG_EXPAND_SZ, new_path)
    path_key.Close()
except:
    path_key = reg.OpenKey(reg.HKEY_LOCAL_MACHINE, path_key_path, 0, reg.KEY_WRITE)
    reg.SetValueEx(path_key, "PYTHONPATH", 0, reg.REG_EXPAND_SZ, directory_to_add)

decompress_key_path = r'Directory\\shell\\Decompression\\'
compress_key_path = r'txtfile\\shell\\Compression\\'
compress_advanced_key_path = r'txtfile\\shell\\Compression advanced\\'

# Create outer key
decompress_key = reg.CreateKey(reg.HKEY_CLASSES_ROOT, decompress_key_path)
compress_key = reg.CreateKey(reg.HKEY_CLASSES_ROOT, compress_key_path)
compress_advanced_key = reg.CreateKey(reg.HKEY_CLASSES_ROOT, compress_advanced_key_path)
reg.SetValue(decompress_key, '', reg.REG_SZ, '&Decompress as LSTM Compressed txt')
reg.SetValue(compress_key, '', reg.REG_SZ, '&Compress With LSTM Model')
reg.SetValue(compress_advanced_key, '', reg.REG_SZ, '&Advanced Compress With LSTM Model')

# create inner key
decompress_key1 = reg.CreateKey(decompress_key, r"command")
compress_key1 = reg.CreateKey(compress_key, r"command")
compress_advanced_key1 = reg.CreateKey(compress_advanced_key, r"command")
reg.SetValue(decompress_key1, '', reg.REG_SZ, '"' + python_exe + '"' + f' "{commands_root}\\cmd_decompress.py"' + ' "%1"')
reg.SetValue(compress_key1, '', reg.REG_SZ, '"' + python_exe + '"' + f' "{commands_root}\\cmd_compress.py"' + ' "%1"')
reg.SetValue(compress_advanced_key1, '', reg.REG_SZ, '"' + python_exe + '"' + f' "{commands_root}\\cmd_compress_advanced.py"' + ' "%1"')
print('"' + python_exe + '"' + f' "{commands_root}\\cmd_compress_advanced.py"' + ' "%1"')