# PSM-Fuzzer
模糊测试 (fuzzing)-PSM Fuzzer

paper: An adaptive fuzzing method based on transformer and protocol similarity mutation

doi：https://doi.org/10.1016/j.cose.2023.103197

注意，在seq2seq_.1.py中我们引用了modbus_tk库。我们更改了modbus_tk\modbus.py文件。请将文件中的modbus.py与modbus_tk\modbus.py替换

在check_tcp.py文件中，我们给出判断测试用例是否符合Modbus TCP协议规范的代码
