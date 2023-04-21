import torch
if torch.cuda.is_available():
    for idx in torch.cuda.device_count():
        print(torch.cuda.get_device_name(idx))
else:
    print('Unable to use GPUs with torch!')