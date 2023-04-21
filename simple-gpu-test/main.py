import torch
if torch.cuda.is_available():
    print(str(torch.cuda.device_count()) + ' GPUs availabe')
else:
    print('Unable to use GPUs with torch!')
