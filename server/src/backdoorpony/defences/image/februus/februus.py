import torch
from grad_cam import GradCam
##################################################
# PARAMETER SETTING
##################################################
MASK_COND = 0.7
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4
N = 0
##################################################

def run(poisoned_model):
    return
    # net = poisoned_model
    # net = net.to(device)
    # net.load_state_dict(torch.load())
    # net.eval()
    # gcam = GradCam(net, True, device)
    # print("Loading model successfully\n")