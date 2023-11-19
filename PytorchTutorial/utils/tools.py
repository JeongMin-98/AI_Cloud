from torch.backends.mps import is_available as mps_available
from torch.cuda import is_available as cuda_available


def get_attributes_of_tensor(tensor):
    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")


def check_device():
    device = (
        "cuda"
        if cuda_available()
        else "mps"
        if mps_available()
        else "cpu"
    )
    return device


def read_config(path):
    """ read config files """
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]

    return lines


def parse_model_config(path):
    """ Parse your model of configuration files and return module defines """
    lines = read_config(path)
    module_configs = []

    for line in lines:
        if line.startswith('['):
            layer_name = line[1:-1].rstrip()
            if layer_name == "net":
                continue
            module_configs.append({})
            module_configs[-1]['type'] = layer_name

            if module_configs[-1]['type'] == 'convolutional':
                module_configs[-1]['batch_normalize'] = 0
        else:
            if layer_name == "net":
                continue
            key, value = line.split("=")
            value = value.strip()
            module_configs[-1][key.rstrip()] = value.strip()

    return module_configs


