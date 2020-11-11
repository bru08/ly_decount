import torch
from collections import OrderedDict

def load_lysto_weights(model, state_path, encoder_arch):
    assert encoder_arch == "resnet50"
    state = torch.load(state_path)["model_state"]
    #
    state_keys = [x for x in state.keys() if x.startswith("base_modules")]
    model_keys = list(model.encoder.state_dict().keys())
    #
    state_renamed = OrderedDict()
    for i, skey in enumerate(state_keys[1:]):
        try:
            state_renamed[model_keys[i+1]] = state[skey]
        except Exception as e:
            print(e)
    # needed for toch segmentation models that by defualt assume resnets have fc layers at the end
    state_renamed["fc.bias"] = 1
    state_renamed["fc.weight"] = 1
    #
    model.encoder.load_state_dict(state_renamed, strict=False)
    print("Succesfully converted and loaded resnet50 weights from lysto pretraining")
