from utils import get_model_complexity_info
import models as models
import torchvision.models

model = models.__dict__['pf10_2_resnet50'](num_classes=1000)

flops, params = get_model_complexity_info(model, (224, 224), as_strings=False, print_per_layer_stat=False)
print('Flops:  %.3f' % (flops / 1e9))
print('Params: %.2fM' % (params / 1e6))
# print(model)
print((flops-4121924096.0)/(1024*1024))
print((params-25557032)/1024)

# model = models.__dict__['se_mnasnet1_0'](num_classes=1000)
# print(model)
