from torchreid import models, utils
import torchreid
from dataset_torchreid import *

loss = "triplet"
model_name = "pcb_p6"

vis_topk = 10

weights_path = './model/torchreid_models/' + model_name + '_' + loss + '/model.pth.tar-1'
save_path = './model/torchreid_models/' + model_name + '_' + loss + '/'
height = 160 if model_name == "hacnn" else 256
width = 64 if model_name == "hacnn" else 128


if loss == "triplet":
    sampler = 'RandomIdentitySampler'
else:
    sampler = 'RandomSampler'

# print(weights_path, save_path, height, width, model_name, sampler)

torchreid.data.register_image_dataset('veri_dataset', VeRiDataset)

datamanager = torchreid.data.ImageDataManager(
    root='..Dataset/VeRi_with_plate/',
    sources='veri_dataset',
    height=height,
    width=width,
    train_sampler=sampler
)

model = models.build_model(name=model_name, num_classes=575)
model = model.cuda()

torchreid.utils.load_pretrained_weights(model, weights_path)
optimizer = torchreid.optim.build_optimizer(
    model, optim='adam', lr=0.0003
)

if loss == "triplet":
    engine = torchreid.engine.ImageTripletEngine(datamanager, model, optimizer)
else:
    engine = torchreid.engine.ImageSoftmaxEngine(datamanager, model, optimizer)

engine.run(test_only=True, save_dir=save_path, visrank=True, visrank_topk=vis_topk)
