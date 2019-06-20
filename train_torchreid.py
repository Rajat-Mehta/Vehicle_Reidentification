from dataset_torchreid import *
import argparse

base_save_path = './model/torchreid_models/'
parser = argparse.ArgumentParser(description='Training torchreid')
parser.add_argument('--engine', default='softmax', type=str, help='which engine to use: softmax or triplet')
parser.add_argument('--model_name', default='hacnn', type=str, help='which model to use for training')
parser.add_argument('--epochs', default=60, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')


opt = parser.parse_args()

engine = opt.engine
model_name = opt.model_name
epochs = opt.epochs
batch_size = opt.batch_size
save_path = base_save_path + model_name + '_' + engine + '/'

print(engine, model_name, epochs, batch_size, save_path)

torchreid.data.register_image_dataset('veri_dataset', VeRiDataset)

if engine == "triplet":
    sampler = 'RandomIdentitySampler'
else:
    sampler = 'RandomSampler'

height = 160 if model_name == "hacnn" else 256
width = 64 if model_name == "hacnn" else 128

datamanager = torchreid.data.ImageDataManager(
    root='..Dataset/VeRi_with_plate/',
    sources='veri_dataset',
    random_erase=True,
    train_sampler=sampler,
    batch_size=batch_size,
    height=height,
    width=width
)

model = torchreid.models.build_model(
    name=model_name,
    num_classes=datamanager.num_train_pids,
    loss=engine
)

model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model, optim='adam', lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

if engine == "triplet":
    model_engine = torchreid.engine.ImageTripletEngine(
        datamanager, model, optimizer, margin=0.3,
        weight_t=1, weight_x=0, scheduler=scheduler
    )
elif engine == "softmax":
    model_engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager, model, optimizer, scheduler=scheduler
    )

model_engine.run(
    max_epoch=epochs,
    save_dir=save_path,
    print_freq=10
)


"""
datamanager = torchreid.data.ImageDataManager(
    root='/home/rajat/MyPC/DFKI/MasterThesis/Datasets/delme',
    sources='market1501',
    height=160,
    width=64,
    combineall=False,
    batch_size=32
)

data_loader, _ = datamanager.return_dataloaders()
i=0
labelset = []
for data, label, camera, path in data_loader:
    label = label.data.numpy().tolist()
    labelset += label
    i+=1
labelset = set(labelset)
print(labelset)
print(len(labelset))

exit()
"""