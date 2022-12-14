import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

#from afm_addrbpool_resnet18 import ResNet18
#from afm_mp_resnet import resnet18
from afm_resnet import resnet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# data_transform = transforms.Compose(
#     [transforms.Resize(256),
#      transforms.CenterCrop(224),
#      transforms.ToTensor(),
#      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# create model
model = resnet18(num_classes=6)
#model =ResNet18()
# model = resnet34(num_classes=5)

# load model weights

#model_weight_path = "kaggle//200spv-resnet//200spv-addrbpool-resnet18.pth"
#model_weight_path = "kaggle//kspvdata82//100-pvdata82-mpbzresnet18.pth"
model_weight_path = "kaggle//kspvdata82//100-pvdata82-lkbzresnet18.pth"

model.load_state_dict(torch.load(model_weight_path, map_location=device))
print(model)

# load image
img = Image.open("pictures//scratch.jpg")
#img = np.array(Image.open("pictures//scratch.jpg").convert("RGB"))
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# forward
out_put = model(img)
for feature_map in out_put:
    # [N, C, H, W] -> [C, H, W]
    im = np.squeeze(feature_map.detach().numpy())
    print(im.shape)
    # [C, H, W] -> [H, W, C]
    #im = np.transpose(im, [1, 2, 0])
    im = np.transpose(im, [1, 2, 0])
    # show top 12 feature maps
    plt.figure()
    for i in range(12):
        ax = plt.subplot(3, 4, i+1)
        # [H, W, C]
        #plt.imshow(im[:, :, i], cmap='gray')
        plt.imshow(im[:, :, i])
    plt.show()

