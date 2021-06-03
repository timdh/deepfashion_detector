
import json
import torch

from code import options
from code import model
from code import dataloader


# load the discriminator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = options.get_args(['network', 'discriminator'])
discriminator = model.get_network(net_type='discriminator', args=args).to(device)
try:
    discriminator.load_state_dict(torch.load(args.model_path))
except:
    print('error: could not load the discriminator weights, please check the model path: [%s]' % args.model_path)
    exit()

# load the image
im = dataloader.load_im(args).to(device)
label = torch.Tensor([args.im_label]).type(torch.LongTensor).to(device)
prediction = discriminator(im, label).item()
model_probability = round(prediction, 2)

if round(prediction):
    label = 'real'
else:
    label = 'fake'
    model_probability = 1 - model_probability

json_out = {
  "label": label,
  "confidence": model_probability
}
print(json.dumps(json_out))
