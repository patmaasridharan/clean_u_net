import torch
from torch.autograd import Variable
from tqdm import tqdm

class segment:
    def __init__(self, net):
        self.net = net
        self.use_cuda = torch.cuda.is_available()

    def predict(self, test_loader, callbacks=None):

        self.net.eval()

        it_count = len(test_loader)

        with tqdm(total=it_count, desc="Classifying") as pbar:
            for ind, (images, files_name) in enumerate(test_loader):
                if self.use_cuda:
                    images = images.cuda()
                    self.net = self.net.cuda()

                with torch.no_grad():
                    images = Variable(images)

                # forward
                logits = self.net(images)
                probs = torch.sigmoid(logits)
                probs = probs.data.cpu().numpy()

                # If there are callback call their __call__ method and pass in some arguments
                if callbacks:
                    for cb in callbacks:
                        cb(step_name="predict",
                           net=self.net,
                           probs=probs,
                           files_name=files_name
                           )

                pbar.update(1)
