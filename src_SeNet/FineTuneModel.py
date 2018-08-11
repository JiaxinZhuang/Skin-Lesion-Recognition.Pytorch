import torch
import torch.nn as nn

class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch):
        super(FineTuneModel, self).__init__()

        if arch.startswith('resnet') :
            # Everything except the last linear layer
            self.features = original_model
            self.classifier = nn.Sequential(
                nn.Linear(1000, 7)
            )
            self.modelName = 'resnet'
        else :
            raise("Finetuning not supported on this architecture yet")

        # Freeze those weights
        #for p in self.features.parameters():
        #    p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y