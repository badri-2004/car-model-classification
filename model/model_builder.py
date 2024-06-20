from transformers import ViTImageProcessor, ViTForImageClassification
class transform:
    def __init__(self,process):
        self.process = process
    def __call__(self,image):
        return self.process(image,return_tensors = 'pt')['pixel_values'].squeeze()

def get_model(num_classes):
  processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
  transform_ = transform(processor)
  model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',num_labels = num_classes,ignore_mismatched_sizes=True)
  return model,transform_
