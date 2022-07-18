from fastai.vision.all import *
#import dataset
path = untar_data(URLs.PETS)/'images'
# imagedataloaders to import, clean, and label data
def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(path,get_image_files(path),
						valid_pct=.2,seed=42,
						label_func = is_cat, item_tfms=Resize(224))
#cnn learner for building a model
learn = cnn_learner(dls,resnet34, metrics = error_rate)
learn.fine_tune(1)
