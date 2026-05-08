This module is used to manually annotate road data.  The annotated road data is then used as the training signal to train a deep convolutional neural network to estimate the road centerline.  The tool works by allowing the user to pick a point that represents the road centerline along three horizontal reference lines.  It is possible, in the case of a fork in the road, that there are two road centerline points on a single horizontal reference line.  It is also possible if there is no road in the scene, that there are no centerline points on any of the horizontal reference lines.

The tool will build up a file called labels.csv that lists each file along with its centerline point locations along the horizontal reference lines that are all represented as percentages of image height and image width.

The tool is expecting the images to be in a directory under training_data which is a subdirectory under this module.  I have made this a softlink in most cases to the actual data on a removable SSD external drive.  The image directory is specified with a -ds <image-directory-name>

The row fractions and some other info is specified in a file called dataset.info.  This file will live with the resulting data where ever it is used by subsequent modules.

Some other annotation parameters are in the config.ini file.

