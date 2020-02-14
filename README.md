# Object_Detection_PRIma
This Repository contains the pytorch code for object detection

## Objective
To build a model that can detect and localize specific objects in images.

### Dataset :- Layout Analysis Dataset
This dataset is used for evaluation of layout analysis (physical and logical) methods. It contains realistic documents with a wide variety of layouts, reflecting the various challenges in layout analysis. Particular emphasis is placed on magazines and technical/scientific publications which are likely to be the focus of digitisation efforts.
Each image in the dataset has associated comprehensive and detailed ground truth enabling in-depth evaluation.

### Overview
  **Some definitions**
  
  **Bounding Box**:
  A bounding box is a box that wraps around an object i.e. represents its bounds.
  
  **Boundary coordinates**:
  The most obvious way to represent a box is by the pixel coordinates of the x and y lines that constitute its boundaries.
  The boundary coordinates of a box are simply (x_min, y_min, x_max, y_max).
  The center-size coordinates of a box are (c_x, c_y, w, h).
  
  **Jaccard Index**:
  The Jaccard Index or Jaccard Overlap or Intersection-over-Union (IoU) measure the degree or extent to which two boxes         overlaps.
  ![Jaccard](/images/jaccard.png)
  An IoU of 1 implies they are the same box, while a value of 0 indicates they're mutually exclusive spaces.
  It's a simple metric, but also one that finds many applications in our model.
  
  **MultiBox Loss**:
  Multibox is a technique for detecting objects where a prediction consists of two components –
  Coordinates of a box that may or may not contain an object. This is a regression task.
  Scores for various object types for this box, including a background class which implies there is no object in the box. This   is a classification task.
  
  **Single Shot Detector (SSD)**:
  The SSD is a purely convolutional neural network (CNN) that we can organize into three parts –
  
  *Base convolutions* derived from an existing image classification architecture that will provide lower-level feature maps.
  
  *Auxiliary convolutions* added on top of the base network that will provide higher-level feature maps.\
  
  *Prediction convolutions* that will locate and identify objects in these feature maps.
  
  **Base Convolutions Part - 1 **
  
  I have used VGG-16 as the base network
  ![vgg-16](/images/vgg-16.png)
  
  Some are logical and necessary changes in the architecture
  
  * The input image size will be 300, 300.
  
  * The 3rd pooling layer, which halves dimensions, will use the mathematical ceiling function instead of the default floor       function in determining output size. This is significant only if the dimensions of the preceding feature map are odd and       not even. By looking at the image above, you could calculate that for our input image size of 300, 300, the conv3_3           feature map will be of cross-section 75, 75, which is halved to 38, 38 instead of an inconvenient 37, 37.

  * I modify the 5th pooling layer from a 2, 2 kernel and 2 stride to a 3, 3 kernel and 1 stride. The effect this has is it no     longer halves the dimensions of the feature map from the preceding convolutional layer.

  * We don't need the fully connected (i.e. classification) layers because they serve no purpose here. We will toss fc8 away       completely, but choose to rework fc6 and fc7 into convolutional layers conv6 and conv7.
  
  **Base Convolutions Part -2 **
  
  We now know how to convert fc6 and fc7 in the original VGG-16 architecture into conv6 and conv7 respectively.

In the ImageNet VGG-16, which operates on images of size 224, 224, 3, you can see that the output of conv5_3 will be of size 7, 7, 512. Therefore –
* fc6 with a flattened input size of 7 * 7 * 512 and an output size of 4096 has parameters of dimensions 4096, 7 * 7 * 512. The equivalent convolutional layer conv6 has a 7, 7 kernel size and 4096 output channels, with reshaped parameters of dimensions 4096, 7, 7, 512.

* fc7 with an input size of 4096 (i.e. the output size of fc6) and an output size 4096 has parameters of dimensions 4096, 4096. The input could be considered as a 1, 1 image with 4096 input channels. The equivalent convolutional layer conv7 has a 1, 1 kernel size and 4096 output channels, with reshaped parameters of dimensions 4096, 1, 1, 4096.

We can see that conv6 has 4096 filters, each with dimensions 7, 7, 512, and conv7 has 4096 filters, each with dimensions 1, 1, 4096.

These filters are numerous and large – and computationally expensive.

To reduce both the number and the size of each filter I subsampled the parameters from the converted convolutional layers.

* conv6 will use 1024 filters, each with dimensions 3, 3, 512. Therefore, the parameters are subsampled from 4096, 7, 7, 512 to 1024, 3, 3, 512.

* conv7 will use 1024 filters, each with dimensions 1, 1, 1024. Therefore, the parameters are subsampled from 4096, 1, 1, 4096 to 1024, 1, 1, 1024.

I subsampled by picking every mth parameter along a particular dimension, in a function named decimation.

Since the kernel of conv6 is decimated from 7, 7 to 3, 3 by keeping only every 3rd value, there are now holes in the kernel. Therefore, we would need to make the kernel dilated or atrous.

This corresponds to a dilation of 3 (same as the decimation factor m = 3). 

So the modified VGG-16
![modified_vgg-16](/images/modified_vgg-16.png)

**Auxiliary Convolutions**

We will now stack some more convolutional layers on top of our base network. These convolutions provide additional feature maps, each progressively smaller than the last.

![aux_conv](/images/aux_conv.png)

We introduce four convolutional blocks, each with two layers. While size reduction happened through pooling in the base network, here it is facilitated by a stride of 2 in every second layer.

**A detour**

Before we move on to the prediction convolutions, we must first understand are we predicting. Sure, it's objects and their positions, but in what form?

It is here that we must learn about priors and the crucial role they play in the SSD.

**Priors**

Object predictions can be quite diverse, and I don't just mean their type. They can occur at any position, with any size and shape. Mind you, we shouldn't go as far as to say there are infinite possibilities for where and how an object can occur. While this may be true mathematically, many options are simply improbable or uninteresting. Furthermore, we needn't insist that boxes are pixel-perfect.

In effect, we can discretize the mathematical space of potential predictions into just thousands of possibilities.

Priors are precalculated, fixed boxes which collectively represent this universe of probable and approximate box predictions.

Priors are manually but carefully chosen based on the shapes and sizes of ground truth objects in our dataset. By placing these priors at every possible location in a feature map, we also account for variety in position.

* they will be applied to various low-level and high-level feature maps, viz. those from conv4_3, conv7, conv8_2, conv9_2, conv10_2, and conv11_2. These are the same feature maps indicated on the figures before.

* if a prior has a scale s, then its area is equal to that of a square with side s. The largest feature map, conv4_3, will have priors with a scale of 0.1, i.e. 10% of image's dimensions, while the rest have priors with scales linearly increasing from 0.2 to 0.9. As you can see, larger feature maps have priors with smaller scales and are therefore ideal for detecting smaller objects.

* At each position on a feature map, there will be priors of various aspect ratios. All feature maps will have priors with ratios 1:1, 2:1, 1:2. The intermediate feature maps of conv7, conv8_2, and conv9_2 will also have priors with ratios 3:1, 1:3. Moreover, all feature maps will have one extra prior with an aspect ratio of 1:1 and at a scale that is the geometric mean of the scales of the current and subsequent feature map.

There are a total of 8732 priors defined for the SSD300!

**Visualizing Priors**

We defined the priors in terms of their scales and aspect ratios.

![wh2](/images/wh1.png)

Solving these equations yields a prior's dimensions w and h.

![wh2](/images/wh2.png)

We're now in a position to draw them on their respective feature maps.

For example, let's try to visualize what the priors will look like at the central tile of the feature map from conv9_2.

![wh3](/images/wh3.png)

The same priors also exist for each of the other tiles.

![wh4](/images/wh4.png)

**Predictions vis-à-vis Priors**

Earlier, we said we would use regression to find the coordinates of an object's bounding box. But then, surely, the priors can't represent our final predicted boxes?

They don't.

Again, I would like to reiterate that the priors represent, approximately, the possibilities for prediction.

This means that we use each prior as an approximate starting point and then find out how much it needs to be adjusted to obtain a more exact prediction for a bounding box.

So if each predicted bounding box is a slight deviation from a prior, and our goal is to calculate this deviation, we need a way to measure or quantify it.

**Prediction convolutions**

Earlier, we earmarked and defined priors for six feature maps of various scales and granularity, viz. those from conv4_3, conv7, conv8_2, conv9_2, conv10_2, and conv11_2.

Then, for each prior at each location on each feature map, we want to predict –

* the offsets (g_c_x, g_c_y, g_w, g_h) for a bounding box.

* a set of n_classes scores for the bounding box, where n_classes represents the total number of object types (including a background class).

To do this in the simplest manner possible, we need two convolutional layers for each feature map –

* a localization prediction convolutional layer with a 3, 3 kernel evaluating at each location (i.e. with padding and stride of 1) with 4 filters for each prior present at the location.

* The 4 filters for a prior calculate the four encoded offsets (g_c_x, g_c_y, g_w, g_h) for the bounding box predicted from that prior.

* a class prediction convolutional layer with a 3, 3 kernel evaluating at each location (i.e. with padding and stride of 1) with n_classes filters for each prior present at the location.

The n_classes filters for a prior calculate a set of n_classes scores for that prior.

![wh5](/images/wh5.png)

All our filters are applied with a kernel size of 3, 3.

## Data Pipeline
**Parse raw data**
* xml_parser() function in main.ipynb is used to process the data in xml format.
* convex hull annotations are converted to bounding box.
* labels given in the form of strings in converted into int by assigning each label an int value

**PyTorch Dataset**

See PascalVOCDataset in main.ipynb

This is a subclass of PyTorch Dataset, used to define our training and test datasets. It needs a __len__ method defined, which returns the size of the dataset, and a __getitem__ method which returns the ith image, bounding boxes of the objects in this image, and labels for the objects in this image.

**Data Transforms**

See transform() in utils.py.

This function applies the following transformations to the images and the objects in them –

* Randomly adjust brightness, contrast, saturation, and hue, each with a 50% chance and in random order.

* With a 50% chance, perform a zoom out operation on the image. This helps with learning to detect small objects. The zoomed out image must be between 1 and 4 times as large as the original. The surrounding space could be filled with the mean of the ImageNet data.

* Randomly crop image, i.e. perform a zoom in operation. This helps with learning to detect large or partial objects. Some objects may even be cut out entirely. Crop dimensions are to be between 0.3 and 1 times the original dimensions. The aspect ratio is to be between 0.5 and 2. Each crop is made such that there is at least one bounding box remaining that has a Jaccard overlap of either 0, 0.1, 0.3, 0.5, 0.7, or 0.9, randomly chosen, with the cropped image. In addition, any bounding boxes remaining whose centers are no longer in the image as a result of the crop are discarded. There is also a chance that the image is not cropped at all.

* With a 50% chance, horizontally flip the image.

* Resize the image to 300, 300 pixels. This is a requirement of the SSD300.

* Convert all boxes from absolute to fractional boundary coordinates. At all stages in our model, all boundary and center-size coordinates will be in their fractional forms.

* Normalize the image with the mean and standard deviation of the ImageNet data that was used to pretrain our VGG base.

**PyTorch DataLoader**

The Dataset described above, PascalVOCDataset, will be used by a PyTorch DataLoader in main.ipynb to create and feed batches of data to the model for training or validation.

Since the number of objects vary across different images, their bounding boxes, labels, and difficulties cannot simply be stacked together in the batch. There would be no way of knowing which objects belong to which image.

Instead, we need to pass a collating function to the collate_fn argument, which instructs the DataLoader about how it should combine these varying size tensors. The simplest option would be to use Python lists.

**Base Convolutions**

See VGGBase in model.py.

Here, we create and apply base convolutions.

The layers are initialized with parameters from a pretrained VGG-16 with the load_pretrained_layers() method.

We're especially interested in the lower-level feature maps that result from conv4_3 and conv7, which we return for use in subsequent stages.

**Auxiliary Convolutions**

See AuxiliaryConvolutions in model.py.

Here, we create and apply auxiliary convolutions.

Use a uniform Xavier initialization for the parameters of these layers.

We're especially interested in the higher-level feature maps that result from conv8_2, conv9_2, conv10_2 and conv11_2, which we return for use in subsequent stages.

**Prediction Convolutions**

See PredictionConvolutions in model.py.

Here, we create and apply localization and class prediction convolutions to the feature maps from conv4_3, conv7, conv8_2, conv9_2, conv10_2 and conv11_2.

These layers are initialized in a manner similar to the auxiliary convolutions.

We also reshape the resulting prediction maps and stack them as discussed. Note that reshaping in PyTorch is only possible if the original tensor is stored in a contiguous chunk of memory.

As expected, the stacked localization and class predictions will be of dimensions 8732, 4 and 8732, 21 respectively.

**Putting it all together**

See SSD300 in model.py.

Here, the base, auxiliary, and prediction convolutions are combined to form the SSD.

There is a small detail here – the lowest level features, i.e. those from conv4_3, are expected to be on a significantly different numerical scale compared to its higher-level counterparts. Therefore, the authors recommend L2-normalizing and then rescaling each of its channels by a learnable value.

**Priors**

See create_prior_boxes() under SSD300 in model.py.

This function creates the priors in center-size coordinates as defined for the feature maps from conv4_3, conv7, conv8_2, conv9_2, conv10_2 and conv11_2, in that order. Furthermore, for each feature map, we create the priors at each tile by traversing it row-wise.

This ordering of the 8732 priors thus obtained is very important because it needs to match the order of the stacked predictions.

**Multibox Loss**

See MultiBoxLoss in model.py.

Two empty tensors are created to store localization and class prediction targets, i.e. ground truths, for the 8732 predicted boxes in each image.

We find the ground truth object with the maximum Jaccard overlap for each prior, which is stored in object_for_each_prior.

We want to avoid the rare situation where not all of the ground truth objects have been matched. Therefore, we also find the prior with the maximum overlap for each ground truth object, stored in prior_for_each_object. We explicitly add these matches to object_for_each_prior and artificially set their overlaps to a value above the threshold so they are not eliminated.

Based on the matches in object_for_each prior, we set the corresponding labels, i.e. targets for class prediction, to each of the 8732 priors. For those priors that don't overlap significantly with their matched objects, the label is set to background.

Also, we encode the coordinates of the 8732 matched objects in object_for_each prior in offset form (g_c_x, g_c_y, g_w, g_h) with respect to these priors, to form the targets for localization. Not all of these 8732 localization targets are meaningful. As we discussed earlier, only the predictions arising from the non-background priors will be regressed to their targets.

The localization loss is the Smooth L1 loss over the positive matches.

Perform Hard Negative Mining – rank class predictions matched to background, i.e. negative matches, by their individual Cross Entropy losses. The confidence loss is the Cross Entropy loss over the positive matches and the hardest negative matches. Nevertheless, it is averaged only by the number of positive matches.

The Multibox Loss is the aggregate of these two losses, combined in the ratio α. In our case, they are simply being added because α = 1.

