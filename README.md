<h2>TensorFlow-FlexUNet-Image-Segmentation-Breast-Cancer-INbreast (2025/08/05)</h2>

This is the first experiment of Image Segmentation for Breast Cancer INbreast Singleclass,
 based on our 
TensorFlowFlexUNet (TensorFlow Flexible UNet Image Segmentation Model for Multiclass) 
and a 512x512 pixels 
<a href="https://drive.google.com/file/d/1Nzw2b4W3Y4cJNww3wV9-EtjAaZv2MCWf/view?usp=sharing">
Augmented-INbreast-ImageMask-Dataset.zip</a>.
which was derived by us from <br><br>
<b>INBREAST-SELECTED-IMGS</b> and <b>INBREAST-SELECTED-MSKS</b> <a href="https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/README.md#INbreast">
INbreast  
</a> 
<br><br>
<a href="https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/README.md">
<b>
Breast-Cancer-Segmentation-Datasets
</b>
<br>
Curated collection of datasets for breast cancer segmentation
</a>
<br>
<br>
As demonstrated in <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel">
TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel</a>, 
our Multiclass TensorFlowFlexUNet, which uses categorized masks, can also be applied to single-class image segmentation models. 
This is because it inherently treats the background as one category and your single-class mask data as a second category. 
In essence, your single-class segmentation model will operate with two categorized classes within our Multiclass UNet framework.
<br>
<br>
<b>Acutual Image Segmentation for 512x512 INbreast Infection images</b><br>

As shown below, the inferred masks predicted by our segmentation model trained on the 
PNG dataset appear similar to the ground truth masks, but lack precision in some areas,
<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test/images/14.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test/masks/14.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test_output/14.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test/images/barrdistorted_1001_0.3_0.3_23.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test/masks/barrdistorted_1001_0.3_0.3_23.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test_output/barrdistorted_1001_0.3_0.3_23.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test/images/distorted_0.02_rsigma0.5_sigma40_5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test/masks/distorted_0.02_rsigma0.5_sigma40_5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test_output/distorted_0.02_rsigma0.5_sigma40_5.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used here has been taken from the web-site:<br>
which was derived by us from <br><br>

<b>INBREAST-SELECTED-IMGS</b> and <b>INBREAST-SELECTED-MSKS</b> <a href="https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/README.md#INbreast">
INbreast  
</a> 
<br><br>
<a href="https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/README.md">
<b>
Breast-Cancer-Segmentation-Datasets
</b>
<br>
Curated collection of datasets for breast cancer segmentation
</a>
<br>
<br>
<b><a href="https://biokeanos.com/source/INBreast">INbreast</b></a><br>

The INbreast database is a mammographic database, with images acquired at a Breast Centre, 
located in a Hospital de São João, Breast Centre, Porto, Portugal. INbreast has a total of 115 cases (410 images) 
of which 90 cases are from women with both breasts (4 images per case) and 25 cases are from mastectomy patients 
(2 images per case). Several types of lesions (masses, 
calcifications, asymmetries, and distortions) are included. Accurate contours made by specialists are also provided in XML format. 
<br>
<br>
<b>Webpage:</b><br>
<a href="https://www.kaggle.com/martholi/inbreast">
https://www.kaggle.com/martholi/inbreast
</a> 
<br>
<br>
<b>Licence:</b><br>
Name: Resource specific <br>
<a href="http://medicalresearch.inescporto.pt/breastresearch/index.php/Get_INbreast_Database">
http://medicalresearch.inescporto.pt/breastresearch/index.php/Get_INbreast_Database"
</a> 
<br>
<br>
<b>Publications:</b><br>
<a href="https://pubmed.ncbi.nlm.nih.gov/22078258/">Nbreast: toward a full-field digital mammographic database PubMed</a>
<br>
<br>

<h3>
<a id="2">
2 INbreast ImageMask Dataset
</a>
</h3>
 If you would like to train this INbreast Segmentation model by yourself,
 please download the dataset from the google drive 
<a href="https://drive.google.com/file/d/1Nzw2b4W3Y4cJNww3wV9-EtjAaZv2MCWf/view?usp=sharing">
Augmented-INbreast-ImageMask-Dataset.zip</a>.
<br>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─INbreast
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>INbreast Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/INbreast/INbreast_Statistics.png" width="512" height="auto"><br>
<br>

On the derivation of the 512x512 pixels augmented dataset, please refer to the following Python scripts:<br>
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
<br>

As shown above, the number of images of train and valid datasets is not so large to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/INbreast/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/INbreast/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained INbreast TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/INbreast/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/INbreast and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 2

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for INbreast 1+3 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
; 1+1 classes
; RGB colors   cancer:white     
rgb_map = {(0,0,0):0,(255,255,255):1 }


</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInfereuncer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/INbreast/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 18,19,20)</b><br>
<img src="./projects/TensorFlowFlexUNet/INbreast/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 38,39,40)</b><br>
<img src="./projects/TensorFlowFlexUNet/INbreast/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was stopped at epoch 40 by EearlyStopping callback.<br><br>
<img src="./projects/TensorFlowFlexUNet/INbreast/asset/train_console_output_at_epoch40.png" width="920" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/INbreast/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/INbreast/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/INbreast/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/INbreast/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/INbreast</b> folder,<br>
and run the following bat file to evaluate TensorFlowFlexUNet model for INbreast.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/INbreast/asset/evaluate_console_output_at_epoch40.png" width="920" height="auto">
<br><br>

<a href="./projects/TensorFlowFlexUNet/INbreast/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this INbreast/test was low and dice_coef_multiclass 
high as shown below.
<br>
<pre>
categorical_crossentropy,0.0117
dice_coef_multiclass,0.9969
</pre>
<br>

<h3>
5 Inference
</h3>
Please move <b>./projects/TensorFlowFlexUNet/INbreast</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for INbreast.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/INbreast/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/INbreast/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/INbreast/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 512x512 pixels</b><br>
<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test/images/42.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test/masks/42.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test_output/42.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test/images/barrdistorted_1002_0.3_0.3_45.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test/masks/barrdistorted_1002_0.3_0.3_45.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test_output/barrdistorted_1002_0.3_0.3_45.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test/images/barrdistorted_1003_0.3_0.3_8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test/masks/barrdistorted_1003_0.3_0.3_8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test_output/barrdistorted_1003_0.3_0.3_8.png" width="320" height="auto"></td>
</tr>



<tr>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test/images/barrdistorted_1003_0.3_0.3_22.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test/masks/barrdistorted_1003_0.3_0.3_22.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test_output/barrdistorted_1003_0.3_0.3_22.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test/images/barrdistorted_1004_0.3_0.3_8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test/masks/barrdistorted_1004_0.3_0.3_8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test_output/barrdistorted_1004_0.3_0.3_8.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test/images/distorted_0.02_rsigma0.5_sigma40_5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test/masks/distorted_0.02_rsigma0.5_sigma40_5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/INbreast/mini_test_output/distorted_0.02_rsigma0.5_sigma40_5.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. INbreast: toward a full-field digital mammographic database </b><br>
Inês C Moreira, Igor Amaral, Inês Domingues, António Cardoso, Maria João Cardoso, Jaime S Cardoso<br>
<a href="https://pubmed.ncbi.nlm.nih.gov/22078258/">https://pubmed.ncbi.nlm.nih.gov/22078258/</a>

<br>
<br><br>
<b>2. Breast-Cancer-Segmentation-Datasets</b>
<br>
Curated collection of datasets for breast cancer segmentation
</a>
<br>
pablogiaccaglia<br>

<a href="https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/README.md">
Breast-Cancer-Segmentation-Datasets
</a>
