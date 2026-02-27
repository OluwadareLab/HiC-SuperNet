# HiC-SuperNet: Multi-Scale Attention-Based Deep Learning for High-Resolution Chromatin Contact Map Enhancement





## Data Pre-processing
Access the GSE62525 GEO entry for Hi-C data from (Rao et al., 2014) [here](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525). In our work, we made use of the [GM12878](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FGM12878%5Fprimary%5Fintrachromosomal%5Fcontact%5Fmatrices%2Etar%2Egz) primary intrachromosomal, [K562](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FK562%5Fintrachromosomal%5Fcontact%5Fmatrices%2Etar%2Egz) intrachromosomal, [HMEC](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525%5FHMEC%5Fintrachromosomal%5Fcontact%5Fmatrices.tar.gz) intrachromosomal, and [NHEK](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525%5FNHEK%5Fintrachromosomal%5Fcontact%5Fmatrices.tar.gz) intrachromosomal matrices.

 - Set your project's root directory by defining it as a string in `Data/Arg_Parser.py`. For instance, we have root_dir = './Data'.
 - Create a folder named raw within the root directory to hold your unprocessed datasets. Use this command: `mkdir $root_dir/raw`.
 - Download and extract your datasets into the `$root_dir/raw` directory. For GM12878 data, an appropriately named folder will be created containing the contact matrices for every chromosome and available resolution. Refer to the README for additional guidance.

To generate .npz formatted datasets for DiCARN, follow the outlined steps:

### 1. Processing Raw Data
This step will generate a new directory $root_dir/mat/<cell_line_name> containing all chrN_[HR].npz files.

`python3 Read_Data.py -c GM12878`

Mandatory argument:

`-c`: Specify the name of the folder where you saved the unzipped cell line data under $root_dir/raw/<cell_line_name>. In this example, <cell_line_name> is GM12878.
Optional arguments:

`-hr`: Resolution setting. Options include 5kb, 10kb, 25kb, 50kb, 100kb, 250kb, 500kb, and 1mb. Default is set to 10kb.

`-q`: Map quality setting. Choices are MAPQGE30 or MAPQG0. Default is MAPQGE30.

`-n`: Normalization approach. Choose from KRnorm, SQRTVCnorm, or VCnorm. Default is KRnorm.


### 2. Downsampling Data Randomly
This step stores downsampled HR data in $root_dir/mat/<cell_line_name> as chrN_[LR].npz.

`python Downsample.py -hr 10kb -lr 40kb -r 16 -c GM12878`

Arguments:

`-hr`: Resolution from the previous step. Default is 10kb.

`-lr`: Specifies the resolution for [LR] in chrN_[LR].npz. Default is 40kb.

`-r`: Downsampling factor. Default value is 16.

`-c`: Cell line identifier.

### 3. Creating Training, Validation, and Testing Datasets
 - Specify chromosomes for each dataset category in `./Arg_Parser.py` within the `set_dict` dictionary.

 - The example below generates a file in `$root_dir/Data with the name dicarn_10kb40kb_c40_s40_b201_nonpool_train.npz`:

`python Generate.py -hr 10kb -lr 40kb -lrc 100 -s train -chunk 40 -stride 40 -bound 201 -scale 1 -c GM12878`

Arguments:

`-hr`: High resolution for chrN_[HR].npz, which serves as the target data for training. Default is 10kb.

`-lr`: Low resolution for chrN_[LR].npz used as input data during training. Default is 40kb.

`-lrc`: Sets the minimum value in the LR matrix. Default is 100.

`-s`: Type of dataset to generate. Options are train, valid, GM12878_test, K562_test, NHEK_test and HMEC_test. Default is train.

`-chunk`: Defines the submatrix size (nxn). Default is 40.

`-stride`: Set equal to -chunk. Default is 40.

`-bound`: Maximum genomic distance limit. Default is 201.

`-scale`: Determines whether input submatrices should be pooled. Default is 1.

`-c`: Specify the cell line name again.

Note: To proceed with training, ensure that both training and validation files are present in $root_dir/data. Adjust the -s option to produce validation and additional required datasets.
### Congratulations! Your datasets are now ready.
