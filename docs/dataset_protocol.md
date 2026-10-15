# HAM10000 Dataset Protocol

## Source and permitted use

The project uses HAM10000 from Harvard Dataverse, DOI `10.7910/DVN/DBW86T`.
The downloaded version states a CC BY-NC 4.0 license. This academic project must
attribute the dataset creators and must not use or distribute the data for a
commercial purpose.

Dataset citation:

> Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a
> large collection of multi-source dermatoscopic images of common pigmented skin
> lesions. *Scientific Data, 5*, 180161.

## Audit result

The official metadata contains 10,015 images, each matched to a local JPEG file.
No duplicate `image_id` records were found.

| Class | Meaning | Images |
|---|---|---:|
| akiec | Actinic keratoses / intraepithelial carcinoma | 327 |
| bcc | Basal cell carcinoma | 514 |
| bkl | Benign keratosis-like lesions | 1,099 |
| df | Dermatofibroma | 115 |
| mel | Melanoma | 1,113 |
| nv | Melanocytic nevi | 6,705 |
| vasc | Vascular lesions | 142 |

`nv` represents 66.95% of all images, while `df` represents 1.15%; this confirms
severe class imbalance.

## Split protocol

The official metadata includes `lesion_id`, but no `patient_id`. It contains 7,470
unique lesions, and 4,501 images belong to repeated lesion groups. Therefore, a
random image-level split would be unsafe.

The split script uses two-stage `StratifiedGroupKFold`, grouping by `lesion_id`
with seed 42. It creates approximately 71.4% training, 14.3% validation, and
14.3% test data while ensuring each lesion appears in exactly one partition.

| Split | Images | Unique lesions |
|---|---:|---:|
| Train | 7,152 | 5,335 |
| Validation | 1,432 | 1,068 |
| Test | 1,431 | 1,067 |

The script asserts that no `lesion_id` overlaps across the splits. It does **not**
justify a claim of a patient-wise split because patient identifiers are absent.
