# Searching Efficient Semantic Segmentation Architectures via Dynamic Path Selection

## Usage
We provide the best models discovered by our DPS, along with their pre-trained weights, for easy verification and further research.

### Installation
The code is compatible with **Python 3.7**. To install the required dependencies, run:
````bash
pip install -r requirements.txt
````

### Evaluation on Cityscapes validation set
````bash
python eval_squeeze.py --cfg experiments/squeeze/city_val.yaml
````
````bash
python eval_faster.py --cfg experiments/faster/city_val.yaml
````

### Evaluation on CamVid test set
````bash
python eval_squeeze.py --cfg experiments/squeeze/camvid.yaml
````
````bash
python eval_faster.py --cfg experiments/faster/camvid.yaml
````

### Evaluation on Cityscapes test set:
If you want to verify our results on the Cityscapes test set, first generate the `testing results` using the following command:
````bash
python eval_squeeze.py --cfg experiments/squeeze/city_test.yaml
````
Next, compress the generated result folder into a `.zip` file and submit it to the official Cityscapes evaluation server at https://www.cityscapes-dataset.com/ . The final test performance will be evaluated and reported by the server.

